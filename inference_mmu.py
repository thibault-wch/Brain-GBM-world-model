import os
import numpy as np
from typing import Union
from PIL import Image
from einops import rearrange
from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig

import torch
from peft import LoraConfig, get_peft_model, PeftModel
from accelerate.logging import get_logger

# Fix tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from models import Showo2Qwen2_5, omni_attn_mask_naive
from models.misc import get_text_tokenizer
from utils import flatten_omega_conf, get_hyper_params, path_to_llm_name, set_seed
from datasets.glioma_dataset import create_medical_dataloader
from transport import create_transport


# ==========================================
# Global Settings & Initialization
# ==========================================

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weight_type = torch.bfloat16
set_seed(10)

save_name = 'all_new'
data_mode = 'external'
lora_path = '/autodl-fs/data/experiments/glioma-class-final/checkpoint-79000/unwrapped_model/'

# Ensure consistent label mapping between GT and Pred
LABEL_TO_ID = {
    "<SURGERY>": 0,
    "<CRT>": 1,
    "<RT>": 2,
    "<TMZ>": 3,
    "<AM>": 4,
}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

logger = get_logger(__name__, log_level="INFO")


def answer_to_index(answer_text: str) -> int:
    """
    Extracts the label token from text and maps it to an integer (0-4).
    Returns -1 if no valid label is found to prevent evaluation crashes.
    """
    for token, idx in LABEL_TO_ID.items():
        if token in answer_text:
            return idx
    return -1


def get_config(path: str) -> DictConfig:
    """Loads a YAML configuration file."""
    return OmegaConf.load(path)


if __name__ == '__main__':
    # Load config
    config = get_config('/root/remoteproject/glioma/configs/showo2_1.5b_stage_2_a.yaml')
    preproc_config = config.dataset.preprocessing

    # Initialize VAE
    if config.model.vae_model.type == 'wan21':
        from models import WanVAE

        vae_model = WanVAE(
            vae_pth=config.model.vae_model.pretrained_model_path,
            dtype=weight_type,
            device=device
        )
    else:
        raise NotImplementedError("Only wan21 VAE is supported currently.")

    # Initialize Tokenizer & Model
    text_tokenizer, showo_token_ids = get_text_tokenizer(
        config.model.showo.llm_model_path,
        add_showo_tokens=True,
        return_showo_token_ids=True,
        llm_name=path_to_llm_name[config.model.showo.llm_model_path]
    )
    config.model.showo.llm_vocab_size = len(text_tokenizer)
    logger.info(f"Config Vocab Size: {config.model.showo.llm_vocab_size}")

    model = Showo2Qwen2_5(**config.model.showo)
    load_path = config.model.showo.pretrained_model_path or '/autodl-fs/data/pytorch_model.bin'

    from transport.utils import convert_qwen2_to_qwen2_dual

    state_dict = convert_qwen2_to_qwen2_dual(
        torch.load(load_path, map_location="cpu"),
        config.model.showo.share_layer_num[0],
        config.model.showo.total_layer_num
    )
    model.load_state_dict(state_dict, strict=False)
    del state_dict

    model.reset_vocbulary(text_tokenizer)

    logger.info(f"Loading LoRA from: {lora_path}")
    model = PeftModel.from_pretrained(model, lora_path)

    model.to(device=device, dtype=weight_type)
    model.eval()

    try:
        model.set_adapter("default")
        logger.info("Adapter 'default' successfully set.")
    except Exception as e:
        logger.warning(f"Failed to set adapter: {e}")

    model.enable_adapter_layers()

    # Hyperparameters & Sequence Length Calculation
    spatial_size_my = config.dataset.params.spatial_size
    num_image_tokens_my = int((spatial_size_my[0] // 16) * (spatial_size_my[1] // 16) * (spatial_size_my[2] // 4))
    max_seq_len_my = num_image_tokens_my + 200

    # Include time embedding if configured
    if config.model.showo.add_time_embeds:
        preproc_config.num_t2i_image_tokens += 1
        preproc_config.num_mmu_image_tokens += 1
        preproc_config.num_video_tokens += 1
        num_image_tokens_my += 1
        max_seq_len_my += 1

    # Dataloader Setup
    test_dataloader_mmu = create_medical_dataloader(
        root="/root/autodl-tmp/dataset",
        batch_size=config.training.batch_size_mmu,
        text_tokenizer=text_tokenizer,
        showo_token_ids=showo_token_ids,
        spatial_size=spatial_size_my,
        num_image_tokens=num_image_tokens_my,
        max_seq_len=max_seq_len_my,
        mode=data_mode,
        is_captioning=True,
        use_seg_mask=True,
        drop_last=False,
        shuffle=False,
    )

    transport = create_transport(
        path_type=config.transport.path_type,
        prediction=config.transport.prediction,
        loss_weight=config.transport.loss_weight,
        train_eps=config.transport.train_eps,
        sample_eps=config.transport.sample_eps,
        snr_type=config.transport.snr_type,
        do_shift=config.transport.do_shift,
        seq_len=preproc_config.num_t2i_image_tokens,
    )

    # Evaluation Loop Preparation
    gt_indices = []
    pred_indices = []

    out_dir = f'/root/autodl-tmp/chwang/experiments/results/mmu_und_output/{data_mode}/{save_name}'
    os.makedirs(f'{out_dir}/recons_xt/', exist_ok=True)
    log_file_path = f'{out_dir}/log.txt'


    # Define processing function OUTSIDE the loop for efficiency
    @torch.no_grad()
    def prepare_latents_and_labels(pixel_values, data_types, image_masks):
        if len(pixel_values.shape) == 4:
            pixel_values = pixel_values.unsqueeze(2)

        image_latents = vae_model.sample(pixel_values, deterministic=True)
        if pixel_values.shape[2] == 1:
            image_latents = image_latents.squeeze(2)

        t_list, xt_list, ut_list, masks = [], [], [], []

        for i, tp in enumerate(data_types):
            max_t0 = config.training.und_max_t0 if tp in ['mmu', 'mmu_vid'] else None
            t_val, x0, x1 = transport.sample(image_latents[i][None], max_t0)
            t_val, xt_val, ut_val = transport.path_sampler.plan(t_val, x0, x1)

            t_list.append(t_val)
            xt_list.append(xt_val)
            ut_list.append(ut_val)

            if tp in ['mmu', 'mmu_vid'] and config.training.und_max_t0 == 1.0:
                masks.append(image_masks[i][None] * 0.0)
            else:
                masks.append(image_masks[i][None])

        t_tensor = torch.stack(t_list, dim=0).squeeze(-1)
        xt_tensor = torch.cat(xt_list, dim=0)

        final_masks = torch.cat(masks, dim=0) if masks else image_masks
        return image_latents, t_tensor, xt_tensor, final_masks


    # Start Inference Loop
    # Open file once before loop to reduce I/O overhead
    with open(log_file_path, 'a+') as log_file:
        for idx, batch in enumerate(tqdm(test_dataloader_mmu, desc="Evaluating")):

            text_tokens = batch['text_tokens'].to(device)
            pixel_values = batch['images'].to(device=device, dtype=weight_type)
            image_masks = batch['image_masks'].to(device)
            modality_positions = batch['modality_positions'].to(device)
            texts = batch['texts']
            data_types = batch['data_type']

            # Handle interleaved data shape adjustments
            if data_types[0] == 'interleaved_data':
                b, n = pixel_values.shape[:2]
                pixel_values = rearrange(pixel_values, "b n c h w -> (b n) c h w")
                data_types = data_types * n

            # Cond processing (Not explicitly used downstream in the loop, kept for structural integrity)
            if data_types[0] == 't2i':
                pixel_values_cond = batch['images_cond'].to(device=device, dtype=weight_type)
                if len(pixel_values_cond.shape) == 4:
                    pixel_values_cond = pixel_values_cond.unsqueeze(2)
                _ = vae_model.sample(pixel_values_cond)  # image_latents_cond

            image_latents, t, xt, final_masks = prepare_latents_and_labels(pixel_values, data_types, image_masks)

            block_mask = omni_attn_mask_naive(
                text_tokens.size(0),
                text_tokens.size(1),
                modality_positions,
                device
            ).to(weight_type)

            with torch.no_grad():
                # Note: seg_mask and probs are ignored here if not needed
                _, _, output_tokens = model.mmu_generate(
                    text_tokens=text_tokens,
                    text_tokenizer=text_tokenizer,
                    image_latents=image_latents,
                    attention_mask=block_mask,
                    modality_positions=modality_positions,
                    max_seq_len=text_tokens.size(1),
                    spatial_size_my=spatial_size_my,
                    device=device,
                    max_new_tokens=1,
                    temperature=1.0,
                    top_k=None,
                    t=t,
                    eos_token=text_tokenizer.eos_token_id,
                )

            if not isinstance(output_tokens, torch.Tensor):
                output_tokens = torch.tensor(output_tokens, dtype=torch.long, device=device)
            else:
                output_tokens = output_tokens.long().to(device)

            if output_tokens.ndim == 1:
                output_tokens = output_tokens.unsqueeze(0)

            # Add Qwen tokenizer vocabulary offset
            output_tokens += 151669

            decoded_text = text_tokenizer.batch_decode(output_tokens, skip_special_tokens=False)

            # Label Extraction with Fallback
            pred_idx = answer_to_index(decoded_text[0])
            gt_idx = answer_to_index(texts[0][1])

            if pred_idx == -1 or gt_idx == -1:
                logger.warning(f"Label parse failure. Pred: '{decoded_text[0]}', GT: '{texts[0][1]}'")

            pred_indices.append(pred_idx)
            gt_indices.append(gt_idx)

            # Logging formatted output
            piece = f"[Q] {texts[0][0]}\n[A] {decoded_text[0]}\t [GT] {texts[0][1]}\n"
            log_file.write(piece)
            log_file.flush()  # Ensure data is saved iteratively

            # Print to console (optional, can be removed to clean up stdout)
            print(f"Batch {idx}: {piece.strip()}")

    # Save Metrics & Post-processing
    txt_dir = f"{out_dir}/txt_answers"
    os.makedirs(txt_dir, exist_ok=True)

    gt_txt_path = os.path.join(txt_dir, "answers_index_gt.txt")
    pred_txt_path = os.path.join(txt_dir, "answers_index_pred.txt")

    np.savetxt(gt_txt_path, np.array(gt_indices, dtype=int), fmt="%d")
    np.savetxt(pred_txt_path, np.array(pred_indices, dtype=int), fmt="%d")

    logger.info(f"GT labels saved to: {gt_txt_path}")
    logger.info(f"Pred labels saved to: {pred_txt_path}")

    