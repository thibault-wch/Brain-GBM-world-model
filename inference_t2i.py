import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"

from typing import Union
from PIL import Image
from einops import rearrange
import wandb
import torch
from tqdm import tqdm
from accelerate.logging import get_logger
from omegaconf import OmegaConf, DictConfig
from peft import LoraConfig, get_peft_model, PeftModel

from models.modeling_showo2_qwen2_5 import Showo2Qwen2_5
from models import omni_attn_mask, omni_attn_mask_naive
from models.misc import get_text_tokenizer, prepare_gen_input
from utils import flatten_omega_conf, denorm, get_hyper_params, path_to_llm_name, load_state_dict, set_seed, \
    collect_lora_targets_full, collect_modules_to_save_for_full_ft, add_default_before_last
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from datasets.utils import image_transform, resize_and_pad_image, to_tensor_and_normalize
from transport.utils import convert_qwen2_to_qwen2_dual
from transport import Sampler, create_transport
from datasets.glioma_dataset import create_medical_dataloader, MedicalPairImageTextDataset
from datasets.mixed_dataloader import MixedDataLoader

# -------------------------------------------------------------------
# Basic Information & Configurations
# -------------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weight_type = torch.bfloat16
set_seed(10)

save_name = 'all_new2'
data_mode = 'test'
lora_path = '/autodl-fs/data/experiments/glioma-class-final/checkpoint-70000/unwrapped_model/'
config_path = '/root/remoteproject/glioma/configs/showo2_1.5b_stage_2_a.yaml'
dataset_root = '/root/autodl-tmp/dataset'
output_dir_base = f'/root/autodl-tmp/chwang/experiments/results/mmu_t2i_output/{data_mode}/{save_name}'

logger = get_logger(__name__, log_level="INFO")


def get_config(path: str) -> DictConfig:
    """Load a single YAML file into an OmegaConf DictConfig."""
    return OmegaConf.load(path)


if __name__ == '__main__':
    config = get_config(config_path)
    preproc_config = config.dataset.preprocessing
    resume_wandb_run = config.wandb.resume

    # Init VQ model for processing images into discrete tokens
    if config.model.vae_model.type == 'wan21':
        from models import WanVAE

        vae_model = WanVAE(vae_pth=config.model.vae_model.pretrained_model_path, dtype=weight_type, device=device)
    else:
        raise NotImplementedError("VAE model type not supported.")

    # Init Show-o model
    text_tokenizer, showo_token_ids = get_text_tokenizer(
        config.model.showo.llm_model_path,
        add_showo_tokens=True,
        return_showo_token_ids=True,
        llm_name=path_to_llm_name[config.model.showo.llm_model_path]
    )
    config.model.showo.llm_vocab_size = len(text_tokenizer)

    model = Showo2Qwen2_5(**config.model.showo)
    load_path = '/autodl-fs/data/pytorch_model.bin' if config.model.showo.pretrained_model_path is None else config.model.showo.pretrained_model_path

    state_dict = convert_qwen2_to_qwen2_dual(
        torch.load(load_path, map_location="cpu"),
        config.model.showo.share_layer_num[0],
        config.model.showo.total_layer_num
    )

    model.load_state_dict(state_dict, strict=False)
    del state_dict

    model.reset_vocbulary(text_tokenizer)

    # Load LoRA
    logger.info(f"Loading LoRA from: {lora_path}")
    model = PeftModel.from_pretrained(model, lora_path)

    model.to(weight_type)
    model.to(device)
    model.eval()

    logger.info(f"Active adapters: {getattr(model, 'active_adapters', None)}")
    try:
        model.set_adapter("default")
        logger.info("Successfully set adapter to 'default'.")
    except Exception as e:
        logger.error(f"Failed to set adapter: {e}")

    model.enable_adapter_layers()

    # Spatial configuration
    spatial_size_my = config.dataset.params.spatial_size
    num_image_tokens_my = int((spatial_size_my[0] // 16) * (spatial_size_my[1] // 16) * (spatial_size_my[2] // 4))
    max_seq_len_my = num_image_tokens_my + 200
    logger.info(f"num_image_tokens_my: {num_image_tokens_my}, max_seq_len_my: {max_seq_len_my}")

    # Fetch hyper parameters
    hyper_params = get_hyper_params(config, text_tokenizer, showo_token_ids)
    num_t2i_image_tokens, num_mmu_image_tokens, num_video_tokens, max_seq_len, max_text_len, image_latent_dim, \
        patch_size, latent_width, latent_height, pad_id, bos_id, eos_id, boi_id, eoi_id, bov_id, eov_id, img_pad_id, \
        vid_pad_id, guidance_scale = hyper_params

    logger.info(f"Add time embeds: {config.model.showo.add_time_embeds}")

    # Process time embeddings logic
    if config.model.showo.add_time_embeds:
        config.dataset.preprocessing.num_t2i_image_tokens += 1
        config.dataset.preprocessing.num_mmu_image_tokens += 1
        config.dataset.preprocessing.num_video_tokens += 1
        num_image_tokens_my += 1
        max_seq_len_my += 1

    # Dataloader setup
    test_dataloader_t2i = create_medical_dataloader(
        root=dataset_root,
        batch_size=config.training.batch_size_mmu,
        text_tokenizer=text_tokenizer,
        showo_token_ids=showo_token_ids,
        spatial_size=spatial_size_my,
        num_image_tokens=num_image_tokens_my,
        max_seq_len=max_seq_len_my,
        mode=data_mode,
        is_captioning=False,
        use_seg_mask=True,
        drop_last=False,
        shuffle=False,
    )

    # Generate output directories
    sub_dirs = ['input_cond', 'gt', 'predicted', 'seg', 'recons_xt']
    for sub_dir in sub_dirs:
        os.makedirs(os.path.join(output_dir_base, sub_dir), exist_ok=True)

    # Setup transport & sampler
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
    sampler = Sampler(transport)


    @torch.no_grad()
    def prepare_latents_and_labels(pixel_values: torch.Tensor, data_type: list, image_masks: torch.Tensor):
        """Prepare latent variables, time steps, and noisy images."""
        if config.model.vae_model.type == 'wan21':
            if len(pixel_values.shape) == 4:
                pixel_values = pixel_values.unsqueeze(2)

            image_latents = vae_model.sample(pixel_values)
            recons_images = vae_model.batch_decode(image_latents)

            if pixel_values.shape[2] == 1:
                image_latents = image_latents.squeeze(2)
                recons_images = recons_images.squeeze(2)
        else:
            raise NotImplementedError("Only wan21 VAE is supported.")

        t_list, xt_list, ut_list, masks = [], [], [], []
        for i, tp in enumerate(data_type):
            t, x0, x1 = transport.sample(image_latents[i][None], config.training.und_max_t0)
            t, xt, ut = transport.path_sampler.plan(t, x0, x1)
            alpha_t = transport.path_sampler.compute_alpha_t(t)

            t_list.append(t)
            xt_list.append(xt)
            ut_list.append(ut)

            if tp in ['mmu', 'mmu_vid'] and config.training.und_max_t0 == 1.0:
                masks.append(image_masks[i][None] * 0.0)
            else:
                masks.append(image_masks[i][None])

        t = torch.stack(t_list, dim=0).squeeze(-1)
        xt = torch.cat(xt_list, dim=0)
        ut = torch.cat(ut_list, dim=0)
        masks = torch.cat(masks, dim=0) if masks else image_masks
        recons_images = vae_model.batch_decode(xt)

        return xt, t, ut, recons_images, masks, alpha_t


    # Main evaluation loop
    for idx, batch in enumerate(tqdm(test_dataloader_t2i, desc="Processing batches")):
        # Target specific patients for testing
        if batch['pid'][0] not in ['Patient-051', 'Patient-059']:
            logger.debug(f"Skipping {batch['pid'][0]}")
            continue

        pid = str(batch['pid'][0]) + str(batch['file_id'][0])
        text_tokens = batch['text_tokens'].to(device)
        pixel_values = batch['images'].to(device).to(weight_type)

        if batch['data_type'][0] == 'interleaved_data':
            b, n = pixel_values.shape[:2]
            pixel_values = rearrange(pixel_values, "b n c h w -> (b n) c h w")
            batch['data_type'] = batch['data_type'] * n

        # Process conditional inputs
        image_latents_cond = None
        if batch['data_type'][0] == 't2i':
            pixel_values_cond = batch['images_cond'].to(device).to(weight_type)
            if len(pixel_values_cond.shape) == 4:
                pixel_values_cond = pixel_values_cond.unsqueeze(2)
            image_latents_cond = vae_model.sample(pixel_values_cond)

        image_masks = batch['image_masks'].to(device)
        modality_positions = batch['modality_positions'].to(device)
        texts = batch['texts']

        # Extract latents
        image_latents, t, _, recons_images, image_masks, alpha_t = prepare_latents_and_labels(
            pixel_values=pixel_values,
            data_type=batch['data_type'],
            image_masks=image_masks
        )

        # Define latent space dimensions dynamically
        latent_depth = spatial_size_my[2] // 4
        latent_height_dim = spatial_size_my[0] // 16
        latent_width_dim = spatial_size_my[1] // 16

        z = torch.randn((
            len(text_tokens),
            image_latent_dim,
            latent_depth,
            latent_height_dim * patch_size,
            latent_width_dim * patch_size
        ), dtype=torch.bfloat16, device=device)

        # Generate Attention Block Mask
        block_mask = omni_attn_mask_naive(
            text_tokens.size(0),
            max_seq_len_my,
            modality_positions,
            device
        ).to(weight_type)

        model_kwargs = dict(
            text_tokens=text_tokens,
            attention_mask=block_mask,
            modality_positions=modality_positions,
            output_hidden_states=True,
            max_seq_len=max_seq_len_my,
            guidance_scale=guidance_scale,
            image_latents_cond=image_latents_cond
        )

        sample_fn = sampler.sample_ode(
            sampling_method=config.transport.sampling_method,
            num_steps=config.transport.num_inference_steps,
            atol=config.transport.atol,
            rtol=config.transport.rtol,
            reverse=config.transport.reverse,
            time_shifting_factor=config.transport.time_shifting_factor
        )

        with torch.no_grad():
            samples = sample_fn(z, model.t2i_generate, **model_kwargs)[-1]

        # Decode output
        if config.model.vae_model.type == 'wan21':
            images = vae_model.batch_decode(samples)
            images = images.squeeze(2)
        else:
            raise NotImplementedError

        # Console logging
        if batch['data_type'][0] == 't2i':
            logger.info(
                f"Cond shape: {batch['images_cond'][0].shape} | Target shape: {batch['images'][0].shape} | Output shape: {images.shape} | Recons shape: {recons_images.shape}")

        logger.info(f"Processed texts: {texts}")

        # Save predictions
        save_path = os.path.join(output_dir_base, f"predicted/{pid}.pt")
        torch.save(images.cpu(), save_path)
        logger.info(f"Saved inference for {pid} to {save_path}")