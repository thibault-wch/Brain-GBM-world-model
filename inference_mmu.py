# coding=utf-8
# Copyright 2025 NUS Show Lab.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Union
os.environ["TOKENIZERS_PARALLELISM"] = "true"
from PIL import Image
from einops import rearrange
import wandb
import torch
from tqdm import tqdm
from accelerate.logging import get_logger
from models import Showo2Qwen2_5, omni_attn_mask, omni_attn_mask_naive
from models.misc import get_text_tokenizer, prepare_gen_input
from utils import  logits_to_mask_np,save_numpy,flatten_omega_conf, denorm, get_hyper_params, path_to_llm_name, load_state_dict, set_seed, collect_lora_targets_full,collect_modules_to_save_for_full_ft,add_default_before_last
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from datasets.utils import image_transform, resize_and_pad_image, to_tensor_and_normalize
from transport.utils import convert_qwen2_to_qwen2_dual
from omegaconf import OmegaConf, DictConfig
from peft import LoraConfig, get_peft_model,PeftModel
# from train_stage_two import prepare_latents_and_labels
from transport import Sampler, create_transport
# from eval_5_way import  FiveWayEval


################.<  basic information   >.#########################

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weight_type=torch.bfloat16
set_seed(10)
save_name='all_new'
# save_name='all_new'
data_mode='external'
# lora_path='/autodl-fs/data/experiments/glioma-class-final-wo-seg/checkpoint-43000/unwrapped_model/'
lora_path='/autodl-fs/data/experiments/glioma-class-final/checkpoint-79000/unwrapped_model/'

# lora_path= '/data_hdd/chwang/experiments/glioma/glioma-class-seg-1112_lr3/checkpoint-22500/unwrapped_model/adapter_model.bin'
# lora_path= '/data_hdd/chwang/experiments/glioma/glioma-class-seg/checkpoint-1000/unwrapped_model/adapter_model.bin'

# lora_path='/data_hdd/chwang/experiments/glioma/glioma-template-seg-5-classification/checkpoint-125000/unwrapped_model/adapter_model.bin'
###################################################################

from datasets.glioma_dataset import create_medical_dataloader,MedicalPairImageTextDataset
from datasets.mixed_dataloader import MixedDataLoader


# 五类标签的映射：一定要保证 gt 和 pred 用的是同一套
LABEL_TO_ID = {
    "<SURGERY>": 0,
    "<CRT>": 1,
    "<RT>": 2,
    "<TMZ>": 3,
    "<AM>": 4,
}

ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}


def answer_to_index(answer_text: str) -> int:
    """
    从模型输出 / GT 文本中找到属于五类标签之一的 token，
    并转换成 0~4 的整数编码。

    要求 answer_text 至少包含其中一个标签；如果有多个，取第一个匹配到的。
    """
    for token, idx in LABEL_TO_ID.items():
        if token in answer_text:
            return idx
    raise ValueError(f"无法在答案中识别标签: {answer_text!r}")



# set_seed(10)

logger = get_logger(__name__, log_level="INFO")


def get_config(path: str) -> DictConfig:
    """Load a single YAML file into an OmegaConf DictConfig."""
    return OmegaConf.load(path)



if __name__ == '__main__':

    config=get_config('/root/remoteproject/glioma/configs/showo2_1.5b_stage_2_a.yaml')
    preproc_config = config.dataset.preprocessing
    resume_wandb_run = config.wandb.resume
    # evaluator = FiveWayEval()  

    # wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}

    # wandb.init(
    #     project="demo",
    #     name=config.experiment.name,
    #     config=wandb_config,
    # )

    # VQ model for processing image into discrete tokens
    if config.model.vae_model.type == 'wan21':
        from models import WanVAE
        vae_model = WanVAE(vae_pth=config.model.vae_model.pretrained_model_path, dtype=weight_type, device=device)
    else:
        raise NotImplementedError

    # Initialize Show-o model
    text_tokenizer, showo_token_ids = get_text_tokenizer(config.model.showo.llm_model_path, 
                                                            add_showo_tokens=True,
                                                            return_showo_token_ids=True,
                                                            llm_name=path_to_llm_name[config.model.showo.llm_model_path])
    config.model.showo.llm_vocab_size = len(text_tokenizer)
    print(f'config voc size: {config.model.showo.llm_vocab_size}')

    
    model = Showo2Qwen2_5(**config.model.showo)
    load_path = '/autodl-fs/data/pytorch_model.bin' if config.model.showo.pretrained_model_path is None else config.model.showo.pretrained_model_path
    state_dict = convert_qwen2_to_qwen2_dual(torch.load(load_path, map_location="cpu"),
                                                    config.model.showo.share_layer_num[0],
                                                    config.model.showo.total_layer_num)
        
    model.load_state_dict(state_dict, strict=False)
    del state_dict

    model.reset_vocbulary(text_tokenizer)
    print(f'load LoRA from: {lora_path}')
    model = PeftModel.from_pretrained(model, lora_path)

   
    model.to(weight_type)
    model.to(device)
    model.eval()
    print("active_adapters:", getattr(model, "active_adapters", None))
    try:
        model.set_adapter("default")   # 确保启用名为 default 的适配器
        print("set_adapter('default') ok")
    except Exception as e:
        print("set_adapter failed:", e)

    model.enable_adapter_layers()

    spatial_size_my=config.dataset.params.spatial_size
    
    num_image_tokens_my=int((spatial_size_my[0]//16)*(spatial_size_my[1]//16)*(spatial_size_my[2]//4))
    max_seq_len_my= num_image_tokens_my+200
    print(f'num_image_tokens_my: {num_image_tokens_my}, max_seq_len_my: {max_seq_len_my}')


    num_t2i_image_tokens, num_mmu_image_tokens, num_video_tokens, max_seq_len, max_text_len, image_latent_dim, patch_size, latent_width, \
    latent_height, pad_id, bos_id, eos_id, boi_id, eoi_id, bov_id, eov_id, img_pad_id, vid_pad_id, guidance_scale \
        = get_hyper_params(config, text_tokenizer, showo_token_ids)
    
    # for time embedding
    if config.model.showo.add_time_embeds:
        # we prepend the time embedding to vision tokens
        config.dataset.preprocessing.num_t2i_image_tokens += 1
        config.dataset.preprocessing.num_mmu_image_tokens += 1
        config.dataset.preprocessing.num_video_tokens += 1
        num_image_tokens_my+=1
        max_seq_len_my+=1


    temperature = 1.0  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    top_k = None  # retain only the top_k most likely tokens, clamp others to have 0 probability


    test_dataloader_mmu =  create_medical_dataloader(
    root="/root/autodl-tmp/dataset",
    batch_size=config.training.batch_size_mmu,
    text_tokenizer=text_tokenizer,
    showo_token_ids=showo_token_ids,
    spatial_size=spatial_size_my,
    num_image_tokens=num_image_tokens_my,
    max_seq_len=max_seq_len_my,  
    mode=data_mode,
    is_captioning=True,    # t2i
    use_seg_mask=True,
     drop_last=False,
        shuffle=False,
    )
        # 用来收集所有样本的 GT / Pred 编码
    gt_indices = []
    pred_indices = []


    responses = ['' for j in range(len(test_dataloader_mmu))]
    model.eval()
    os.makedirs(f'/root/autodl-tmp/chwang/experiments/results/mmu_und_output/{data_mode}/{save_name}/recons_xt/', exist_ok=True)

    # sys_prompt_ids = text_tokenizer("system\nYou are a helpful assistant.<|im_end|>",
                                    # add_special_tokens=False)['input_ids']
    # role_a = text_tokenizer("\n<|im_start|>user\n", add_special_tokens=False)['input_ids']
    # role_b = text_tokenizer("\n<|im_start|>assistant\n", add_special_tokens=False)['input_ids']

    for idx,batch in enumerate(test_dataloader_mmu):

        text_tokens = batch['text_tokens'].to(device)
        # print(batch['text_labels'])
        pid=batch['pid'][0]+batch['file_id'][0]
        seg_true=batch['seg_masks'][0]
        if batch['text_labels'][0] is not None:
            text_labels = batch['text_labels'].to(device)
        pixel_values = batch['images'].to(device).to(weight_type)
        if batch['data_type'][0] == 'interleaved_data':
            b, n = pixel_values.shape[:2]
            pixel_values = rearrange(pixel_values, "b n c h w -> (b n) c h w")
            batch['data_type'] = batch['data_type'] * n
        else:
            b, n = 0, 0
        if batch['data_type'][0]=='t2i':
            pixel_values_cond=batch['images_cond'].to(device).to(weight_type)
            if len(pixel_values_cond.shape) == 4:
                pixel_values_cond = pixel_values_cond.unsqueeze(2)
            image_latents_cond = vae_model.sample(pixel_values_cond)
        else:
            image_latents_cond = None

        # text_masks = batch['text_masks'].to(device)
        image_masks = batch['image_masks'].to(device)
        # seg_masks= batch['seg_masks'].to(device)
        modality_positions = batch['modality_positions'].to(device)
        texts=batch['texts']
        # prepare image latents and labels
        # default: 1000 steps, linear noise schedule
        transport = create_transport(
            path_type=config.transport.path_type,
            prediction=config.transport.prediction,
            loss_weight=config.transport.loss_weight,
            train_eps=config.transport.train_eps,
            sample_eps=config.transport.sample_eps,
            snr_type=config.transport.snr_type,
            do_shift=config.transport.do_shift,
            seq_len=preproc_config.num_t2i_image_tokens,
        )  # default: velocity;

        @torch.no_grad()
        def prepare_latents_and_labels(
                    pixel_values: Union[torch.FloatTensor, torch.LongTensor],
                    data_type,
                    shape,
                    image_masks,
                    modality_positions
            ):
            # print(pixel_values.shape)

            if config.model.vae_model.type == 'wan21':
                if len(pixel_values.shape) == 4:
                    pixel_values = pixel_values.unsqueeze(2)
                # print(f'pixel shape : {pixel_values.shape}')
                image_latents = vae_model.sample(pixel_values,deterministic=True)
                recons_images = vae_model.batch_decode(image_latents)
                if pixel_values.shape[2] == 1:
                    image_latents = image_latents.squeeze(2)
                    recons_images = recons_images.squeeze(2)
            else:
                raise NotImplementedError

            # c, h, w = image_latents.shape[1:]
            # timesteps, noise, original image
            # each for loop takes around 0.002, which is affordable
            t_list, xt_list, ut_list, masks = [], [], [], []
            for i, tp in enumerate(data_type):
                # x0->noise x1->image
                t, x0, x1 = transport.sample(image_latents[i][None],
                    config.training.und_max_t0 if tp in ['mmu', 'mmu_vid'] else None)
                # timesteps, noised image, velocity
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

            if len(masks) != 0:
                masks = torch.cat(masks, dim=0)
            else:
                masks = image_masks

            recons_images = vae_model.batch_decode(xt)

            return xt, t, ut, recons_images, masks,alpha_t

            
        image_latents, t, image_labels, recons_images, image_masks,alpha_t = prepare_latents_and_labels(pixel_values,
                                                                                                    batch['data_type'],
                                                                                                    (b, n),
                                                                                                    image_masks,
                                                                                                    modality_positions)
            
        # text_masks = (batch['text_tokens'] != text_tokenizer.pad_token_id).long().to(device)
        block_mask = omni_attn_mask_naive(
            text_tokens.size(0),
            text_tokens.size(1),
            modality_positions,
            device
        ).to(weight_type)

        # image_latents=image_labels
        with torch.no_grad():
            # print(text_tokens.shape,image_latents.shape)
            seg_mask,probs,output_tokens = model.mmu_generate(
                text_tokens=text_tokens,
                text_tokenizer=text_tokenizer,
                image_latents=image_latents,
                attention_mask= block_mask ,
                modality_positions=modality_positions,
                max_seq_len=text_tokens.size(1),
                spatial_size_my= spatial_size_my,
                device=device,
                max_new_tokens=1,
                temperature=1.0,
                top_k=top_k,
                t=t,
                eos_token=text_tokenizer.eos_token_id,
                # amp_dtype=weight_type
                )
        # print(output_tokens)

        if isinstance(output_tokens, torch.Tensor):
            output_tokens = output_tokens.long().to(device)
        else:
            output_tokens = torch.tensor(output_tokens, dtype=torch.long, device=device)

        if output_tokens.ndim == 1:  # [T] -> [1, T]
            output_tokens = output_tokens.unsqueeze(0)


        # print(output_tokens)
        output_tokens+=151669
        text = text_tokenizer.batch_decode(output_tokens,skip_special_tokens=False)
        print(texts)
        # evaluator.add(y_pred_text=text[0], y_true_text=texts[0][1],y_score=probs.detach().cpu().float())
        # ----- 新增：把 GT / Pred 文本转成 0~4 的整数，并收集起来 ----- #
        # text[0] 是模型回答，texts[0][1] 是 GT（你原来传给 FiveWayEval 的那一项）
        try:
            pred_idx = answer_to_index(text[0])
            gt_idx = answer_to_index(texts[0][1])
        except ValueError as e:
            print("标签解析失败：", e)
            # 你可以选择 continue / raise，这里我直接 raise 出去
            raise

        pred_indices.append(pred_idx)
        gt_indices.append(gt_idx)
        # -------------------------------------------------------------- #

        piece = f"[Q] {texts[0][0]}\n[A] {text[0]}\t [GT] {texts[0][1]}\n"
        if isinstance(responses[idx], list):
            responses[idx].append(piece)
        else:
            responses[idx] += piece
        # responses[idx]+=text+'\n Answer :'+texts[0]

        print(responses[idx])
        # print(seg_true.shape)
        # print(logits_to_mask_np(seg_mask).shape,pixel_values.shape)
        # torch.save(recons_images[0].cpu(), f"/root/autodl-tmp/chwang/experiments/results/mmu_und_output/{data_mode}/{save_name}/recons_xt/{pid}.pt")
        # save_numpy(logits_to_mask_np(seg_mask),f'/root/autodl-tmp/chwang/experiments/results/mmu_und_output/{data_mode}/{save_name}/seg_mask_mmu/{pid}.npy')
        # save_numpy(seg_true.squeeze(0).float().cpu().numpy(),f'/root/autodl-tmp/chwang/experiments/results/mmu_und_output/{data_mode}/{save_name}/seg_mask_mmu_true/{pid}.npy')
        # save_numpy(pixel_values.squeeze(0).float().cpu().numpy(),f'/root/autodl-tmp/chwang/experiments/results/mmu_und_output/{data_mode}/{save_name}/input_image_mmu/{pid}.npy')
        with open(f'/root/autodl-tmp/chwang/experiments/results/mmu_und_output/{data_mode}/{save_name}/log.txt', 'a+') as file:
            file.write(responses[idx])

    # ---------- 所有样本处理完毕后，保存 gt / pred 到 txt ---------- #
    import numpy as np

    txt_dir = f"/root/autodl-tmp/chwang/experiments/results/mmu_und_output/{data_mode}/{save_name}/txt_answers"
    os.makedirs(txt_dir, exist_ok=True)

    gt_txt_path = os.path.join(txt_dir, "answers_index_gt.txt")
    pred_txt_path = os.path.join(txt_dir, "answers_index_pred.txt")

    gt_arr = np.array(gt_indices, dtype=int)
    pred_arr = np.array(pred_indices, dtype=int)

    # 每行一个整数
    np.savetxt(gt_txt_path, gt_arr, fmt="%d")
    np.savetxt(pred_txt_path, pred_arr, fmt="%d")

    print(f"GT 标签保存到: {gt_txt_path}")
    print(f"Pred 标签保存到: {pred_txt_path}")

    from eval_5_way import evaluate_from_txt  # 如果就在同一个文件里，这行可以省略

    metrics, metrics_df = evaluate_from_txt(
        gt_path=gt_txt_path,
        pred_path=pred_txt_path,
        model_name=f"glioma_{save_name}",
        save_csv_path=os.path.join(txt_dir, "classification_metrics.csv"),
    )


    #     # break
    #     # torch.save(images.cpu(), "images.pt")

    # results = evaluator.report(title="MMU 5-way Classification")


        # images = torch.cat(images, dim=0)
        # images = denorm(images)
        # pil_images = [Image.fromarray(image) for image in images]

        # wandb_images = [wandb.Image(image, caption=responses[i]) for i, image in enumerate(pil_images)]
        # wandb.log({"Multimodal understanding responses": wandb_images}, step=step)


