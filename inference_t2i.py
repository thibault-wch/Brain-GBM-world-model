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
from models.modeling_showo2_qwen2_5 import Showo2Qwen2_5
from models import  omni_attn_mask, omni_attn_mask_naive
from models.misc import get_text_tokenizer, prepare_gen_input
from utils import  flatten_omega_conf, denorm, get_hyper_params, path_to_llm_name, load_state_dict, set_seed, collect_lora_targets_full,collect_modules_to_save_for_full_ft,add_default_before_last
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from datasets.utils import image_transform, resize_and_pad_image, to_tensor_and_normalize
from transport.utils import convert_qwen2_to_qwen2_dual
from omegaconf import OmegaConf, DictConfig
from peft import LoraConfig, get_peft_model
# from train_stage_two import prepare_latents_and_labels
from transport import Sampler, create_transport
from peft import PeftModel
from tqdm import tqdm
################.<  basic information   >.#########################

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weight_type=torch.bfloat16
set_seed(10)
# save_name='wo_seg_new'
save_name='all_new2'
data_mode='test'
# lora_path='/autodl-fs/data/experiments/glioma-class-final-wo-seg/checkpoint-43000/unwrapped_model/'
lora_path='/autodl-fs/data/experiments/glioma-class-final/checkpoint-70000/unwrapped_model/'
# lora_path= '/data_hdd/chwang/experiments/glioma/glioma-class-seg-1117_control_full_cnn3/checkpoint-32700/unwrapped_model/'
# lora_path= '/data_hdd/chwang/experiments/glioma/glioma-class-seg-unpaired-noaug/checkpoint-3500/unwrapped_model/adapter_model.bin'
# lora_path= '/data_hdd/chwang/experiments/glioma/glioma-class-seg/checkpoint-1000/unwrapped_model/adapter_model.bin'

# lora_path='/data_hdd/chwang/experiments/glioma/glioma-template-seg-5-classification/checkpoint-125000/unwrapped_model/adapter_model.bin'
###################################################################

from datasets.glioma_dataset import create_medical_dataloader,MedicalPairImageTextDataset
from datasets.mixed_dataloader import MixedDataLoader




logger = get_logger(__name__, log_level="INFO")


def get_config(path: str) -> DictConfig:
    """Load a single YAML file into an OmegaConf DictConfig."""
    return OmegaConf.load(path)



if __name__ == '__main__':

    config=get_config('/root/remoteproject/glioma/configs/showo2_1.5b_stage_2_a.yaml')
    preproc_config = config.dataset.preprocessing
    resume_wandb_run = config.wandb.resume

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
    # targets = collect_lora_targets_full(model)
    # modules_to_save = collect_modules_to_save_for_full_ft(model)  # lm_head + cond_proj 全量
    # print(f"[LoRA targets] {len(targets)} layers")
    # peft_config = LoraConfig(
    #     r=224,
    #     lora_alpha=112,
    #     target_modules=targets, # optionally indicate target modules
    #     modules_to_save=modules_to_save,  # ★ lm_head/cond_proj 全量
    # )   
    # model = get_peft_model(model, peft_config)
    # lora_state_dict=add_default_before_last(torch.load(lora_path))
    # model.load_state_dict(lora_state_dict,strict=False)


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
    # print(spatial_size_my)
    # config.model.showo.spatial_size_my= spatial_size_my
    
    num_image_tokens_my=int((spatial_size_my[0]//16)*(spatial_size_my[1]//16)*(spatial_size_my[2]//4))
    
    max_seq_len_my= num_image_tokens_my+200
    print(f'num_image_tokens_my: {num_image_tokens_my}, max_seq_len_my: {max_seq_len_my}')


    num_t2i_image_tokens, num_mmu_image_tokens, num_video_tokens, max_seq_len, max_text_len, image_latent_dim, patch_size, latent_width, \
    latent_height, pad_id, bos_id, eos_id, boi_id, eoi_id, bov_id, eov_id, img_pad_id, vid_pad_id, guidance_scale \
        = get_hyper_params(config, text_tokenizer, showo_token_ids)
    
    print('add time embeds',config.model.showo.add_time_embeds)
    
    # for time embedding
    if config.model.showo.add_time_embeds:
        # we prepend the time embedding to vision tokens
        config.dataset.preprocessing.num_t2i_image_tokens += 1
        config.dataset.preprocessing.num_mmu_image_tokens += 1
        config.dataset.preprocessing.num_video_tokens += 1
        num_image_tokens_my+=1
        max_seq_len_my+=1


    temperature = 1.0  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    top_k = 1  # retain only the top_k most likely tokens, clamp others to have 0 probability


    test_dataloader_t2i =  create_medical_dataloader(
    root="/root/autodl-tmp/dataset",
    batch_size=config.training.batch_size_mmu,
    text_tokenizer=text_tokenizer,
    showo_token_ids=showo_token_ids,
    spatial_size=spatial_size_my,
    num_image_tokens=num_image_tokens_my,
    max_seq_len=max_seq_len_my,  
    mode=data_mode,
    is_captioning=False,    # t2i
    use_seg_mask=True,
     drop_last=False,
        shuffle=False,
    )

    responses = ['' for j in range(len(test_dataloader_t2i))]
    model.eval()

    os.makedirs(f'/root/autodl-tmp/chwang/experiments/results/mmu_t2i_output/{data_mode}/{save_name}/input_cond/', exist_ok=True)
    os.makedirs(f'/root/autodl-tmp/chwang/experiments/results/mmu_t2i_output/{data_mode}/{save_name}/gt/', exist_ok=True)
    os.makedirs(f'/root/autodl-tmp/chwang/experiments/results/mmu_t2i_output/{data_mode}/{save_name}/predicted/', exist_ok=True)
    os.makedirs(f'/root/autodl-tmp/chwang/experiments/results/mmu_t2i_output/{data_mode}/{save_name}/seg/', exist_ok=True)
    os.makedirs(f'/root/autodl-tmp/chwang/experiments/results/mmu_t2i_output/{data_mode}/{save_name}/recons_xt/', exist_ok=True)
   


    for idx,batch in enumerate(tqdm(test_dataloader_t2i)):
        if batch['pid'][0]!='Patient-051' and batch['pid'][0]!='Patient-059':
            print(batch['pid'][0])
            continue
        pid=batch['pid'][0]+batch['file_id'][0]
        text_tokens = batch['text_tokens'].to(device)
        # print(batch['text_labels'])
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
        sampler = Sampler(transport)

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
                image_latents = vae_model.sample(pixel_values)
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
                    config.training.und_max_t0)
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
            
        latent_depth=spatial_size_my[2]//4
        latent_height=spatial_size_my[0]//16
        latent_width=spatial_size_my[1]//16
        z = torch.randn((len(text_tokens),
                         image_latent_dim, 
                         latent_depth,
                         latent_height * patch_size,
                         latent_width * patch_size)).to(torch.bfloat16).to(device)

        # if guidance_scale > 0:
        #     z = torch.cat([z, z], dim=0)
        #     text_tokens = torch.cat([text_tokens, text_tokens], dim=0)
        #     modality_positions = torch.cat([modality_positions, modality_positions], dim=0)
        #     # B=None would potentially induce loss spike when there are a lot of ignored labels (-100) in the batch
        #     # we must set B=text_tokens.shape[0] (loss spike may still happen sometimes)
        #     # omni_mask_fn = omni_attn_mask(modality_positions)
        #     # block_mask = create_block_mask(omni_mask_fn, B=z.size(0), H=None, Q_LEN=max_seq_len,
        #     #                                KV_LEN=max_seq_len, device=device)
        #     # or use naive omni attention mask, which is more stable
        #     block_mask = omni_attn_mask_naive(text_tokens.size(0),
        #                                       max_seq_len,
        #                                       modality_positions,
        #                                       device).to(weight_type)
        # else:
            # B=None would potentially induce loss spike when there are a lot of ignored labels (-100) in the batch
            # we must set B=text_tokens.shape[0] (loss spike may still happen sometimes)
            # omni_mask_fn = omni_attn_mask(modality_positions)
            # block_mask = create_block_mask(omni_mask_fn, B=z.size(0), H=None, Q_LEN=max_seq_len,
            #                                KV_LEN=max_seq_len, device=device)
        block_mask = omni_attn_mask_naive(text_tokens.size(0),
                                              max_seq_len_my,
                                              modality_positions,
                                              device).to(weight_type)

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
            # samples = torch.chunk(samples, 2)[0]

        if config.model.vae_model.type == 'wan21':
            # samples = samples.unsqueeze(2)
            # print(samples.shape)
            images = vae_model.batch_decode(samples)
            images = images.squeeze(2)
        else:
            raise NotImplementedError
        print(batch['images_cond'][0].shape,batch['images'][0].shape,images.shape,recons_images.shape)
        # seg
        print(texts)
        # torch.save(recons_images[0].cpu(), f"/root/autodl-tmp/chwang/experiments/results/mmu_t2i_output/{data_mode}/{save_name}/recons_xt/{pid}.pt")
        # torch.save(batch['images_cond'][0].cpu(), f"/root/autodl-tmp/chwang/experiments/results/mmu_t2i_output/{data_mode}/{save_name}/input_cond/{pid}.pt")
        # torch.save(batch['images'][0].cpu(), f"/root/autodl-tmp/chwang/experiments/results/mmu_t2i_output/{data_mode}/{save_name}/gt/{pid}.pt")
        torch.save(images.cpu(), f"/root/autodl-tmp/chwang/experiments/results/mmu_t2i_output/{data_mode}/{save_name}/predicted/{pid}.pt")
        print(f"/root/autodl-tmp/chwang/experiments/results/mmu_t2i_output/{data_mode}/{save_name}/predicted", pid)
        # if idx ==3:
        #     break
        


        # Convert to PIL images
        # images = denorm(images)
        # pil_images = [Image.fromarray(image) for image in images]

        # Log images
        # wandb_images = [wandb.Image(image, caption=prompts[i]) for i, image in enumerate(pil_images)]
        # wandb.log({"Generated images": wandb_images}, step=step)