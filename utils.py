from typing import Any, List, Tuple
from omegaconf import DictConfig, ListConfig, OmegaConf
import torch
import numpy as np
from PIL import Image
import os
from copy import deepcopy
from collections import OrderedDict
import random
from decord import VideoReader, cpu
import torch.nn as nn


import os
from typing import Union
import numpy as np
import torch
import torch.nn.functional as F



def logits_to_mask_np(
    logits: Union[torch.Tensor, np.ndarray],
    apply_softmax: bool = True,
) -> np.ndarray:
    """
    将形状 (1,4,D,H,W) 的分割 logits 转为 (D,H,W) 的 0/1/2/3 掩膜（numpy, uint8）。
    默认先 softmax 再 argmax。
    """
    # 支持 numpy / torch，统一成 torch 处理
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits)
    if not isinstance(logits, torch.Tensor):
        raise TypeError("logits 必须是 torch.Tensor 或 np.ndarray")
    with torch.no_grad():
        x = F.softmax(logits, dim=1) if apply_softmax else logits
        mask_t = x.argmax(dim=1).squeeze(0).to(torch.uint8)  # (D,H,W), 0..3
        mask_np = mask_t.cpu().numpy()
    return mask_np


def save_numpy(arr: np.ndarray, path: str) -> str:
    """
    将 numpy 数组保存为 .npy 文件；自动创建目录，返回实际保存路径。
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError("arr 必须是 numpy.ndarray")
    # 确保扩展名
    if not path.endswith(".npy"):
        path = path + ".npy"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.save(path, arr)
    return path


##################################################
#              lora utils
##################################################

import re
import torch.nn as nn
from collections import OrderedDict


def mask_grad_last_k_rows(module: nn.Linear, k: int = 5):
    W = module.weight
    def _hook(g):
        # g 形状与 W 相同，清零前部梯度
        mask = torch.zeros_like(g)
        mask[-k:, :] = 1
        return g * mask
    h = W.register_hook(_hook)
    # 对 bias 也可同理：
    if module.bias is not None:
        def _hook_b(gb):
            maskb = torch.zeros_like(gb); maskb[-k:] = 1
            return gb * maskb
        module.bias.register_hook(_hook_b)
    return h  # 如需以后移除：h.remove()

# 用法


def add_default_before_last(sd, only_weight=True):
    """
    将键的倒数第二个位置插入 'default'。
    - only_weight=True 仅处理以 '.weight' 结尾的键；False 则处理所有键。
    - 已含 '.default.' 的键会跳过，避免重复插入。
    """
    new_sd = OrderedDict()
    for k, v in sd.items():
        if only_weight and not k.endswith('.weight'):
            new_sd[k] = v
            continue
        parts = k.split('.')
        if len(parts) >= 2 and parts[-2] == 'default':
            new_sd[k] = v  # 已经有 default，保持不变
        else:
            parts.insert(-1, 'default')
            new_k = '.'.join(parts)
            new_sd[new_k] = v
    return new_sd
import torch.nn as nn

def _has_params(mod: nn.Module) -> bool:
    return any(p is not None and p.numel() > 0 for p in mod.parameters())

def collect_modules_to_save_for_full_ft(model):
    candidates = set()

    # A) lm_head
    for name, mod in model.named_modules():
        if name and name.endswith("lm_head") and _has_params(mod):
            candidates.add(name)

    # B) cond_proj
    for name, mod in model.named_modules():
        if name and "cond_proj" in name and _has_params(mod):
            candidates.add(name)

    # C) segmentor
    for name, mod in model.named_modules():
        if name and "segmentor" in name and _has_params(mod):
            candidates.add(name)

    # spe
    for name, mod in model.named_modules():
        if name and "spe" in name and _has_params(mod):
            candidates.add(name)

    # fusion_proj
    for name, mod in model.named_modules():
        if name and "fusion_proj" in name and _has_params(mod):
            candidates.add(name)

    # diff_proj
    for name, mod in model.named_modules():
        if name and "diff_proj" in name and _has_params(mod):
            candidates.add(name)

    # ✅ E) diffusion_head_a：如果是 ModuleList，就展开成 diffusion_head_a.0/1/...
    for name, mod in model.named_modules():
        if not name:
            continue
        if name == "diffusion_head_a" or name.endswith(".diffusion_head_a"):
            if isinstance(mod, nn.ModuleList):
                for i, child in enumerate(mod):
                    if _has_params(child):
                        candidates.add(f"{name}.{i}")
            else:
                # 如果你的实现里 diffusion_head_a 不是 ModuleList（少见），才允许直接加
                if _has_params(mod):
                    candidates.add(name)

    # diffusion_head_b
    for name, mod in model.named_modules():
        if name and "diffusion_head_b" in name and _has_params(mod):
            candidates.add(name)

    # 去重&只保留顶层
    cand_sorted = sorted(candidates, key=lambda s: s.count("."))
    kept = []
    for n in cand_sorted:
        if not any(n == k or n.startswith(k + ".") for k in kept):
            kept.append(n)

    # # ✅ 兜底：最终 kept 里不允许容器（防止其他地方也匹配到 ModuleList）
    # name_to_mod = dict(model.named_modules())
    # bad = (nn.ModuleList, nn.ModuleDict, nn.ParameterList, nn.ParameterDict)
    # kept = [n for n in kept if not isinstance(name_to_mod.get(n, None), bad)]

    kept = [n for n in kept if not re.match(r"^showo\.model\.(layers|dual_layers)\.\d+$", n)]
    kept = [n for n in kept if not re.match(r"^showo\.model\.(layers|dual_layers)$", n)]


    print(f"[modules_to_save] {len(kept)}")
    for s in kept:
        print("  -", s)

    return kept


import re
import torch.nn as nn

def collect_lora_targets_full(model):
    """
    Collect the full list of LoRA targets (only nn.Linear layers):
    - showo.model.layers.* / showo.model.dual_layers.* q/k/v/o_proj, gate/up/down_proj
    - diff_proj.{0,2}
    - fusion_proj
    - diffusion_head_a.<idx>.self_attn.(q/k/v/o)_proj
    - diffusion_head_a.<idx>.mlp.(gate/up/down)_proj
    - diffusion_head_a.<idx>.adaLN_modulation.1 (if Linear)
    - diffusion_head_b.linear
    Excludes *_norm and *.bias from LoRA.
    """
    keep = set()

    # 1) showo main structure (including dual_layers)
    pat_showo = re.compile(
        r"^showo\.model\.(?:layers|dual_layers)\.\d+\."
        r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp(?:1|2)?\.(?:gate_proj|up_proj|down_proj))$"
    )

    # 2) Explicit leaves for diffusion and time embedding
    leaves_exact = {
        # "fusion_proj.0",
        # "fusion_proj.1",
        # "fusion_proj.3",
        # "diff_proj.0",
        # "diff_proj.2",
        # "diffusion_head_b.linear",
    }

    # 3) diffusion_head_a.<idx> - self_attn and mlp layers
    # pat_head_a = re.compile(
    #     r"^diffusion_head_a\.\d+\."
    #     r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))$"
    # )
    
    # 4) diffusion_head_a.<idx>.adaLN_modulation.1 (if Linear)
    # pat_head_a_ada = re.compile(r"^diffusion_head_a\.\d+\.adaLN_modulation\.1$")

    # 5) for und_trans
    pat_und_trans = re.compile(
    r"^und_trans\.layers\.\d+\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|out_proj)|mlp\.(?:fc1|fc2))$"
    )
    # Iterate through model modules
    for full_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        # Showo main structure
        if pat_showo.match(full_name):
            keep.add(full_name)
            continue

        # Explicitly add known leaves (e.g., fusion_proj, diff_proj)
        if full_name in leaves_exact:
            keep.add(full_name)
            continue

        # diffusion_head_a attn/mlp layers
        # if pat_head_a.match(full_name):
        #     keep.add(full_name)
        #     continue

        # # diffusion_head_a adaLN_modulation.1 (if Linear)
        # if pat_head_a_ada.match(full_name):
        #     keep.add(full_name)
        #     continue

        # if pat_und_trans.match(full_name):
        #     keep.add(full_name)
        #     continue

    targets = sorted(keep)
    print(f"[LoRA targets] {len(targets)}")
    for s in targets:
        print("  -", s)
    if not targets:
        raise RuntimeError("No LoRA targets found, check if the naming is correct.")
    
    return targets


##################################################
#              config utils
##################################################
def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)

    return conf


def flatten_omega_conf(cfg: Any, resolve: bool = False) -> List[Tuple[str, Any]]:
    ret = []

    def handle_dict(key: Any, value: Any, resolve: bool) -> List[Tuple[str, Any]]:
        return [(f"{key}.{k1}", v1) for k1, v1 in flatten_omega_conf(value, resolve=resolve)]

    def handle_list(key: Any, value: Any, resolve: bool) -> List[Tuple[str, Any]]:
        return [(f"{key}.{idx}", v1) for idx, v1 in flatten_omega_conf(value, resolve=resolve)]

    if isinstance(cfg, DictConfig):
        for k, v in cfg.items_ex(resolve=resolve):
            if isinstance(v, DictConfig):
                ret.extend(handle_dict(k, v, resolve=resolve))
            elif isinstance(v, ListConfig):
                ret.extend(handle_list(k, v, resolve=resolve))
            else:
                ret.append((str(k), v))
    elif isinstance(cfg, ListConfig):
        for idx, v in enumerate(cfg._iter_ex(resolve=resolve)):
            if isinstance(v, DictConfig):
                ret.extend(handle_dict(idx, v, resolve=resolve))
            elif isinstance(v, ListConfig):
                ret.extend(handle_list(idx, v, resolve=resolve))
            else:
                ret.append((str(idx), v))
    else:
        assert False

    return ret


##################################################
#              misc
##################################################

def _count_params(module, precision: int = 3):
    Total_params = 0
    Trainable_params = 0
    NonTrainable_params = 0

    for param in module.parameters():
        mulValue = param.numel()
        Total_params += mulValue
        if param.requires_grad:
            Trainable_params += mulValue
        else:
            NonTrainable_params += mulValue

    M = 1e6
    fmt = f"{{:.{precision}f}}M"
    print(f'Total params: {fmt.format(Total_params / M)}')
    print(f'Trainable params: {fmt.format(Trainable_params / M)}')
    print(f'Non-trainable params: {fmt.format(NonTrainable_params / M)}')

def _freeze_params(model, frozen_params=None):
    if frozen_params is not None:
        for n, p in model.named_parameters():
            for name in frozen_params:
                if name in n:
                    p.requires_grad = False


def _weak_params(model, weak_params=None):
    if weak_params is not None:
        for n, p in model.named_parameters():
            for name in weak_params:
                if name in n:
                    p.requires_grad = True


path_to_llm_name = {
    "Qwen/Qwen2.5-7B-Instruct": 'qwen2_5',
    "Qwen/Qwen2.5-1.5B-Instruct": 'qwen2_5',
    "meta-llama/Llama-3.2-1B-Instruct": 'llama3'
}




class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def denorm(images):
    images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0).to(torch.float32)
    images *= 255.0
    images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    return images

def denorm_vid(images):
    images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0).to(torch.float32)
    images *= 255.0
    # B, C, T, H, W --> B, T, C, H, W
    images = images.permute(0, 2, 1, 3, 4).cpu().numpy().astype(np.uint8)
    return images


def get_hyper_params(config, text_tokenizer, showo_token_ids, is_video=False, is_hq=False):
    # [bos_id, text_tokens, im_start, image_tokens, im_end, eos_id, pad_id]
    max_seq_len = config.dataset.preprocessing.max_seq_length
    num_video_tokens = config.dataset.preprocessing.num_video_tokens
    if is_video:
        max_text_len = max_seq_len - num_video_tokens - 4
        latent_width = config.dataset.preprocessing.video_latent_width
        latent_height = config.dataset.preprocessing.video_latent_height
        num_t2i_image_tokens = config.dataset.preprocessing.num_t2i_image_tokens
        num_mmu_image_tokens = config.dataset.preprocessing.num_mmu_image_tokens
    else:
        if is_hq:
            latent_width = config.dataset.preprocessing.hq_latent_width
            latent_height = config.dataset.preprocessing.hq_latent_height
            num_t2i_image_tokens = config.dataset.preprocessing.num_hq_image_tokens
            num_mmu_image_tokens = config.dataset.preprocessing.num_mmu_image_tokens
            max_seq_len = config.dataset.preprocessing.max_hq_seq_length
            max_text_len = max_seq_len - num_t2i_image_tokens - 4
        else:
            num_t2i_image_tokens = config.dataset.preprocessing.num_t2i_image_tokens
            num_mmu_image_tokens = config.dataset.preprocessing.num_mmu_image_tokens
            latent_width = config.dataset.preprocessing.latent_width
            latent_height = config.dataset.preprocessing.latent_height
            max_text_len = max_seq_len - num_t2i_image_tokens - 4

    image_latent_dim = config.model.showo.image_latent_dim
    patch_size = config.model.showo.patch_size

    pad_id = text_tokenizer.pad_token_id
    bos_id = showo_token_ids['bos_id']
    eos_id = showo_token_ids['eos_id']
    boi_id = showo_token_ids['boi_id']
    eoi_id = showo_token_ids['eoi_id']
    bov_id = showo_token_ids['bov_id']
    eov_id = showo_token_ids['eov_id']
    img_pad_id = showo_token_ids['img_pad_id']
    vid_pad_id = showo_token_ids['vid_pad_id']

    guidance_scale = config.transport.guidance_scale

    return num_t2i_image_tokens, num_mmu_image_tokens, num_video_tokens, max_seq_len, max_text_len, image_latent_dim, patch_size, \
           latent_width, latent_height, pad_id, bos_id, eos_id, boi_id, eoi_id, bov_id, eov_id, img_pad_id, \
           vid_pad_id, guidance_scale


# these save and recover functions are based on our internal packages
# please modified them when necessary
def save_dataloader_state(rank, loader, ckpt_path="./"):
    ckpt_path = os.path.join(ckpt_path, f"loader_{rank}.ckpt")
    saved_state = deepcopy(loader.__getstate__())
    torch.save(saved_state, ckpt_path)

def recover_dataloader_state(rank, loader, ckpt_path='./'):
    ckpt_path = os.path.join(ckpt_path, f"loader_{rank}.ckpt")
    if os.path.exists(ckpt_path):
        with open(ckpt_path, 'rb') as f:
            loader_state_dict = torch.load(f)
            loader.__setstate__(loader_state_dict)
        print(f"rank {rank} loader state dict loaded successfully!")


def save_images_as_grid(pil_images, fn, path, grid_size=(2, 2)):

    os.makedirs(path, exist_ok=True)

    rows, cols = grid_size

    num_images = len(pil_images)
    if num_images > rows * cols:
        raise ValueError(f"Number of images ({num_images}) exceeds grid capacity ({rows * cols}).")

    img_width, img_height = pil_images[0].size

    grid_width = cols * img_width
    grid_height = rows * img_height
    grid_image = Image.new("RGB", (grid_width, grid_height), color=(255, 255, 255))  # 白色背景

    for idx, image in enumerate(pil_images):
        row = idx // cols
        col = idx % cols
        x_offset = col * img_width
        y_offset = row * img_height
        grid_image.paste(image, (x_offset, y_offset))

    grid_image.save(os.path.join(path, f"{fn}.png"))

    return grid_image


def load_state_dict(model_path):
    if model_path.endswith(".bin"):
        state_dict = torch.load(model_path)
    else:
        checkpoint_files = sorted(
            [os.path.join(model_path, f) for f in os.listdir(model_path) if f.endswith('.bin')]
        )

        state_dict = OrderedDict()
        for checkpoint_file in checkpoint_files:
            print(f"Loading checkpoint: {checkpoint_file}")
            checkpoint = torch.load(checkpoint_file)
            state_dict.update(checkpoint)

    return state_dict

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def load_video(video_path, max_frames_num, fps, force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 432, 432, 3))

    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps() / fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i / fps for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i / vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = [Image.fromarray(item) for item in vr.get_batch(frame_idx).asnumpy()]

    return spare_frames, frame_time, video_time