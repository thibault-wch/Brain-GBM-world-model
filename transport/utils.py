import torch as th
import math
import re

class EasyDict:
    def __init__(self, sub_dict):
        for k, v in sub_dict.items():
            setattr(self, k, v)

    def __getitem__(self, key):
        return getattr(self, key)


def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return th.mean(x, dim=list(range(1, len(x.size()))))


def log_state(state):
    result = []

    sorted_state = dict(sorted(state.items()))
    for key, value in sorted_state.items():
        # Check if the value is an instance of a class
        if "<object" in str(value) or "object at" in str(value):
            result.append(f"{key}: [{value.__class__.__name__}]")
        else:
            result.append(f"{key}: {value}")

    return "\n".join(result)

def time_shift(mu: float, sigma: float, t: th.Tensor):
    # the following implementation was original for t=0: clean / t=1: noise
    # Since we adopt the reverse, the 1-t operations are needed
    t = 1 - t
    t = math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)
    t = 1 - t
    return t

def get_lin_function(x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b

def expand_dims(v, dims):
    """
    Expand the tensor `v` to the dim `dims`.

    Args:
        `v`: a PyTorch tensor with shape [N].
        `dim`: a `int`.
    Returns:
        a PyTorch tensor with shape [N, 1, 1, ..., 1] and the total dimension is `dims`.
    """
    return v[(...,) + (None,) * (dims - 1)]


from collections import OrderedDict

def infer_num_layers_from_state_dict(sd):
    # Try to infer the maximum layer index from keys like: model.layers.{i}.*
    pattern = re.compile(r"^model\.layers\.(\d+)\.")
    max_idx = -1
    for k in sd.keys():
        m = pattern.match(k)
        if m:
            idx = int(m.group(1))
            if idx > max_idx:
                max_idx = idx
    if max_idx < 0:
        raise ValueError("Could not infer number of layers from state_dict keys.")
    return max_idx + 1

import re
from collections import OrderedDict

def infer_num_layers_from_state_dict(sd):
    pat = re.compile(r"^model\.layers\.(\d+)\.")
    mx = -1
    for k in sd.keys():
        m = pat.match(k)
        if m:
            mx = max(mx, int(m.group(1)))
    if mx < 0:
        raise ValueError("Cannot infer num_hidden_layers (no keys like 'model.layers.<i>.*').")
    return mx + 1

import re
from collections import OrderedDict

def _infer_num_layers_from_sd(sd: dict) -> int:
    pat = re.compile(r"^model\.layers\.(\d+)\.")
    mx = -1
    for k in sd.keys():
        m = pat.match(k)
        if m:
            mx = max(mx, int(m.group(1)))
    if mx < 0:
        raise ValueError("Cannot infer num_hidden_layers from keys like 'model.layers.<i>.*'.")
    return mx + 1

def convert_qwen2_to_qwen2_dual(src_state_dict: dict, two_thirds, num_hidden_layers: int = None):
    """
    将单-MLP 的 Qwen2 checkpoint（键名形如: model.layers.i.*）转换为 Qwen2-dual 的命名：
      - 目标层命名：showo.model.layers.* 与 showo.model.dual_layers.*；lm_head.* -> showo.lm_head.*
      - 前 two_thirds 层  -> showo.model.layers.[0..two_thirds-1]
      - 后 (L - two_thirds) 层 -> showo.model.dual_layers.[0..(L-two_thirds-1)]
      - 对映射到 dual_layers 的每层：将源层的 MLP 权重复制到 mlp1.* 和 mlp2.*（两者相同）
    典型：原 L=28，two_thirds=14，则 0..13 -> layers，14..27 -> dual_layers[0..13]
    """
    sd = src_state_dict
    if num_hidden_layers is None:
        num_hidden_layers = _infer_num_layers_from_sd(sd)
    L = num_hidden_layers

    if not (0 < two_thirds <= L):
        raise ValueError(f"two_thirds={two_thirds} out of range (L={L}).")
    one_third = L - two_thirds  # 要映射到 dual_layers 的层数

    new_sd = OrderedDict()

    # ---------- 顶层：把非 layers.* 的键重命名到 showo.* ----------
    # 例：model.embed_tokens.weight -> showo.model.embed_tokens.weight
    #     lm_head.weight           -> showo.lm_head.weight
    layer_pat = re.compile(r"^model\.layers\.(\d+)\.")
    for k, v in sd.items():
        if layer_pat.match(k):
            continue
        if k.startswith("model."):
            new_k = "showo." + k       # -> showo.model.*
        elif k.startswith("lm_head."):
            new_k = "showo." + k       # -> showo.lm_head.*
        else:
            new_k = k                  # 其他罕见顶层键，原样保留
        new_sd[new_k] = v

    # ---------- 前 two_thirds 层：直接复制到 showo.model.layers.i.* ----------
    for i in range(two_thirds):
        src_prefix = f"model.layers.{i}."
        tgt_prefix = f"showo.model.layers.{i}."
        for k, v in sd.items():
            if k.startswith(src_prefix):
                new_sd[k.replace(src_prefix, tgt_prefix)] = v

    # ---------- 后 1/3 层：映射到 showo.model.dual_layers.j.* ----------
    # 复制 MLP 到 mlp1 与 mlp2；其余子模块（attn/norm等）直接改前缀
    mlp_suffixes = [
        "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
        "mlp.gate_proj.bias",   "mlp.up_proj.bias",   "mlp.down_proj.bias",
    ]

    for j in range(one_third):
        i = two_thirds + j
        src_layer_prefix = f"showo.model.layers.{i}."
        tgt_dual_prefix  = f"showo.model.dual_layers.{j}."

        # 非 MLP：直接重命名前缀
        for k, v in sd.items():
            if not k.startswith(src_layer_prefix):
                continue
            if ".mlp." in k:
                continue
            new_sd[k.replace(src_layer_prefix, tgt_dual_prefix)] = v

        # MLP：复制到 mlp1.* 和 mlp2.*
        for suf in mlp_suffixes:
            sk = f"{src_layer_prefix}{suf}"
            if sk not in sd:
                continue
            new_sd[f"{tgt_dual_prefix}{suf.replace('mlp.', 'mlp1.')}"] = sd[sk]
            new_sd[f"{tgt_dual_prefix}{suf.replace('mlp.', 'mlp2.')}"] = sd[sk]

    return new_sd
