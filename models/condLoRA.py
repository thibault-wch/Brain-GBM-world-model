# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import math
from linear_attention_transformer import LinearAttentionTransformer

from .modules import  PatchEmbed
from einops import rearrange

# from transformers.models.bert.modeling_bert import BertLayer
# from transformers import BertConfig

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-3):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight

class _ConcatFusion(nn.Module):
    """
    """
    def __init__(
        self,
        dim: int,
        config: None,
        depth: int = 5,
        patch_size=2, image_latent_dim=16, 
        hidden_mult: float = 2.0,
        activation: str = "silu",
        norm: str = "none",                # "none" | "layer" | "rms"
        tiny_B_std: float = 0.0,
        init_on_build: bool = True,
        time_mode: str = "add",          # "add" | "film"
        # 3D APE（必在 input 注入）
        spatial_size=(192, 192, 48),      # (H, W, D)
        comp=(4, 16, 16),                 # (cD, cH, cW)
        pe_num_freqs: int = 8,
        pe_base: float = 2.0,
        pe_learnable_proj: bool = True,
        pe_pos_scale: float = 1.0,
    ):
        
        super().__init__()
        self.dim = dim
        self.image_gen = PatchEmbed(
            patch_size=patch_size,
            in_chans=image_latent_dim,
            embed_dim=dim,
        )


        self.layer1 = LinearAttentionTransformer(dim=dim,depth=2,heads=8,max_seq_len=100000,causal=False)
        # nn.Sequential(*[nn.TransformerEncoderLayer(d_model=dim, nhead=8) for _ in range(2)])

        # self.layer2 = LinearAttentionTransformer(dim=dim,depth=2,heads=8,max_seq_len=100000,causal=False)

        if init_on_build:
            self.init_parameters()

    @torch.no_grad()
    def init_parameters(self):
        """
        统一 Kaiming 初始化所有 Linear 权重；bias 置 0。
        SiLU 用 relu 增益近似即可。
        """
        nonlinearity = 'relu'  # 对 ReLU/GELU/SiLU 都够稳
        a = math.sqrt(5)       # 与 PyTorch 默认 bias 边界计算保持一致

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=a, mode='fan_in', nonlinearity=nonlinearity)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        """
        x, cond: (B, N, D)
        t_embed: (B, D) or None
        """
        #########################################################
        b,T,c,h,w=cond.shape
        cond = rearrange(cond, 'b c t h w -> (b t) c h w')
        cond = self.image_gen(cond)
        # print('cond',cond.shape)
        cond= cond.reshape(b, T, -1, self.dim)
        cond = rearrange(cond, 'b t l d -> b (t l) d')
        #########################################################

        return self.layer1(cond),None


class _ConcatFusionold(nn.Module):
    """
    """
    def __init__(
        self,
        dim: int,
        config: None,
        depth: int = 5,
        hidden_mult: float = 2.0,
        activation: str = "silu",
        norm: str = "none",                # "none" | "layer" | "rms"
        tiny_B_std: float = 0.0,
        init_on_build: bool = True,
        time_mode: str = "add",          # "add" | "film"
        spatial_size=(192, 192, 48),      # (H, W, D)
        comp=(4, 16, 16),                 # (cD, cH, cW)
        pe_num_freqs: int = 8,
        pe_base: float = 2.0,
        pe_learnable_proj: bool = True,
        pe_pos_scale: float = 1.0,
    ):
        
        super().__init__()
        self.dim = dim


        self.layer1 = LinearAttentionTransformer(dim=dim,depth=2,heads=8,max_seq_len=100000,causal=False)
        # nn.Sequential(*[nn.TransformerEncoderLayer(d_model=dim, nhead=8) for _ in range(2)])

        self.layer2 = LinearAttentionTransformer(dim=dim,depth=2,heads=8,max_seq_len=100000,causal=False)

        if init_on_build:
            self.init_parameters()

    @torch.no_grad()
    def init_parameters(self):
        """
        """
        nonlinearity = 'relu'
        a = math.sqrt(5)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=a, mode='fan_in', nonlinearity=nonlinearity)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        """
        x, cond: (B, N, D)
        t_embed: (B, D) or None
        """
        out1= self.layer1(cond)
        out2= self.layer2(out1)
        # return out1, 0
        return out1,out2

# ============ 便捷规格 ============
class ConcatFusion_S(_ConcatFusion):
    def __init__(self, dim: int, **kwargs):
        super().__init__(dim=dim, depth=3, hidden_mult=1.0, **kwargs)

class ConcatFusion_M(_ConcatFusion):
    def __init__(self, dim: int, **kwargs):
        super().__init__(dim=dim, depth=4, hidden_mult=1.5, **kwargs)

class ConcatFusion_L(_ConcatFusion):
    def __init__(self, dim: int, **kwargs):
        super().__init__(dim=dim, depth=5, hidden_mult=2.0, **kwargs)
