# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import math
from linear_attention_transformer import LinearAttentionTransformer

from .modules import PatchEmbed
from einops import rearrange


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
            patch_size=2, image_latent_dim=16,
            init_on_build: bool = True,
    ):

        super().__init__()
        self.dim = dim
        self.image_gen = PatchEmbed(
            patch_size=patch_size,
            in_chans=image_latent_dim,
            embed_dim=dim,
        )

        self.layer1 = LinearAttentionTransformer(dim=dim, depth=2, heads=8, max_seq_len=100000, causal=False)

        if init_on_build:
            self.init_parameters()

    @torch.no_grad()
    def init_parameters(self):
        """
        Uniform Kaiming initialization for all Linear weights; set bias to 0.
        SiLU can be approximated using the ReLU gain.
        """
        nonlinearity = 'relu'  # Stable enough for ReLU/GELU/SiLU
        a = math.sqrt(5)  # Keeps consistent with PyTorch's default bias boundary calculation

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
        b, T, c, h, w = cond.shape
        cond = rearrange(cond, 'b c t h w -> (b t) c h w')
        cond = self.image_gen(cond)
        # print('cond',cond.shape)
        cond = cond.reshape(b, T, -1, self.dim)
        cond = rearrange(cond, 'b t l d -> b (t l) d')
        #########################################################

        return self.layer1(cond), None


# ============ Pre-defined Configurations ============
class ConcatFusion_S(_ConcatFusion):
    def __init__(self, dim: int, **kwargs):
        super().__init__(dim=dim, depth=3, hidden_mult=1.0, **kwargs)


class ConcatFusion_M(_ConcatFusion):
    def __init__(self, dim: int, **kwargs):
        super().__init__(dim=dim, depth=4, hidden_mult=1.5, **kwargs)


class ConcatFusion_L(_ConcatFusion):
    def __init__(self, dim: int, **kwargs):
        super().__init__(dim=dim, depth=5, hidden_mult=2.0, **kwargs)