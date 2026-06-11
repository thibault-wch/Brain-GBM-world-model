import math
from typing import Tuple, Union

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from einops import rearrange


# --- Utils ---
def _to_3d_tuple(v: Union[int, Tuple[int, ...]]) -> Tuple[int, int, int]:
    return (v, v, v) if isinstance(v, int) else tuple(v)


def tokens_to_grid(x_bnc: torch.Tensor, spatial_size: Tuple[int, int, int],
                   comp: Tuple[int, int, int] = (4, 16, 16)) -> torch.Tensor:
    """Reshapes token sequence to 3D grid: (B, N, C) -> (B, C, Dg, Hg, Wg)"""
    B, N, C = x_bnc.shape
    D, H, W = spatial_size
    cD, cH, cW = comp

    Dg, Hg, Wg = D // cD, H // cH, W // cW
    assert Dg * Hg * Wg == N, f"Token count N={N} does not match grid {Dg}x{Hg}x{Wg}"

    return rearrange(x_bnc, 'b (dg hg wg) c -> b c dg hg wg', dg=Dg, hg=Hg, wg=Wg)


def unpatchify3d(x_bcdhw: torch.Tensor, rd: int = 1, rh: int = 2, rw: int = 2) -> torch.Tensor:
    """Unpacks channels into spatial dimensions: (B, C*rd*rh*rw, D, H, W) -> (B, C, D*rd, H*rh, W*rw)"""
    B, Cmul, D, H, W = x_bcdhw.shape
    factor = rd * rh * rw
    assert Cmul % factor == 0, f"Channels ({Cmul}) not divisible by unpatch factor ({factor})"

    return rearrange(x_bcdhw, 'b (c rd rh rw) d h w -> b c (d rd) (h rh) (w rw)', rd=rd, rh=rh, rw=rw)


class Up3D(nn.Module):
    """Upsample -> Conv3d(k, dilation) -> SiLU"""

    def __init__(self, in_ch: int, out_ch: int, scale: Tuple[int, int, int] = (2, 2, 2),
                 k: Union[int, Tuple[int, ...]] = 3, final_act: bool = True,
                 bias: bool = True, dilation: Union[int, Tuple[int, ...]] = 1):
        super().__init__()
        self.final_act = final_act
        self.upsample = nn.Upsample(scale_factor=scale, mode='trilinear', align_corners=False)

        k3 = _to_3d_tuple(k)
        d3 = _to_3d_tuple(dilation)
        pad = tuple((d3[i] * (k3[i] - 1)) // 2 for i in range(3))

        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=k3, padding=pad, dilation=d3, bias=bias)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.conv(x)
        if self.final_act:
            x = self.act(x)
        return x


class TokensToVolume(nn.Module):
    def __init__(self, spatial_size: Tuple[int, int, int] = (192, 192, 40),
                 comp: Tuple[int, int, int] = (4, 16, 16),
                 in_channels: int = 1536, out_channels: int = 4,
                 first_pow2: int = 1024, width_mult: float = 1.0,
                 unpatch: Tuple[int, int, int] = (1, 2, 2),
                 share_in_proj: bool = False,
                 last_use_dilation: bool = True, use_ckpt: bool = True):
        super().__init__()

        # Reorder spatial size internally to (D, H, W)
        self.spatial_size = (spatial_size[2], spatial_size[0], spatial_size[1])
        self.comp = comp
        self.use_ckpt = use_ckpt

        self.rd, self.rh, self.rw = unpatch
        factor = self.rd * self.rh * self.rw

        C0 = max(8, int(first_pow2 * width_mult))
        C1 = max(1, C0 // 2)
        C2 = max(1, C1 // 2)

        assert in_channels % factor == 0, f"in_channels ({in_channels}) not divisible by factor ({factor})"
        base_ch = in_channels // factor

        self.share_in_proj = share_in_proj
        if share_in_proj:
            self.in_proj_shared = nn.Conv3d(base_ch, C0, kernel_size=1, bias=False)
        else:
            self.in_proj = nn.ModuleDict({
                'low': nn.Conv3d(base_ch, C0, kernel_size=1, bias=False),
                'middle': nn.Conv3d(base_ch, C0, kernel_size=1, bias=False),
                'high': nn.Conv3d(base_ch, C0, kernel_size=1, bias=False)
            })

        self.upAB = Up3D(C0, C1, scale=(2, 2, 2), k=3, final_act=True)
        self.upCD = Up3D(C1, C2, scale=(2, 2, 2), k=3, final_act=True)

        if last_use_dilation:
            self.upEF = Up3D(C2, out_channels, scale=(1, 2, 2), k=3, dilation=(1, 2, 2), final_act=False)
        else:
            self.upEF = Up3D(C2, out_channels, scale=(1, 2, 2), k=(3, 5, 5), final_act=False)

        self.lat_mid_to_C1 = Up3D(C0, C1, scale=(2, 2, 2), k=1, final_act=False)
        self.lat_low_to_C2 = Up3D(C0, C2, scale=(4, 4, 4), k=1, final_act=False)

        self._init_weights()

    def _maybe_ckpt(self, module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.use_ckpt and self.training:
            return checkpoint(module, x, use_reentrant=False)
        return module(x)

    def _to_base_feat(self, x_bnc: torch.Tensor, branch: str) -> torch.Tensor:
        x = tokens_to_grid(x_bnc, self.spatial_size, self.comp)
        x = unpatchify3d(x, rd=self.rd, rh=self.rh, rw=self.rw)

        if self.share_in_proj:
            x = self.in_proj_shared(x)
        else:
            x = self.in_proj[branch](x)
        return x

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5), mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_bnc_low: torch.Tensor, x_bnc_middle: torch.Tensor, x_bnc_high: torch.Tensor) -> torch.Tensor:
        D, H, W = self.spatial_size

        f_low = self._to_base_feat(x_bnc_low, 'low')
        f_middle = self._to_base_feat(x_bnc_middle, 'middle')
        f_high = self._to_base_feat(x_bnc_high, 'high')

        x = self._maybe_ckpt(self.upAB, f_high)
        x = x + self.lat_mid_to_C1(f_middle)

        x = self._maybe_ckpt(self.upCD, x)
        x = x + self.lat_low_to_C2(f_low)

        x = self._maybe_ckpt(self.upEF, x)

        assert x.shape[2:] == (D, H, W), f"Output shape mismatch: got {x.shape[2:]}, expected {(D, H, W)}"
        return x