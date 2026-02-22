import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
import math
# --------- 工具（全部用 rearrange） ---------
def tokens_to_grid(x_bnc: torch.Tensor, spatial_size, comp=(4,16,16)):
    """(B,N,C)->(B,C,D',H',W')"""
    B, N, C = x_bnc.shape
    D, H, W = spatial_size
    cD, cH, cW = comp
    Dg, Hg, Wg = D // cD, H // cH, W // cW
    assert Dg * Hg * Wg == N, f"N={N} != {Dg*Hg*Wg}"
    return rearrange(x_bnc, 'b (dg hg wg) c -> b c dg hg wg', dg=Dg, hg=Hg, wg=Wg)

def unpatchify3d(x_bcdhw: torch.Tensor, rd=1, rh=2, rw=2):
    """
    (B, C*rd*rh*rw, D, H, W) -> (B, C, D*rd, H*rh, W*rw)
    """
    B, Cmul, D, H, W = x_bcdhw.shape
    assert Cmul % (rd*rh*rw) == 0, f"C({Cmul}) not divisible by rd*rh*rw={rd*rh*rw}"
    return rearrange(x_bcdhw, 'b (c rd rh rw) d h w -> b c (d rd) (h rh) (w rw)', rd=rd, rh=rh, rw=rw)

class Up3D(nn.Module):
    """
    Upsample -> Conv3d(k, dilation) -> SiLU
    """
    def __init__(self, in_ch, out_ch, scale=(2,2,2), k=3, final_act=True, bias=True, dilation=1):
        super().__init__()
        self.scale = scale
        self.final_act = final_act
        self.upsample = nn.Upsample(scale_factor=scale, mode='trilinear', align_corners=False)
        def to3(v): return (v, v, v) if isinstance(v, int) else tuple(v)
        k3 = to3(k); d3 = to3(dilation)
        pad = tuple((d3[i] * (k3[i] - 1)) // 2 for i in range(3))
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=k3, padding=pad, dilation=d3, bias=bias)
        self.act  = nn.SiLU()
    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        if self.final_act: x = self.act(x)
        return x


class TokensToVolume(nn.Module):
    """
    """
    def __init__(self, spatial_size=(192,192,40), comp=(4,16,16),
                 first_pow2=1024, width_mult=1,
                 unpatch=(1,2,2), share_in_proj=False,
                 last_use_dilation=True, use_ckpt=True):
        super().__init__()

        self.spatial_size = [spatial_size[2], spatial_size[0], spatial_size[1]]
        self.comp = comp
        self.use_ckpt = use_ckpt

        Cin = 1536
        rd, rh, rw = unpatch
        self.rd, self.rh, self.rw = rd, rh, rw


        C0b = max(8, int(first_pow2 * width_mult))
        C0  = C0b
        C1  = max(1, C0 // 2)
        C2  = max(1, C1 // 2)


        assert Cin % (rd*rh*rw) == 0, f"Cin({Cin}) % (rd*rh*rw) != 0"
        in_ch = Cin // (rd*rh*rw)
        self.share_in_proj = share_in_proj
        if share_in_proj:
            self.in_proj_shared = nn.Conv3d(in_ch, C0, 1, bias=False)
        else:
            self.in_proj_low    = nn.Conv3d(in_ch, C0, 1, bias=False)
            self.in_proj_middle = nn.Conv3d(in_ch, C0, 1, bias=False)
            self.in_proj_high   = nn.Conv3d(in_ch, C0, 1, bias=False)

        self.upAB = Up3D(C0, C1, scale=(2,2,2), k=3, final_act=True)
        self.upCD = Up3D(C1, C2, scale=(2,2,2), k=3, final_act=True)

        if last_use_dilation:
            self.upEF = Up3D(C2, 4, scale=(1,2,2), k=3, dilation=(1,2,2), final_act=False)
        else:
            self.upEF = Up3D(C2, 4, scale=(1,2,2), k=(3,5,5), final_act=False)


        self.lat_mid_to_C1 = Up3D(C0, C1, scale=(2,2,2), k=1, final_act=False)
        self.lat_low_to_C2 = Up3D(C0, C2, scale=(4,4,4), k=1, final_act=False)

        self._init_near_zero()

    # --------- 内部工具 ---------
    def _maybe_ckpt(self, module, x):
        if self.use_ckpt and self.training:
            from torch.utils.checkpoint import checkpoint
            return checkpoint(module, x, use_reentrant=False)
        return module(x)

    def _to_base_feat(self, x_bnc: torch.Tensor, branch: str):
        # grid -> unpatchify -> in_proj
        x = tokens_to_grid(x_bnc, self.spatial_size, self.comp)
        x = unpatchify3d(x, rd=self.rd, rh=self.rh, rw=self.rw)  # (B,1536/4,D',2H',2W')
        if self.share_in_proj:
            x = self.in_proj_shared(x)
        else:
            x = getattr(self, f'in_proj_{branch}')(x)
        return x

    # --------- 初始化 ---------
    def _init_near_zero(self) -> None:
        """
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5),
                                        mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


    def forward(self, x_bnc_low: torch.Tensor, x_bnc_middle: torch.Tensor, x_bnc_high: torch.Tensor):
        D, H, W = self.spatial_size

        f_low    = self._to_base_feat(x_bnc_low,    'low')
        f_middle = self._to_base_feat(x_bnc_middle, 'middle')
        f_high   = self._to_base_feat(x_bnc_high,   'high')

        x = self._maybe_ckpt(self.upAB, f_high)                 # (C1, ×2)
        x = x + self.lat_mid_to_C1(f_middle)                    #  middle

        x = self._maybe_ckpt(self.upCD, x)                      # (C2, ×2)
        x = x + self.lat_low_to_C2(f_low)                       #  low

        x = self._maybe_ckpt(self.upEF, x)                      # (4, ×(1,2,2))
        assert x.shape[2:] == (D, H, W), f"got {x.shape[2:]}, want {(D,H,W)}"
        return x