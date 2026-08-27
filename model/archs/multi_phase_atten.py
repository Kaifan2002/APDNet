import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.fft as fft
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
from pdb import set_trace as stx
# import cv2
import numbers

class IG_MSA(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in):
        """
        x_in: [b,c,h,w]         # input_feature
        illu_fea: [b,h,w,c]      
        return out: [b,h,w,c]
        """

        x_in = x_in.permute(0, 2, 3, 1) # [b,c,h,w]  to [b,h,w,c]  
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        # illu_attn = illu_fea_trans # illu_fea: b,c,h,w -> b,h,w,c
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                 (q_inp, k_inp, v_inp))
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))   # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v   # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)    # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p

        # out = out.permute(0, 3, 1, 2)

        return out
    

# --------- 辅助函数 ----------
def fft_low_high_split(x, radius_ratio=0.25):
    # x: (B,C,H,W), 返回 low, high 实域张量
    B,C,H,W = x.shape
    X = fft.rfft2(x, norm='ortho')
    yy = torch.fft.fftfreq(H, d=1.0, device=x.device)[:,None]
    xx = torch.fft.rfftfreq(W, d=1.0, device=x.device)[None,:]
    rr = torch.sqrt(yy*yy + xx*xx)
    radius = radius_ratio * rr.max()
    low_mask = (rr <= radius).to(X.dtype)  # shape (H, W//2+1)
    low = fft.irfft2(X * low_mask[None,None,:,:], s=(H,W), norm='ortho')
    high = x - low
    return low, high

def upsample_to(x, size, mode='bilinear'):
    return F.interpolate(x, size=size, mode=mode, align_corners=False)

# --------- Illumination encoder（生成全局光照条件向量）----------
class IlluminationEncoderGlobal(nn.Module):
    """
    接受 illumination map (B,1,Hl,Wl) 返回一个全局条件向量 c (B, Ccond)
    """
    def __init__(self, in_ch=1, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_dim//2, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_dim//2),
            nn.GELU(),
            nn.Conv2d(out_dim//2, out_dim, 3, padding=1, bias=False),
            nn.AdaptiveAvgPool2d(1),
        )
    def forward(self, l):
        # l: (B,1,H,W) or (B,3,H,W) -> return (B, out_dim)
        if l.shape[1] != 1:
            l = l.mean(dim=1, keepdim=True)
        t = self.net(l)  # B, out_dim, 1, 1
        return t.flatten(1)  # B, out_dim

# --------- IG-DCM Block ----------
class IG_DynamicConvMetaBlock(nn.Module):
    """
    Illumination-Guided Dynamic-Conv MetaFormer Block (no attention)
    Inputs:
        x: (B, C, H, W)
        l: (B, 1 or 3, H_l, W_l)  -- illumination prior (can be lower-res)
    Config:
        basis_k: number of basis depthwise convs (K)
        kernel_size: size of basis kernels (k)
    """
    def __init__(self, C=64, basis_k=4, kernel_size=3, cond_dim=256, res_scale=0.1, freq_radius=0.25):
        super().__init__()
        assert kernel_size % 2 == 1
        self.C = C
        self.K = basis_k
        self.ks = kernel_size
        self.res_scale = res_scale
        self.freq_radius = freq_radius

        # 1) 基础 local processing（在 tokenization 之前）
        self.local_pre = nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1, groups=C, bias=False),  # depthwise
            nn.Conv2d(C, C, 1, bias=False),
            nn.GroupNorm(8, C),
            nn.GELU()
        )

        # 2) basis depthwise conv bank (K 个 depthwise convs, 每个 conv: groups=C)
        #    这里我们使用可学习的 shared basis kernels（不同 basis 代表不同滤波模式）
        self.basis_convs = nn.ModuleList([
            nn.Conv2d(C, C, kernel_size, padding=kernel_size//2, groups=C, bias=False)
            for _ in range(self.K)
        ])
        # 初始化 basis 稍微多样化
        for i, conv in enumerate(self.basis_convs):
            nn.init.kaiming_normal_(conv.weight, a=0.2)

        # 3) illumination encoder -> 生成每个 sample 的组合系数: coeffs shape (B, C, K)
        self.ill_enc = IlluminationEncoderGlobal(in_ch=1, out_dim=cond_dim)
        # 从 cond_dim 投影到 C*K coefficients（先做小瓶颈）
        self.coef_proj = nn.Sequential(
            nn.Linear(cond_dim, cond_dim//2),
            nn.GELU(),
            nn.Linear(cond_dim//2, C * self.K)
        )

        # 4) 频域门控（用局部化的illum来决定高频抑制）
        #    输入: illumination resized -> 输出 gate per channel (B,C,1,1)
        self.freq_gate_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(C, C, 1),
            nn.Sigmoid()
        )

        # 5) channel mixer (MLP style convs)
        self.channel_mixer = nn.Sequential(
            nn.Conv2d(C, C*2, 1),
            nn.GELU(),
            nn.Conv2d(C*2, C, 1)
        )

        # small norm on token tokenspace: use LayerNorm on flattened tokens if needed
        # but for conv pathway we keep GroupNorm already in place.

    def forward(self, x, l):
        """
        x: (B,C,H,W)
        l: (B,1 or 3, Hl, Wl)
        """
        B,C,H,W = x.shape
        assert C == self.C, "channel mismatch"

        # 1) local feature
        x_local = self.local_pre(x)  # (B,C,H,W)

        # 2) apply each basis conv -> get K feature maps of shape (B,C,H,W)
        basis_outputs = []
        for conv in self.basis_convs:
            basis_outputs.append(conv(x_local))  # each (B,C,H,W)
        # stack -> (B, K, C, H, W)
        stacked = torch.stack(basis_outputs, dim=1)

        # 3) get coefficients from illumination (global cond)
        cond = self.ill_enc(l)  # (B, cond_dim)
        coefs = self.coef_proj(cond)  # (B, C*K)
        coefs = coefs.view(B, C, self.K)  # (B, C, K)
        # normalize coefficients across K (softmax across K) for stability
        coefs = torch.softmax(coefs, dim=-1)  # (B,C,K)

        # 4) weighted sum of basis outputs per-channel per-sample
        # rearrange stacked to (B, C, K, H, W) to align
        stacked = stacked.permute(0,2,1,3,4)  # (B,C,K,H,W)
        # expand coefs for broadcasting: (B,C,K,1,1)
        coefs_exp = coefs.unsqueeze(-1).unsqueeze(-1)  # (B,C,K,1,1)
        # weighted sum over K -> (B,C,H,W)
        mixed = (stacked * coefs_exp).sum(dim=2)

        # 5) frequency-aware gating: decide how much to suppress high-frequency
        # resize illumination to x_local spatial size and feed gate net
        if l.shape[1] != 1:
            l_gray = l.mean(dim=1, keepdim=True)
        else:
            l_gray = l
        l_resized = upsample_to(l_gray, (H,W))
        # create per-channel gate from l_resized (we can combine l_resized with mixed)
        gate = self.freq_gate_net(l_resized)  # (B,C,1,1) values in (0,1)
        low, high = fft_low_high_split(mixed, radius_ratio=self.freq_radius)
        # suppress high-frequency where gate is high (illum==bright => keep more high-freq; dark => suppress noise)
        # choose design: let gate ~ illumination intensity -> more illum -> less suppression
        # here use (1 - gate) to suppress when illumination small
        mixed_freq = low + high * (1.0 * gate)  # tune sign if you want inverse behavior

        # 6) residual + channel mixer
        out = x + self.res_scale * mixed_freq
        out = out + self.channel_mixer(out)

        return out