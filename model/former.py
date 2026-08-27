import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from model.archs.RetinexFormer_arch2 import *
from math import sqrt

    

    
class GlobalChannelAttention(nn.Module):
    def __init__(self, dim, reduction=4):
        super().__init__()

        self.channel_mlp = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim, bias=False),
            nn.Sigmoid()
        )
        
        self.global_proj = nn.Linear(dim, dim)
        self.fusion = nn.Linear(2 * dim, dim)

    def forward(self, x):
        b, c, h, w = x.shape
        g = F.adaptive_avg_pool2d(x, 1).view(b, c)
        g = self.global_proj(g)
        w_c = self.channel_mlp(g)

        fused = torch.cat([g, w_c], dim=-1)
        fused = self.fusion(fused).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]

        out = x * fused.expand_as(x)
        return out


class SGAB(nn.Module):
    def __init__(self, n_feats, drop=0.0, k=2, squeeze_factor= 15, attn ='GLKA'):   
        super().__init__()
        i_feats =n_feats*2
        
        self.Conv1 = nn.Conv2d(n_feats, i_feats, 1, 1, 0) 
        self.DWConv1 = nn.Conv2d(n_feats, n_feats, 7, 1, 7//2, groups= n_feats)     
        self.Conv2 = nn.Conv2d(n_feats, n_feats, 1, 1, 0)
        
        self.norm = LayerNorm(n_feats)
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)
        
    def forward(self, x):      
        shortcut = x.clone()
         
        x = self.Conv1(self.norm(x))
        a, x = torch.chunk(x, 2, dim=1) 
        x = x*self.DWConv1(a)
        x = self.Conv2(x)
        
        return  x*self.scale + shortcut       
    

    
class GLiFoM2(nn.Module):
    """
    非自注意力：多尺度局部 + 全局池化 + 通道SE + 空间门控 + 频域门控（低/高频能量自适应）
    Input/Output: [B,C,H,W]
    """
    def __init__(self, dim, rd=4):
        super().__init__()
        self.dim = dim
        mid = max(8, dim//rd)

        self.norm = LayerNorm(dim) 
        self.conv1 = nn.Conv2d(dim, 2*dim, 1)
        self.proj_in = nn.Conv2d(dim, dim, 1)

        self.local3 = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
                                    nn.Conv2d(dim, dim, 1))
        self.local5 = nn.Sequential(nn.Conv2d(dim, dim, 5, padding=2, groups=dim),
                                    nn.Conv2d(dim, dim, 1))
        self.local7 = nn.Sequential(nn.Conv2d(dim, dim, 9, padding=4, groups=dim),
                                    nn.Conv2d(dim, dim, 1))
        
        self.gate_s = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.Sigmoid())
        self.gate_m = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.Sigmoid())
        self.gate_l = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.Sigmoid())

        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, mid, 1), nn.GELU(),
            nn.Conv2d(mid, dim, 1), nn.Sigmoid()
        )

        self.sa = nn.Sequential(
            nn.Conv2d(dim, 1, kernel_size=7, padding=3, groups=1),
            nn.Sigmoid()
        )



        self.proj_out = nn.Conv2d(dim, dim, 1)
        self.scale = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.at = GlobalChannelAttention(dim)
        self.catconv = nn.Conv2d(2*dim, dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        B,C,H,W = x.shape
        shortcut = x
        x = self.norm(x)
        x = self.conv1(x)
        x, z = torch.chunk(x, 2, dim=1)
        x = self.proj_in(x)

        s = self.local3(x)
        m = self.local5(x)
        l = self.local7(x)

        loc = (s + m + l) # 大小核输出

        z = self.at(z) # GFB
        
        # HAFB 
        # loc = self.relu(self.catconv(torch.cat([z, loc], dim=1)))
        # ca = self.ca(loc)
        # sa = self.sa(loc)
        # y = loc * ca + loc * sa
        
        y = self.relu(self.catconv(torch.cat([z, loc], dim=1)))

        y = self.proj_out(y)
        return shortcut + y * self.scale
    





class NMAB(nn.Module):
    def __init__(self, n_feats):   
        super().__init__()
        
        self.LKA = GLiFoM2(n_feats) 
        self.LFE = SGAB(n_feats)
        
    def forward(self, x, s): 
        x  = self.LKA(x)  
        x = self.LFE(x)  
        
        return x   