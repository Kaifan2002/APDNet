import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from model.archs.RetinexFormer_arch2 import *
from model.former import NMAB


class GatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(GatedConv2d, self).__init__()
        self.conv_feature = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding)
        self.conv_gate = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        feature = self.relu(self.conv_feature(x))  
        gate = self.sigmoid(self.conv_gate(x)) 
        return feature * gate  
    

class Dual_path_LLRC(nn.Module):
    def __init__(self, inc = 4, outc = 3):
        super(Dual_path_LLRC, self).__init__()
    
        self.conv1 = nn.Conv2d(in_channels=inc, out_channels=outc*10, kernel_size=1, bias=True)
        self.depth_conv = nn.Conv2d(in_channels=outc*10, out_channels=outc*10, kernel_size=5, padding=2, bias=True, groups=3)
        self.conv2 = nn.Conv2d(in_channels=outc*10, out_channels = outc, kernel_size=1, bias=True)
        self.gamma = nn.Parameter(torch.zeros((1, 3, 1, 1)), requires_grad=True)
        self.relu = nn.ReLU()
    def forward(self, x): # Light Stage中对于光照的表述其实不只是灰度，还有颜色，因此将这里改为三通道实现光照恢复

        mean_c = x.mean(dim=1).unsqueeze(1)
        input = torch.cat([x,mean_c], dim=1)
        D = self.relu(self.conv2(self.relu(self.depth_conv(self.relu(self.conv1(input))))))
        return D
  

class LLEM(nn.Module):
    def __init__(self):
        super(LLEM, self).__init__()
        self.Lightingchannel1 = Dual_path_LLRC()
    def forward(self, x):
        D = self.Lightingchannel1(x)
        return D
    

class amp_processing(nn.Module):
    def __init__(self, inc, outc):
        super(amp_processing, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=inc, out_channels=inc, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=False)
        self.conv3 = nn.Conv2d(in_channels=outc, out_channels=inc, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(in_channels=inc, out_channels=inc, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv5 = nn.Conv2d(in_channels=inc, out_channels=inc, kernel_size=1, stride=1, padding=0, bias=False)
        self.gelu = nn.GELU()

    def forward(self, x):
        _, _, H, W = x.shape
        x = self.conv1(x)
        fft_x = torch.fft.fft2(x, dim=(-2, -1), norm='backward') 
        amp = torch.abs(fft_x)
        phase = torch.angle(fft_x)
        x_amp = self.conv3(self.relu(self.conv2(amp)))
        #phase = self.conv4(phase)
        
        fre_out = torch.fft.ifft2(x_amp * torch.exp(1j * phase), dim=(-2, -1)).real
        fre_out = fre_out + x
        out = self.gelu(self.conv5(fre_out))

        return out
    
class HaarDWT(nn.Module):
    def __init__(self):
        super().__init__()
        # 4 个基础核: LL, LH, HL, HH
        h = 1 / 2**0.5
        self.register_buffer("ll", torch.tensor([[h, h], [h, h]], dtype=torch.float32))
        self.register_buffer("lh", torch.tensor([[-h, -h], [h, h]], dtype=torch.float32))
        self.register_buffer("hl", torch.tensor([[-h, h], [-h, h]], dtype=torch.float32))
        self.register_buffer("hh", torch.tensor([[h, -h], [-h, h]], dtype=torch.float32))

    def forward(self, x):
        B, C, H, W = x.shape

        pad_h = (H % 2)
        pad_w = (W % 2)
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h)) 

        filters = torch.stack([self.ll, self.lh, self.hl, self.hh], dim=0)  # [4,2,2]
        filters = filters.unsqueeze(1).to(x.device)  # [4,1,2,2]
        filters = filters.repeat(C, 1, 1, 1)  # [4C,1,2,2]

        y = F.conv2d(x, filters, stride=2, groups=C)  # [B,4C,H/2,W/2]
        y = y.view(B, C, 4, y.shape[-2], y.shape[-1])
        LL, LH, HL, HH = y[:, :, 0], y[:, :, 1], y[:, :, 2], y[:, :, 3]
        return LL, LH, HL, HH, H, W 
        

class HaarIDWT(nn.Module):
    def __init__(self):
        super().__init__()
        h = 1 / 2**0.5
        self.register_buffer("ll", torch.tensor([[h, h], [h, h]], dtype=torch.float32))
        self.register_buffer("lh", torch.tensor([[-h, -h], [h, h]], dtype=torch.float32))
        self.register_buffer("hl", torch.tensor([[-h, h], [-h, h]], dtype=torch.float32))
        self.register_buffer("hh", torch.tensor([[h, -h], [-h, h]], dtype=torch.float32))

    def forward(self, LL, LH, HL, HH, H, W):
        B, C, h, w = LL.shape
        filters = torch.stack([self.ll, self.lh, self.hl, self.hh], dim=0)  # [4,2,2]
        filters = filters.unsqueeze(1).to(LL.device)  # [4,1,2,2]
        filters = filters.repeat(C, 1, 1, 1)  # [4C,1,2,2]

        y = torch.stack([LL, LH, HL, HH], dim=2).view(B, 4*C, h, w)
        out = F.conv_transpose2d(y, filters, stride=2, groups=C)  # [B,C,H',W']

        return out[:, :, :H, :W]

    
class WaveletFFN(nn.Module):

    def __init__(self, inc, outc):
        super(WaveletFFN, self).__init__()
        self.dwt = HaarDWT()
        self.idwt = HaarIDWT()

        # 低频卷积
        self.low_conv = nn.Sequential(
            nn.Conv2d(inc, outc, 3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(outc, inc, 1, bias=False)
        )

        # 高频卷积
        self.high_conv = nn.Sequential(
            nn.Conv2d(3*inc, outc, 3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(outc, 3*inc, 1, bias=False)
        )

        self.gelu = nn.GELU()
        self.proj = nn.Conv2d(inc, inc, 1)

    def forward(self, x):
        LL, LH, HL, HH, H, W = self.dwt(x)

        LL_out = self.low_conv(LL)

        high = torch.cat([LH, HL, HH], dim=1)   
        high_out = self.high_conv(high)
        LH_out, HL_out, HH_out = torch.chunk(high_out, 3, dim=1)

        out = self.idwt(LL_out, LH_out, HL_out, HH_out, H, W)
        out = out + x
        out = self.gelu(self.proj(out))

        return out  

class MLP(nn.Module):
    def __init__(self, inc, outc):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(inc, outc, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(outc, inc, kernel_size=1, bias=False)
        )
    def forward(self, x):
        return self.mlp(x)


# class space_processing(nn.Module):
#     def __init__(self,inc, outc):
#         super(space_processing, self).__init__()
#         self.body = nn.Sequential(
#             nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.LeakyReLU(0.01),
#             nn.Conv2d(in_channels=outc, out_channels=outc, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.LeakyReLU(0.01),
#             nn.Conv2d(in_channels=outc, out_channels=inc, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.LeakyReLU(0.01)
#         )
#     def forward(self, x):
#         original_size = x.shape[-2:]
#         x = self.body(x)
#         if x.shape[-2:] != original_size:
#             x = F.interpolate(x, size=original_size, mode='bilinear', align_corners=False)
#         return x



class space_processing(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()

        self.body = nn.Sequential(

            # expand
            nn.Conv2d(inc, outc, 1, bias=True),
            nn.LeakyReLU(0.01, inplace=True),

            # depthwise
            nn.Conv2d(
                outc,
                outc,
                kernel_size=3,
                padding=1,
                groups=outc,
                bias=True
            ),
            nn.LeakyReLU(0.01, inplace=True),

            # project
            nn.Conv2d(outc, inc, 1, bias=True),
            nn.LeakyReLU(0.01, inplace=True)
        )

    def forward(self, x):
        return self.body(x)

    
class Cross_attention1(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Cross_attention1, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv= nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.gateconv = GatedConv2d(dim, dim, kernel_size=1)
        self.scale = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.Sigmoid()
        )
        self.cat = nn.Sequential(nn.Conv2d(2*dim, dim, 3, padding=1), nn.ReLU()) 
        
    def forward(self, x, s):
        #x = x.permute(0, 3, 1, 2)
        b,c,h,w = x.shape
        s = self.gateconv(s)
        kv = self.kv_dwconv(self.kv(s))
        k,v = kv.chunk(2, dim=1)   
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        
        q = self.q_dwconv(self.q(x))   
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        #out = out * x * self.scale
        out = self.project_out(out)
        pos = self.pos_emb(x)
        out = out + pos

        return out






class FFTBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.la1 = LayerNorm(dim)
        self.space = space_processing(dim, dim*4)
        self.atten = Cross_attention1(dim, num_heads=8, bias=True)
        self.la2 = LayerNorm(dim)
        self.s_conv=  nn.Conv2d(3, dim, kernel_size=3, stride=1, padding=1)
        self.s_norm = LayerNorm(dim)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim*4, 1), nn.GELU(),
            nn.Conv2d(dim*4, dim, 1), nn.Sigmoid()
        )
        self.scale1 = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.scale2 = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.ffn = WaveletFFN(dim, dim*4) 
        # self.ffn = MLP(dim, dim*4)

    def forward(self, x ,s):
        scale_factor = 24/self.dim
        s_ = F.interpolate(s, scale_factor=scale_factor, mode='bilinear', align_corners=False)
        s_ = self.s_norm(self.s_conv(s_))

        x0 = self.la1(x)

        x1 = self.space(x0)
        x2 = self.atten(x1, s_)
        ca = self.ca(x2)
        x2 = x2 * ca
        x3 = x + self.scale1 * x2
        x4 = self.la2(x3)
        x5 = self.ffn(x4)
        out = self.scale2 * x5 + x3

        return out
    
    
class Downsample(nn.Module):
    def __init__(self, inc):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(inc, inc*2, kernel_size=3, stride=2, padding=1, bias=False))
    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, inc):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(nn.ConvTranspose2d(in_channels=inc, out_channels=inc//2, kernel_size=4,stride=2,  padding=1, output_padding=0))
    def forward(self, x):
        return self.body(x)
    

class SequentialWithS(nn.Sequential):
    def forward(self, x, s):
        for module in self:
            x = module(x, s)
        return x
    

class DDIIR(nn.Module): # Dual-Domain Image Information Reconstruction
    #def __init__(self, inc = 3 , outc = 48, dim = 48, num_blocks = [4,6,6,8], heads = [1,2,4,8]):
    def __init__(self, inc = 3 , outc = 24, dim = 24, num_blocks = [2,2,3,2]): #
        super(DDIIR, self).__init__()
        self.encoder_level1 = SequentialWithS(*[FFTBlock( dim) for i in range(num_blocks[0])])
        self.encoder_level2 = SequentialWithS(*[FFTBlock( dim*2**1) for i in range(num_blocks[1])])
        self.encoder_level3 = SequentialWithS(*[FFTBlock(dim*2**2) for i in range(num_blocks[2])])
        # self.encoder_level1 = SequentialWithS(*[NMAB( n_feats = dim) for i in range(num_blocks[0])])
        # self.encoder_level2 = SequentialWithS(*[NMAB( n_feats = dim*2**1) for i in range(num_blocks[1])])
        # self.encoder_level3 = SequentialWithS(*[NMAB( n_feats = dim*2**2) for i in range(num_blocks[2])])
        
        self.midlayer1 = SequentialWithS(*[FFTBlock(dim*2**3) for i in range(num_blocks[3])])
        # self.midlayer1 = SequentialWithS(*[NMAB( n_feats = dim*2**3) for i in range(num_blocks[3])])
        self.midlayer2 = SequentialWithS(*[NMAB( n_feats = dim*2**3) for i in range(num_blocks[3])])
        # self.midlayer2 = SequentialWithS(*[FFTBlock(dim*2**3) for i in range(num_blocks[3])])
        
        self.conv1 = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=3, stride=1, padding=1)
        
        self.down1_2 = Downsample(dim)
        self.down2_3 = Downsample(int(dim*2**1))
        self.down3_4 = Downsample(int(dim*2**2))
        
        self.decoder_level1 = SequentialWithS(*[NMAB( n_feats = dim)for i in range(num_blocks[0])])
        self.decoder_level2 = SequentialWithS(*[NMAB( n_feats = dim*2**1) for i in range(num_blocks[1])])
        self.decoder_level3 = SequentialWithS(*[NMAB( n_feats = dim*2**2) for i in range(num_blocks[2])])
        # self.decoder_level1 = SequentialWithS(*[FFTBlock( dim) for i in range(num_blocks[0])])
        # self.decoder_level2 = SequentialWithS(*[FFTBlock( dim*2**1) for i in range(num_blocks[1])])
        # self.decoder_level3 = SequentialWithS(*[FFTBlock( dim*2**2) for i in range(num_blocks[2])])


        self.relu = nn.ReLU(inplace=True)

        self.up4_3 = Upsample(int(dim*2**3))
        self.up3_2 = Upsample(int(dim*2**2))
        self.up2_1 = Upsample(int(dim*2)) 
        #self.reduce_chan = nn.Conv2d(int(dim), int(dim), kernel_size=1, bias=False)
        self.out_channel = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=inc, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

        self.connect_conv1 = nn.Conv2d(in_channels=dim*2, out_channels=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.connect_conv2 = nn.Conv2d(in_channels=dim*2**2, out_channels=dim*2**1, kernel_size=1, stride=1, padding=0, bias=True)
        self.connect_conv3 = nn.Conv2d(in_channels=dim*2**3, out_channels=dim*2**2, kernel_size=1, stride=1, padding=0, bias=True)
        
        self.mid_conv = nn.Conv2d(in_channels=dim*2**3, out_channels=3, kernel_size=3, stride=1, padding=1)

        #self.refinement = nn.Sequential(*[restore_block( dim = dim, depth = num_blocks[0], mlp_dim = 64) for i in range(num_blocks[0])])

    def forward(self, x, s):
        en1_in = self.relu(self.conv1(x))#0709加入relu
        en1_out = self.encoder_level1(en1_in,s)
        en2_in = self.down1_2(en1_out)
        en2_out = self.encoder_level2(en2_in, s)
        en3_in = self.down2_3(en2_out)
        en3_out = self.encoder_level3(en3_in, s)


        mid_in = self.down3_4(en3_out)
        mid_out1 = self.midlayer1(mid_in, s)


        mid_out2 = self.midlayer2(mid_out1, s)
        dec3_in = self.up4_3(mid_out2 + mid_out1)

        connect3 = self.connect_conv3(torch.cat([dec3_in ,en3_out], dim=1))
        dec3_out = self.decoder_level3(connect3, s)
        dec2_in = self.up3_2(dec3_out)
 
        connect2 = self.connect_conv2(torch.cat([dec2_in ,en2_out], dim=1))
        dec2_out = self.decoder_level2(connect2, s)
        dec1_in = self.up2_1(dec2_out)

        connect1 = self.connect_conv1(torch.cat([dec1_in ,en1_out], dim=1))
        dec1_out = self.decoder_level1(connect1, s)
        #out = self.reduce_chan(dec1_out)
        out = self.out_channel(dec1_out)
        return  out + x


