import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import torchvision
from thop import profile

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)



 
class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1,
                      bias=False, groups=dim * mult),
            nn.GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2).contiguous())
        return out.permute(0, 2, 3, 1)


class Attention(nn.Module):
    def __init__(self, dim, dim_head=64, heads = 8):
        super().__init__()
        dim_head = dim // heads
        project_out = not (heads == 1 and dim_head == dim)
        
        self.heads = heads
        self.dim_head = dim_head
        self.attend = nn.Softmax(dim = -1)
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        )

    def forward(self, x):
        """
        x_in: [b,h,w,c]         # input_feature
        return out: [b,h,w,c]
        """
        b, h, w, c = x.shape
        n = h*w
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads),
        #                          (q_inp, k_inp, v_inp))
        q = q_inp.view(b, n, self.heads, -1).transpose(1, 2)  # (b, h, n, d)
        k = k_inp.view(b, n, self.heads, -1).transpose(1, 2)  # (b, h, n, d)
        v = v_inp.view(b, n, self.heads, -1).transpose(1, 2) 
        
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1)) 
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v   # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)    # Transpose
        x = x.reshape(b, h * w, self.heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(
            0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p
    
        return out
 
class Transformer(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, num_blocks=2):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(num_blocks):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, fn = Attention(dim=dim, dim_head=dim_head, heads=heads)),
                PreNorm(dim, fn = FeedForward(dim=dim))
            ]))
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out
    
if __name__ == '__main__':
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
   # model = light_model()
    model = Transformer(dim=24, dim_head=64, heads=8, num_blocks=2)
    model.to(device)
    input_image = torch.randn(1, 24, 640, 640) # 示例输入
    input_image = input_image.to(device)
    # LLEM_out, ligt_map, DDIIR_out= model(input_image)
    # print(LLEM_out.shape, ligt_map.shape, DDIIR_out.shape)
    x1 = model(input_image)
    print(x1.shape)
    flops, params = profile(model, (input_image,))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))