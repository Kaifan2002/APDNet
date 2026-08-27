import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class LinearAttention(nn.Module):
    """线性注意力机制（适用于 BCHW 输入）"""
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        # x: (B, H*W, C)
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, h, -1).transpose(1, 2), qkv)

        # 使用特征映射近似 softmax
        q = F.elu(q) + 1
        k = F.elu(k) + 1

        # 线性注意力计算
        k = k * self.scale
        context = torch.einsum('bhnd,bhne->bhde', k, v)
        out = torch.einsum('bhnd,bhde->bhne', q, context)
        out = out.reshape(b, n, -1)
        return self.to_out(out)

class FeedForward(nn.Module):
    """前馈网络"""
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class LinearTransformerBlock(nn.Module):
    """线性 Transformer 块（支持 BCHW）"""
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.attn = LinearAttention(dim, heads=heads, dim_head=dim_head)
        self.ff = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm1(x)
        x = x + self.attn(x)
        x = self.norm2(x)
        x = x + self.ff(x)
        return x

class LinearTransformer2D(nn.Module):
    """支持 BCHW 输入的线性 Transformer"""
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.dim = dim
        self.pos_emb = nn.Parameter(torch.randn(1, dim, 1, 1))  # 可学习的 2D 位置编码

        self.layers = nn.ModuleList([
            LinearTransformerBlock(dim, heads, dim_head, mlp_dim, dropout)
            for _ in range(depth)
        ])

    def forward(self, x):
        # x: (B, C, H, W)
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b, h * w, c)  # BCHW → B (H W) C

        # 添加可学习的位置编码
        pos_emb = self.pos_emb.expand(b, -1, h, w)  # (B, C, H, W)
        pos_emb = pos_emb.permute(0, 2, 3, 1).reshape(b, h * w, c)
        x = x + pos_emb

        # 通过线性 Transformer 块
        for layer in self.layers:
            x = layer(x)

        # 如果需要，可以 reshape 回 BCHW
        x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)  # B (H W) C → B C H W
        return x

# 示例用法
if __name__ == "__main__":
    model = LinearTransformer2D(
        dim=512,      # 输入通道数 (C)
        depth=6,       # Transformer 块层数
        heads=8,       # 注意力头数
        dim_head=64,   # 每个头的维度
        mlp_dim=1024,  # 前馈网络的隐藏层维度
        dropout=0.1    # Dropout 率
    )

    # 模拟输入 (B, C, H, W)
    x = torch.randn(1, 512, 32, 32)  # 假设输入是 32x32 的特征图
    out = model(x)
    print(out.shape)  # 输出仍然是 (1, 512, 32, 32)