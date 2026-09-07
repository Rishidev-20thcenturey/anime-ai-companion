import math
import torch
from torch import nn


class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freq = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / max(half - 1, 1))
        x = t.float()[:, None] * freq[None, :]
        emb = torch.cat([x.sin(), x.cos()], dim=-1)
        return self.proj(emb)


class RAYDiTBlock(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.cond = nn.Linear(dim, dim * 2)

    def forward(self, x, cond):
        scale, shift = self.cond(cond).chunk(2, dim=-1)
        h = self.norm1(x) * (1 + scale[:, None, :]) + shift[:, None, :]
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class RAYDiT(nn.Module):
    """Small latent diffusion transformer for 64x64x4 latents."""

    def __init__(self, latent_channels=4, dim=384, depth=12, heads=6, patch=4, cond_dim=384):
        super().__init__()
        self.patch = patch
        self.in_proj = nn.Conv2d(latent_channels, dim, patch, patch)
        self.pos = nn.Parameter(torch.randn(1, 256, dim) * 0.02)
        self.time = TimestepEmbedding(dim)
        self.text_proj = nn.Linear(cond_dim, dim)
        self.blocks = nn.ModuleList([RAYDiTBlock(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, patch * patch * latent_channels)

    def forward(self, z, t, text):
        x = self.in_proj(z).flatten(2).transpose(1, 2)
        x = x + self.pos[:, :x.shape[1]]
        cond = self.time(t) + self.text_proj(text.mean(dim=1))
        for block in self.blocks:
            x = block(x, cond)
        x = self.out(self.norm(x))
        b, n, d = x.shape
        side = int(n ** 0.5)
        x = x.reshape(b, side, side, self.patch, self.patch, z.shape[1])
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(b, z.shape[1], side * self.patch, side * self.patch)
        return x
