import math
import torch
from torch import nn


class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freq = torch.exp(
            -math.log(10000)
            * torch.arange(half, device=t.device)
            / max(half - 1, 1)
        )
        x = t.float()[:, None] * freq[None, :]
        emb = torch.cat([x.sin(), x.cos()], dim=-1)
        return self.proj(emb)


class RAYDiTBlock(nn.Module):
    """DiT block with global timestep conditioning and token-level text cross-attention."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

        self.norm_cross = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.cond = nn.Linear(dim, dim * 2)

    def forward(self, x, cond, text, text_mask=None):
        scale, shift = self.cond(cond).chunk(2, dim=-1)
        h = self.norm1(x) * (1 + scale[:, None, :]) + shift[:, None, :]
        x = x + self.attn(h, h, h, need_weights=False)[0]

        # Every image token can directly attend to the full text sequence.
        h = self.norm_cross(x)
        text_ctx = text
        x = x + self.cross_attn(
            h,
            text_ctx,
            text_ctx,
            key_padding_mask=text_mask,
            need_weights=False,
        )[0]

        x = x + self.mlp(self.norm2(x))
        return x


class RAYDiT(nn.Module):
    """Tiny latent DiT with token-level text conditioning."""

    def __init__(self, latent_channels=4, dim=256, depth=6, heads=4, patch=2, cond_dim=256):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.patch = patch
        self.in_proj = nn.Conv2d(latent_channels, dim, patch, patch)
        self.time = TimestepEmbedding(dim)
        self.text_proj = nn.Linear(cond_dim, dim)
        self.text_ctx_proj = nn.Linear(cond_dim, dim) if cond_dim != dim else nn.Identity()
        self.blocks = nn.ModuleList([RAYDiTBlock(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, patch * patch * latent_channels)

    def forward(self, z, t, text, text_mask=None):
        h, w = z.shape[-2:]
        if h % self.patch or w % self.patch:
            raise ValueError(f"latent size {(h, w)} must be divisible by patch={self.patch}")

        x = self.in_proj(z)
        ph, pw = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)

        pos = self._sinusoidal_2d(ph, pw, x.shape[-1], x.device, x.dtype)
        x = x + pos[None]

        if text_mask is None:
            pooled = text.mean(dim=1)
        else:
            valid = (~text_mask).to(text.dtype).unsqueeze(-1)
            pooled = (text * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)

        cond = self.time(t) + self.text_proj(pooled)
        text_ctx = self.text_ctx_proj(text)
        for block in self.blocks:
            x = block(x, cond, text_ctx, text_mask=text_mask)

        x = self.out(self.norm(x))
        b, n, _ = x.shape
        x = x.view(b, ph, pw, self.patch, self.patch, z.shape[1])
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(b, z.shape[1], h, w)
        return x

    @staticmethod
    def _sinusoidal_2d(h, w, dim, device, dtype):
        if dim % 4 != 0:
            raise ValueError("model dim must be divisible by 4 for 2D sinusoidal positions")
        quarter = dim // 4
        y = torch.arange(h, device=device, dtype=dtype)
        x = torch.arange(w, device=device, dtype=dtype)
        freq = torch.exp(
            -math.log(10000)
            * torch.arange(quarter, device=device, dtype=dtype)
            / max(quarter - 1, 1)
        )
        yy = y[:, None] * freq[None, :]
        xx = x[:, None] * freq[None, :]
        yemb = torch.cat([yy.sin(), yy.cos()], dim=-1)[:, None, :].expand(h, w, quarter * 2)
        xemb = torch.cat([xx.sin(), xx.cos()], dim=-1)[None, :, :].expand(h, w, quarter * 2)
        return torch.cat([yemb, xemb], dim=-1).reshape(h * w, dim)
