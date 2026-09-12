import math

import torch
from torch import nn


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding followed by an MLP."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        if self.dim % 2 != 0:
            raise ValueError("timestep embedding dim must be even")
        freq = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        x = t.float()[:, None] * freq[None, :]
        emb = torch.cat([x.sin(), x.cos()], dim=-1)
        return self.mlp(emb)


class RAYDiTBlock_v2(nn.Module):
    """Pre-norm DiT block with gated text cross-attention."""

    def __init__(self, dim: int, heads: int, cross_gate: bool = False):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")

        self.cross_gate = None
        if cross_gate:
            self.cross_gate = nn.Parameter(torch.ones(1))

        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)

        self.norm_cross = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

        # Global conditioning from timestep + pooled text.
        self.cond = nn.Linear(dim, dim * 2)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        text: torch.Tensor,
        text_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scale, shift = self.cond(cond).chunk(2, dim=-1)
        h = self.norm1(x)
        h = h * (1.0 + scale[:, None, :]) + shift[:, None, :]
        x = x + self.self_attn(h, h, h, need_weights=False)[0]

        h = self.norm_cross(x)
        cross = self.cross_attn(
            h,
            text,
            text,
            key_padding_mask=text_mask,
            need_weights=False,
        )[0]
        if self.cross_gate is not None:
            x = x + self.cross_gate * cross
        else:
            x = x + cross

        x = x + self.mlp(self.norm2(x))
        return x


class RAYDiT_v2(nn.Module):
    """RAY-IMAGE DiT v2 for 16x32x32 latents and Qwen-sized text context."""

    def __init__(
        self,
        latent_channels=16,
        latent_size=32,
        dim=768,
        depth=18,
        heads=12,
        patch=2,
        cond_dim=2560,
        cross_gate: bool = False,
    ):
        super().__init__()
        if latent_size % patch != 0:
            raise ValueError("latent_size must be divisible by patch")
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")

        self.latent_channels = latent_channels
        self.latent_size = latent_size
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.patch = patch
        self.cond_dim = cond_dim

        # [B, 16, 32, 32] -> [B, 768, 16, 16] -> [B, 256, 768].
        self.in_proj = nn.Conv2d(
            latent_channels,
            dim,
            kernel_size=patch,
            stride=patch,
        )
        self.time = TimestepEmbedding(dim)
        self.text_proj = nn.Linear(cond_dim, dim)
        self.text_ctx_proj = (
            nn.Linear(cond_dim, dim) if cond_dim != dim else nn.Identity()
        )

        self.blocks = nn.ModuleList(
            [
                RAYDiTBlock_v2(dim, heads, cross_gate=cross_gate)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, patch * patch * latent_channels)

    def forward(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        text: torch.Tensor,
        text_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if z.ndim != 4:
            raise ValueError(f"expected latent tensor [B,C,H,W], got {tuple(z.shape)}")
        b, c, h, w = z.shape
        if c != self.latent_channels:
            raise ValueError(
                f"expected {self.latent_channels} latent channels, got {c}"
            )
        if h != self.latent_size or w != self.latent_size:
            raise ValueError(
                f"expected latent spatial size {self.latent_size}x{self.latent_size}, "
                f"got {h}x{w}"
            )
        if text.ndim != 3 or text.shape[0] != b or text.shape[2] != self.cond_dim:
            raise ValueError(
                f"expected text [B,L,{self.cond_dim}], got {tuple(text.shape)}"
            )
        if t.ndim != 1 or t.shape[0] != b:
            raise ValueError(f"expected timestep [B], got {tuple(t.shape)}")
        if text_mask is not None and text_mask.shape != text.shape[:2]:
            raise ValueError(
                f"expected text_mask {tuple(text.shape[:2])}, got {tuple(text_mask.shape)}"
            )

        x = self.in_proj(z)
        ph, pw = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        x = x + self._sinusoidal_2d(ph, pw, self.dim, x.device, x.dtype)[None]

        text = text.to(dtype=x.dtype)
        if text_mask is None:
            pooled = text.mean(dim=1)
        else:
            valid = (~text_mask).to(dtype=text.dtype).unsqueeze(-1)
            pooled = (text * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)

        cond = self.time(t) + self.text_proj(pooled)
        text_ctx = self.text_ctx_proj(text)
        for block in self.blocks:
            x = block(x, cond, text_ctx, text_mask=text_mask)

        x = self.out(self.norm(x))
        x = x.view(
            b,
            ph,
            pw,
            self.patch,
            self.patch,
            self.latent_channels,
        )
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(
            b, self.latent_channels, h, w
        )
        return x

    @staticmethod
    def _sinusoidal_2d(
        h: int,
        w: int,
        dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if dim % 4 != 0:
            raise ValueError("model dim must be divisible by 4 for 2D sinusoidal positions")

        quarter = dim // 4
        y = torch.arange(h, device=device, dtype=dtype)
        x = torch.arange(w, device=device, dtype=dtype)
        freq = torch.exp(
            -math.log(10000.0)
            * torch.arange(quarter, device=device, dtype=dtype)
            / max(quarter - 1, 1)
        )

        yy = y[:, None] * freq[None, :]
        xx = x[:, None] * freq[None, :]
        yemb = torch.cat([yy.sin(), yy.cos()], dim=-1)[:, None, :].expand(
            h, w, quarter * 2
        )
        xemb = torch.cat([xx.sin(), xx.cos()], dim=-1)[None, :, :].expand(
            h, w, quarter * 2
        )
        return torch.cat([yemb, xemb], dim=-1).reshape(h * w, dim)
