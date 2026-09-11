"""Shared helpers for RAY-IMAGE training, checkpointing, and seeding."""
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import RAYConfig
from .dit import RAYDiT
from .text_encoder import LegacyRAYTextEncoder, RAYTextEncoder
from .vae import RAYVAE


def load_checkpoint(path, device=None):
    """Load one of our own project checkpoints."""
    return torch.load(path, map_location=device, weights_only=False)


def set_seed(seed: int) -> None:
    """Seed Python/torch (and cuda) for reproducible runs."""
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_models(cfg: RAYConfig, device, *, vae=True, text_encoder=True, dit=True):
    """Instantiate the requested modules.

    ``text_encoder_backend='qwen3'`` selects the frozen Qwen3-4B encoder;
    the default legacy backend preserves existing N0-N7 checkpoints/smokes.
    """
    modules = {}
    if vae:
        modules["vae"] = RAYVAE(cfg.latent_channels, cfg.vae_base).to(device)
    if text_encoder:
        backend = getattr(cfg, "text_encoder_backend", "legacy")
        if backend == "qwen3":
            modules["text_encoder"] = RAYTextEncoder(max_tokens=cfg.max_tokens)
        else:
            modules["text_encoder"] = LegacyRAYTextEncoder(
                cfg.vocab_size, cfg.text_dim, cfg.max_tokens
            ).to(device)
    if dit:
        modules["dit"] = RAYDiT(
            cfg.latent_channels,
            cfg.model_dim,
            cfg.depth,
            cfg.heads,
            cfg.patch_size,
            cond_dim=cfg.text_dim,
            cross_gate=getattr(cfg, "dit_cross_gate", False),
        ).to(device)
    return modules


def vae_loss(recon, image, mean, logvar, beta=1e-4):
    recon_term = F.mse_loss(recon, image)
    kl = -0.5 * torch.mean(1 + logvar - mean.square() - logvar.exp())
    return recon_term + beta * kl, recon_term, kl


def save_checkpoint(path, cfg: RAYConfig, step: int, **state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"config": cfg.__dict__, "step": step}
    for key, value in state.items():
        if value is None:
            continue
        if isinstance(value, (torch.nn.Module, torch.optim.Optimizer)):
            payload[key] = value.state_dict()
        else:
            payload[key] = value
    torch.save(payload, path)
    return path


def load_pretrained(modules, checkpoint, device):
    """Load state dicts from a checkpoint into modules in place."""
    for name, module in modules.items():
        if name in checkpoint:
            module.load_state_dict(checkpoint[name])
    return modules
