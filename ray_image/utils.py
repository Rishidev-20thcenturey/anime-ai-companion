"""Shared helpers for RAY-IMAGE training, checkpointing, and seeding.

These utilities live in one place so the single-stage ``train`` entry point and
the staged ``train_vae`` / ``train_generator`` scripts stay consistent (same VAE
objective, same checkpoint format, same model construction).
"""
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import RAYConfig
from .dit import RAYDiT
from .text_encoder import RAYTextEncoder
from .vae import RAYVAE


def load_checkpoint(path, device=None):
    """Load one of our own project checkpoints.

    RAY-IMAGE checkpoints may embed arbitrary project state (an optimizer state
    dict, the vocabulary dict, config), so ``weights_only`` must be disabled:
    torch >= 2.6 defaults to ``weights_only=True`` and would reject those
    objects. These files are written by our own scripts, so this is trusted.
    """
    return torch.load(path, map_location=device, weights_only=False)


def set_seed(seed: int) -> None:
    """Seed Python/torch (and cuda) for reproducible runs."""
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_models(cfg: RAYConfig, device, *, vae=True, text_encoder=True, dit=True):
    """Instantiate the modules requested for a given training/generation stage.

    Returns a dict mapping module name -> module. Modules not requested (passing
    ``vae=False`` etc.) are omitted so a stage never holds unused weights.
    """
    modules = {}
    if vae:
        modules["vae"] = RAYVAE(cfg.latent_channels, cfg.vae_base).to(device)
    if text_encoder:
        modules["text_encoder"] = RAYTextEncoder(
            cfg.vocab_size, cfg.text_dim, cfg.max_tokens
        ).to(device)
    if dit:
        modules["dit"] = RAYDiT(
            cfg.latent_channels,
            cfg.model_dim,
            cfg.depth,
            cfg.heads,
            cfg.patch_size,
            cfg.text_dim,
            cross_gate=getattr(cfg, "dit_cross_gate", False),
        ).to(device)
    return modules


def vae_loss(recon, image, mean, logvar, beta=1e-4):
    """Per-image reconstruction loss + KL for the RAY VAE.

    Returns ``(total, recon_term, kl)`` so callers can log each term.
    """
    recon_term = F.mse_loss(recon, image)
    kl = -0.5 * torch.mean(1 + logvar - mean.square() - logvar.exp())
    return recon_term + beta * kl, recon_term, kl


def save_checkpoint(path, cfg: RAYConfig, step: int, **state):
    """Write a checkpoint containing only the objects provided via keywords.

    Each ``torch.nn.Module`` value is stored as its ``state_dict``; everything
    else (e.g. ``vocab``, ``stage``, an ``optimizer`` state dict) is stored
    as-is. ``None`` values are skipped, so a VAE-only stage can pass just the
    VAE while a full stage passes all three models plus the optimizer.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"config": cfg.__dict__, "step": step}
    for key, value in state.items():
        if value is None:
            continue
        if isinstance(value, (torch.nn.Module, torch.optim.Optimizer)):
            # Store serializable state dicts, not the live objects.
            payload[key] = value.state_dict()
        else:
            payload[key] = value
    torch.save(payload, path)
    return path


def load_pretrained(modules, checkpoint, device):
    """Load state dicts from a checkpoint into a dict of modules (in place).

    Only keys present in both the checkpoint and the module are loaded, so a
    checkpoint can carry extra/missing pieces (e.g. a VAE-only file) safely.
    """
    for name, module in modules.items():
        if name in checkpoint:
            module.load_state_dict(checkpoint[name])
    return modules
