"""Fast architecture smoke test for RAY-IMAGE v0.1.

This test uses random tensors only. It verifies tensor geometry, forward passes,
and gradient flow; it does not measure image quality.
"""
import torch
import torch.nn.functional as F

from .config import RAYConfig
from .flow import sample_flow_pair
from .utils import build_models


def main():
    cfg = RAYConfig()
    assert cfg.latent_channels == 16, cfg.latent_channels
    device = "cuda" if torch.cuda.is_available() else "cpu"

    models = build_models(cfg, device)
    vae, text, dit = models["vae"], models["text_encoder"], models["dit"]

    images = torch.rand(2, 3, cfg.image_size, cfg.image_size, device=device)
    tokens = torch.randint(0, cfg.vocab_size, (2, cfg.max_tokens), device=device)

    recon, z, mean, logvar = vae(images)
    assert z.shape[1] == cfg.latent_channels, (z.shape, cfg.latent_channels)
    assert z.shape[2:] == (cfg.latent_size, cfg.latent_size), (z.shape, cfg.latent_size)
    assert recon.shape == images.shape, (recon.shape, images.shape)

    cond = text(tokens)
    xt, t, target = sample_flow_pair(z.detach())
    pred = dit(xt, t, cond)
    assert pred.shape == target.shape, (pred.shape, target.shape)

    recon_loss = F.mse_loss(recon, images)
    kl = -0.5 * torch.mean(1 + logvar - mean.square() - logvar.exp())
    flow_loss = F.mse_loss(pred, target)
    loss = recon_loss + 1e-4 * kl + flow_loss
    loss.backward()

    print(f"device={device}")
    print(f"image={tuple(images.shape)} latent={tuple(z.shape)} tokens={cfg.latent_tokens}")
    print(f"pred={tuple(pred.shape)} loss={loss.item():.6f}")
    print("RAY-IMAGE v0.1 smoke test: PASS")


if __name__ == "__main__":
    main()
