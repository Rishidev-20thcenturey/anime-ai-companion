"""Tiny smoke test: proves the RAY v0.1 modules can execute and backpropagate.

This is NOT the production training script. It deliberately uses random images
and random token IDs so the architecture can be validated before dataset work.
"""
import torch
import torch.nn.functional as F

from .config import RAYConfig
from .dit import RAYDiT
from .text_encoder import RAYTextEncoder
from .vae import RAYVAE
from .flow import sample_flow_pair


def main():
    cfg = RAYConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = RAYVAE(cfg.latent_channels).to(device)
    text = RAYTextEncoder(cfg.vocab_size, cfg.text_dim, cfg.max_tokens).to(device)
    dit = RAYDiT(cfg.latent_channels, cfg.model_dim, cfg.depth, cfg.heads, cfg.patch_size, cfg.text_dim).to(device)

    images = torch.rand(2, 3, cfg.image_size, cfg.image_size, device=device)
    tokens = torch.randint(0, cfg.vocab_size, (2, cfg.max_tokens), device=device)

    recon, z, _, _ = vae(images)
    cond = text(tokens)
    xt, t, target = sample_flow_pair(z.detach())
    pred = dit(xt, t, cond)
    loss = F.mse_loss(pred, target)
    loss.backward()

    print(f"device={device}")
    print(f"image={tuple(images.shape)} latent={tuple(z.shape)}")
    print(f"pred={tuple(pred.shape)} loss={loss.item():.6f}")
    print("RAY-IMAGE v0.1 smoke test: PASS")


if __name__ == "__main__":
    main()
