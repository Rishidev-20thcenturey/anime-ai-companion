"""Stage 1: train the RAY-IMAGE VAE reconstruction model."""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import RAYConfig
from .dataset import RAYCaptionDataset
from .vae import RAYVAE


def loss_fn(recon, image, mean, logvar, beta=1e-4):
    recon_loss = F.mse_loss(recon, image)
    kl = -0.5 * torch.mean(1 + logvar - mean.square() - logvar.exp())
    return recon_loss + beta * kl, recon_loss, kl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--save", default="/content/ray_vae_v0_1.pt")
    args = parser.parse_args()

    cfg = RAYConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = RAYCaptionDataset(args.manifest, cfg.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    vae = RAYVAE(cfg.latent_channels, cfg.vae_base).to(device)
    optimizer = AdamW(vae.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.01)

    pbar = tqdm(total=args.steps, desc=f"RAY-VAE ({device})")
    step = 0
    vae.train()
    while step < args.steps:
        for images, _ in loader:
            if step >= args.steps:
                break
            images = images.to(device)
            recon, _, mean, logvar = vae(images)
            total, recon_loss, kl = loss_fn(recon, images, mean, logvar)
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            optimizer.step()
            step += 1
            pbar.update(1)
            pbar.set_postfix(loss=f"{total.item():.4f}", recon=f"{recon_loss.item():.4f}", kl=f"{kl.item():.4f}")

    pbar.close()
    path = Path(args.save)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": cfg.__dict__, "vae": vae.state_dict(), "step": step}, path)
    print(f"saved VAE checkpoint: {path}")


if __name__ == "__main__":
    main()
