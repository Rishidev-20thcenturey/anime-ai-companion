"""Stage 1: train the RAY-IMAGE VAE reconstruction model."""
import argparse

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import RAYConfig
from .dataset import RAYCaptionDataset
from .utils import build_models, save_checkpoint, set_seed, vae_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--save", default="checkpoints/ray_vae_v0_1.pt")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = RAYConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    dataset = RAYCaptionDataset(args.manifest, cfg.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    vae = build_models(cfg, device, text_encoder=False, dit=False)["vae"]
    optimizer = AdamW(vae.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.01)

    vae.train()
    pbar = tqdm(total=args.steps, desc=f"RAY-VAE ({device})")
    step = 0
    while step < args.steps:
        for images, _ in loader:
            if step >= args.steps:
                break
            images = images.to(device)
            recon, _, mean, logvar = vae(images)
            total, recon_loss, kl = vae_loss(recon, images, mean, logvar)
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            optimizer.step()
            step += 1
            pbar.update(1)
            pbar.set_postfix(loss=f"{total.item():.4f}", recon=f"{recon_loss.item():.4f}", kl=f"{kl.item():.4f}")

    pbar.close()
    save_checkpoint(args.save, cfg, step, vae=vae)
    print(f"saved VAE checkpoint: {args.save}")


if __name__ == "__main__":
    main()
