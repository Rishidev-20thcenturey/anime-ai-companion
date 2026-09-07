"""Train the RAY-IMAGE v0.1 prototype on an image/caption dataset.

This is intentionally small and educational. It trains the VAE reconstruction
path plus the latent flow-matching objective for the DiT/text encoder.
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import RAYConfig
from .dataset import RAYCaptionDataset, build_vocab, encode_text
from .dit import RAYDiT
from .flow import sample_flow_pair
from .text_encoder import RAYTextEncoder
from .vae import RAYVAE


def vae_loss(recon, image, mean, logvar, beta=1e-4):
    recon_term = F.mse_loss(recon, image)
    kl = -0.5 * torch.mean(1 + logvar - mean.square() - logvar.exp())
    return recon_term + beta * kl, recon_term, kl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--save", default="checkpoints/ray_image_v0_1.pt")
    args = parser.parse_args()

    cfg = RAYConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = RAYCaptionDataset(args.manifest, cfg.image_size)
    vocab = build_vocab((text for _, text in dataset.items), cfg.vocab_size)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    vae = RAYVAE(cfg.latent_channels).to(device)
    text_encoder = RAYTextEncoder(cfg.vocab_size, cfg.text_dim, cfg.max_tokens).to(device)
    dit = RAYDiT(cfg.latent_channels, cfg.model_dim, cfg.depth, cfg.heads, cfg.patch_size, cfg.text_dim).to(device)

    optimizer = AdamW(
        list(vae.parameters()) + list(text_encoder.parameters()) + list(dit.parameters()),
        lr=args.lr,
        betas=(0.9, 0.99),
        weight_decay=0.01,
    )

    step = 0
    pbar = tqdm(total=args.steps, desc=f"RAY-IMAGE ({device})")
    while step < args.steps:
        for images, captions in loader:
            if step >= args.steps:
                break
            images = images.to(device)
            tokens = torch.stack([encode_text(x, vocab, cfg.max_tokens) for x in captions]).to(device)

            recon, z, mean, logvar = vae(images)
            text = text_encoder(tokens)
            xt, t, target = sample_flow_pair(z.detach())
            pred = dit(xt, t, text)

            loss, rec, kl = vae_loss(recon, images, mean, logvar)
            flow = F.mse_loss(pred, target)
            total = loss + flow

            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                list(vae.parameters()) + list(text_encoder.parameters()) + list(dit.parameters()), 1.0
            )
            optimizer.step()

            step += 1
            pbar.update(1)
            pbar.set_postfix(total=f"{total.item():.4f}", recon=f"{rec.item():.4f}", flow=f"{flow.item():.4f}")

            if step % 100 == 0 or step == args.steps:
                path = Path(args.save)
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "config": cfg.__dict__,
                    "vocab": vocab,
                    "vae": vae.state_dict(),
                    "text_encoder": text_encoder.state_dict(),
                    "dit": dit.state_dict(),
                    "step": step,
                }, path)
    pbar.close()
    print(f"saved checkpoint: {args.save}")


if __name__ == "__main__":
    main()
