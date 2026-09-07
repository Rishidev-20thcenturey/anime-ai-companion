"""Train the RAY-IMAGE v0.1 prototype on an image/caption dataset.

The training loop keeps the project checkpointable so short free-GPU sessions
can be resumed later. v0.1 trains the VAE reconstruction path together with
the latent flow-matching objective.
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


def save_checkpoint(path, cfg, vocab, vae, text_encoder, dit, optimizer, step):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": cfg.__dict__,
            "vocab": vocab,
            "vae": vae.state_dict(),
            "text_encoder": text_encoder.state_dict(),
            "dit": dit.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--save", default="checkpoints/ray_image_v0_1.pt")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    cfg = RAYConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = RAYCaptionDataset(args.manifest, cfg.image_size)
    if len(dataset) < args.batch_size:
        raise ValueError(f"dataset has {len(dataset)} samples but batch size is {args.batch_size}")

    vocab = build_vocab((item["text"] for item in dataset.items), cfg.vocab_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    vae = RAYVAE(cfg.latent_channels, cfg.vae_base).to(device)
    text_encoder = RAYTextEncoder(cfg.vocab_size, cfg.text_dim, cfg.max_tokens).to(device)
    dit = RAYDiT(
        cfg.latent_channels,
        cfg.model_dim,
        cfg.depth,
        cfg.heads,
        cfg.patch_size,
        cfg.text_dim,
    ).to(device)

    optimizer = AdamW(
        list(vae.parameters()) + list(text_encoder.parameters()) + list(dit.parameters()),
        lr=args.lr,
        betas=(0.9, 0.99),
        weight_decay=0.01,
    )

    step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        vae.load_state_dict(checkpoint["vae"])
        text_encoder.load_state_dict(checkpoint["text_encoder"])
        dit.load_state_dict(checkpoint["dit"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        step = int(checkpoint.get("step", 0))
        saved_vocab = checkpoint.get("vocab")
        if saved_vocab:
            vocab = saved_vocab
        print(f"resumed from step={step}: {args.resume}")

    pbar = tqdm(total=args.steps, initial=step, desc=f"RAY-IMAGE ({device})")
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

            vae_total, rec, kl = vae_loss(recon, images, mean, logvar)
            flow = F.mse_loss(pred, target)
            total = vae_total + flow

            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                list(vae.parameters()) + list(text_encoder.parameters()) + list(dit.parameters()), 1.0
            )
            optimizer.step()

            step += 1
            pbar.update(1)
            pbar.set_postfix(
                total=f"{total.item():.4f}",
                recon=f"{rec.item():.4f}",
                kl=f"{kl.item():.4f}",
                flow=f"{flow.item():.4f}",
            )

            if step % 100 == 0 or step == args.steps:
                save_checkpoint(args.save, cfg, vocab, vae, text_encoder, dit, optimizer, step)

    pbar.close()
    print(f"saved checkpoint: {args.save}")


if __name__ == "__main__":
    main()
