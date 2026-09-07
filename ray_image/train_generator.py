"""Stage 2: train the text-conditioned latent flow generator with a frozen VAE."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--vae", required=True)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--save", default="/content/ray_image_v0_2_trained.pt")
    args = parser.parse_args()

    cfg = RAYConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = RAYCaptionDataset(args.manifest, cfg.image_size)
    if len(dataset) < args.batch_size:
        raise ValueError(f"dataset has {len(dataset)} samples but batch size is {args.batch_size}")
    vocab = build_vocab((item["text"] for item in dataset.items), cfg.vocab_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    vae_ckpt = torch.load(args.vae, map_location=device)
    vae = RAYVAE(cfg.latent_channels, cfg.vae_base).to(device)
    vae.load_state_dict(vae_ckpt["vae"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

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
        list(text_encoder.parameters()) + list(dit.parameters()),
        lr=args.lr,
        betas=(0.9, 0.99),
        weight_decay=0.01,
    )

    pbar = tqdm(total=args.steps, desc=f"RAY-IMAGE v0.2 generator ({device})")
    step = 0
    while step < args.steps:
        for images, captions in loader:
            if step >= args.steps:
                break
            images = images.to(device)
            tokens = torch.stack(
                [encode_text(x, vocab, cfg.max_tokens) for x in captions]
            ).to(device)
            text_mask = tokens.eq(0)

            # Use the VAE mean latent, not a fresh stochastic sample, so the
            # tiny semantic experiment gets a deterministic image target.
            with torch.no_grad():
                _, mean, _ = vae.encode(images)
                z = mean

            text = text_encoder(tokens, mask=text_mask)
            xt, t, target = sample_flow_pair(z)
            pred = dit(xt, t, text, text_mask=text_mask)
            loss = F.mse_loss(pred, target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            trainable = list(text_encoder.parameters()) + list(dit.parameters())
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            step += 1
            pbar.update(1)
            pbar.set_postfix(flow=f"{loss.item():.4f}")

    pbar.close()
    path = Path(args.save)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": cfg.__dict__,
            "vocab": vocab,
            "vae": vae.state_dict(),
            "text_encoder": text_encoder.state_dict(),
            "dit": dit.state_dict(),
            "step": step,
            "stage": "generator_v0.2",
        },
        path,
    )
    print(f"saved generator checkpoint: {path}")


if __name__ == "__main__":
    main()
