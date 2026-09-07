"""Stage 2: train the text-conditioned latent flow generator with a frozen VAE."""
import argparse

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import RAYConfig
from .dataset import RAYCaptionDataset, build_vocab, encode_text
from .flow import sample_flow_pair
from .utils import build_models, load_checkpoint, load_pretrained, save_checkpoint, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--vae", required=True)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--save", default="checkpoints/ray_image_v0_2_trained.pt")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = RAYConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    dataset = RAYCaptionDataset(args.manifest, cfg.image_size)
    if len(dataset) < args.batch_size:
        raise ValueError(f"dataset has {len(dataset)} samples but batch size is {args.batch_size}")
    vocab = build_vocab((item["text"] for item in dataset.items), cfg.vocab_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # Load the frozen VAE checkpoint into a matching VAE instance.
    vae_ckpt = load_checkpoint(args.vae, device)
    vae = build_models(cfg, device, text_encoder=False, dit=False)["vae"]
    load_pretrained({"vae": vae}, vae_ckpt, device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    modules = build_models(cfg, device, vae=False)
    text_encoder, dit = modules["text_encoder"], modules["dit"]
    optimizer = AdamW(
        [*text_encoder.parameters(), *dit.parameters()],
        lr=args.lr,
        betas=(0.9, 0.99),
        weight_decay=0.01,
    )

    text_encoder.train()
    dit.train()
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
            trainable = [*text_encoder.parameters(), *dit.parameters()]
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            step += 1
            pbar.update(1)
            pbar.set_postfix(flow=f"{loss.item():.4f}")

    pbar.close()
    save_checkpoint(
        args.save,
        cfg,
        step,
        vocab=vocab,
        vae=vae,
        text_encoder=text_encoder,
        dit=dit,
        stage="generator_v0.2",
    )
    print(f"saved generator checkpoint: {args.save}")


if __name__ == "__main__":
    main()
