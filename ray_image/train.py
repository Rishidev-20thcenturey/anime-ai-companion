"""Train the RAY-IMAGE v0.1 prototype on an image/caption dataset.

The training loop keeps the project checkpointable so short free-GPU sessions
can be resumed later. v0.1 trains the VAE reconstruction path together with
the latent flow-matching objective.
"""
import argparse

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import RAYConfig
from .dataset import RAYCaptionDataset, build_vocab, encode_text
from .flow import sample_flow_pair
from .utils import build_models, load_checkpoint, save_checkpoint, set_seed, vae_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--save", default="checkpoints/ray_image_v0_1.pt")
    parser.add_argument("--resume", default=None)
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

    models = build_models(cfg, device)
    vae, text_encoder, dit = models["vae"], models["text_encoder"], models["dit"]
    trainable = [*vae.parameters(), *text_encoder.parameters(), *dit.parameters()]
    optimizer = AdamW(trainable, lr=args.lr, betas=(0.9, 0.99), weight_decay=0.01)

    step = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume, device)
        for name, module in models.items():
            if name in checkpoint:
                module.load_state_dict(checkpoint[name])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        step = int(checkpoint.get("step", 0))
        saved_vocab = checkpoint.get("vocab")
        if saved_vocab:
            vocab = saved_vocab
        print(f"resumed from step={step}: {args.resume}")

    vae.train()
    text_encoder.train()
    dit.train()
    pbar = tqdm(total=args.steps, initial=step, desc=f"RAY-IMAGE ({device})")
    while step < args.steps:
        for images, captions in loader:
            if step >= args.steps:
                break
            images = images.to(device)
            tokens = torch.stack([encode_text(x, vocab, cfg.max_tokens) for x in captions]).to(device)
            # Pad positions (id 0) must be ignored by self/cross-attention and by
            # the pooled text conditioning in the DiT.
            text_mask = tokens.eq(0)

            recon, z, mean, logvar = vae(images)
            text = text_encoder(tokens, mask=text_mask)
            xt, t, target = sample_flow_pair(z.detach())
            pred = dit(xt, t, text, text_mask=text_mask)

            vae_total, rec, kl = vae_loss(recon, images, mean, logvar)
            flow = F.mse_loss(pred, target)
            total = vae_total + flow

            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
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
                save_checkpoint(
                    args.save,
                    cfg,
                    step,
                    vocab=vocab,
                    vae=vae,
                    text_encoder=text_encoder,
                    dit=dit,
                    optimizer=optimizer,
                )

    pbar.close()
    print(f"saved checkpoint: {args.save}")


if __name__ == "__main__":
    main()
