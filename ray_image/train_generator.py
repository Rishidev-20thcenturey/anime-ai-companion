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


def mse_breakdown(pred, target, center=4):
    """Per-latent-channel and center-vs-border MSE of the flow objective.

    Diagnostic only -- never used to change the loss. ``pred``/``target`` are
    [B, C, H, W] flow velocities. Returns a dict of python scalars:
      flow_mse       : overall MSE
      per_channel    : list of per-channel MSE
      center_mse     : MSE over the central ``center x center`` latent block
      border_mse     : MSE over the surrounding border pixels
      border_ratio   : border_mse / flow_mse (how much error lives at edges)
    """
    err = (pred - target) ** 2
    flow_mse = float(err.mean())
    per_channel = err.mean(dim=(0, 2, 3)).tolist()
    h, w = err.shape[-2], err.shape[-1]
    ch = min(center, h)
    cw = min(center, w)
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2
    center_region = err[:, :, y0:y0 + ch, x0:x0 + cw]
    center_mse = float(center_region.mean())
    # Border = everything not in the central block.
    total = err.mean()
    border_mse = float((total * h * w - center_mse * ch * cw) / (h * w - ch * cw))
    return {
        "flow_mse": flow_mse,
        "per_channel": per_channel,
        "center_mse": center_mse,
        "border_mse": border_mse,
        "border_ratio": border_mse / flow_mse if flow_mse > 0 else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--vae", required=True)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--save", default="checkpoints/ray_image_v0_2_trained.pt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--diagnostics", action="store_true",
                        help="log per-channel / center-border flow MSE every "
                             "--diag-every steps (no effect on the loss)")
    parser.add_argument("--diag-every", type=int, default=100)
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

            if args.diagnostics and (step == 1 or step % args.diag_every == 0):
                db = mse_breakdown(pred, target)
                pbar.write(
                    f"[diag] step={step} flow_mse={db['flow_mse']:.4f} "
                    f"center={db['center_mse']:.4f} border={db['border_mse']:.4f} "
                    f"border_ratio={db['border_ratio']:.3f} "
                    f"per_channel=[{', '.join(f'{v:.4f}' for v in db['per_channel'])}]"
                )

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
