"""N8 VAE v2 training on streamed MONET images via the HF Datasets API.

Dataset: jasperai/monet, streamed from Hugging Face. Each sample exposes a
``thumbnail`` PIL image; it is resized to 256x256 on the fly.

Loss schedule:
  0-1999  : L1
  2000-3999: L1 + 0.1 * LPIPS
  4000+    : L1 + 0.1 * LPIPS + GAN, ramping GAN weight 0 -> 0.02

Every 500 steps the script logs metrics and writes a local checkpoint. When
HF_TOKEN is available, the same checkpoint is uploaded to the requested HF repo.
Checkpoints include VAE + optimizer state and can be resumed with --resume.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import lpips
from datasets import load_dataset
from huggingface_hub import HfApi
from PIL import Image
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm

from .vae_v2 import RAYVAE_v2


MONET_REPO = "jasperai/monet"
HF_DEFAULT_REPO = "Rishix500/ray-image-vae-n8"


class PatchGAN(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 4, 1, 4, 2, 1),
        )

    def forward(self, x):
        return self.net(x)


def preprocess_thumbnail(image, size=256):
    """Convert MONET sample['thumbnail'] to a [3,size,size] float tensor."""
    if not isinstance(image, Image.Image):
        raise TypeError(f"expected PIL image in sample['thumbnail'], got {type(image)!r}")
    image = image.convert("RGB")
    image = image.resize((size, size), Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def next_batch(iterator, batch_size):
    """Read batch_size valid MONET thumbnails from the streaming iterator."""
    images = []
    while len(images) < batch_size:
        sample = next(iterator)
        image = sample.get("thumbnail")
        if image is None:
            continue
        try:
            images.append(preprocess_thumbnail(image, 256))
        except (TypeError, ValueError):
            continue
    return torch.stack(images)


def upload_checkpoint(local_path: Path, repo_id: str, hf_token: str | None):
    if not hf_token:
        print("[HF] HF_TOKEN not set; keeping checkpoint local only")
        return
    api = HfApi(token=hf_token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=local_path.name,
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"[HF] uploaded {local_path.name} -> {repo_id}")


def save_checkpoint(path: Path, vae, discriminator, g_optimizer, d_optimizer, step: int):
    """Save VAE, discriminator, and optimizer states for exact training resume."""
    torch.save(
        {
            "stage": "vae_v2_n8",
            "step": step,
            "config": {
                "image_size": 256,
                "latent_channels": 16,
                "latent_size": 32,
                "base_channels": 64,
            },
            "vae": vae.state_dict(),
            "optimizer": g_optimizer.state_dict(),
            "g_optimizer": g_optimizer.state_dict(),
            "discriminator": discriminator.state_dict(),
            "d_optimizer": d_optimizer.state_dict(),
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start-step", type=int, default=0,
                        help="Treat training as starting after this step when not resuming.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a .pt checkpoint created by this script.")
    parser.add_argument("--output-dir", default="/kaggle/working/ray_image_vae_n8")
    parser.add_argument("--hf-repo", default=HF_DEFAULT_REPO)
    parser.add_argument("--no-hf-upload", action="store_true")
    args = parser.parse_args()

    if args.start_step < 0:
        raise ValueError("--start-step must be >= 0")
    if args.resume and args.start_step:
        raise ValueError("Use either --resume or --start-step, not both")
    if args.steps < 0:
        raise ValueError("--steps must be >= 0")

    if not torch.cuda.is_available():
        raise RuntimeError("N8 VAE v2 training requires a CUDA GPU")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # MONET is a parquet-backed HF dataset, not WebDataset tar shards.
    dataset = load_dataset(MONET_REPO, split="train", streaming=True)
    iterator = iter(dataset)

    vae = RAYVAE_v2(latent_channels=16, base_channels=64).to(device)
    discriminator = PatchGAN().to(device)
    perceptual = lpips.LPIPS(net="vgg").to(device).eval()
    for p in perceptual.parameters():
        p.requires_grad_(False)

    g_optimizer = AdamW(
        vae.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.01
    )
    d_optimizer = AdamW(
        discriminator.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.01
    )
    adv_criterion = nn.BCEWithLogitsLoss()

    last_checkpoint_step = args.start_step
    if args.resume:
        checkpoint_path = Path(args.resume)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        vae.load_state_dict(checkpoint["vae"])
        if "g_optimizer" in checkpoint:
            g_optimizer.load_state_dict(checkpoint["g_optimizer"])
        elif "optimizer" in checkpoint:
            g_optimizer.load_state_dict(checkpoint["optimizer"])
        if "discriminator" in checkpoint:
            discriminator.load_state_dict(checkpoint["discriminator"])
        if "d_optimizer" in checkpoint:
            d_optimizer.load_state_dict(checkpoint["d_optimizer"])
        last_checkpoint_step = int(checkpoint.get("step", 0))
        print(f"[resume] loaded {checkpoint_path} at step {last_checkpoint_step}")

    if args.steps <= last_checkpoint_step:
        print(
            f"[done] target steps={args.steps} already reached at step={last_checkpoint_step}"
        )
        return

    remaining_steps = args.steps - last_checkpoint_step
    pbar = tqdm(
        total=remaining_steps,
        desc=f"RAY-VAE v2 N8 ({device})",
    )

    for current_step in range(last_checkpoint_step + 1, args.steps + 1):
        try:
            images = next_batch(iterator, args.batch_size).to(device, non_blocking=True)
        except StopIteration:
            iterator = iter(dataset)
            images = next_batch(iterator, args.batch_size).to(device, non_blocking=True)

        recon = vae(images)[0]
        l1 = F.l1_loss(recon, images)

        if current_step <= 2000:
            lpips_loss = l1.new_tensor(0.0)
            gan_loss = l1.new_tensor(0.0)
            adv_w = l1.new_tensor(0.0)
            g_loss = l1
        else:
            lpips_loss = perceptual(recon * 2.0 - 1.0, images * 2.0 - 1.0).mean()
            recon_total = l1 + 0.1 * lpips_loss

            if current_step <= 4000:
                gan_loss = l1.new_tensor(0.0)
                adv_w = l1.new_tensor(0.0)
                g_loss = recon_total
            else:
                adv_w = l1.new_tensor(min(0.02, 0.02 * (current_step - 4000) / 2000.0))

                d_optimizer.zero_grad(set_to_none=True)
                real_logits = discriminator(images)
                fake_logits = discriminator(recon.detach())
                d_real = adv_criterion(real_logits, torch.ones_like(real_logits))
                d_fake = adv_criterion(fake_logits, torch.zeros_like(fake_logits))
                d_loss = 0.5 * (d_real + d_fake)
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
                d_optimizer.step()

                adv_logits = discriminator(recon)
                gan_loss = adv_criterion(adv_logits, torch.ones_like(adv_logits))
                g_loss = recon_total + adv_w * gan_loss

        g_optimizer.zero_grad(set_to_none=True)
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
        g_optimizer.step()

        pbar.update(1)
        pbar.set_postfix(
            step=current_step,
            l1=f"{l1.item():.4f}",
            lpips=f"{lpips_loss.item():.4f}",
            gan=f"{gan_loss.item():.4f}",
            adv_w=f"{adv_w.item():.4f}",
        )

        if current_step % 500 == 0:
            print(
                f"step={current_step} "
                f"L1={l1.item():.6f} "
                f"LPIPS={lpips_loss.item():.6f} "
                f"GAN={gan_loss.item():.6f} "
                f"recon_mean={recon.mean().item():.6f} "
                f"recon_std={recon.std(unbiased=False).item():.6f}"
            )
            ckpt = outdir / f"ray_vae_v2_step_{current_step:07d}.pt"
            save_checkpoint(ckpt, vae, discriminator, g_optimizer, d_optimizer, current_step)
            print(f"[checkpoint] {ckpt}")
            if not args.no_hf_upload:
                upload_checkpoint(ckpt, args.hf_repo, os.getenv("HF_TOKEN"))

    pbar.close()
    if last_checkpoint_step != args.steps:
        ckpt = outdir / f"ray_vae_v2_step_{args.steps:07d}.pt"
        save_checkpoint(ckpt, vae, discriminator, g_optimizer, d_optimizer, args.steps)
        print(f"[checkpoint] {ckpt}")
        if not args.no_hf_upload:
            upload_checkpoint(ckpt, args.hf_repo, os.getenv("HF_TOKEN"))


if __name__ == "__main__":
    main()
