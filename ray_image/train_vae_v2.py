"""N8 VAE v2 training on streamed MONET images.

Dataset: jasperai/monet, webdataset/full-resolution image shards streamed from
Hugging Face. Images are resized/cropped to 256x256 on the fly.

Loss schedule:
  0-1999  : L1
  2000-3999: L1 + 0.1 * LPIPS
  4000+    : L1 + 0.1 * LPIPS + GAN, ramping GAN weight 0 -> 0.05

Every 500 steps the script logs metrics and writes a local checkpoint. When
HF_TOKEN is available, the same checkpoint is uploaded to the requested HF repo.
"""

import argparse
import io
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import lpips
from PIL import Image
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm

try:
    import webdataset as wds
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Install webdataset before running N8 VAE v2 training") from exc

try:
    from huggingface_hub import HfApi
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Install huggingface_hub before running N8 VAE v2 training") from exc

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


def make_monet_urls(version: str):
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    pattern = f"datasets/{MONET_REPO}/{version}/**/*.tar"
    paths = fs.glob(pattern)
    if not paths:
        raise RuntimeError(f"No MONET tar shards found for {version}")
    return [
        f"pipe:curl -s -L https://huggingface.co/datasets/{MONET_REPO}/resolve/main/"
        f"{p.removeprefix(f'datasets/{MONET_REPO}/')}"
        for p in paths
    ]


def build_stream(version: str, shuffle_buffer: int):
    urls = make_monet_urls(version)
    return (
        wds.WebDataset(urls, shardshuffle=False, nodesplitter=wds.split_by_node)
        .shuffle(shuffle_buffer)
        .decode("pil")
        .to_tuple("jpg")
    )


def preprocess(image, size=256):
    if isinstance(image, bytes):
        image = Image.open(io.BytesIO(image))
    image = image.convert("RGB")
    w, h = image.size
    scale = size / min(w, h)
    image = image.resize((round(w * scale), round(h * scale)), Image.Resampling.LANCZOS)
    left = (image.width - size) // 2
    top = (image.height - size) // 2
    image = image.crop((left, top, left + size, top + size))
    x = torch.from_numpy(__import__("numpy").array(image)).permute(2, 0, 1).float() / 255.0
    return x


def collate_stream(batch, batch_size):
    images = []
    for sample in batch:
        try:
            images.append(preprocess(sample, 256))
        except Exception:
            continue
    if len(images) != batch_size:
        return None
    return torch.stack(images)


def lpips_input(x):
    return x * 2.0 - 1.0


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--monet-version", default="v2.0.0")
    parser.add_argument("--shuffle-buffer", type=int, default=512)
    parser.add_argument("--output-dir", default="/kaggle/working/ray_image_vae_n8")
    parser.add_argument("--hf-repo", default=HF_DEFAULT_REPO)
    parser.add_argument("--no-hf-upload", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("N8 VAE v2 training requires a CUDA GPU")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    vae = RAYVAE_v2(latent_channels=16, base_channels=64).to(device)
    discriminator = PatchGAN().to(device)
    perceptual = lpips.LPIPS(net="vgg").to(device).eval()
    for p in perceptual.parameters():
        p.requires_grad_(False)

    g_optimizer = AdamW(vae.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.01)
    d_optimizer = AdamW(discriminator.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.01)
    adv_criterion = nn.BCEWithLogitsLoss()

    stream = build_stream(args.monet_version, args.shuffle_buffer)
    iterator = iter(stream)
    pbar = tqdm(total=args.steps, desc=f"RAY-VAE v2 N8 ({device})")
    last_log = None

    for step in range(args.steps):
        while True:
            batch = []
            while len(batch) < args.batch_size:
                try:
                    batch.append(next(iterator)[0])
                except StopIteration:
                    iterator = iter(build_stream(args.monet_version, args.shuffle_buffer))
            images = collate_stream(batch, args.batch_size)
            if images is not None:
                break

        images = images.to(device, non_blocking=True)
        recon = vae(images)[0]
        l1 = F.l1_loss(recon, images)

        if step < 2000:
            lpips_loss = l1.new_tensor(0.0)
            gan_loss = l1.new_tensor(0.0)
            adv_w = l1.new_tensor(0.0)
            g_loss = l1
        else:
            lpips_loss = perceptual(lpips_input(recon), lpips_input(images)).mean()
            recon_total = l1 + 0.1 * lpips_loss

            if step < 4000:
                gan_loss = l1.new_tensor(0.0)
                adv_w = l1.new_tensor(0.0)
                g_loss = recon_total
            else:
                adv_w = l1.new_tensor(min(0.05, 0.05 * (step - 4000) / 2000.0))

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
            l1=f"{l1.item():.4f}",
            lpips=f"{lpips_loss.item():.4f}",
            gan=f"{gan_loss.item():.4f}",
            adv_w=f"{adv_w.item():.4f}",
        )

        current_step = step + 1
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
            torch.save(
                {
                    "stage": "vae_v2_n8",
                    "step": current_step,
                    "config": {
                        "image_size": 256,
                        "latent_channels": 16,
                        "latent_size": 32,
                        "base_channels": 64,
                    },
                    "vae": vae.state_dict(),
                },
                ckpt,
            )
            print(f"[checkpoint] {ckpt}")
            if not args.no_hf_upload:
                upload_checkpoint(ckpt, args.hf_repo, os.getenv("HF_TOKEN"))
            last_log = current_step

    pbar.close()
    if last_log != args.steps:
        ckpt = outdir / f"ray_vae_v2_step_{args.steps:07d}.pt"
        torch.save(
            {
                "stage": "vae_v2_n8",
                "step": args.steps,
                "config": {
                    "image_size": 256,
                    "latent_channels": 16,
                    "latent_size": 32,
                    "base_channels": 64,
                },
                "vae": vae.state_dict(),
            },
            ckpt,
        )
        if not args.no_hf_upload:
            upload_checkpoint(ckpt, args.hf_repo, os.getenv("HF_TOKEN"))


if __name__ == "__main__":
    main()
