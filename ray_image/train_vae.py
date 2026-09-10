"""N6: train the upgraded 16-channel RAY-IMAGE VAE with perceptual + adversarial loss."""
import argparse

import torch
import torch.nn.functional as F
import lpips
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import RAYConfig
from .dataset import RAYCaptionDataset
from .utils import build_models, save_checkpoint, set_seed
from .vae import RAYVAE


class PatchGAN(nn.Module):
    """Small PatchGAN discriminator returning a 2D real/fake logit map."""

    def __init__(self, in_channels: int = 3, base_channels: int = 64):
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


def lpips_input(x):
    """LPIPS expects image values in [-1, 1]."""
    return x * 2.0 - 1.0


def reconstruction_loss(recon, images, perceptual):
    l1 = F.l1_loss(recon, images)
    perceptual_loss = perceptual(lpips_input(recon), lpips_input(images)).mean()
    return l1 + perceptual_loss, l1, perceptual_loss


def adaptive_adv_weight(recon_term, adv_term, last_layer, max_weight=0.5):
    """Balance GAN pressure against reconstruction pressure using gradient norms."""
    recon_grad = torch.autograd.grad(
        recon_term, last_layer, retain_graph=True, allow_unused=True
    )[0]
    adv_grad = torch.autograd.grad(
        adv_term, last_layer, retain_graph=True, allow_unused=True
    )[0]
    if recon_grad is None or adv_grad is None:
        return recon_term.new_tensor(0.0)
    ratio = recon_grad.norm() / (adv_grad.norm() + 1e-4)
    return ratio.clamp(0.0, max_weight).detach()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--save", default="checkpoints/ray_vae_n6_16ch.pt")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = RAYConfig()
    if cfg.latent_channels != 16:
        raise ValueError("N6 VAE requires cfg.latent_channels == 16")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    dataset = RAYCaptionDataset(args.manifest, cfg.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    vae = RAYVAE(cfg.latent_channels, cfg.vae_base).to(device)
    discriminator = PatchGAN().to(device)
    perceptual = lpips.LPIPS(net="vgg").to(device).eval()
    for p in perceptual.parameters():
        p.requires_grad_(False)

    g_optimizer = AdamW(vae.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.01)
    d_optimizer = AdamW(discriminator.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.01)
    adv_criterion = nn.BCEWithLogitsLoss()
    last_layer = vae.decoder[-2].weight

    vae.train()
    discriminator.train()
    pbar = tqdm(total=args.steps, desc=f"RAY-VAE N6 ({device})")
    step = 0
    while step < args.steps:
        for images, _ in loader:
            if step >= args.steps:
                break
            images = images.to(device)
            recon = vae(images)[0]

            recon_total, l1, perceptual_loss = reconstruction_loss(recon, images, perceptual)

            if step < 1000:
                stage = 1
                lambda_adv = recon_total.new_tensor(0.0)
            elif step < 3000:
                stage = 2
                lambda_adv = recon_total.new_tensor(0.5 * (step - 1000) / 2000.0)
            else:
                stage = 3
                adv_for_g = discriminator(recon)
                g_adv = adv_criterion(adv_for_g, torch.ones_like(adv_for_g))
                lambda_adv = adaptive_adv_weight(recon_total, g_adv, last_layer)

            if stage >= 2:
                d_optimizer.zero_grad(set_to_none=True)
                real_logits = discriminator(images)
                fake_logits = discriminator(recon.detach())
                d_loss_real = adv_criterion(real_logits, torch.ones_like(real_logits))
                d_loss_fake = adv_criterion(fake_logits, torch.zeros_like(fake_logits))
                d_loss = 0.5 * (d_loss_real + d_loss_fake)
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
                d_optimizer.step()

            if stage == 1:
                g_loss = recon_total
                g_adv = recon_total.new_tensor(0.0)
            else:
                adv_logits = discriminator(recon)
                g_adv = adv_criterion(adv_logits, torch.ones_like(adv_logits))
                g_loss = recon_total + lambda_adv * g_adv

            g_optimizer.zero_grad(set_to_none=True)
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            g_optimizer.step()

            step += 1
            pbar.update(1)
            postfix = {
                "stage": stage,
                "loss": f"{g_loss.item():.4f}",
                "l1": f"{l1.item():.4f}",
                "lpips": f"{perceptual_loss.item():.4f}",
                "adv_w": f"{lambda_adv.item():.4f}",
            }
            if stage >= 2:
                postfix["d"] = f"{d_loss.item():.4f}"
            pbar.set_postfix(**postfix)

    pbar.close()
    save_checkpoint(
        args.save,
        cfg,
        step,
        vae=vae,
        stage="vae_n6_16ch_lpips_patchgan",
    )
    print(f"saved N6 VAE checkpoint: {args.save} (step={step})")


if __name__ == "__main__":
    main()
