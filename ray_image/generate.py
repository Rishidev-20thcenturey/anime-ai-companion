"""Generate an image from a trained RAY-IMAGE checkpoint."""
import argparse
from pathlib import Path

import torch
from PIL import Image

from .config import RAYConfig
from .dataset import encode_text
from .flow import euler_sample
from .utils import build_models, load_checkpoint, load_pretrained
from .whiten import LatentNormalizer


def load_models(checkpoint_path, device, vae_checkpoint=None, latent_stats=None):
    checkpoint = load_checkpoint(checkpoint_path, device)
    raw_cfg = checkpoint.get("config", {})
    cfg = RAYConfig(**{k: v for k, v in raw_cfg.items() if k in RAYConfig.__dataclass_fields__})

    # N7 path: use the separately trained N6 VAE and its matching 16-channel
    # latent statistics instead of the VAE embedded in the DiT checkpoint.
    if vae_checkpoint is not None:
        cfg.latent_channels = 16
        if latent_stats is None:
            raise ValueError("--latent-stats is required when --vae-checkpoint is provided")

        vae_checkpoint_data = load_checkpoint(vae_checkpoint, device)
        vae = build_models(cfg, device, text_encoder=False, dit=False)["vae"]
        load_pretrained({"vae": vae}, vae_checkpoint_data, device)

        text_encoder = build_models(cfg, device, vae=False, dit=False)["text_encoder"]
        dit = build_models(cfg, device, vae=False, text_encoder=False)["dit"]
        load_pretrained({"text_encoder": text_encoder, "dit": dit}, checkpoint, device)

        whiten = LatentNormalizer.from_stats_json(latent_stats)
        whiten.validate_channels(16)
    else:
        # Legacy path: load the VAE, text encoder, DiT, and whitening state from
        # the generator checkpoint exactly as before (including 4-channel N5).
        modules = build_models(cfg, device)
        load_pretrained(modules, checkpoint, device)
        vae = modules["vae"]
        text_encoder = modules["text_encoder"]
        dit = modules["dit"]

        whiten = None
        if checkpoint.get("whiten") is not None:
            whiten = LatentNormalizer.from_state(checkpoint["whiten"])
            whiten.validate_channels(cfg.latent_channels)

    for module in (vae, text_encoder, dit):
        module.eval()

    return cfg, checkpoint["vocab"], vae, text_encoder, dit, whiten


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vae-checkpoint", default=None,
                        help="Optional N6 VAE checkpoint. When provided, generation uses "
                             "the 16-channel N6 VAE instead of the embedded VAE.")
    parser.add_argument("--latent-stats", default=None,
                        help="Optional N6 latent stats JSON. Required with --vae-checkpoint "
                             "for matching 16-channel whitening inversion.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", default="generated.png")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    cfg, vocab, vae, text_encoder, dit, whiten = load_models(
        args.checkpoint,
        device,
        vae_checkpoint=args.vae_checkpoint,
        latent_stats=args.latent_stats,
    )

    tokens = encode_text(args.prompt, vocab, cfg.max_tokens).unsqueeze(0).to(device)
    text_mask = tokens.eq(0)
    with torch.no_grad():
        text = text_encoder(tokens, mask=text_mask)
        latent = euler_sample(
            dit,
            text,
            (1, cfg.latent_channels, cfg.latent_size, cfg.latent_size),
            steps=args.steps,
            device=device,
            text_mask=text_mask,
        )
        # The sampler operates in whitened space when the checkpoint was trained
        # with whitening; invert (z = z_norm*std + mean) before decoding.
        if whiten is not None:
            latent = whiten.denormalize(latent)
        image = vae.decode(latent).clamp(0, 1)[0]

    array = (image.permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(output)
    print(f"saved image: {output}")


if __name__ == "__main__":
    main()
