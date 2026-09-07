"""Generate an image from a trained RAY-IMAGE checkpoint."""
import argparse
from pathlib import Path

import torch
from PIL import Image

from .config import RAYConfig
from .dataset import encode_text
from .dit import RAYDiT
from .flow import euler_sample
from .text_encoder import RAYTextEncoder
from .vae import RAYVAE


def load_models(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    raw_cfg = checkpoint.get("config", {})
    cfg = RAYConfig(**{k: v for k, v in raw_cfg.items() if k in RAYConfig.__dataclass_fields__})

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

    vae.load_state_dict(checkpoint["vae"])
    text_encoder.load_state_dict(checkpoint["text_encoder"])
    dit.load_state_dict(checkpoint["dit"])
    vae.eval()
    text_encoder.eval()
    dit.eval()
    return cfg, checkpoint["vocab"], vae, text_encoder, dit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", default="generated.png")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    cfg, vocab, vae, text_encoder, dit = load_models(args.checkpoint, device)

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
        image = vae.decode(latent).clamp(0, 1)[0]

    array = (image.permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(output)
    print(f"saved image: {output}")


if __name__ == "__main__":
    main()
