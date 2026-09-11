"""Train the gated-cross-attention generator for N5/N7/N8.

Backwards-compatible N5 mode:
  load an N2 generator checkpoint (including its legacy 4-channel VAE and
  lightweight text encoder) and warm-start the gated DiT.

N7/N8 mode:
  provide --vae-checkpoint and --latent-stats to train against the N6
  16-channel VAE latents. This path uses the frozen Qwen3-4B text encoder
  introduced in N8; the Qwen tokenizer replaces the toy vocabulary.
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
from .utils import build_models, load_checkpoint, load_pretrained, save_checkpoint, set_seed
from .whiten import LatentNormalizer


def config_from_checkpoint(raw_cfg: dict) -> RAYConfig:
    cfg = RAYConfig(**{k: v for k, v in raw_cfg.items()
                       if k in RAYConfig.__dataclass_fields__})
    cfg.dit_cross_gate = True
    cfg.text_encoder_backend = getattr(cfg, "text_encoder_backend", "legacy")
    return cfg


def build_n5_from_n2(n2_checkpoint: str, device, cfg=None):
    """Load legacy N2 weights into an N5 gated model."""
    n2 = load_checkpoint(n2_checkpoint, device)
    if cfg is None:
        cfg = config_from_checkpoint(n2.get("config", {}))
    vocab = n2["vocab"]

    vae = build_models(cfg, device, text_encoder=False, dit=False)["vae"]
    text_encoder = build_models(cfg, device, vae=False, dit=False)["text_encoder"]
    vae.load_state_dict(n2["vae"])
    text_encoder.load_state_dict(n2["text_encoder"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    dit = build_models(cfg, device, vae=False, text_encoder=False)["dit"]
    missing, unexpected = dit.load_state_dict(n2["dit"], strict=False)

    new_params = []
    gate_names = [f"blocks.{i}.cross_gate" for i in range(len(dit.blocks))]
    sd = dit.state_dict()
    for gname in gate_names:
        if gname in sd:
            new_params.append((gname, float(sd[gname].item())))
    loaded = {"missing_keys": sorted(missing),
              "unexpected_keys": sorted(unexpected),
              "new_params": new_params}

    whiten = None
    if n2.get("whiten") is not None:
        whiten = LatentNormalizer.from_state(n2["whiten"])
        whiten.validate_channels(cfg.latent_channels)

    return cfg, vocab, vae, text_encoder, dit, whiten, new_params, loaded


def build_n7_from_n6(n6_vae_checkpoint, latent_stats, device, n2_checkpoint=None):
    """Build a 16-channel N7/N8 generator around a frozen N6 VAE.

    The N8 path always uses Qwen3-4B for text conditioning. If an old N2
    checkpoint is supplied, only shape-compatible DiT weights are reused;
    the legacy toy text encoder is not loaded into the Qwen encoder.
    """
    n2 = load_checkpoint(n2_checkpoint, device) if n2_checkpoint else None
    cfg = config_from_checkpoint(n2.get("config", {})) if n2 is not None else RAYConfig()
    cfg.latent_channels = 16
    cfg.dit_cross_gate = True
    cfg.text_encoder_backend = "qwen3"
    cfg.text_dim = 2560

    vae_ckpt = load_checkpoint(n6_vae_checkpoint, device)
    vae = build_models(cfg, device, text_encoder=False, dit=False)["vae"]
    load_pretrained({"vae": vae}, vae_ckpt, device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    text_encoder = build_models(cfg, device, vae=False, dit=False)["text_encoder"]
    dit = build_models(cfg, device, vae=False, text_encoder=False)["dit"]
    loaded = {"missing_keys": [], "unexpected_keys": [], "skipped_shape_keys": []}

    if n2 is not None:
        # Qwen3 tokenizer/encoder replaces the legacy toy text stack in N8.
        # Only reuse DiT tensors with exactly matching names and shapes.
        dit_sd = dit.state_dict()
        compatible = {
            k: v for k, v in n2["dit"].items()
            if k in dit_sd and tuple(v.shape) == tuple(dit_sd[k].shape)
        }
        skipped = sorted(set(n2["dit"]) - set(compatible))
        missing, unexpected = dit.load_state_dict(compatible, strict=False)
        loaded["missing_keys"] = sorted(missing)
        loaded["unexpected_keys"] = sorted(unexpected)
        loaded["skipped_shape_keys"] = skipped

    whiten = LatentNormalizer.from_stats_json(latent_stats)
    whiten.validate_channels(16)
    return cfg, None, vae, text_encoder, dit, whiten, loaded


def train(cfg, vocab, vae, text_encoder, dit, whiten, manifest, steps, batch_size,
          lr, save, seed, device):
    set_seed(seed)
    dataset = RAYCaptionDataset(manifest, cfg.image_size)
    legacy_text = getattr(cfg, "text_encoder_backend", "legacy") != "qwen3"
    if legacy_text and vocab is None:
        vocab = build_vocab((item["text"] for item in dataset.items), cfg.vocab_size)
    if len(dataset) < batch_size:
        raise ValueError(f"dataset has {len(dataset)} samples but batch size is {batch_size}")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Qwen3 is frozen; only the DiT is optimized in N8.
    trainable = list(dit.parameters())
    optimizer = AdamW(trainable, lr=lr, betas=(0.9, 0.99), weight_decay=0.01)
    if legacy_text:
        text_encoder.train()
        trainable = [*text_encoder.parameters(), *dit.parameters()]
        optimizer = AdamW(trainable, lr=lr, betas=(0.9, 0.99), weight_decay=0.01)
    else:
        text_encoder.eval()
        for p in text_encoder.parameters():
            p.requires_grad_(False)

    dit.train()
    pbar = tqdm(total=steps, desc=f"RAY-IMAGE {'N5' if legacy_text else 'N8'} ({device})")
    step = 0
    while step < steps:
        for images, captions in loader:
            if step >= steps:
                break
            images = images.to(device)

            if legacy_text:
                tokens = torch.stack(
                    [encode_text(x, vocab, cfg.max_tokens) for x in captions]
                ).to(device)
                text_mask = tokens.eq(0)
            else:
                text, text_mask = text_encoder.encode_texts(
                    list(captions), max_length=cfg.max_tokens
                )
                text = text.to(device)
                text_mask = text_mask.to(device)

            with torch.no_grad():
                _, mean, _ = vae.encode(images)
                z = mean
            if whiten is not None:
                z = whiten.normalize(z)

            if legacy_text:
                text = text_encoder(tokens, mask=text_mask)
            xt, t, target = sample_flow_pair(z)
            pred = dit(xt, t, text, text_mask=text_mask)
            loss = F.mse_loss(pred, target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            step += 1
            pbar.update(1)
            pbar.set_postfix(flow=f"{loss.item():.4f}")
    pbar.close()

    save_checkpoint(
        save, cfg, step,
        vocab=vocab if legacy_text else None,
        vae=vae,
        text_encoder=None,
        dit=dit,
        whiten=whiten.state() if whiten is not None else None,
        stage="generator_v0.3_n8" if not legacy_text else "generator_v0.3_n5",
    )
    return step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", default=None,
                        help="Optional N2 generator checkpoint for legacy N5 warm-starting "
                             "or shape-compatible DiT initialization in N8.")
    parser.add_argument("--vae-checkpoint", default=None,
                        help="N6 VAE checkpoint (.pt); selects the 16-channel N8/Qwen3 path.")
    parser.add_argument("--latent-stats", default=None,
                        help="N6 latent stats JSON; required with --vae-checkpoint.")
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--save", default="/content/ray_image_v0_3_n5.pt")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.vae_checkpoint:
        if not args.latent_stats:
            parser.error("--latent-stats is required when --vae-checkpoint is provided")
        cfg, vocab, vae, text_encoder, dit, whiten, loaded = build_n7_from_n6(
            args.vae_checkpoint, args.latent_stats, device, args.checkpoint
        )
        print(f"[N8] device={device} n6_vae={args.vae_checkpoint}")
        print(f"[N8] latent_channels={cfg.latent_channels}; text_dim={cfg.text_dim}; backend=Qwen3-4B")
        print(f"[N8] warmstart={args.checkpoint or 'none'}")
        if args.checkpoint:
            print(f"[N8] compatible DiT weights loaded; skipped keys={loaded['skipped_shape_keys']}")
        print(f"[N8] whitening loaded from {args.latent_stats}")
        save_default = "/content/ray_image_v0_4_n8.pt"
    else:
        if not args.checkpoint:
            parser.error("--checkpoint is required unless --vae-checkpoint is provided")
        cfg, vocab, vae, text_encoder, dit, whiten, new_params, loaded = \
            build_n5_from_n2(args.checkpoint, device)
        print(f"[N5] device={device} warmstart={args.checkpoint}")
        print(f"[N5] dit_cross_gate=True; load missing keys={loaded['missing_keys']}")
        print(f"[N5] newly initialized params={new_params}")
        if loaded["unexpected_keys"]:
            print(f"[N5] WARNING unexpected keys (dropped): {loaded['unexpected_keys']}")
        if whiten is not None:
            print("[N5] whitening convention inherited from N2 checkpoint")
        save_default = "/content/ray_image_v0_3_n5.pt"

    if args.vae_checkpoint and args.save == "/content/ray_image_v0_3_n5.pt":
        save_path = save_default
    else:
        save_path = args.save

    final_step = train(cfg, vocab, vae, text_encoder, dit, whiten,
                       args.manifest, args.steps, args.batch_size, args.lr,
                       save_path, args.seed, device)
    print(f"saved checkpoint: {save_path} (step={final_step})")


if __name__ == "__main__":
    main()
