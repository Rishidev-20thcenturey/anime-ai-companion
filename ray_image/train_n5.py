"""Train the gated-cross-attention generator for N5/N7.

Backwards-compatible N5 mode:
  load an N2 generator checkpoint (including its 4-channel VAE and whitening)
  and warm-start the gated DiT.

N7 mode:
  provide --vae-checkpoint and --latent-stats to train against the N6
  16-channel VAE latents. If --checkpoint is omitted, the text encoder and DiT
  start from scratch; the N2 warm-start is skipped entirely.
"""
import argparse
from pathlib import Path

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
    cfg.dit_cross_gate = True  # N5/N7 uses the gated DiT.
    return cfg


def build_n5_from_n2(n2_checkpoint: str, device, cfg=None):
    """Load N2 weights into an N5 (gated) model.

    Returns (cfg, vocab, vae, text_encoder, dit, whiten, new_params, loaded).
    This function intentionally preserves the legacy 4-channel N2 path for
    smoke tests and backwards compatibility.
    """
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
    """Build an N7 model around a frozen 16-channel N6 VAE.

    If an N2 checkpoint is supplied, compatible text/DiT weights are reused,
    but the incompatible 4-channel VAE weights are never loaded. Without an
    N2 checkpoint, text encoder and DiT are initialized from scratch.
    """
    n2 = load_checkpoint(n2_checkpoint, device) if n2_checkpoint else None
    if n2 is not None:
        cfg = config_from_checkpoint(n2.get("config", {}))
    else:
        cfg = RAYConfig()
        cfg.dit_cross_gate = True
    # N6 is the explicit source of truth for the latent width.
    cfg.latent_channels = 16
    cfg.dit_cross_gate = True

    vae_ckpt = load_checkpoint(n6_vae_checkpoint, device)
    vae = build_models(cfg, device, text_encoder=False, dit=False)["vae"]
    load_pretrained({"vae": vae}, vae_ckpt, device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    if n2 is not None:
        vocab = n2["vocab"]
    else:
        # N7-from-scratch still needs the manifest vocabulary; train() receives
        # the dataset separately, so the caller replaces this with build_vocab.
        vocab = None

    text_encoder = build_models(cfg, device, vae=False, dit=False)["text_encoder"]
    dit = build_models(cfg, device, vae=False, text_encoder=False)["dit"]
    loaded = {"missing_keys": [], "unexpected_keys": [], "skipped_shape_keys": []}

    if n2 is not None:
        text_encoder.load_state_dict(n2["text_encoder"])
        # Reuse only DiT tensors whose names AND shapes match. The N2 input/output
        # projections are 4-channel-specific and must not be forced into N7.
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
    return cfg, vocab, vae, text_encoder, dit, whiten, loaded


def train(cfg, vocab, vae, text_encoder, dit, whiten, manifest, steps, batch_size,
          lr, save, seed, device):
    set_seed(seed)
    dataset = RAYCaptionDataset(manifest, cfg.image_size)
    if vocab is None:
        vocab = build_vocab((item["text"] for item in dataset.items), cfg.vocab_size)
    if len(dataset) < batch_size:
        raise ValueError(f"dataset has {len(dataset)} samples but batch size is {batch_size}")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    optimizer = AdamW(
        [*text_encoder.parameters(), *dit.parameters()],
        lr=lr, betas=(0.9, 0.99), weight_decay=0.01,
    )
    text_encoder.train()
    dit.train()
    pbar = tqdm(total=steps, desc=f"RAY-IMAGE N7 ({device})")
    step = 0
    while step < steps:
        for images, captions in loader:
            if step >= steps:
                break
            images = images.to(device)
            tokens = torch.stack(
                [encode_text(x, vocab, cfg.max_tokens) for x in captions]
            ).to(device)
            text_mask = tokens.eq(0)

            with torch.no_grad():
                _, mean, _ = vae.encode(images)
                z = mean
            if whiten is not None:
                z = whiten.normalize(z)

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
        save, cfg, step,
        vocab=vocab,
        vae=vae,
        text_encoder=text_encoder,
        dit=dit,
        whiten=whiten.state() if whiten is not None else None,
        stage="generator_v0.3_n7" if cfg.latent_channels == 16 else "generator_v0.3_n5",
    )
    return step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", default=None,
                        help="Optional N2 generator checkpoint for backwards-compatible "
                             "warm-starting. Omit for an N7 DiT-from-scratch run.")
    parser.add_argument("--vae-checkpoint", default=None,
                        help="N6 VAE checkpoint (.pt). When provided, the frozen VAE "
                             "is loaded from this file and the model uses 16-channel latents.")
    parser.add_argument("--latent-stats", default=None,
                        help="N6 latent stats JSON. Required with --vae-checkpoint for "
                             "the matching 16-channel whitening convention.")
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
        print(f"[N7] device={device} n6_vae={args.vae_checkpoint}")
        print(f"[N7] latent_channels={cfg.latent_channels}; warmstart={args.checkpoint or 'none'}")
        if args.checkpoint:
            print(f"[N7] compatible DiT weights loaded; skipped shape-mismatched keys={loaded['skipped_shape_keys']}")
        print(f"[N7] whitening loaded from {args.latent_stats}")
        save_default = "/content/ray_image_v0_3_n7.pt"
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

    save_path = args.save if args.save != "/content/ray_image_v0_3_n5.pt" or not args.vae_checkpoint else save_default
    final_step = train(cfg, vocab, vae, text_encoder, dit, whiten,
                       args.manifest, args.steps, args.batch_size, args.lr,
                       save_path, args.seed, device)
    print(f"[N{'7' if args.vae_checkpoint else '5'}] saved checkpoint: {save_path} (step={final_step})")


if __name__ == "__main__":
    main()
