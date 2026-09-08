"""N5: train the gated-cross-attention generator, warm-started from the N2 baseline.

Single controlled architecture change (N5):
  add a per-block learnable scalar gate to the token-level text cross-attention
  residual in the DiT (x = x + gate * cross), gate init = 1.0 so the model
  starts mathematically identical to N2 before training.

This entry point does NOT train from scratch:
  - loads the N2 generator checkpoint (vae, text encoder, dit, vocab, whiten),
  - reconstructs the DiT WITH the new gates (dit_cross_gate=True),
  - loads every compatible N2 weight (strict=False), leaving only the NEW gate
    parameters at their init value 1.0,
  - then trains the text encoder + DiT on the same flow objective / whitened
    latent convention / budget as N2.

No VAE / dataset / tokenizer / flow / sampler / resolution / model-size change.
CFG is NOT added. Nothing here silently launches training -- it must be invoked.
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import RAYConfig
from .dataset import RAYCaptionDataset, encode_text
from .flow import sample_flow_pair
from .utils import build_models, load_checkpoint, save_checkpoint, set_seed
from .whiten import LatentNormalizer


def config_from_checkpoint(raw_cfg: dict) -> RAYConfig:
    cfg = RAYConfig(**{k: v for k, v in raw_cfg.items()
                       if k in RAYConfig.__dataclass_fields__})
    cfg.dit_cross_gate = True  # N5 forces the gated DiT.
    return cfg


def build_n5_from_n2(n2_checkpoint: str, device, cfg=None):
    """Load N2 weights into an N5 (gated) model.

    Returns (cfg, vocab, vae, text_encoder, dit, whiten, new_params, loaded)
      new_params : list of (param_name, init_value) for parameters that were NOT
                   in the N2 checkpoint (the newly initialized gates).
      loaded     : info on the strict=False load (missing/unexpected keys).
    Raises if the N2 checkpoint cannot be read.
    """
    n2 = load_checkpoint(n2_checkpoint, device)
    if cfg is None:
        cfg = config_from_checkpoint(n2.get("config", {}))
    vocab = n2["vocab"]

    # VAE + text encoder: identical in N5, load all N2 weights strictly.
    vae = build_models(cfg, device, text_encoder=False, dit=False)["vae"]
    text_encoder = build_models(cfg, device, vae=False, dit=False)["text_encoder"]
    vae.load_state_dict(n2["vae"])
    text_encoder.load_state_dict(n2["text_encoder"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    # DiT: gated architecture; load every compatible N2 weight.
    dit = build_models(cfg, device, vae=False, text_encoder=False)["dit"]
    missing, unexpected = dit.load_state_dict(n2["dit"], strict=False)

    # Report the newly initialized parameters (the gates, present at init 1.0).
    new_params = []
    gate_names = [f"blocks.{i}.cross_gate" for i in range(len(dit.blocks))]
    sd = dit.state_dict()
    for gname in gate_names:
        if gname in sd:
            new_params.append((gname, float(sd[gname].item())))
    loaded = {"missing_keys": sorted(missing),
              "unexpected_keys": sorted(unexpected),
              "new_params": new_params}

    # Whitening stats embedded in N2 (same convention must be kept for N5).
    whiten = None
    if n2.get("whiten") is not None:
        whiten = LatentNormalizer.from_state(n2["whiten"])
        whiten.validate_channels(cfg.latent_channels)

    return cfg, vocab, vae, text_encoder, dit, whiten, new_params, loaded


def train(cfg, vocab, vae, text_encoder, dit, whiten, manifest, steps, batch_size,
          lr, save, seed, device):
    set_seed(seed)
    dataset = RAYCaptionDataset(manifest, cfg.image_size)
    if len(dataset) < batch_size:
        raise ValueError(f"dataset has {len(dataset)} samples but batch size is {batch_size}")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    optimizer = AdamW(
        [*text_encoder.parameters(), *dit.parameters()],
        lr=lr, betas=(0.9, 0.99), weight_decay=0.01,
    )
    text_encoder.train()
    dit.train()
    pbar = tqdm(total=steps, desc=f"RAY-IMAGE N5 ({device})")
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
        stage="generator_v0.3_n5",
    )
    return step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True,
                        help="N2 generator checkpoint to warm-start from "
                             "(e.g. /content/ray_image_v0_2_whiten.pt)")
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--save", default="/content/ray_image_v0_3_n5.pt")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, vocab, vae, text_encoder, dit, whiten, new_params, loaded = \
        build_n5_from_n2(args.checkpoint, device)

    print(f"[N5] device={device} warmstart={args.checkpoint}")
    print(f"[N5] dit_cross_gate=True; load missing keys={loaded['missing_keys']}")
    print(f"[N5] newly initialized params={new_params}")
    if loaded["unexpected_keys"]:
        print(f"[N5] WARNING unexpected keys (dropped): {loaded['unexpected_keys']}")
    if whiten is not None:
        print("[N5] whitening convention inherited from N2 checkpoint")

    final_step = train(cfg, vocab, vae, text_encoder, dit, whiten,
                       args.manifest, args.steps, args.batch_size, args.lr,
                       args.save, args.seed, device)
    print(f"[N5] saved checkpoint: {args.save} (step={final_step})")


if __name__ == "__main__":
    main()
