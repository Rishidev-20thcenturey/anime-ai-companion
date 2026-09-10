"""CPU smoke tests for N2 latent whitening (no training).

Checks, deterministically on CPU:
  1. LatentNormalizer.normalize centers/scales each channel toward N(0,1) and
     normalize∘denormalize is the identity.
  2. A generator checkpoint produced with whitening embeds the normalizer, and
     generate.load_models recovers it so the sampler's output is inverted
     before VAE decode (training/generation consistency).
  3. The same path with whitening disabled yields a None normalizer (baseline
     preserved).

Run with:  python -m ray_image.whiten_smoke
"""
import json
import tempfile
from pathlib import Path

import torch

from .config import RAYConfig
from .generate import load_models
from .utils import build_models, save_checkpoint
from .whiten import LatentNormalizer


def _save_generator_ckpt(path, whiten_state=None, seed=0):
    torch.manual_seed(seed)
    cfg = RAYConfig()
    device = torch.device("cpu")
    mods = build_models(cfg, device)
    vocab = {"<pad>": 0, "<unk>": 1, "a": 2, "blue": 3, "circle": 4}
    save_checkpoint(
        path, cfg, step=0,
        vocab=vocab, whiten=whiten_state,
        **{k: v for k, v in mods.items()},
    )
    return cfg


def _norm_stats(channels=16, seed=0):
    torch.manual_seed(seed)
    mean4 = torch.tensor([1.2, -1.6, 1.9, 1.0])
    std4 = torch.tensor([1.4, 1.8, 2.2, 1.5])
    repeats = channels // 4
    return LatentNormalizer(mean4.repeat(repeats), std4.repeat(repeats))


def main():
    device = torch.device("cpu")
    tmp = Path(tempfile.mkdtemp(prefix="ray_whiten_smoke_"))

    mean4 = torch.tensor([1.2, -1.6, 1.9, 1.0])
    std4 = torch.tensor([1.4, 1.8, 2.2, 1.5])
    mean = mean4.repeat(4)
    std = std4.repeat(4)
    norm = LatentNormalizer(mean, std)

    torch.manual_seed(1)
    z = torch.randn(32, 16, 8, 8)
    z = z * std.view(1, 16, 1, 1) + mean.view(1, 16, 1, 1)
    zn = norm.normalize(z)
    per_ch_mean = zn.mean(dim=(0, 2, 3))
    per_ch_std = zn.std(dim=(0, 2, 3), unbiased=False)
    assert torch.allclose(per_ch_mean, torch.zeros(16), atol=0.1), per_ch_mean
    assert torch.allclose(per_ch_std, torch.ones(16), atol=0.1), per_ch_std

    z_round = norm.denormalize(zn)
    assert torch.allclose(z_round, z, atol=1e-5), "normalize/denormalize not identity"

    for whiten_state, expect_norm in [(norm.state(), True), (None, False)]:
        ckpt = tmp / ("gen_w.pt" if whiten_state else "gen_raw.pt")
        _save_generator_ckpt(ckpt, whiten_state=whiten_state)
        _, _, vae, text_enc, dit, recovered = load_models(str(ckpt), device)

        if expect_norm:
            assert recovered is not None, "normalizer not recovered"
            assert recovered.state() == whiten_state, "recovered stats differ"
            latent_norm = torch.randn(1, 16, 8, 8)
            raw = recovered.denormalize(latent_norm)
            assert torch.allclose(
                raw, latent_norm * std.view(1, 16, 1, 1) + mean.view(1, 16, 1, 1),
                atol=1e-5,
            )
            with torch.no_grad():
                img = vae.decode(raw)
            assert img.shape == (1, 3, 64, 64)
        else:
            assert recovered is None, "baseline unexpectedly has a normalizer"

    stats_json = tmp / "vae_latent_stats.json"
    stats_json.write_text(json.dumps({
        "per_channel_mean": mean.tolist(),
        "per_channel_std": std.tolist(),
        "latent_shape": [16, 8, 8],
    }))
    from_stats = LatentNormalizer.from_stats_json(str(stats_json))
    assert torch.allclose(from_stats.mean, mean) and torch.allclose(from_stats.std, std)

    print("whiten_smoke: PASS")
    print("  channel whitening -> per-channel mean~0, std~1  [verified]")
    print("  normalize/denormalize round-trip identity       [verified]")
    print("  checkpoint stores/reloads normalizer            [verified]")
    print("  baseline (whiten=None) preserved                [verified]")


if __name__ == "__main__":
    main()
