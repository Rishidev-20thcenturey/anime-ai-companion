"""CPU smoke test for the N4 separability probe (no training)."""
import tempfile
from pathlib import Path

import torch

from . import probe_latent_separability as M
from .config import RAYConfig
from .toy import COLORS, SHAPES
from .utils import build_models, save_checkpoint
from .whiten import LatentNormalizer


def _build_ckpt(path):
    cfg = RAYConfig()
    assert cfg.latent_channels == 16, cfg.latent_channels
    device = torch.device("cpu")
    mods = build_models(cfg, device)
    vocab = {"<pad>": 0, "<unk>": 1}
    vocab.update({w: i + 2 for i, w in enumerate(["a"] + list(COLORS) + list(SHAPES))})
    mean4 = torch.tensor([1.2, -1.6, 1.9, 1.0])
    std4 = torch.tensor([1.4, 1.8, 2.2, 1.5])
    repeats = cfg.latent_channels // mean4.numel()
    whiten = LatentNormalizer(mean4.repeat(repeats), std4.repeat(repeats))
    save_checkpoint(path, cfg, step=0, vocab=vocab, whiten=whiten.state(),
                    **{k: v for k, v in mods.items()})
    return cfg


def main():
    orig_seeds = M.SWAP_SEEDS
    M.SWAP_SEEDS = [0]
    try:
        tmp = Path(tempfile.mkdtemp(prefix="ray_n4_smoke_"))
        ckpt = tmp / "gen_random.pt"
        cfg = _build_ckpt(ckpt)
        outdir = tmp / "n4"
        report = M.run_probe(str(ckpt), str(outdir), size=cfg.image_size,
                             canonical_seed=1234, steps=6,
                             device=torch.device("cpu"))
    finally:
        M.SWAP_SEEDS = orig_seeds

    assert report["whitening_embedded"] is True
    ls = report["latent_separability"]
    for k in ("same_color_diff_shape", "same_shape_diff_color"):
        assert ls[k]["mean"] is not None and ls[k]["n"] > 0, ls
    assert set(report["flow_target_separability"]) == {"0.25", "0.5", "0.75"}
    assert set(report["prediction_separability"]) == {"0.25", "0.5", "0.75"}
    ratios = report["shape_color_distance_ratio"]
    assert ratios["latent"] is not None
    sw = report["fixed_noise_shape_swap"]
    assert "0" in sw["seeds"] and sw["paths"], sw

    assert (outdir / "n4_report.json").exists()
    grid = outdir / "grids" / "shape_swap"
    pngs = list(grid.glob("*.png"))
    assert pngs, "no shape-swap PNGs written"

    print(f"n4_smoke: PASS (artifacts under {tmp})")
    print(f"  latent channels = {cfg.latent_channels}")
    print(f"  report keys: {sorted(k for k in report if k not in ('flow_target_separability',))}")
    print(f"  shape-swap PNGs: {len(pngs)}")


if __name__ == "__main__":
    main()
