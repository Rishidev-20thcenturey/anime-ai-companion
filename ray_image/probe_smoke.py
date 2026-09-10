"""CPU smoke test for the N1 VAE-latent probe (no real training)."""
import json
import tempfile
from pathlib import Path

import torch

from .config import RAYConfig
from .dataset import build_vocab, encode_text  # noqa: F401
from .probe_vae_latents import run_probe, summarize_statistics
from .toy import CLASS_COMBOS, draw_shape
from .utils import build_models, save_checkpoint


def _make_manifest(root: Path, size: int = 64, n: int = 24, seed: int = 7):
    image_dir = root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for i in range(n):
        color, shape = CLASS_COMBOS[i % len(CLASS_COMBOS)]
        draw_shape(size, color, shape, seed + i).save(image_dir / f"{i:06d}.png")
        lines.append(f'{{"image": "images/{i:06d}.png", "text": "a {color} {shape}"}}')
    manifest = root / "manifest.jsonl"
    manifest.write_text("\n".join(lines) + "\n")
    return manifest


def main():
    cfg = RAYConfig()
    assert cfg.latent_channels == 16, cfg.latent_channels
    device = torch.device("cpu")
    tmp = Path(tempfile.mkdtemp(prefix="ray_probe_smoke_"))

    vae = build_models(cfg, device, text_encoder=False, dit=False)["vae"]
    vae_ckpt = tmp / "vae.pt"
    save_checkpoint(vae_ckpt, cfg, step=0, vae=vae)

    manifest = _make_manifest(tmp / "data")
    outdir = tmp / "probe"
    report = run_probe(
        str(vae_ckpt), str(outdir), size=64, seed=123,
        manifest=str(manifest), stats_samples=24, stats_batch=8,
        eval_recon=True, device=device,
    )

    recon_dir = Path(report["recon_dir"])
    names = [f"{c}_{s}" for c, s in CLASS_COMBOS]
    missing = [n for n in names if not (recon_dir / f"{n}.png").exists()]
    assert not missing, f"missing reconstructions: {missing}"

    stats_path = Path(report["stats_path"])
    assert stats_path.exists()
    stats = json.loads(stats_path.read_text())
    assert len(stats["per_channel_mean"]) == cfg.latent_channels, stats
    assert {"global_mean", "global_std", "per_channel_std"} <= set(stats)

    report_path = outdir / "probe_report.json"
    assert report_path.exists()
    ev = report["evaluation"]
    assert ev["returncode"] == 0, ev.get("stderr")
    for key in ("color_accuracy", "shape_accuracy", "suite_accuracy"):
        assert ev.get(key) is not None, f"evaluator did not emit {key}"

    _ = json.dumps(summarize_statistics(
        torch.randn(4, cfg.latent_channels, cfg.latent_size, cfg.latent_size), source="unit-test"))

    print(f"probe_smoke: PASS (artifacts under {tmp})")
    print(f"  latent channels = {cfg.latent_channels}")
    print(f"  suite_accuracy parsed = {ev['suite_accuracy']}")


if __name__ == "__main__":
    main()
