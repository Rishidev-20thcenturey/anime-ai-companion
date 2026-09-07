"""N1 diagnostic -- VAE latent sanity + latent statistics probe.

Answers the question: *"Does the current VAE latent representation preserve
color AND shape well enough for the generator to learn them?"*

Pipeline (read-only w.r.t. any model weights; the VAE is used in eval mode):
  1. Load a trained VAE checkpoint.
  2. Draw one deterministic, representative 64x64 reference image for each of
     the 12 toy classes (same distribution as ``tools/make_toy_dataset.py``).
  3. Encode each reference with the *deterministic mean latent* (no sampling
     noise), decode that mean latent, and save the reconstruction PNGs.
  4. Run the EXISTING 12-class evaluator (tools/evaluate_toy_suite.py) on the
     reconstruction directory and report color/shape/suite accuracy.
  5. Estimate latent statistics (per-channel and global mean/std/min/max) over
     the dataset (or, if no manifest is given, over the reference images) and
     write them to ``vae_latent_stats.json`` so N2 can reuse them later.

No architecture is modified here.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import RAYConfig
from .dataset import RAYCaptionDataset
from .utils import build_models, load_checkpoint, set_seed
from .vae import RAYVAE
from .toy import CLASS_COMBOS, draw_shape

REPO_ROOT = Path(__file__).resolve().parents[1]


def _tensor_to_pil(t):
    """Convert a [C,H,W] tensor in [0,1] to a PIL RGB image."""
    from PIL import Image

    arr = (t.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy() * 255).round().astype("uint8")
    return Image.fromarray(arr)


def _load_vae(checkpoint_path, device):
    ckpt = load_checkpoint(checkpoint_path, device)
    raw_cfg = ckpt.get("config", {})
    cfg = RAYConfig(**{k: v for k, v in raw_cfg.items() if k in RAYConfig.__dataclass_fields__})
    vae = RAYVAE(cfg.latent_channels, cfg.vae_base).to(device)
    vae.load_state_dict(ckpt["vae"])
    vae.eval()
    return cfg, vae


def build_reference_tensors(size: int, seed: int):
    """Return list of (name, image_tensor[1,3,size,size]) for the 12 classes.

    Deterministic: each class is drawn with seed ``seed + class_index``.
    """
    from torchvision import transforms

    to_tensor = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
    refs = []
    for i, (color, shape) in enumerate(CLASS_COMBOS):
        img = draw_shape(size, color, shape, seed + i)
        name = f"{color}_{shape}"
        refs.append((name, to_tensor(img).unsqueeze(0)))
    return refs


def reconstruct(vae, refs, recon_dir: Path, device):
    """Encode each reference with the mean latent, decode, save PNG.

    Returns list of saved paths.
    """
    recon_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    with torch.no_grad():
        for name, img in refs:
            img = img.to(device)
            _, mean, _ = vae.encode(img)  # deterministic mean latent, no noise
            recon = vae.decode(mean).clamp(0, 1)[0]
            out = recon_dir / f"{name}.png"
            _tensor_to_pil(recon).save(out)
            saved.append(out)
    return saved


def _means_from_images(vae, images, device):
    """Encode a batch of [B,C,H,W] images and return their mean latents."""
    with torch.no_grad():
        _, mean, _ = vae.encode(images.to(device))
    return mean


def compute_latent_statistics_from_dataset(vae, manifest, size, device,
                                           max_samples=512, batch_size=32, seed=0):
    """Estimate latent stats over a sample of dataset images."""
    set_seed(seed)
    dataset = RAYCaptionDataset(manifest, size)
    if max_samples and max_samples < len(dataset):
        from torch.utils.data import Subset
        import random
        rng = random.Random(seed)
        idx = sorted(rng.sample(range(len(dataset)), max_samples))
        dataset = Subset(dataset, idx)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    means = []
    with torch.no_grad():
        for images, _ in loader:
            means.append(_means_from_images(vae, images, device))
    if not means:
        raise ValueError("no images encoded from dataset")
    means = torch.cat(means, dim=0)
    source = f"dataset manifest={manifest} (n={means.shape[0]})"
    return means, source


def compute_latent_statistics_from_references(vae, refs, device):
    """Estimate latent stats over the 12 reference images (fallback)."""
    with torch.no_grad():
        means = torch.cat([_means_from_images(vae, img, device) for _, img in refs], dim=0)
    return means, f"12 clean reference images (n={means.shape[0]})"


def summarize_statistics(means, source):
    """Given stacked mean latents [N,C,H,W], produce a JSON-serializable dict."""
    means = means.double()
    c = means.shape[1]
    per_channel_mean = means.mean(dim=(0, 2, 3)).tolist()
    per_channel_std = means.std(dim=(0, 2, 3), unbiased=False).tolist()
    per_channel_min = means.amin(dim=(0, 2, 3)).tolist()
    per_channel_max = means.amax(dim=(0, 2, 3)).tolist()
    stats = {
        "source": source,
        "latent_shape": list(means.shape[1:]),
        "n_images": int(means.shape[0]),
        "global_mean": float(means.mean()),
        "global_std": float(means.std(unbiased=False)),
        "global_min": float(means.min()),
        "global_max": float(means.max()),
        "per_channel_mean": per_channel_mean,
        "per_channel_std": per_channel_std,
        "per_channel_min": per_channel_min,
        "per_channel_max": per_channel_max,
    }
    return stats


def run_evaluator(recon_dir: Path):
    """Run the existing 12-class evaluator via subprocess and parse metrics.

    The evaluator is intentionally NOT reimplemented here so the probe always
    measures accuracy the exact same way as the headline toy experiment.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "tools.evaluate_toy_suite", "--dir", str(recon_dir)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    text = proc.stdout + "\n" + proc.stderr
    metrics = {}
    for key in ("color_accuracy", "shape_accuracy", "suite_accuracy"):
        m = re.search(rf"{key}=([\d.]+)\s+\((\d+)/(\d+)\)", text)
        if m:
            metrics[key] = {"value": float(m.group(1)),
                            "correct": int(m.group(2)), "total": int(m.group(3))}
        else:
            metrics[key] = None
    metrics["returncode"] = proc.returncode
    if proc.returncode != 0:
        metrics["stderr"] = text[-4000:]
    return metrics, text


def run_probe(vae_checkpoint, outdir, *, size=64, seed=1337, manifest=None,
              stats_samples=512, stats_batch=32, eval_recon=True, device=None):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg, vae = _load_vae(vae_checkpoint, device)
    print(f"[probe] device={device} checkpoint={vae_checkpoint}")
    print(f"[probe] image_size={cfg.image_size} latent={cfg.latent_channels}@{cfg.latent_size}x{cfg.latent_size}")

    # 1) Reconstruct one clean reference per class via the deterministic mean latent.
    refs = build_reference_tensors(size, seed)
    recon_dir = outdir / "reconstructions"
    saved = reconstruct(vae, refs, recon_dir, device)
    print(f"[probe] saved {len(saved)} reconstructions -> {recon_dir}")

    # 2) Latent statistics (dataset if available, else the references).
    if manifest:
        means, source = compute_latent_statistics_from_dataset(
            vae, manifest, size, device, max_samples=stats_samples,
            batch_size=stats_batch, seed=seed)
    else:
        means, source = compute_latent_statistics_from_references(vae, refs, device)
    stats = summarize_statistics(means, source)
    stats_path = outdir / "vae_latent_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")
    print(f"[probe] latent stats ({source}) -> {stats_path}")
    for i in range(len(stats["per_channel_mean"])):
        print(f"  ch{i}: mean={stats['per_channel_mean'][i]:+.4f} "
              f"std={stats['per_channel_std'][i]:.4f} "
              f"[{stats['per_channel_min'][i]:.3f},{stats['per_channel_max'][i]:.3f}]")
    print(f"  global: mean={stats['global_mean']:.4f} std={stats['global_std']:.4f} "
          f"[{stats['global_min']:.3f},{stats['global_max']:.3f}]")

    # 3) Evaluate reconstructions with the existing evaluator.
    report = {"stats_path": str(stats_path), "recon_dir": str(recon_dir),
              "latent_stats": stats, "config": cfg.__dict__}
    if eval_recon and recon_dir:
        metrics, raw = run_evaluator(recon_dir)
        report["evaluation"] = metrics
        print("[probe] reconstruction accuracy (existing evaluator):")
        for k in ("color_accuracy", "shape_accuracy", "suite_accuracy"):
            m = metrics.get(k)
            if m:
                print(f"  {k} = {m['value']:.3f} ({m['correct']}/{m['total']})")
            else:
                print(f"  {k} = MISSING")
    report_path = outdir / "probe_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[probe] report -> {report_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vae-checkpoint", required=True)
    parser.add_argument("--outdir", default="diagnostics/n1")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--manifest", default=None,
                        help="toy manifest.jsonl; if given, latent stats are "
                             "estimated over a dataset sample, else over references")
    parser.add_argument("--stats-samples", type=int, default=512)
    parser.add_argument("--stats-batch", type=int, default=32)
    parser.add_argument("--no-eval", action="store_true",
                        help="skip running the existing 12-class evaluator")
    args = parser.parse_args()
    run_probe(
        args.vae_checkpoint,
        args.outdir,
        size=args.size,
        seed=args.seed,
        manifest=args.manifest,
        stats_samples=args.stats_samples,
        stats_batch=args.stats_batch,
        eval_recon=not args.no_eval,
    )


if __name__ == "__main__":
    main()
