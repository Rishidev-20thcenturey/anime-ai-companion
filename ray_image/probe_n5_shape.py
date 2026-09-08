"""N5.1 diagnostic -- geometric audit of the generated 12-image suite.

DIAGNOSTIC ONLY. No training, no checkpoint modification, no model-code change.

Context: N5 automated `shape_accuracy` is 0.333 (4/12) while a Gemini visual
inspection reported 12/12 correct shapes. This module resolves that discrepancy
by measuring, for every generated image, the exact geometric features the
evaluator uses -- bbox fill, bbox width/height, aspect ratio, object area,
centroid, and the four corner-occupancy values -- plus the per-shape
circle/square/triangle scores and the evaluator's predicted shape.

It reproduces the SAME generation settings as the N2/N5 runs (12 prompts,
seed=42, steps=50) so outputs are directly comparable.

Usage:
  python -m ray_image.probe_n5_shape \
      --n5-checkpoint /content/ray_image_v0_3_n5.pt \
      --outdir /content/n5_1 \
      [--n2-checkpoint /content/ray_image_v0_2_whiten.pt]   # optional N2 comparison
      [--n2-dir /path/to/n2/pngs]                            # alternative: reuse N2 PNGs

Writes runs.N5.1-style report (n5_1_report.json) under --outdir.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from . import toy as ray_toy

# Fixed 12-prompt suite (same order/names as generate/evaluate notebooks).
SUITE = [
    ("red_circle", "a red circle"), ("red_square", "a red square"),
    ("red_triangle", "a red triangle"),
    ("green_circle", "a green circle"), ("green_square", "a green square"),
    ("green_triangle", "a green triangle"),
    ("blue_circle", "a blue circle"), ("blue_square", "a blue square"),
    ("blue_triangle", "a blue triangle"),
    ("yellow_circle", "a yellow circle"), ("yellow_square", "a yellow square"),
    ("yellow_triangle", "a yellow triangle"),
]
COLORS = {
    "red": np.array([220, 60, 60], dtype=np.float32) / 255.0,
    "green": np.array([60, 190, 100], dtype=np.float32) / 255.0,
    "blue": np.array([70, 110, 220], dtype=np.float32) / 255.0,
    "yellow": np.array([230, 190, 60], dtype=np.float32) / 255.0,
}
SHAPES = ("circle", "square", "triangle")
FILL_TARGET = {"circle": 0.785, "square": 1.0, "triangle": 0.50}
CORNER_TARGET = {
    "circle": np.array([0.03, 0.03, 0.03, 0.03], dtype=np.float32),
    "square": np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32),
    "triangle": np.array([0.0, 0.0, 0.18, 0.18], dtype=np.float32),
}


def generate_suite(checkpoint, outdir, steps=50, seed=42, device=None):
    """Generate the 12-prompt suite via ray_image.generate (subprocess).

    Each prompt is a separate process call with the given seed/steps, matching
    exactly how the N2 and N5 Colab notebooks generate the suite.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for name, prompt in SUITE:
        out = outdir / f"{name}.png"
        cmd = [sys.executable, "-m", "ray_image.generate",
               "--checkpoint", checkpoint, "--prompt", prompt,
               "--steps", str(steps), "--seed", str(seed), "--output", str(out)]
        subprocess.run(cmd, check=True)
    return outdir


def _features_from_array(a):
    """Full geometric features for an image array in [0,1] HxWx3.

    Mirrors tools/evaluate_toy_suite.features but exposes more detail.
    """
    bg = np.array([245, 245, 245], dtype=np.float32) / 255.0
    mask = np.linalg.norm(a - bg, axis=2) > 0.10
    h, w = mask.shape
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return {
            "empty": True, "bbox_fill": 0.0, "bbox_w": 0, "bbox_h": 0,
            "aspect_ratio": None, "area_frac": 0.0,
            "centroid": [0.5, 0.5],
            "corners": [0.0, 0.0, 0.0, 0.0],
            "foreground_mean": bg.tolist(), "n_pixels": 0,
        }
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    q = max(2, min(h, w) // 4)
    corners = np.array([
        mask[:q, :q].mean(), mask[:q, -q:].mean(),
        mask[-q:, :q].mean(), mask[-q:, -q:].mean(),
    ], dtype=np.float32)
    return {
        "empty": False,
        "bbox_fill": float(mask.sum() / (bw * bh)),
        "bbox_w": int(bw), "bbox_h": int(bh),
        "aspect_ratio": round(bw / bh, 4) if bh > 0 else None,
        "area_frac": float(mask.mean()),
        "centroid": [round(float(xs.mean() / w), 4), round(float(ys.mean() / h), 4)],
        "corners": [round(float(c), 4) for c in corners],
        "foreground_mean": [round(float(v), 4) for v in a[mask].mean(axis=0)],
        "n_pixels": int(mask.sum()),
    }


def color_score(fg_mean, color):
    err = float(np.linalg.norm(np.asarray(fg_mean, dtype=np.float32) - COLORS[color]))
    return max(0.0, 1.0 - err / 0.60)


def shape_score(corners, bbox_fill, shape):
    """Same scoring as tools/evaluate_toy_suite.shape_score."""
    fill_err = abs(bbox_fill - FILL_TARGET[shape])
    corner_err = float(np.mean(np.abs(np.asarray(corners, dtype=np.float32)
                                      - CORNER_TARGET[shape])))
    return max(0.0, 1.0 - (fill_err / 0.55 + corner_err) / 2.0)


def analyze_png(path, target_color, target_shape):
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    f = _features_from_array(a)
    color_scores = {c: round(color_score(f["foreground_mean"], c), 4) for c in COLORS}
    shape_scores = {
        s: round(shape_score(f["corners"], f["bbox_fill"], s), 4) for s in SHAPES
    }
    pred_color = max(color_scores, key=color_scores.get)
    pred_shape = max(shape_scores, key=shape_scores.get)
    return {
        "target": f"{target_color}_{target_shape}",
        "prompt_color": target_color, "prompt_shape": target_shape,
        **{k: f[k] for k in
           ("empty", "bbox_fill", "bbox_w", "bbox_h", "aspect_ratio",
            "area_frac", "centroid", "corners", "n_pixels")},
        "foreground_mean": f["foreground_mean"],
        "circle_score": shape_scores["circle"],
        "square_score": shape_scores["square"],
        "triangle_score": shape_scores["triangle"],
        "predicted_shape": pred_shape,
        "predicted_color": pred_color,
        "shape_correct": pred_shape == target_shape,
        "color_correct": pred_color == target_color,
    }


def analyze_dir(image_dir, label):
    """Analyze all 12 images in a directory (must use our color_shape.png names)."""
    image_dir = Path(image_dir)
    results = []
    for color in COLORS:
        for shape in SHAPES:
            path = image_dir / f"{color}_{shape}.png"
            if not path.exists():
                results.append({"target": f"{color}_{shape}", "missing": True})
                continue
            results.append(analyze_png(path, color, shape))
    summary = summarize(results)
    return results, summary


def summarize(results):
    valid = [r for r in results if not r.get("missing") and not r.get("empty")]
    shape_correct = sum(1 for r in valid if r.get("shape_correct"))
    color_correct = sum(1 for r in valid if r.get("color_correct"))
    joint = sum(1 for r in valid if r.get("shape_correct") and r.get("color_correct"))
    n = len(valid)
    return {
        "n": n,
        "color_accuracy": round(color_correct / n, 4) if n else None,
        "shape_accuracy": round(shape_correct / n, 4) if n else None,
        "suite_accuracy": round(joint / n, 4) if n else None,
        # Aggregate geometry per target shape to expose bias.
        "per_shape": {
            s: _agg_shape([r for r in valid if r["prompt_shape"] == s])
            for s in SHAPES
        },
    }


def _agg_shape(rows):
    if not rows:
        return None
    def mean(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(float(np.mean(vals)), 4) if vals else None
    return {
        "n": len(rows),
        "shape_correct": round(sum(r["shape_correct"] for r in rows) / len(rows), 4),
        "bbox_fill_mean": mean("bbox_fill"),
        "area_frac_mean": mean("area_frac"),
        "aspect_ratio_mean": mean("aspect_ratio"),
        "corner_occ_mean": [round(float(np.mean([r["corners"][i] for r in rows])), 4)
                            for i in range(4)],
        # What fraction were classified as each shape.
        "classified_as": {s: round(sum(r["predicted_shape"] == s for r in rows) / len(rows), 4)
                          for s in SHAPES},
    }


def build_oracle(outdir, size=64, seed=0):
    """Render geometrically ideal toy shapes (no model) as a calibration oracle.

    Uses ray_image.toy.draw_shape -- the exact rasterizer behind the toy
    dataset -- to establish reference features for a "perfect" version of each
    class, then runs the evaluator's metric on them. This exposes whether the
    evaluator's shape-score targets (fill + corner occupancy) actually agree
    with ideal geometry under the evaluator's own whole-image corner definition.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for color in COLORS:
        for shape in SHAPES:
            img = ray_toy.draw_shape(size, color, shape, seed=seed)
            path = outdir / f"{color}_{shape}.png"
            img.save(path)
            rows.append(analyze_png(path, color, shape))
    return rows, summarize(rows)


def shape_score_from_fill(fill, shape):
    """Evaluate shape_score using only bbox_fill, with corners=0 (centered shape).

    Whole-image corner occupancy is ~0 for any centered shape on the 64x64
    canvas (verified by the oracle), so the classifier effectively reduces to a
    threshold on bbox_fill.
    """
    fill_err = abs(fill - FILL_TARGET[shape])
    corner_err = float(np.mean(np.abs(0.0 - CORNER_TARGET[shape])))
    return max(0.0, 1.0 - (fill_err / 0.55 + corner_err) / 2.0)


def decision_boundaries():
    """Return the bbox_fill ranges where each shape wins the argmax (corners=0).

    Because whole-image corners are ~0 for centered shapes, this is a pure
    1-D classification over bbox_fill. It quantifies how wide each class's
    "winning band" is -- e.g. whether real (slightly imperfect) shapes are
    classified correctly or not.
    """
    grid = np.arange(0.20, 1.001, 0.001)
    band = {}
    for shp in SHAPES:
        band[shp] = []
    cur = None
    for fill in grid:
        s = {x: shape_score_from_fill(float(fill), x) for x in SHAPES}
        winner = max(s, key=s.get)
        if winner != cur:
            if cur is not None:
                band[cur].append(fill)  # mark start of previous
            cur = winner
    band[cur].append(1.001)
    # Convert runs into [lo, hi] intervals.
    runs = []
    starts = []
    prev_w = None
    for fill in grid:
        s = {x: shape_score_from_fill(float(fill), x) for x in SHAPES}
        w = max(s, key=s.get)
        if w != prev_w:
            starts.append((float(fill), w))
            prev_w = w
    intervals = []
    for i, (lo, w) in enumerate(starts):
        hi = starts[i + 1][0] if i + 1 < len(starts) else 1.001
        intervals.append({"shape": w, "fill_lo": round(lo, 3), "fill_hi": round(hi, 3)})
    return intervals


def run(checkpoint_n5=None, outdir="/content/n5_1", oracle_only=False,
        n2_checkpoint=None, n2_dir=None, steps=50, seed=42, device=None):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Deterministic ideal-geometry oracle (always produced; no model needed).
    oracle_rows, oracle_summary = build_oracle(outdir / "oracle", size=64, seed=0)
    report = {
        "experiment": "N5.1",
        "n5_checkpoint": checkpoint_n5,
        "n2_checkpoint": n2_checkpoint,
        "generation": {"seed": seed, "steps": steps, "prompts": len(SUITE)},
        "oracle": {"rows": oracle_rows, "summary": oracle_summary},
        "evaluator_calibration": {
            "whole_image_corner_occupancy_for_centered_shapes": 0.0,
            "note": ("evaluate_toy_suite.features computes corners at the four "
                     "WHOLE-IMAGE corners (mask[:q,:q] etc, q=min(h,w)//4). "
                     "For centered 64x64 shapes those patches are empty "
                     "(corners=0 for all shapes), so corner_targets in "
                     "shape_score (square 0.25, triangle 0/.18) never apply and "
                     "classification reduces to a threshold on bbox_fill."),
            "bbox_fill_decision_bands_corners_zero": decision_boundaries(),
        },
        "n5": None, "n2": None,
    }

    # 2) Real N5 suite from a trained checkpoint (only available on Drive).
    if checkpoint_n5:
        n5_img_dir = outdir / "images" / "n5"
        generate_suite(checkpoint_n5, n5_img_dir, steps=steps, seed=seed, device=device)
        n5_rows, n5_summary = analyze_dir(n5_img_dir, "n5")
        report["n5"] = {"rows": n5_rows, "summary": n5_summary,
                        "source": "generated_from_checkpoint"}
    else:
        report["n5"] = {
            "source": None,
            "note": ("No N5 checkpoint given. Re-run with --n5-checkpoint on the "
                     "machine that holds the trained .pt (Drive /content) to "
                     "obtain measured rows; this run only produced the ideal-"
                     "geometry oracle."),
        }

    # 3) N2 comparison (optional): either a checkpoint, or an existing dir.
    if n2_checkpoint:
        n2_img_dir = outdir / "images" / "n2"
        generate_suite(n2_checkpoint, n2_img_dir, steps=steps, seed=seed, device=device)
        n2_rows, n2_summary = analyze_dir(n2_img_dir, "n2")
        report["n2"] = {"rows": n2_rows, "summary": n2_summary,
                        "source": "generated_from_checkpoint"}
    elif n2_dir:
        n2_rows, n2_summary = analyze_dir(n2_dir, "n2")
        report["n2"] = {"rows": n2_rows, "summary": n2_summary, "source": str(n2_dir)}

    report_path = outdir / "n5_1_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print_table(report)
    print(f"\n[N5.1] report saved -> {report_path}")
    return report


def print_table(report):
    print("===== N5.1 GEOMETRIC AUDIT =====")
    for tag in ("oracle", "n5", "n2"):
        block = report.get(tag)
        if not block or not isinstance(block, dict) or "summary" not in block:
            continue
        print(f"\n--- {tag.upper()}"
              + ("  (ideal-geometry oracle)" if tag == "oracle" else ""))
        s = block["summary"]
        if s.get("n") is None:
            print("  no measured images (no checkpoint available)")
            continue
        print(f"  (n={s['n']}) color_accuracy={s['color_accuracy']}  "
              f"shape_accuracy={s['shape_accuracy']}  "
              f"suite_accuracy={s['suite_accuracy']}")
        for r in block["rows"]:
            if r.get("missing"):
                print(f"  {r['target']:<16} MISSING")
                continue
            if r.get("empty"):
                print(f"  {r['target']:<16} EMPTY")
                continue
            print(
                f"  {r['target']:<16} fill={r['bbox_fill']:.3f} "
                f"bbox={r['bbox_w']}x{r['bbox_h']} ar={r['aspect_ratio']} "
                f"area={r['area_frac']:.3f} cent={r['centroid']} "
                f"corn={r['corners']} "
                f"| circle={r['circle_score']:.2f} square={r['square_score']:.2f} "
                f"tri={r['triangle_score']:.2f} -> {r['predicted_shape']}"
                f"{'  <== CORRECT' if r['shape_correct'] else '  (WRONG)'}")
        print("  per-target-shape geometry:")
        for shape, agg in s["per_shape"].items():
            if agg:
                print(f"    {shape:<9} n={agg['n']} correct={agg['shape_correct']:.2f} "
                      f"fill={agg['bbox_fill_mean']} ar={agg['aspect_ratio_mean']} "
                      f"corners={agg['corner_occ_mean']} classified={agg['classified_as']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n5-checkpoint", default=None,
                        help="Trained N5 .pt (on Drive) to generate the suite from.")
    parser.add_argument("--outdir", default="/content/n5_1")
    parser.add_argument("--n2-checkpoint", default=None)
    parser.add_argument("--n2-dir", default=None)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.n5_checkpoint, args.outdir, n2_checkpoint=args.n2_checkpoint,
        n2_dir=args.n2_dir, steps=args.steps, seed=args.seed)


if __name__ == "__main__":
    main()
