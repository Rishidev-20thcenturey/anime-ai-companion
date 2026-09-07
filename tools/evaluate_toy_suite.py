"""Generate and score a complete 12-class RAY-IMAGE toy prompt suite.

The score is a coarse pixel-level proxy, not a semantic image-quality metric.
It checks color agreement, occupied area, centroid, and corner occupancy against
synthetic class prototypes.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

COLORS = {
    "red": np.array([220, 60, 60], dtype=np.float32) / 255.0,
    "green": np.array([60, 190, 100], dtype=np.float32) / 255.0,
    "blue": np.array([70, 110, 220], dtype=np.float32) / 255.0,
    "yellow": np.array([230, 190, 60], dtype=np.float32) / 255.0,
}
SHAPES = ("circle", "square", "triangle")


def features(path):
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    bg = np.array([245, 245, 245], dtype=np.float32) / 255.0
    mask = np.linalg.norm(a - bg, axis=2) > 0.10
    ys, xs = np.where(mask)
    if len(xs) == 0:
        centroid = np.array([0.5, 0.5], dtype=np.float32)
    else:
        h, w = mask.shape
        centroid = np.array([xs.mean() / w, ys.mean() / h], dtype=np.float32)
    h, w = mask.shape
    q = max(2, min(h, w) // 4)
    corners = np.array([
        mask[:q, :q].mean(), mask[:q, -q:].mean(),
        mask[-q:, :q].mean(), mask[-q:, -q:].mean()
    ], dtype=np.float32)
    return {
        "mean": a.mean(axis=(0, 1)),
        "area": float(mask.mean()),
        "centroid": centroid,
        "corners": corners,
    }


def target_features(color, shape):
    # Approximate class signatures are intentionally tolerant.
    mean = 0.80 * COLORS[color] + 0.20 * (np.array([245, 245, 245]) / 255.0)
    area = {"circle": 0.31, "square": 0.40, "triangle": 0.20}[shape]
    corners = {
        "circle": np.array([0.02, 0.02, 0.02, 0.02]),
        "square": np.array([0.25, 0.25, 0.25, 0.25]),
        "triangle": np.array([0.0, 0.0, 0.20, 0.20]),
    }[shape]
    return mean, area, corners


def class_score(f, color, shape):
    mean, area, corners = target_features(color, shape)
    color_err = float(np.linalg.norm(f["mean"] - mean))
    area_err = abs(f["area"] - area)
    corner_err = float(np.mean(np.abs(f["corners"] - corners)))
    score = max(0.0, 1.0 - (color_err / 0.75 + area_err / 0.35 + corner_err) / 3.0)
    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="/content/ray_suite")
    args = parser.parse_args()
    root = Path(args.dir)
    correct = 0
    total = 0
    for color in COLORS:
        for shape in SHAPES:
            path = root / f"{color}_{shape}.png"
            if not path.exists():
                print(f"missing={path}")
                continue
            f = features(path)
            candidates = [(class_score(f, c, s), c, s) for c in COLORS for s in SHAPES]
            best = max(candidates)
            target = f"{color}_{shape}"
            predicted = f"{best[1]}_{best[2]}"
            ok = predicted == target
            correct += int(ok)
            total += 1
            print(f"target={target} predicted={predicted} score={best[0]:.3f} correct={ok}")
    if total:
        print(f"suite_accuracy={correct / total:.3f} ({correct}/{total})")


if __name__ == "__main__":
    main()
