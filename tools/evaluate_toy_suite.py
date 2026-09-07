"""Score the fixed 12-class RAY-IMAGE toy prompt suite.

This evaluator is intentionally a coarse proxy. It separates color and shape
classification so model progress is easier to diagnose. Foreground color is
measured only on detected object pixels, while shape uses scale-invariant
bounding-box fill ratio plus corner occupancy.
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
    # The training backgrounds vary only slightly around this neutral color.
    bg = np.array([245, 245, 245], dtype=np.float32) / 255.0
    mask = np.linalg.norm(a - bg, axis=2) > 0.10
    ys, xs = np.where(mask)
    h, w = mask.shape
    if len(xs) == 0:
        centroid = np.array([0.5, 0.5], dtype=np.float32)
        foreground_mean = bg.copy()
        bbox_fill = 0.0
    else:
        centroid = np.array([xs.mean() / w, ys.mean() / h], dtype=np.float32)
        foreground_mean = a[mask].mean(axis=0)
        bw = max(1, int(xs.max() - xs.min() + 1))
        bh = max(1, int(ys.max() - ys.min() + 1))
        bbox_fill = float(mask.sum() / (bw * bh))

    q = max(2, min(h, w) // 4)
    corners = np.array([
        mask[:q, :q].mean(), mask[:q, -q:].mean(),
        mask[-q:, :q].mean(), mask[-q:, -q:].mean(),
    ], dtype=np.float32)
    return {
        "foreground_mean": foreground_mean,
        "area": float(mask.mean()),
        "bbox_fill": bbox_fill,
        "centroid": centroid,
        "corners": corners,
    }


def color_score(f, color):
    err = float(np.linalg.norm(f["foreground_mean"] - COLORS[color]))
    return max(0.0, 1.0 - err / 0.60)


def shape_score(f, shape):
    # Idealized fill ratios inside the object bounding box.
    fill_targets = {"circle": 0.785, "square": 1.0, "triangle": 0.50}
    fill_err = abs(f["bbox_fill"] - fill_targets[shape])
    corner_targets = {
        "circle": np.array([0.03, 0.03, 0.03, 0.03], dtype=np.float32),
        "square": np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32),
        "triangle": np.array([0.0, 0.0, 0.18, 0.18], dtype=np.float32),
    }
    corner_err = float(np.mean(np.abs(f["corners"] - corner_targets[shape])))
    return max(0.0, 1.0 - (fill_err / 0.55 + corner_err) / 2.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="/content/ray_suite")
    args = parser.parse_args()
    root = Path(args.dir)

    joint_correct = 0
    color_correct = 0
    shape_correct = 0
    total = 0

    for color in COLORS:
        for shape in SHAPES:
            path = root / f"{color}_{shape}.png"
            if not path.exists():
                print(f"missing={path}")
                continue

            f = features(path)
            best_color = max((color_score(f, c), c) for c in COLORS)
            best_shape = max((shape_score(f, s), s) for s in SHAPES)
            predicted = f"{best_color[1]}_{best_shape[1]}"
            target = f"{color}_{shape}"
            ok_color = best_color[1] == color
            ok_shape = best_shape[1] == shape
            ok_joint = ok_color and ok_shape

            color_correct += int(ok_color)
            shape_correct += int(ok_shape)
            joint_correct += int(ok_joint)
            total += 1
            print(
                f"target={target} predicted={predicted} "
                f"color_score={best_color[0]:.3f} shape_score={best_shape[0]:.3f} "
                f"color_correct={ok_color} shape_correct={ok_shape} joint_correct={ok_joint}"
            )

    if total:
        print(f"color_accuracy={color_correct / total:.3f} ({color_correct}/{total})")
        print(f"shape_accuracy={shape_correct / total:.3f} ({shape_correct}/{total})")
        print(f"suite_accuracy={joint_correct / total:.3f} ({joint_correct}/{total})")


if __name__ == "__main__":
    main()
