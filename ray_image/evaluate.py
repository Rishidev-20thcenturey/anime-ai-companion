"""Evaluate generated RAY-IMAGE PNGs with simple pixel diagnostics.

This is not a semantic image-quality metric. It provides repeatable signals for
our tiny toy experiment: image dimensions, range, mean/std, blue dominance,
and fraction of pixels close to the light background.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def analyze(path: str):
    p = Path(path)
    image = Image.open(p).convert("RGB")
    a = np.asarray(image, dtype=np.float32) / 255.0
    mean = a.mean(axis=(0, 1))
    std = a.std(axis=(0, 1))
    blue_score = float(mean[2] - 0.5 * (mean[0] + mean[1]))
    bg = np.array([245, 245, 245], dtype=np.float32) / 255.0
    bg_dist = np.linalg.norm(a - bg, axis=2)
    non_bg = float((bg_dist > 0.10).mean())

    print(f"file={p}")
    print(f"size={image.width}x{image.height}")
    print(f"mean_rgb={mean.round(4).tolist()}")
    print(f"std_rgb={std.round(4).tolist()}")
    print(f"min={a.min():.4f} max={a.max():.4f}")
    print(f"blue_score={blue_score:.4f}")
    print(f"non_background_fraction={non_bg:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", help="PNG/JPEG paths to evaluate")
    args = parser.parse_args()
    for image in args.images:
        analyze(image)


if __name__ == "__main__":
    main()
