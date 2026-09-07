"""Create a tiny synthetic captioned dataset for end-to-end RAY-IMAGE tests.

This dataset is deliberately simple: colored geometric shapes with captions.
It is useful for checking that training learns a signal before spending GPU
hours on a real image dataset.
"""
import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw


COLORS = {
    "red": (220, 60, 60),
    "green": (60, 190, 100),
    "blue": (70, 110, 220),
    "yellow": (230, 190, 60),
}
SHAPES = ("circle", "square", "triangle")


def draw_shape(size, color_name, shape, seed):
    image = Image.new("RGB", (size, size), (245, 245, 245))
    draw = ImageDraw.Draw(image)
    margin = 10 + (seed % 8)
    x0, y0 = margin, margin
    x1, y1 = size - margin, size - margin
    color = COLORS[color_name]

    if shape == "circle":
        draw.ellipse((x0, y0, x1, y1), fill=color)
    elif shape == "square":
        draw.rectangle((x0, y0, x1, y1), fill=color)
    else:
        cx = size // 2
        draw.polygon([(cx, y0), (x1, y1), (x0, y1)], fill=color)
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/toy")
    parser.add_argument("--samples", type=int, default=120)
    parser.add_argument("--size", type=int, default=64)
    args = parser.parse_args()

    root = Path(args.output)
    image_dir = root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest = root / "manifest.jsonl"

    combinations = [(c, s) for c in COLORS for s in SHAPES]
    with manifest.open("w", encoding="utf-8") as f:
        for i in range(args.samples):
            color, shape = combinations[i % len(combinations)]
            # Slowly vary the geometry while keeping the caption class stable.
            seed = int(abs(math.sin(i * 12.9898)) * 10000)
            image = draw_shape(args.size, color, shape, seed)
            name = f"{i:06d}.png"
            image.save(image_dir / name)
            caption = f"a {color} {shape}"
            f.write(json.dumps({"image": f"images/{name}", "text": caption}) + "\n")

    print(f"created {args.samples} samples")
    print(f"manifest: {manifest}")


if __name__ == "__main__":
    main()
