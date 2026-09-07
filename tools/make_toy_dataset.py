"""Create a richer synthetic captioned dataset for RAY-IMAGE toy training.

Four colors x three shapes are rendered with randomized position, scale, and
background jitter. The caption remains the semantic class label.
"""
import argparse
import json
import random
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
    rng = random.Random(seed)
    bg = tuple(rng.randint(238, 250) for _ in range(3))
    image = Image.new("RGB", (size, size), bg)
    draw = ImageDraw.Draw(image)
    box_size = rng.randint(int(size * 0.42), int(size * 0.70))
    cx = rng.randint(size // 3, size * 2 // 3)
    cy = rng.randint(size // 3, size * 2 // 3)
    half = box_size // 2
    x0 = max(2, cx - half)
    y0 = max(2, cy - half)
    x1 = min(size - 2, cx + half)
    y1 = min(size - 2, cy + half)
    color = COLORS[color_name]

    if shape == "circle":
        draw.ellipse((x0, y0, x1, y1), fill=color)
    elif shape == "square":
        draw.rectangle((x0, y0, x1, y1), fill=color)
    else:
        draw.polygon([(cx, y0), (x1, y1), (x0, y1)], fill=color)
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/toy")
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    root = Path(args.output)
    image_dir = root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest = root / "manifest.jsonl"

    combinations = [(c, s) for c in COLORS for s in SHAPES]
    with manifest.open("w", encoding="utf-8") as f:
        for i in range(args.samples):
            color, shape = combinations[i % len(combinations)]
            image = draw_shape(args.size, color, shape, args.seed + i)
            name = f"{i:06d}.png"
            image.save(image_dir / name)
            caption = f"a {color} {shape}"
            f.write(json.dumps({"image": f"images/{name}", "text": caption}) + "\n")

    print(f"created {args.samples} samples")
    print(f"classes={len(combinations)}")
    print(f"manifest: {manifest}")


if __name__ == "__main__":
    main()
