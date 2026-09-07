"""Shared deterministic toy-image helpers for RAY-IMAGE diagnostics.

These draw the same colored-shape images the toy dataset generator produces, so
a diagnostic (e.g. the N1 VAE latent probe) can reconstruct a clean,
representative reference image for each of the 12 classes without depending on
a particular pre-generated dataset directory.

The drawing logic mirrors ``tools/make_toy_dataset.py`` exactly (same palettes,
same shape rasterization, same deterministic RNG driven by the seed).
"""
import random

from PIL import Image, ImageDraw

# Color / shape vocabulary -- keep in sync with tools/evaluate_toy_suite.py.
COLORS = {
    "red": (220, 60, 60),
    "green": (60, 190, 100),
    "blue": (70, 110, 220),
    "yellow": (230, 190, 60),
}
SHAPES = ("circle", "square", "triangle")

# (color, shape) in the exact cyclic order used by the dataset generator so the
# first 12 dataset samples are one image per class in this order.
CLASS_COMBOS = [(c, s) for c in COLORS for s in SHAPES]


def draw_shape(size: int, color_name: str, shape: str, seed: int):
    """Render one deterministic ``size x size`` colored shape.

    Returns a PIL RGB image on a light, slightly-jittered background.
    """
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
