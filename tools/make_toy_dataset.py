"""Create a richer synthetic captioned dataset for RAY-IMAGE toy training.

Four colors x three shapes are rendered with randomized position, scale, and
background jitter. The caption remains the semantic class label.

Drawing helpers live in ``ray_image.toy`` so diagnostics can reuse the exact
same image distribution.
"""
import argparse
import json
import sys
from pathlib import Path

# Allow running as a plain script (`python tools/make_toy_dataset.py`) while
# still importing the shared package that sits at the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ray_image.toy import CLASS_COMBOS, SHAPES, draw_shape  # noqa: E402


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

    with manifest.open("w", encoding="utf-8") as f:
        for i in range(args.samples):
            color, shape = CLASS_COMBOS[i % len(CLASS_COMBOS)]
            image = draw_shape(args.size, color, shape, args.seed + i)
            name = f"{i:06d}.png"
            image.save(image_dir / name)
            caption = f"a {color} {shape}"
            f.write(json.dumps({"image": f"images/{name}", "text": caption}) + "\n")

    print(f"created {args.samples} samples")
    print(f"classes={len(CLASS_COMBOS)}")
    print(f"manifest: {manifest}")


if __name__ == "__main__":
    main()
