# RAY-IMAGE

RAY-IMAGE is an original, from-scratch text-to-image research project. The v0.1 stack is deliberately tiny so the architecture can be trained and iterated on free GPU sessions before scaling toward larger models.

## Current pipeline

```text
image + caption
      |
      +--> RAY VAE ----> latent [B, 4, 8, 8]
      |                     |
      |                     +--> flow matching --> RAY DiT
      |                                      ^
      +--> vocabulary --> RAY text encoder --+

noise + prompt --> RAY DiT --> latent --> RAY VAE decoder --> image
```

The default configuration is 64x64 RGB, 4 latent channels, a 256-dimensional DiT, 6 transformer blocks, and 4 attention heads. This is an architecture/proof-of-learning target, not a production-quality 2B model.

## 1. Install

```bash
pip install -r requirements.txt
```

## 2. Verify the architecture

```bash
python -m ray_image.train_smoke
```

## 3. Build a toy dataset

```bash
python tools/make_toy_dataset.py --output data/toy --samples 120
```

This creates simple colored-shape images and a JSONL caption manifest so the whole pipeline can be tested without sourcing a large dataset.

## 4. Train

```bash
python -m ray_image.train \
  --manifest data/toy/manifest.jsonl \
  --steps 1000 \
  --batch-size 8 \
  --save checkpoints/ray_image_v0_1.pt
```

Every 100 steps a full checkpoint is written. A later free-GPU session can resume:

```bash
python -m ray_image.train \
  --manifest data/toy/manifest.jsonl \
  --steps 2000 \
  --batch-size 8 \
  --resume checkpoints/ray_image_v0_1.pt \
  --save checkpoints/ray_image_v0_1.pt
```

## 5. Generate

After a checkpoint exists:

```bash
python -m ray_image.generate \
  --checkpoint checkpoints/ray_image_v0_1.pt \
  --prompt "a blue circle" \
  --output generated.png \
  --steps 30
```

## Dataset format

`manifest.jsonl` contains one JSON object per line:

```json
{"image":"images/000001.png","text":"a blue circle"}
```

Image paths are relative to the manifest directory.

## Roadmap

1. Prove learning on the synthetic dataset.
2. Separate VAE pretraining from DiT/text training.
3. Add deterministic tokenizer serialization and richer caption conditioning.
4. Add mixed precision, gradient accumulation, EMA, and validation sampling.
5. Scale from 64px to 128px and 256px.
6. Add anime-focused training and character identity conditioning.
7. Extend the latent architecture with temporal modules for RAY-VIDEO.
8. Scale the architecture toward a larger RAY-IMAGE model, eventually targeting the ~2B parameter class.
