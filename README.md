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

Every 100 steps a full checkpoint is written. All training entry points take an
optional `--seed` (default 0) for reproducible runs. A later free-GPU session
can resume (checkpoints embed the optimizer state and vocabulary):

```bash
python -m ray_image.train \
  --manifest data/toy/manifest.jsonl \
  --steps 2000 \
  --batch-size 8 \
  --seed 0 \
  --resume checkpoints/ray_image_v0_1.pt \
  --save checkpoints/ray_image_v0_1.pt
```

### Two-stage training (roadmap item 2)

For stable latents before learning the generator, first train the VAE alone,
then train the text->latent flow generator with that frozen VAE:

```bash
python -m ray_image.train_vae \
  --manifest data/toy/manifest.jsonl \
  --steps 1000 --batch-size 16 --save checkpoints/vae.pt

python -m ray_image.train_generator \
  --manifest data/toy/manifest.jsonl \
  --vae checkpoints/vae.pt \
  --steps 8000 --batch-size 8 --save checkpoints/ray_image_stage2.pt
```

A `.pt` checkpoint from either stage is interchangeable with `generate.py` as
long as it carries the VAE, text encoder, DiT and `vocab`.

## 5. Generate

After a checkpoint exists:

```bash
python -m ray_image.generate \
  --checkpoint checkpoints/ray_image_v0_1.pt \
  --prompt "a blue circle" \
  --output generated.png \
  --steps 30
```

## 6. Diagnostics (N1) — VAE latent sanity + statistics

Diagnostic only: no architecture change, no training. It checks whether the VAE
latent preserves color and shape well enough for the generator to learn them.

```bash
# Reconstruct one clean reference image per class via the deterministic mean
# latent, score them with the existing 12-class evaluator, and estimate latent
# statistics for later whitening (N2).
python -m ray_image.probe_vae_latents \
  --vae-checkpoint checkpoints/vae.pt \
  --manifest data/toy/manifest.jsonl \
  --outdir diagnostics/n1 \
  --size 64 --seed 1337 --stats-samples 512 --stats-batch 32
```

Outputs under `--outdir`:
- `reconstructions/{color}_{shape}.png` — 12 deterministic reconstructions.
- `vae_latent_stats.json` — per-channel & global mean/std/min/max (reused by N2).
- `probe_report.json` — machine-readable summary incl. evaluator accuracy.

`tools/make_toy_dataset.py` and the probe share one drawing implementation
(`ray_image/toy.py`), so reconstruction references match the training
distribution.

Optional generator diagnostic logging (does NOT change the objective) can be
enabled while training the generator to watch per-latent-channel and
center-vs-border flow MSE:

```bash
python -m ray_image.train_generator \
  --manifest data/toy/manifest.jsonl --vae checkpoints/vae.pt \
  --steps 8000 --batch-size 8 --save checkpoints/ray_image_stage2.pt \
  --diagnostics --diag-every 100
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
