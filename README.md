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

## 7. N2 — channel-wise latent whitening (optional, one switch)

Single controlled change: normalize each latent channel to ~N(0,1) before the
flow objective, using the N1 statistics. It rebalances the flow MSE across the
strongly channel-imbalanced VAE latents. VAE / tokenizer / text encoder / DiT /
flow objective / sampler / dataset / evaluator are unchanged.

```bash
# 1) get stats: python -m ray_image.probe_vae_latents ... (see N1 above)
#    -> produces vae_latent_stats.json with per_channel_mean / per_channel_std

# 2) train the generator with whitening enabled:
python -m ray_image.train_generator \
  --manifest data/toy/manifest.jsonl --vae checkpoints/vae.pt \
  --whiten-stats diagnostics/n1/vae_latent_stats.json \
  --steps 8000 --batch-size 8 --save checkpoints/ray_image_stage2_whiten.pt

# 3) generate (whitening is embedded in the checkpoint and auto-inverted):
python -m ray_image.generate \
  --checkpoint checkpoints/ray_image_stage2_whiten.pt \
  --prompt "a blue circle" --output generated.png --steps 50
```

Convention: `z_norm = (z - mean_c) / std_c` applied identically in training and
generation; `generate.py` inverts (`z = z_norm*std + mean`) before VAE decode.
The unwhitened baseline is preserved — simply omit `--whiten-stats`.

Verification on CPU (no training):

```bash
python -m ray_image.whiten_smoke
```

## 8. N3 — text-conditioning probe (diagnostic only, no training)

Investigates why the generator learns color but collapses shape. It reads a
trained generator checkpoint and measures, with NO training:
- tokenizer distinctness for color/shape words,
- token-level text embeddings and padding-mask validity (is the shape token
  visible?),
- pooled-embedding distance split (same-shape/diff-color vs
  same-color/diff-shape),
- the DiT conditioning-vector effect for prompts differing only in shape,
- an estimate of whether swapping the shape token materially changes the DiT
  output (vs swapping the color token).

```bash
python -m ray_image.probe_text_conditioning \
  --checkpoint checkpoints/ray_image_stage2_whiten.pt \
  --outdir diagnostics/n3
```

Writes `text_conditioning_report.json`. Random-init checkpoints yield NO
meaningful numbers (smoke only):

```bash
python -m ray_image.probe_text_smoke
```

## 9. N4 — latent/velocity separability probe (diagnostic only, no training)

Measures whether the color-1.0/shape-0.33 collapse is visible on the latent /
velocity side, on the trained generator checkpoint:
- VAE latent separability (mean latent L2 for same-color/diff-shape vs
  same-shape/diff-color),
- flow-target separability (matched-noise velocity `noise - z` on the t grid),
- trained-prediction separability (real DiT velocity distances on a shared
  latent state),
- a fixed-noise shape-swap probe (red circle/square/triangle over seeds).

```bash
python -m ray_image.probe_latent_separability \
  --checkpoint checkpoints/ray_image_stage2_whiten.pt \
  --outdir /content/n4
```

Writes `n4_report.json` + `grids/shape_swap/*.png`. Smoke (random-init only):

```bash
python -m ray_image.n4_smoke
```

## 10. N5 — gated cross-attention (controlled architecture experiment)

Single controlled change to the DiT: a per-block learnable scalar gate scales
the token-level text cross-attention residual, `x = x + gate * cross`. The gate
is a scalar, independent of token identity/category, and is initialized to `1.0`
so the gated model starts mathematically identical to the previous architecture.

This is a real training experiment, but the code does not launch training on its
own. The N5 entry point warm-starts from the N2 baseline checkpoint: it loads all
compatible N2 weights (VAE, text encoder, DiT, vocab, whitening) and initializes
only the new gate parameters.

```bash
# Train N5 (warm-started from N2, same 8000-step budget):
python -m ray_image.train_n5 \
  --manifest data/toy/manifest.jsonl \
  --checkpoint checkpoints/ray_image_stage2_whiten.pt \
  --steps 8000 --batch-size 16 \
  --save /content/ray_image_v0_3_n5.pt --seed 0

# Generate + evaluate exactly as N2 (steps 50, seed 42):
python -m ray_image.generate --checkpoint /content/ray_image_v0_3_n5.pt \
  --prompt "a blue circle" --output gen.png --steps 50 --seed 42
```

Checkpoint compatibility is explicit: the N2 checkpoint loads directly (all shared
weights), and only the new per-block `cross_gate` parameters are newly initialized
to `1.0`. Smoke / regression (CPU, random-init, no scientific meaning):

```bash
python -m ray_image.n5_smoke
```

A standalone one-experiment Colab notebook
(`notebooks/RAY_IMAGE_N5_STANDALONE_Colab.ipynb`) restores N2 from Drive, runs the
N5 compatibility smoke, trains only N5, saves the N5 checkpoint + evaluator output
+ image grid under Drive `runs/N5/`.

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
8. Scale the architecture toward a larger RAY-IMAGE model, eventually targeting the ~2B parameter class that beat 10b class family..
