# RAY-IMAGE PROJECT STATE

> Authoritative continuation checkpoint for the RAY-IMAGE project.
> Read this first before resuming any work. Do not blindly re-run setup.

## Mission

Build an original image-generation model, ultimately targeting ~2B parameters,
with the long-term goal of outperforming larger models on selected
image-generation benchmarks/tasks through architecture, data, training
efficiency, specialization and research.

## Current phase

**FIRST_IMAGE**

Immediate milestone:

> Generate the first recognizable image using RAY-IMAGE itself.

Priorities for this phase are deliberately narrow:

- Make the model produce a *recognizable* (semantically correct) image on the
  toy task before any larger architecture/scale changes.
- Treat the known failure mode (shape conditioning collapse) as a real research
  signal, **not** a reason to simply train longer or scale parameters.
- Prove the smallest experiment that yields the first recognizable image, then
  carry that lesson forward into later phases (128px → 256px → anime data → scale).

## Repository state

- **Branch (active):** `arena/01a07cdc-anime-ai-companion`
- **HEAD commit:** `44fe812` — "Add N1 VAE latent sanity probe + latent stats
  diagnostics"
- **Parent commit:** `c984832` — "Refactor training/generation and fix mask +
  resume bugs"
- **main:** `5ceeea0` — "Improve toy evaluator: separate color/shape metrics and
  make features scale-invariant" (`origin/main`). The active branch is `main` +
  the two commits above (`c984832`, `44fe812`).
- **Project version:** package `__version__ = "0.1.0"` (`ray_image/__init__.py`).
  The current toy *generator* experiment is internally labeled **v0.2**
  (notebook title, and default save name `ray_image_v0_2_trained.pt`).
- **Remote:** `https://github.com/Rishidev-20thcenturey/anime-ai-companion.git`
- Working tree at HEAD is clean.

Note: the local clone may on occasion reset toward `main`. If so, restore the
authoritative branch state with:

```bash
git fetch origin arena/01a07cdc-anime-ai-companion
git reset --hard FETCH_HEAD
```

## Completed engineering

Only verified, working items are listed.

- **VAE** — `ray_image/vae.py`, 8× spatial compression; 64×64 RGB ↔ [4, 8, 8]
  latent; mean/logvar reparam; transpose-conv decoder.
- **Trainable text encoder** — `ray_image/text_encoder.py`, learned token +
  position embeddings, TransformerEncoder stack (not a frozen pretrained model).
- **DiT** — `ray_image/dit.py`, latent diffusion transformer over patch tokens.
- **Flow matching** — `ray_image/flow.py`, linear data↔noise interpolation;
  target velocity = noise − data.
- **Euler sampler** — `ray_image/flow.py::euler_sample`, noise→data integration.
- **2D positional encoding fix** — `ray_image/dit.py::_sinusoidal_2d`,
  hand-rolled separable Y/X sinusoidal positions.
- **8× VAE latent geometry** — 4 channels @ latent_size = image_size / 8.
- **Token-level text cross-attention** — per-block masked cross-attention to full
  text sequence; pooled-text + timestep → adaLN global conditioning.
- **Checkpoint / resume** — `ray_image/utils.py` (`save_checkpoint`,
  `load_checkpoint` with `weights_only=False`), optimizer state + vocab embedded;
  resume verified working.
- **Toy dataset** — `tools/make_toy_dataset.py` + shared drawing in
  `ray_image/toy.py`; 4 colors × 3 shapes = 12 classes; deterministic.
- **Evaluator** — `tools/evaluate_toy_suite.py`; separate color and shape
  metrics (scale-invariant fill ratio + corner occupancy); color/shape/suite
  accuracy.
- **N1 VAE latent probe** — `ray_image/probe_vae_latents.py`; reconstructs one
  reference image per class via the mean latent, runs the existing evaluator,
  and writes latent statistics JSON.
- **Colab notebook** — `notebooks/RAY_IMAGE_Colab.ipynb` with a clearly
  separated N1 diagnostic section.
- **Smoke tests** — `ray_image.train_smoke` (architecture) and
  `ray_image.probe_smoke` (N1 CPU path), both wired into CI
  (`.github/workflows/test.yml`).

## Verified experimental results

Known validated result on the fixed 12-class toy prompt suite (previous full
Colab generator run):

- color_accuracy = **1.000**
- shape_accuracy = **0.333**
- suite_accuracy = **0.333**

Brief explanation: the current generator learned **color conditioning well**
but **shape conditioning is weak / collapsed toward triangle** — every prompt
was generated with the correct color but the same (triangle-leaning) shape, so
only true-triangle prompts scored. This is interpreted as an architectural
research signal (a conditioning / objective / spatial-resolution issue), not a
"train longer" issue.

## Current unfinished work

- **N1 VAE latent sanity probe**: fully implemented, committed (`44fe812`), and
  CI-covered — but the latest fresh **Colab N1 run has NOT yet been completed**.
  No N1 result numbers have been recorded yet.
- Pending N1 outputs to capture when run: reconstruction color/shape/suite
  accuracy and `vae_latent_stats.json` (feeds N2 latent whitening decision).

## Next experiment

**FIRST_IMAGE**: use the smallest controlled experiment necessary to obtain the
first recognizable generated image before pursuing larger architecture changes.

Plan of record:

1. Complete the N1 Colab run (probe the current VAE) and record its output here.
   - If reconstruction *shape_accuracy* is already high → the VAE is fine; the
     generator is the target (latent whitening / loss rebalancing, then CFG as
     needed).
   - If reconstruction *shape_accuracy* is already low → the VAE is the first
     bottleneck; address it before the generator.
2. Run the smallest generator experiment that produces a recognizable image on
   the toy task (milestone: clearly beat the 3/12 baseline and collapse).
3. Only after a recognizable image is obtained, revisit architecture/scale
   toward the larger mission.

Do **not** jump to parameter scaling or large datasets before FIRST_IMAGE is
met.

## Do not repeat

Future Colab sessions should **NOT** automatically re-run completed work:

- Do **not** retrain the VAE when a valid VAE checkpoint already exists.
- Do **not** regenerate the toy dataset unnecessarily (recreate only if the
  manifest/images are missing or intentionally changed).
- Do **not** repeat completed engineering (smoke checks are cheap and fine;
  VAE/generator re-training and dataset regeneration are not to be repeated by
  default).

When in doubt: read this file, verify what artifacts already exist, and resume
from the current unfinished work instead of restarting from scratch.
