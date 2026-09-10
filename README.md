# RAY-IMAGE: An Efficient Text-to-Image Generator

**Goal:** Build a 2B-parameter image generator from scratch that outperforms 10B+ parameter models through architectural efficiency, not brute force.

**Status:** N6 complete (16-channel VAE upgrade). Ready for DiT retraining.

---

## 📖 Overview

RAY-IMAGE is a from-scratch implementation of a text-to-image diffusion pipeline, designed to run entirely on free-tier GPU hardware (Kaggle T4/P100, Google Colab T4). The project explores whether **smart architecture + efficient inference** can close the gap between small and massive models.

The core thesis: **A 2B model with Sparse Mixture-of-Experts (MoE) can match or beat a dense 10B model** by activating only a fraction of its parameters per forward pass.

This project is built openly, with every experiment (N1 → N12) committed and documented.

---

## 🏗️ Architecture

RAY-IMAGE is a **latent diffusion model** composed of three core components:

### 1. VAE (Variational Autoencoder)
- **Latent shape:** `[B, 16, 8, 8]` (N6 upgrade)
- **Compression:** 64×64×3 → 16×8×8 (48× spatial compression, 16 latent channels)
- **Losses:** L1 + LPIPS (perceptual) + PatchGAN (adversarial)
- **Training:** Three-stage recipe (warmup → GAN ramp-up → adaptive weighting)

### 2. DiT (Diffusion Transformer)
- **Architecture:** Transformer blocks with patch-based input projection
- **Conditioning:** Token-level text cross-attention
- **N5 Feature:** Learnable per-block scalar gate on the cross-attention residual (`x = x + gate * cross`)
- **N6 Compatibility:** Input/output layers updated to accept 16-channel latents

### 3. Text Encoder
- Lightweight transformer encoder for prompt conditioning
- Tokenized via a simple vocabulary for the toy dataset phase

---

## 📊 Experiment Log (N-Series)

| ID | Name | Description | Status |
| :--- | :--- | :--- | :--- |
| **N0** | VAE Baseline | Initial VAE with 4-channel latents | ✅ Complete |
| **N1** | Latent Probe | Per-channel statistics & reconstruction accuracy | ✅ Complete |
| **N2** | Baseline DiT | 8,000-step training on toy dataset with whitening | ✅ Complete |
| **N3** | Text Conditioning | Diagnostic probe for text conditioning | ✅ Complete |
| **N4** | Separability Probe | Latent/velocity/prediction separability diagnostic | ✅ Complete |
| **N5** | Gated Cross-Attention | Learnable scalar gate on cross-attention residual | ✅ Complete |
| **N5.1** | Shape Diagnostic | Evaluator calibration audit (found shape-eval bug) | ✅ Complete |
| **N5.2** | Shape Perfection | Iterating toward 100% geometric shape accuracy | ✅ Complete |
| **N6** | **VAE Upgrade** | **16-channel VAE + LPIPS + PatchGAN** | ✅ **Complete** |
| **N7** | DiT Retrain | Retrain DiT on N6's sharper latents | 🔜 Next |
| **N8** | MONET Streaming | Swap toy data for MONET (104.9M images) | ⏳ Planned |
| **N9** | Progressive Resolution | 256 → 512 → 1024 training | ⏳ Planned |
| **N10** | Usable AI Generator | Stable 1024×1024 model | ⏳ Planned |
| **N11** | MoE 2B Activation | Sparse MoE with 2B active parameters | ⏳ Planned |
| **N12** | Reasoning & Deployment | Layout guidance + GGUF export | ⏳ Planned |

---

## 🔬 Key Discoveries

### The VAE Bottleneck (N5.2 → N6)
After training N5.2, shapes remained blurry despite perfect color accuracy. A diagnostic test isolated the issue:
- **Original image:** Sharp red circle (crisp 90° edges)
- **VAE reconstruction:** Soft, blurry circle with a halo
- **Root cause:** 48× compression (4 channels × 8×8 latents) mathematically cannot preserve high-frequency edge details

**Fix:** Upgraded to 16 latent channels + LPIPS perceptual loss + PatchGAN discriminator. This is the same recipe used by SD3 and FLUX.

### The Evaluator Bug (N5.1)
The auto-evaluator reported `shape_accuracy = 0.333` (4/12). Investigation revealed:
- The evaluator computed "corner occupancy" on the **whole image**, not the object's bounding box
- For centered shapes, all four whole-image corners are empty → the feature is dead
- Classification collapsed to a single threshold on `bbox_fill`, incorrectly penalizing real squares

**Verdict:** The 0.333 score was a measurement artifact, not a model failure. The evaluator needs recalibration (planned for a future patch).

---

## 🚀 Roadmap to a Shareable Model

1. **N6 VAE Training** (current): Train the 16-channel VAE on the toy dataset and verify sharp reconstruction.
2. **N7 DiT Retrain:** Retrain the DiT on N6's sharp latents; verify sharp shapes at 64×64.
3. **N8 MONET:** Stream the MONET dataset (104.9M image-text pairs) for real training.
4. **N9 Progressive Resolution:** 256 → 512 → 1024 training with progressive upsizing.
5. **N10 Usable Model:** A stable 1024×1024 generator deployed on Hugging Face Spaces.
6. **N11 Sparse MoE Scaling:** Activate only ~2B of ~17B total parameters per forward pass.
7. **N12 Reasoning + Deployment:** Add layout-guided conditioning; export to GGUF for consumer hardware.

---

## 🛠️ Setup & Usage

### Prerequisites
- Python 3.11+
- PyTorch 2.x with CUDA
- 16 GB VRAM (T4, P100, or better)

### Installation
```bash
git clone https://github.com/Rishidev-20thcenturey/anime-ai-companion.git
cd anime-ai-companion
pip install -r requirements.txt
```

### Training the VAE (N6)

```bash
python -m ray_image.train_vae \
    --manifest data/toy/manifest.jsonl \
    --steps 5000 \
    --batch-size 8 \
    --save checkpoints/ray_vae_v1_n6.pt
```

### Training the DiT (N7)

```bash
python -m ray_image.train_n5 \
    --manifest data/toy/manifest.jsonl \
    --vae-checkpoint checkpoints/ray_vae_v1_n6.pt \
    --steps 8000 \
    --batch-size 8 \
    --save checkpoints/ray_dit_v1_n7.pt
```

### Generating Images

```bash
python -m ray_image.generate \
    --checkpoint checkpoints/ray_dit_v1_n7.pt \
    --vae-checkpoint checkpoints/ray_vae_v1_n6.pt \
    --prompt "a red square" \
    --steps 50 \
    --output outputs/red_square.png
```

---

## 📂 Repository Structure

```text
anime-ai-companion/
├── ray_image/
│   ├── vae.py                  # VAE architecture (16-channel latent)
│   ├── dit.py                  # DiT with gated cross-attention
│   ├── train_vae.py            # N6 training (LPIPS + PatchGAN)
│   ├── train_n5.py             # DiT training with cross-gate
│   ├── generate.py             # Inference script
│   ├── probe_vae_latents.py    # N1 diagnostic
│   ├── probe_n5_shape.py       # N5.1 shape diagnostic
│   ├── config.py               # Global configuration
│   └── utils.py                # Checkpoint helpers
├── tools/
│   └── make_toy_dataset.py     # Synthetic 12-class dataset generator
├── runs/                       # Experiment logs and reports
├── notebooks/                  # Kaggle/Colab notebooks
└── README.md
```

---

## 🎯 Design Principles

1. **Diagnose before you train.** Every failure (blur, wrong shape, bad metrics) gets a diagnostic script before any retraining.
2. **Small models, smart architectures.** Efficiency beats brute force.
3. **Reproducibility above all.** Every checkpoint, metric, and config is versioned.
4. **Free hardware, serious results.** Kaggle's 30-hour weekly quota is enough if the code is efficient.

---

## 🤝 Contributing

This is a personal research project, but the code is open. If you build on it, please:

· Cite the original paper inspirations (Nucleus-Image, HiDream-O1, MegaFusion)
· Report issues with diagnostic evidence (not just "it doesn't work")
· Share your experiment results — the field moves faster when we document failures too

---

## 📜 License

Apache 2.0 — free for research and commercial use.

---

## 🙏 Acknowledgements

· Nucleus-Image for the Sparse MoE blueprint
· SD3 / FLUX for the 16-channel VAE recipe
· MONET for the open-source dataset
· Kaggle for the free 30-hour weekly GPU quota that makes this possible
