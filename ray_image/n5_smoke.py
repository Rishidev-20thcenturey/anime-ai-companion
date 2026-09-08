"""CPU smoke tests for N5 gated cross-attention (no real training).

Validates (all on random-init / tiny models):
  1. Gate parameter existence: a `cross_gate=True` DiT has per-block gates; the
     default gate-free DiT has none.
  2. Gate init == 1.0.
  3. Forward pass: gated and gate-free models produce IDENTICAL outputs when the
     gated model carries the same shared weights and gate=1.0 (no shape change,
     mask behavior intact).
  4. N2 checkpoint compatibility: loading a gate-free N2 DiT state into the
     gated architecture loads all shared weights and only leaves the new gates
     missing (strict=False), initialized to 1.0.
  5. Checkpoint save/load round trip preserves gate params.
  6. train_n5 warm-start entrypoint runs end-to-end on a tiny manifest and saves
     a gated checkpoint.

WARNING: these are random-init / a couple of steps; the numbers carry NO
scientific meaning. They only prove the code paths and compatibility.
"""
import json  # noqa: F401
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import RAYConfig
from .dit import RAYDiT
from .toy import CLASS_COMBOS, draw_shape
from .utils import build_models, load_checkpoint, save_checkpoint
from .whiten import LatentNormalizer
from . import train_n5


def _make_manifest(root: Path, n: int = 24, size: int = 64, seed: int = 7):
    image_dir = root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for i in range(n):
        color, shape = CLASS_COMBOS[i % len(CLASS_COMBOS)]
        draw_shape(size, color, shape, seed + i).save(image_dir / f"{i:06d}.png")
        lines.append(f'{{"image": "images/{i:06d}.png", "text": "a {color} {shape}"}}')
    manifest = root / "manifest.jsonl"
    manifest.write_text("\n".join(lines) + "\n")
    return manifest


def _n2_ckpt(path, device, seed=0):
    """Save a gate-free N2-style generator checkpoint (random-init)."""
    cfg = RAYConfig()  # dit_cross_gate defaults False
    torch.manual_seed(seed)
    mods = build_models(cfg, device)
    vocab = {"<pad>": 0, "<unk>": 1, "a": 2, "red": 3, "blue": 4,
             "circle": 5, "square": 6, "triangle": 7}
    whiten = LatentNormalizer(torch.tensor([1.2, -1.6, 1.9, 1.0]),
                              torch.tensor([1.4, 1.8, 2.2, 1.5]))
    save_checkpoint(path, cfg, step=0, vocab=vocab,
                    whiten=whiten.state(), **{k: v for k, v in mods.items()})
    return cfg


def main():
    device = torch.device("cpu")
    tmp = Path(tempfile.mkdtemp(prefix="ray_n5_smoke_"))
    n2_path = tmp / "n2.pt"
    cfg = _n2_ckpt(n2_path, device)

    # --- 1 & 2: gate existence + init ---
    torch.manual_seed(0)
    gated = RAYDiT(cfg.latent_channels, cfg.model_dim, cfg.depth, cfg.heads,
                   cfg.patch_size, cfg.text_dim, cross_gate=True)
    ungated = RAYDiT(cfg.latent_channels, cfg.model_dim, cfg.depth, cfg.heads,
                     cfg.patch_size, cfg.text_dim, cross_gate=False)
    gate_names = [f"blocks.{i}.cross_gate" for i in range(cfg.depth)]
    gsd = gated.state_dict()
    for gn in gate_names:
        assert gn in gsd, f"missing gate {gn}"
        assert float(gsd[gn].item()) == 1.0, f"gate {gn} not init to 1.0"
    for gn in gate_names:
        assert gn not in ungated.state_dict(), "gate-free model must not have gates"
    print("[1][2] gates exist (init=1.0) on gated model; absent on gate-free model")

    # --- 3 & 4: N2-load compatibility => gated(copy) == ungated output ---
    cfg_n5, vocab, vae, te, dit_n5, whiten, new_params, loaded = \
        train_n5.build_n5_from_n2(str(n2_path), device)
    assert cfg_n5.dit_cross_gate is True
    assert len(new_params) == cfg.depth, new_params
    assert all(v == 1.0 for _, v in new_params), new_params
    assert sorted(n for n, _ in new_params) == sorted(gate_names)
    assert not loaded["unexpected_keys"], loaded
    # every gate should be reported missing from the N2 (gate-free) state
    assert all(gn in loaded["missing_keys"] for gn in gate_names), loaded

    # Load the SAME N2 (gate-free) DiT weights into `ungated` so the comparison
    # below is same-weights (gated with gate=1 vs ungated), not random-vs-random.
    n2_ck = load_checkpoint(n2_path, device)
    ungated.load_state_dict(n2_ck["dit"])

    # deterministic identical input
    torch.manual_seed(1)
    z = torch.randn(1, cfg.latent_channels, cfg.latent_size, cfg.latent_size)
    t = torch.tensor([0.5])
    text = torch.randn(1, 8, cfg.text_dim)  # L=8 valid tokens, no mask
    with torch.no_grad():
        o_g = dit_n5(z, t, text)
        o_u = ungated(z, t, text)
    assert o_g.shape == o_u.shape == (1, cfg.latent_channels,
                                      cfg.latent_size, cfg.latent_size)
    assert torch.allclose(o_g, o_u, atol=1e-6), "gated(gate=1, copied) != ungated"
    print("[3][4] N2 weights load into gated model; output identical at gate=1")

    # --- 4b: mask behavior preserved ---
    mask = torch.zeros(1, 8, dtype=torch.bool); mask[0, 7] = True  # pad last token
    with torch.no_grad():
        o_m = dit_n5(z, t, text, text_mask=mask)
    assert o_m.shape == o_g.shape
    print("[mask] gated model forward with key padding mask OK")

    # --- 5: checkpoint save/load round trip ---
    save_checkpoint(tmp / "n5.pt", cfg_n5, step=3, vocab=vocab, vae=vae,
                    text_encoder=te, dit=dit_n5, whiten=whiten.state())
    ck2 = load_checkpoint(tmp / "n5.pt", device)
    assert "dit" in ck2
    # reload into a fresh gated model and compare
    dit_re = RAYDiT(cfg_n5.latent_channels, cfg_n5.model_dim, cfg_n5.depth,
                    cfg_n5.heads, cfg_n5.patch_size, cfg_n5.text_dim,
                    cross_gate=True)
    dit_re.load_state_dict(ck2["dit"])
    with torch.no_grad():
        o_r = dit_re(z, t, text)
    assert torch.allclose(o_g, o_r, atol=1e-6)
    print("[5] gated checkpoint save/load round trip OK")

    # --- 6: train_n5 warm-start tiny run ---
    manifest = _make_manifest(tmp / "data")
    out = tmp / "n5_trained.pt"
    train_n5.train(cfg_n5, vocab, vae, te, dit_n5, whiten, str(manifest),
                   steps=2, batch_size=8, lr=1e-4, save=str(out), seed=0,
                   device=device)
    ck3 = load_checkpoint(out, device)
    for gn in gate_names:
        assert gn in ck3["dit"], f"trained ckpt missing gate {gn}"
    assert ck3["config"]["dit_cross_gate"] is True
    print("[6] train_n5 warm-start tiny run saved a gated checkpoint")

    # --- 7: N5 checkpoint loads through the generation path (gated DiT) ---
    from .generate import load_models as gen_load
    cfg_r, vocab_r, vae_r, te_r, dit_r, whiten_r = gen_load(str(out), device)
    assert cfg_r.dit_cross_gate is True
    assert any(gn in dit_r.state_dict() for gn in gate_names)
    with torch.no_grad():
        o_gen = dit_r(z, t, text)
    assert o_gen.shape == (1, cfg.latent_channels, cfg.latent_size,
                           cfg.latent_size)
    print("[7] N5 checkpoint loads through generate.load_models (gated DiT)")

    print(f"\nn5_smoke: PASS (artifacts under {tmp})")


if __name__ == "__main__":
    main()
