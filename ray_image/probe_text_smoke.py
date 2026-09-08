"""CPU smoke test for the N3 text-conditioning probe (no training).

Builds a random-init generator checkpoint (toy vocab) and runs the full probe
path on CPU. This validates that every analysis step executes and that the
report JSON has the expected structure.

WARNING: because the checkpoint is random-init, the numbers carry NO scientific
meaning. This only proves the code path. Real interpretation requires the
trained 8000-step generator checkpoint.
"""
import json
import tempfile
from pathlib import Path

import torch

from .config import RAYConfig
from .probe_text_conditioning import (COLOR_WORDS, DEFAULT_PROMPTS, SHAPE_WORDS,
                                      run_probe)
from .utils import build_models, save_checkpoint


def _build_ckpt(path):
    cfg = RAYConfig()
    device = torch.device("cpu")
    mods = build_models(cfg, device)
    vocab = {"<pad>": 0, "<unk>": 1}
    vocab.update({w: i + 2 for i, w in enumerate(["a"] + COLOR_WORDS + SHAPE_WORDS)})
    save_checkpoint(path, cfg, step=0,
                    vocab=vocab, **{k: v for k, v in mods.items()})
    return cfg


def main():
    tmp = Path(tempfile.mkdtemp(prefix="ray_probe_text_smoke_"))
    ckpt = tmp / "gen_random.pt"
    _build_ckpt(ckpt)

    outdir = tmp / "n3"
    report = run_probe(str(ckpt), str(outdir),
                       prompts=DEFAULT_PROMPTS, n_samples=2, seed=0,
                       device=torch.device("cpu"))

    # Structural assertions on the report.
    assert report["tokenizer"]["all_distinct_non_special"] is True, report["tokenizer"]
    ds = report["distance_split"]
    for k in ("same_shape_diff_color", "same_color_diff_shape"):
        assert "n_pairs" in ds[k] and "cosine_mean" in ds[k]
    cd = report["conditioning"]
    assert set(cd["shape_token_effect_on_cond"]) == {
        "a red circle -> a red square", "a red circle -> a red triangle"}
    oi = report["output_influence"]
    assert {"shape_token_effect_l2", "color_token_effect_l2"} <= set(oi)

    # Report file written.
    assert (outdir / "text_conditioning_report.json").exists()

    print(f"probe_text_smoke: PASS (artifacts under {tmp})")
    print(f"  tokenizer distinct: {report['tokenizer']['all_distinct_non_special']}")
    print(f"  distance split keys ok, influence keys ok")


if __name__ == "__main__":
    main()
