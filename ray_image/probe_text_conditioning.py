"""N3 diagnostic -- text-conditioning investigation (diagnostic only).

Why: the generator learns color but collapses shape (color_accuracy=1.000,
shape_accuracy=0.333). N1/N2 ruled out VAE latent whitening as the fix. This
probe inspects the *text side* of conditioning on a TRAINED generator
checkpoint to locate where color-vs-shape signal diverges.

It loads the trained text encoder + DiT from a generator checkpoint and, with
NO training, reports:

  1. Tokenizer behavior for color/shape words (distinct IDs? not <unk>/<pad>?).
  2. Token-level text embeddings for the requested prompts.
  3. Embedding distances/similarities, split into
       same-shape/different-color   vs   same-color/different-shape.
  4. Padding-mask correctness and whether shape tokens are valid (visible).
  5. The DiT conditioning vector (pooled text -> text_proj) for prompts
     differing ONLY in shape ("a red circle/square/triangle").
  6. A gradient/influence estimate: does swapping the shape token materially
     change the DiT output, and how does that compare to swapping the color
     token?

Nothing here modifies the VAE/text-encoder/DiT/objective/sampler/dataset or
evaluator. It only reads a checkpoint and measures.

WARNING (experimental discipline): a random-init checkpoint used for smoke
testing yields NO meaningful signal -- it only proves the code path runs.
The scientific reading requires the real trained 8000-step checkpoint.
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .dataset import encode_text
from .generate import load_models

# The exact prompts requested by the research plan.
COLOR_WORDS = ["red", "green", "blue", "yellow"]
SHAPE_WORDS = ["circle", "square", "triangle"]

DEFAULT_PROMPTS = [
    "a red circle",
    "a red square",
    "a red triangle",
    "a green circle",
    "a blue circle",
    "a yellow circle",
]

_EXTRA_FOR_PAIRS = [
    "a green square", "a green triangle",
    "a blue square", "a blue triangle",
    "a yellow square", "a yellow triangle",
]


def _prompt_words(prompt):
    return prompt.lower().split()


def _tokens(vocab, prompt, max_tokens):
    toks = encode_text(prompt, vocab, max_tokens)
    words = _prompt_words(prompt)
    return toks, words


def _mask(tokens):
    return tokens.eq(0)  # True == pad (masked out)


def _pooled(text_emb, mask):
    valid = (~mask).to(text_emb.dtype).unsqueeze(-1)
    pooled = (text_emb * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
    return pooled


def _normalize(x, dim=-1):
    return F.normalize(x, p=2, dim=dim)


def cosine_sim(a, b):
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1)[0])


def l2(a, b):
    return float(torch.norm(a - b, p=2))


def tokenizer_report(vocab):
    """Check color/shape words produce distinct, real (non-unk/pad) IDs."""
    rows = {}
    ok = True
    for w in COLOR_WORDS + SHAPE_WORDS:
        i = vocab.get(w)
        row = {"id": i}
        if i is None:
            row["status"] = "OOV (missing from vocab)"
            ok = False
        elif i in (0, 1):
            row["status"] = f"collides with special id {i}"
            ok = False
        else:
            row["status"] = "ok"
        rows[w] = row
    ids = [rows[w]["id"] for w in COLOR_WORDS + SHAPE_WORDS if rows[w]["id"] is not None]
    distinct = len(set(ids)) == len(ids)
    return {"rows": rows, "all_distinct_non_special": distinct and ok}


def encode_prompt_set(vocab, cfg, prompts):
    """Return per-prompt dicts with tokens, words, valid mask."""
    out = {}
    for p in prompts:
        toks, words = _tokens(vocab, p, cfg.max_tokens)
        out[p] = {"tokens": toks, "words": words, "valid": (~_mask(toks)).tolist()[:len(words)]}
    return out


@torch.no_grad()
def embedding_report(text_encoder, vocab, cfg, prompts, device):
    """Token-level embeddings + pooled embeddings per prompt."""
    rep = {}
    for p in prompts:
        toks = encode_text(p, vocab, cfg.max_tokens).unsqueeze(0).to(device)
        m = _mask(toks[0]).unsqueeze(0)  # [1, L]
        emb = text_encoder(toks, mask=m)[0]  # [L, dim]
        words = _prompt_words(p)
        rep[p] = {
            "words": words,
            "pooled": _pooled(emb.unsqueeze(0), m)[0],
            "token_norms": [float(emb[i].norm().item()) for i in range(min(len(words), emb.shape[0]))],
        }
    return rep


def _pair_avg(rep, prompts):
    """Pairwise cosine+L2 among pooled embeddings of a group of prompts."""
    if len(prompts) < 2:
        return None
    cos, l2s = [], []
    for i in range(len(prompts)):
        for j in range(i + 1, len(prompts)):
            cos.append(cosine_sim(rep[prompts[i]]["pooled"], rep[prompts[j]]["pooled"]))
            l2s.append(l2(rep[prompts[i]]["pooled"], rep[prompts[j]]["pooled"]))
    return {"n_pairs": len(cos), "cosine_mean": float(sum(cos) / len(cos)),
            "l2_mean": float(sum(l2s) / len(l2s))}


def distance_split_report(rep):
    """Compare same-shape/diff-color vs same-color/diff-shape pooled distance."""
    # same shape = circle, across colors:
    same_shape_groups = {s: [f"a {c} {s}" for c in COLOR_WORDS] for s in SHAPE_WORDS}
    same_color_groups = {c: [f"a {c} {s}" for s in SHAPE_WORDS] for c in COLOR_WORDS}

    def aggregate(groups):
        pairs, cos, l2s = 0, [], []
        for name, prompts in groups.items():
            avail = [p for p in prompts if p in rep]
            if len(avail) >= 2:
                for i in range(len(avail)):
                    for j in range(i + 1, len(avail)):
                        cos.append(cosine_sim(rep[avail[i]]["pooled"], rep[avail[j]]["pooled"]))
                        l2s.append(l2(rep[avail[i]]["pooled"], rep[avail[j]]["pooled"]))
                        pairs += 1
        return {"n_pairs": pairs,
                "cosine_mean": float(sum(cos) / max(pairs, 1)),
                "l2_mean": float(sum(l2s) / max(pairs, 1))}

    ss = aggregate(same_shape_groups)   # different color, same shape
    sc = aggregate(same_color_groups)   # different shape, same color
    return {"same_shape_diff_color": ss, "same_color_diff_shape": sc}


@torch.no_grad()
def conditioning_report(text_encoder, dit, vocab, cfg, prompts, device):
    """Compare pooled-text conditioning (excl. timestep) for shape-only prompts.

    Recomputes the DiT cond path (text_proj(pooled)) for the red-circle family.
    """
    red_family = [f"a red {s}" for s in SHAPE_WORDS]
    # Include green circle to represent same-shape/diff-color contrast.
    conds = {}
    for p in red_family + ["a green circle"]:
        toks = encode_text(p, vocab, cfg.max_tokens).unsqueeze(0).to(device)
        m = _mask(toks[0]).unsqueeze(0)  # [1, L]
        emb = text_encoder(toks, mask=m)
        pooled = _pooled(emb, m)
        conds[p] = dit.text_proj(pooled)[0]  # the text part of the DiT cond

    shape_changes = {}
    base = "a red circle"
    for s in SHAPE_WORDS:
        p = f"a red {s}"
        if p == base:
            continue
        shape_changes[f"{base} -> {p}"] = {
            "cosine": cosine_sim(conds[base], conds[p]),
            "l2": l2(conds[base], conds[p]),
        }
    color_contrast = cosine_sim(conds[base], conds["a green circle"])
    color_l2 = l2(conds[base], conds["a green circle"])
    return {
        "red_circle_family_cond_vectors": {
            p: {"norm": float(conds[p].norm().item())} for p in conds
        },
        "shape_token_effect_on_cond": shape_changes,
        "color_token_effect_on_cond_red_vs_green": {"cosine": color_contrast, "l2": color_l2},
    }


@torch.no_grad()
def output_influence(text_encoder, dit, cfg, vocab, device,
                     n_samples=8, seed=0):
    """Estimate whether swapping the shape vs color token changes DiT output.

    Fixes a random clean latent ``z`` (in the whitened space the generator was
    trained in) and integrates a few timesteps; measures the mean L2 change in
    the predicted velocity when only the shape token differs (red circle ->
    red square) versus only the color token differs (red circle -> green
    circle). Uses several seeds for z and a fixed t for comparability.

    Returns {shape_token_effect, color_token_effect, ratio_shape_over_color}.
    """
    rng = torch.Generator(device=device).manual_seed(seed)
    base_toks = encode_text("a red circle", vocab, cfg.max_tokens).unsqueeze(0).to(device)
    shape_toks = encode_text("a red square", vocab, cfg.max_tokens).unsqueeze(0).to(device)
    color_toks = encode_text("a green circle", vocab, cfg.max_tokens).unsqueeze(0).to(device)
    t = torch.tensor([0.5], device=device)  # middle of the flow trajectory

    def emb_of(toks):
        m = _mask(toks[0]).unsqueeze(0)
        return text_encoder(toks, mask=m)

    eb, es, ec = emb_of(base_toks), emb_of(shape_toks), emb_of(color_toks)
    mb, ms, mc = _mask(base_toks[0]).unsqueeze(0), _mask(shape_toks[0]).unsqueeze(0), _mask(color_toks[0]).unsqueeze(0)

    out_base, out_shape, out_color = [], [], []
    for _ in range(n_samples):
        z = torch.randn(1, cfg.latent_channels, cfg.latent_size, cfg.latent_size,
                        generator=rng, device=device)
        v_base = dit(z, t, eb, text_mask=mb)
        v_shape = dit(z, t, es, text_mask=ms)
        v_color = dit(z, t, ec, text_mask=mc)
        out_base.append(v_base); out_shape.append(v_shape); out_color.append(v_color)

    def mean_l2(a_list, b_list):
        return float(sum(torch.norm(a - b).item() for a, b in zip(a_list, b_list)) / len(a_list))

    shape_effect = mean_l2(out_base, out_shape)
    color_effect = mean_l2(out_base, out_color)
    return {
        "shape_token_effect_l2": shape_effect,
        "color_token_effect_l2": color_effect,
        "ratio_shape_over_color": shape_effect / max(color_effect, 1e-12),
        "n_samples": n_samples, "t": 0.5,
    }


def run_probe(checkpoint_path, outdir, prompts=None, n_samples=8, seed=0,
              device=None):
    if prompts is None:
        prompts = DEFAULT_PROMPTS
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg, vocab, vae, text_encoder, dit, whiten = load_models(checkpoint_path, device)
    text_encoder.eval()
    dit.eval()

    report = {"checkpoint": str(checkpoint_path), "device": str(device),
              "prompts": prompts,
              "latent_whitening_embedded": whiten is not None}

    report["tokenizer"] = tokenizer_report(vocab)

    enc = encode_prompt_set(vocab, cfg, prompts)
    report["token_ids_and_validity"] = {
        p: {"words": enc[p]["words"],
            "ids": enc[p]["tokens"][:len(enc[p]["words"])].tolist(),
            "valid_flags": enc[p]["valid"]}
        for p in prompts
    }

    # Widen the embedding report to all 12 class prompts so pair metrics are full.
    all12 = DEFAULT_PROMPTS + _EXTRA_FOR_PAIRS
    emb_rep = embedding_report(text_encoder, vocab, cfg, all12, device)
    report["embeddings"] = {p: {"words": emb_rep[p]["words"],
                                "token_norms": emb_rep[p]["token_norms"]} for p in all12}
    report["distance_split"] = distance_split_report(emb_rep)
    report["conditioning"] = conditioning_report(text_encoder, dit, vocab, cfg,
                                                 all12, device)
    report["output_influence"] = output_influence(
        text_encoder, dit, cfg, vocab, device, n_samples=n_samples, seed=seed)

    report_path = outdir / "text_conditioning_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    _print_summary(report)
    print(f"[probe] full report -> {report_path}")
    return report


def _print_summary(r):
    print("=== N3 TEXT-CONDITIONING PROBE ===")
    print(f"device={r['device']} checkpoint={r['checkpoint']}")
    print(f"latent_whitening_embedded={r['latent_whitening_embedded']}")
    print("\n[1] tokenizer")
    tz = r["tokenizer"]
    print(f"  all color/shape words distinct & non-special: {tz['all_distinct_non_special']}")
    for w, row in tz["rows"].items():
        print(f"    {w:<10} id={row['id']}  {row['status']}")
    print("\n[2] token validity (shape token visible to cross-attention?)")
    for p, info in list(r["token_ids_and_validity"].items()):
        print(f"    {p!r:<22} ids={info['ids']} valid={info['valid_flags']}")
    print("\n[3] pooled-embedding distance split")
    ds = r["distance_split"]
    print("  same shape, diff color:", ds["same_shape_diff_color"])
    print("  same color, diff shape:", ds["same_color_diff_shape"])
    print("\n[4] DiT cond effect, red-circle family (only shape token changes)")
    cd = r["conditioning"]
    for change, v in cd["shape_token_effect_on_cond"].items():
        print(f"    {change}: cosine={v['cosine']:.4f} l2={v['l2']:.4f}")
    cc = cd["color_token_effect_on_cond_red_vs_green"]
    print(f"  red circle -> green circle (color token): cosine={cc['cosine']:.4f} l2={cc['l2']:.4f}")
    print("\n[5] DiT output influence (mean |L2| velocity change)")
    oi = r["output_influence"]
    print(f"  shape-token swap effect: {oi['shape_token_effect_l2']:.6f}")
    print(f"  color-token swap effect: {oi['color_token_effect_l2']:.6f}")
    print(f"  ratio shape/color:       {oi['ratio_shape_over_color']:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True,
                        help="trained generator .pt checkpoint to inspect")
    parser.add_argument("--outdir", default="diagnostics/n3")
    parser.add_argument("--influence-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompts", nargs="*", default=None,
                        help="optional custom prompt list")
    args = parser.parse_args()
    run_probe(args.checkpoint, args.outdir, prompts=args.prompts,
              n_samples=args.influence_samples, seed=args.seed)


if __name__ == "__main__":
    main()
