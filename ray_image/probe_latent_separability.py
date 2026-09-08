"""N4 diagnostic -- latent / flow / prediction separability probe (diagnostic only).

Context: N2 whitening changed nothing (color 1.000 / shape 0.333), and N3 showed
the text side is *not* obviously broken (distinct tokens, shape token does move
pooled embeddings and DiT conditioning), but the shape/color velocity-influence
ratio was 0.64 (shape<color). N4 therefore asks a sharper question on the
LATENT / VELOCITY side:

  For the balanced 12-class toy set, how separable are classes that share a
  color but differ in shape, versus classes that share a shape but differ in
  color -- in (1) the VAE latent targets, (2) the flow-matching target
  velocities, and (3) the velocity predictions of the trained N2 generator?

It additionally runs a fixed-noise "shape-swap" generation probe (same initial
noise/timestep, prompts red circle/square/triangle) across several seeds.

Constraints honored: no architecture change, no loss change, no dataset change,
no training. Only reads a trained checkpoint and measures. Nothing here is a
claim -- a random-init checkpoint (smoke) yields meaningless numbers by design.

Convention note: the N2 generator is trained on WHITENED mean latents, so every
distance/probe here operates in that same space when the checkpoint embeds a
whitening normalizer (recovered from ``generate.load_models``); ``vae.decode``
always receives the de-normalized latent.

Reference-geometry note: each of the 12 classes is rendered with the SAME
canonical seed so object position/scale are identical across classes -- the
only difference between two classes is the color and/or the shape silhouette.
This isolates the color axis and the shape axis rather than object placement.
"""
import argparse
import json
from pathlib import Path

import torch

from .generate import load_models
from .toy import COLORS, SHAPES, draw_shape

COLOR_ORDER = list(COLORS.keys())
SHAPE_ORDER = list(SHAPES)
T_VALUES = [0.25, 0.50, 0.75]
SWAP_PROMPTS = ["a red circle", "a red square", "a red triangle"]
SWAP_SEEDS = [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# small numeric helpers
# ---------------------------------------------------------------------------
def _to_tensor(img):
    import numpy as np
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]


def _l2(a, b):
    return float(torch.norm(a.reshape(-1) - b.reshape(-1), p=2).item())


def _mean_std(xs):
    if not xs:
        return {"n": 0, "mean": None, "std": None}
    x = torch.tensor(xs, dtype=torch.float32)
    return {"n": int(x.numel()), "mean": float(x.mean().item()),
            "std": float(x.std(unbiased=True).item() if x.numel() > 1 else 0.0)}


def _save_tensor(t, path):
    from PIL import Image
    import numpy as np
    arr = (t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
    Image.fromarray(arr).save(path)


def _group_distance_stats(class_vecs):
    """Aggregate per-pair L2 distances over (name, color, shape, vec) entries.

    Returns two buckets:
      same_color_diff_shape : pairs that keep color, vary shape
      same_shape_diff_color : pairs that keep shape, vary color
    each as {n, mean, std}.
    """
    same_color, same_shape = {}, {}
    for _name, color, shape, vec in class_vecs:
        same_color.setdefault(color, []).append((shape, vec))
        same_shape.setdefault(shape, []).append((color, vec))
    shape_axis, color_axis = [], []
    for color, members in same_color.items():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                shape_axis.append(_l2(members[i][1], members[j][1]))  # diff shape
    for shape, members in same_shape.items():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                color_axis.append(_l2(members[i][1], members[j][1]))  # diff color
    return {"same_color_diff_shape": _mean_std(shape_axis),
            "same_shape_diff_color": _mean_std(color_axis)}


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------
class N4Probe:
    def __init__(self, checkpoint_path, device=None, size=64, canonical_seed=1234,
                 steps=30):
        self.checkpoint_path = str(checkpoint_path)
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.size = size
        self.canonical_seed = canonical_seed
        self.steps = steps
        self.cfg, self.vocab, self.vae, self.text_encoder, self.dit, self.whiten = \
            load_models(str(checkpoint_path), self.device)
        self.vae.eval(); self.text_encoder.eval(); self.dit.eval()

    def _class_latent(self, color, shape):
        """Deterministic VAE mean latent (optionally whitened) [1,C,H,W]."""
        img = _to_tensor(draw_shape(self.size, color, shape,
                                    self.canonical_seed)).to(self.device)
        with torch.no_grad():
            _, mean, _ = self.vae.encode(img)
            z = mean
            if self.whiten is not None:
                z = self.whiten.normalize(z)
        return z

    def _text(self, prompt):
        from .dataset import encode_text
        toks = encode_text(prompt, self.vocab, self.cfg.max_tokens) \
            .unsqueeze(0).to(self.device)
        mask = toks.eq(0)
        emb = self.text_encoder(toks, mask=mask)
        return emb, mask

    def _fixed_noise(self, seed):
        g = torch.Generator(device=self.device).manual_seed(seed)
        shape = (1, self.cfg.latent_channels,
                 self.cfg.latent_size, self.cfg.latent_size)
        return torch.randn(shape, generator=g, device=self.device)

    # ---- component 1: VAE latent targets ----------------------------------
    def latent_separability(self):
        vecs = []
        for c in COLOR_ORDER:
            for s in SHAPE_ORDER:
                vecs.append((f"{c}_{s}", c, s, self._class_latent(c, s)[0]))
        return _group_distance_stats(vecs)

    # ---- component 2: flow-target velocities ------------------------------
    def flow_target_separability(self):
        # target velocity v = noise - z (t-independent in this linear flow).
        # Use one shared reference noise so every per-pair distance is measured
        # under matched noise. Report on the requested t grid (identical since
        # v does not depend on t).
        noise_ref = self._fixed_noise(7777)
        per_t = {}
        for t in T_VALUES:
            vecs = []
            for c in COLOR_ORDER:
                for s in SHAPE_ORDER:
                    z = self._class_latent(c, s)
                    v = (noise_ref - z)[0]
                    vecs.append((f"{c}_{s}", c, s, v))
            per_t[str(t)] = _group_distance_stats(vecs)
        return per_t

    # ---- component 3: trained-prediction separability ---------------------
    def prediction_separability(self):
        # Shared latent state xt(t) is used for all prompts at each t, so any
        # predicted-velocity difference is purely prompt (conditioning) driven.
        ref = self._class_latent("red", "circle")  # [1,C,H,W]
        noise = self._fixed_noise(9999)
        per_t = {}
        for t in T_VALUES:
            xt = (1.0 - t) * ref + t * noise
            vecs = []
            for c in COLOR_ORDER:
                for s in SHAPE_ORDER:
                    emb, mask = self._text(f"a {c} {s}")
                    tv = torch.tensor([t], device=self.device)
                    with torch.no_grad():
                        v = self.dit(xt, tv, emb, text_mask=mask)[0]
                    vecs.append((f"{c}_{s}", c, s, v))
            per_t[str(t)] = _group_distance_stats(vecs)
        return per_t

    # ---- component 4: shape-swap fixed-noise generation -------------------
    def shape_swap_probe(self, outdir):
        grids_dir = Path(outdir) / "grids" / "shape_swap"
        grids_dir.mkdir(parents=True, exist_ok=True)
        results = {"seeds": {}, "paths": []}
        steps = self.steps
        dt = 1.0 / steps
        for seed in SWAP_SEEDS:
            noise = self._fixed_noise(seed)
            images = {}
            for prompt in SWAP_PROMPTS:
                x = noise.clone()
                emb, mask = self._text(prompt)
                with torch.no_grad():
                    for i in range(steps):
                        t = 1.0 - i * dt
                        tv = torch.tensor([t], device=self.device)
                        v = self.dit(x, tv, emb, text_mask=mask)
                        x = x - dt * v
                    img = self._decode(x)  # [C,H,W]
                images[prompt] = img
                fname = grids_dir / f"seed{seed}_{prompt.replace(' ', '_')}.png"
                _save_tensor(img, fname)
                results["paths"].append(str(fname))
            pd = {}
            for i in range(len(SWAP_PROMPTS)):
                for j in range(i + 1, len(SWAP_PROMPTS)):
                    pd[f"{SWAP_PROMPTS[i]} vs {SWAP_PROMPTS[j]}"] = round(
                        (images[SWAP_PROMPTS[i]] - images[SWAP_PROMPTS[j]])
                        .abs().mean().item(), 5)
            results["seeds"][str(seed)] = pd
        return results

    def _decode(self, latent):
        z = latent
        if self.whiten is not None:
            z = self.whiten.denormalize(z)
        with torch.no_grad():
            return self.vae.decode(z).clamp(0, 1)[0]


# ---------------------------------------------------------------------------
# report driver
# ---------------------------------------------------------------------------
def _ratio(sep):
    """mean(same_color_diff_shape) / mean(same_shape_diff_color).

    >1 => shape-varying pairs farther apart than color-varying pairs (shape more
    separated in the measured representation); <1 => color dominates.
    """
    scds = sep["same_color_diff_shape"]
    ssdc = sep["same_shape_diff_color"]
    if scds["mean"] is None or ssdc["mean"] is None or not ssdc["mean"]:
        return None
    return round(scds["mean"] / ssdc["mean"], 4)


def _fmt(sep):
    return (f"same_color_diff_shape m={sep['same_color_diff_shape']['mean']} "
            f"(n={sep['same_color_diff_shape']['n']}) | "
            f"same_shape_diff_color m={sep['same_shape_diff_color']['mean']} "
            f"(n={sep['same_shape_diff_color']['n']})")


def run_probe(checkpoint, outdir, size=64, canonical_seed=1234, steps=30,
              device=None):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    probe = N4Probe(checkpoint, device=device, size=size,
                    canonical_seed=canonical_seed, steps=steps)

    latent = probe.latent_separability()
    flow = probe.flow_target_separability()
    pred = probe.prediction_separability()
    swap = probe.shape_swap_probe(outdir)

    report = {
        "checkpoint": str(checkpoint),
        "device": str(probe.device),
        "whitening_embedded": probe.whiten is not None,
        "reference_geometry_seed": canonical_seed,
        "latent_separability": latent,
        "flow_target_separability": flow,
        "prediction_separability": pred,
        "shape_color_distance_ratio": {
            "latent": _ratio(latent),
            "per_t_flow_target": {t: _ratio(v) for t, v in flow.items()},
            "per_t_prediction": {t: _ratio(v) for t, v in pred.items()},
        },
        "fixed_noise_shape_swap": swap,
    }

    report_path = outdir / "n4_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    _print_summary(report)
    print(f"[N4] full report -> {report_path}")
    return report


def _print_summary(r):
    print("=== N4 LATENT/VELOCITY SEPARABILITY PROBE ===")
    print(f"checkpoint={r['checkpoint']} device={r['device']} "
          f"whitening_embedded={r['whitening_embedded']}")
    print(f"\n[1] VAE latent separability\n  {_fmt(r['latent_separability'])}")
    print("\n[2] flow-target separability (velocity = noise - z)")
    for t, v in r["flow_target_separability"].items():
        print(f"  t={t}: {_fmt(v)}")
    print("\n[3] trained-prediction separability (shared noise, prompt-driven)")
    for t, v in r["prediction_separability"].items():
        print(f"  t={t}: {_fmt(v)}")
    sc = r["shape_color_distance_ratio"]
    print("\n[4] shape/color distance ratio (same_color_diff_shape / "
          "same_shape_diff_color)")
    print(f"  latent:            {sc['latent']}")
    print("  per_t flow_target: ", {k: sc["per_t_flow_target"][k]
                                    for k in sc["per_t_flow_target"]})
    print("  per_t prediction:  ", {k: sc["per_t_prediction"][k]
                                    for k in sc["per_t_prediction"]})
    print("\n[5] fixed-noise shape-swap (red circle/square/triangle)")
    sw = r["fixed_noise_shape_swap"]
    for seed, pd in sw["seeds"].items():
        print(f"  seed {seed}: {pd}")
    print("\n  grid paths under <outdir>/grids/shape_swap/")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--outdir", default="/content/n4")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--canonical-seed", type=int, default=1234)
    parser.add_argument("--steps", type=int, default=30,
                        help="sampling steps for the shape-swap generation probe")
    args = parser.parse_args()
    run_probe(args.checkpoint, args.outdir, size=args.size,
              canonical_seed=args.canonical_seed, steps=args.steps)


if __name__ == "__main__":
    main()
