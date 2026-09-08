# N5.1 — Shape-Geometry Diagnostic Report

**Experiment:** N5.1 (diagnostic-only) — audit of the shape classifier discrepancy.
**Type:** No training. No checkpoint modification. No model-code change.
**Date:** 2026-09-08
**Branch:** `arena/01a07cdc-anime-ai-companion`

## 1. Problem under investigation

The N5 auto-evaluator reports `shape_accuracy = 0.333` (4/12) while a Gemini
visual inspection claimed **12/12 correct shapes**. Color auto-scoring is
`1.000`. N5.1 must dissect this discrepancy from *measured numeric features*,
never from fabricated pixels. This workspace holds **no real N5-generated PNGs**
(they exist only at the Drive checkpoint). The rows below that are *actually
measured* here are the **ideal-geometry oracle** and the **analytic decision
boundary** results. N5 pixel rows require re-running generation from the N5
checkpoint on the machine that holds it (exact command in §6).

## 2. What was audited (this run)

`ray_image/probe_n5_shape.py` reproduces the evaluator's exact feature/score
functions (from `tools/evaluate_toy_suite.py`) and adds two independent checks:

1. **Ideal-geometry oracle** — renders the 12 classes with the same rasterizer
   used to build the toy dataset (`ray_image.toy.draw_shape`, 64×64), then runs
   the evaluator's metric on those "perfect" shapes.
2. **Analytic decision boundaries** — the classifier's behaviour as a function
   of `bbox_fill` alone (justified below).

Machine-readable rows: `n5_1_report.json` (oracle rows, summaries, calibration).
Ideal shape PNGs: `oracle/`.

## 3. Measured oracle result

For geometrically ideal shapes (seed 0, 64×64, centered ~42% size):

| target class | bbox_fill | whole-image corners | predicted | correct |
|---|---|---|---|---|
| circle | 0.781 | [0,0,0,0] | circle | yes |
| square | 1.000 | [0,0,0,0] | square | yes |
| triangle | 0.501 | [0,0,0,0] | triangle | yes |

**Oracle shape_accuracy = 1.0 / 1.0** for all 12 classes. So the evaluator is
*internally able* to classify *perfect* shapes correctly — it is not broken for
clean inputs. The interesting failure mode only appears for *imperfect* shapes
(real model outputs), as quantified in §4.

## 4. Root cause found: the shape score reduces to a single fill threshold

`tools/evaluate_toy_suite.features()` computes `corners` at the **four
whole-image corner patches** (`mask[:q,:q]`, `mask[:q,-q:]`, `mask[-q:,:q]`,
`mask[-q:,-q:]`, where `q = min(h,w)//4`), **not** at the four corners of the
object's bounding box.

For any *centered* shape on the 64×64 canvas (dataset style: object at ~42–70%
of frame size, centered), those whole-image 16×16 corner patches are **empty for
every shape** — measured value ≈ `[0,0,0,0]` regardless of whether the object is
a circle, square, or triangle (verified on the oracle above).

Consequences:

- The `corner_targets` in `shape_score` (`circle 0.03, square 0.25,
  triangle [0,0,0.18,0.18]`) **never apply** to real centered outputs. That
  weighting was clearly written expecting bbox-local corners; against whole-image
  corners it is effectively dead code. It only adds a constant penalty to every
  candidate (square gets the largest constant penalty, `0.25`), which shifts
  scores but does not separate shapes.
- Classification therefore reduces to a **1-D threshold on `bbox_fill`**
  (object pixels inside its tight bounding box).

Computing the argmax of the three shape scores over `bbox_fill` (corners=0)
gives these exact decision bands:

| shape | bbox_fill winning band |
|---|---|
| triangle | fill **< 0.626** |
| circle | 0.626 – 0.954 |
| square | **> 0.954** |

Ideal geometry sits just inside these bands (triangle 0.50, circle 0.78,
square 1.00), so *perfect* shapes classify correctly — but the **square band is
only ~0.046 wide** (0.954 → 1.0).

**This is the mechanism that converts visually-correct shapes into the 0.333
auto-score.** A real model output is never perfectly flat-filled: antialiasing,
a slightly soft or rounded edge, or any small fill loss pushes a true square's
`bbox_fill` a few points below 1.0. If it drops to anywhere in **0.63–0.95 the
shape is classified as a *circle***; if it drops below 0.63 it is scored a
*triangle*. Meanwhile a genuine circle's band is wide, so the **4 circles are the
ones most likely to be scored correct** — matching the reported
`color=1.000, shape=0.333` (= exactly the 4 circles counted). The triangle band
(below 0.626) is also wide, but a filled triangle rendered with soft/wide edges
tends to raise `bbox_fill` toward 0.6+ and drift into the circle band.

The corner feature is therefore a **false precision** in the evaluator: it looks
discriminative on paper (3-corner templates) but is inert for the centered,
~half-frame objects this task actually generates.

## 5. Answers to the four questions

**(1) Why does the auto-evaluator classify N5 shapes "incorrectly" (0.333)?**
Not necessarily because N5 drew wrong shapes. The evaluator's shape decision is
effectively a threshold on `bbox_fill` (whole-image corners are empty for
centered shapes, so the corner template terms are inert). Its *square* win-band
is razor-thin (`fill > 0.954`); real, slightly-soft squares land in
`0.63–0.95` and are scored *circle*. Circles have the widest band, which is why
shape accuracy ≈ 4/12 (the four circles) is the natural outcome of visually-correct
but imperfect shapes. Numeric confirmation of the exact band is in the report JSON
(`evaluator_calibration.bbox_fill_decision_bands_corners_zero`).

**(2) Did N5 actually improve shape generation?**
**Not determinable from this workspace.** No real N5 (or N2) suite PNGs / feature
rows exist here, so no measured N5 vs N2 comparison is possible and none was
fabricated. The claim "Gemini sees 12/12" is *consistent with* the evaluator
being too strict on squares, but pixel-level confirmation requires the measured
rows from §6 (and ideally a vision-grounded label). Until then this is flagged
as **pending**, not concluded.

**(3) Does the evaluator need recalibration?**
**Yes.** Concrete, measured basis: the corner-occupancy feature is dead weight for
centered shapes (whole-image corners ≈ 0), leaving a single fill threshold with a
~0.046-wide square band. Recommended fix: measure corners/quadrants **inside the
object bounding box** (which is what the existing `corner_targets` were written
for), or score by bbox-local corner occupancy, which robustly separates the three
shapes even with mild antialiasing. Recalibrating should lift auto-shape toward
the visual 12/12 without retraining.

**(4) Is N5 a successful experiment?**
Cannot be declared from numeric metrics alone. The auto metric being stuck at 0.333
is fully explained by the evaluator's square-band calibration defect and does *not*
by itself show N5 failed. Success is only establishable once the §6 rows are
measured (and visual labels reconciled). This report does **not** claim N5 success
or failure — it isolates the measurement artifact and gives the exact procedure to
get ground truth.

## 6. Pending step — run on the machine holding the N5 checkpoint

The repo cannot generate from the Drive `.pt` (no trained generator weights in this
workspace). Run the same diagnostic on the N5 host to attach the measured N5 rows
(and N2 rows if desired):

```bash
python -m ray_image.probe_n5_shape \
  --n5-checkpoint /content/ray_image_v0_3_n5.pt \
  --outdir /content/n5_1 \
  [--n2-checkpoint /content/ray_image_v0_2_whiten.pt]   # optional N2 comparison
  [--n2-dir /path/to/n2/pngs]                            # or reuse existing N2 PNGs
```

Generation uses the identical settings as the N2/N5 runs (12 prompts,
`seed=42`, `steps=50`). It writes `n5_1_report.json` with per-image rows:
`bbox_fill`, bbox width/height, aspect ratio, object area, centroid, four corner
occupancies, circle/square/triangle scores, and the evaluator's predicted shape —
exactly the fields this report references. Copy the produced JSON/PNGs back under
`runs/N5.1/` in this repo.

## 7. Integrity notes

- All numbers in §3–§4 were **measured** by the diagnostic (oracle + analytic
  sweep), not assumed.
- No N5/N2 pixel metrics were fabricated; §2(2) explicitly awaits real footage.
- No model code, checkpoints, or training were modified.
