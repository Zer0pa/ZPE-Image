# ZPE-Image

## What This Is

ZPE-Image is a deterministic sparse-stroke image encoder — one of 17 independent encoding products in the Zer0pa portfolio. It encodes bounded sparse figures (glyphs, flow graphs, mazes, structural skeletons) at the geometry layer rather than the pixel layer, keeping what matters and discarding what doesn't. Broad natural-image coverage is out of scope by design.

Two routes ship: a primary sparse-stroke route and a narrower hole-bearing bundle route for exactly three hole-bearing forms. Both are CI-falsified.

| Field | Value |
|-------|-------|
| Architecture | IMAGE_STREAM |
| Encoding | IMAGE_SPARSE_GEOMETRY_V1 |

**Strongest intrinsic result (CI-anchored, no external baseline):** 5.75× byte reduction on the five accepted sparse figures versus the internal quadtree-enhanced fallback, with 100% accept and reject rates and perturbation floors clearing documented thresholds.

## Key Metrics

| Metric | Value | Baseline | Notes |
|---|---:|---|---|
| SPARSE_ACCEPTS | 5/5 | bounded pack | 100% positive-accept rate |
| REJECT_RATE | 100% | 7 retained negatives | mixed + natural-image buckets |
| SPARSE_WORST_PERTURB_IOU | 0.632 | floor threshold ≥0.62 | worst case: maze_turns under salt-pepper |
| SPARSE_WORST_PERTURB_SKELETON_F1 | 0.741 | floor threshold ≥0.74 | worst case: glyph_a under salt-pepper |
| SPARSE_MEAN_BYTES | 1,362 | 7,839 quadtree baseline | 5.75× byte reduction on accepted figures |
| BUNDLE_ACCEPTS | 3/3 | hole-bearing subset | projection IoU = 1.0 under all perturbations |
| BUNDLE_MEAN_BYTES | 3,655 | 8,459 quadtree baseline | 2.31× byte reduction on bundle route |

> Source: `proofs/artifacts/fresh_falsification_packet.json`, `validation/results/fresh_falsification_check.json`
> Baseline is ZPE-Image's internal quadtree-enhanced codec. No external codec (JPEG/WebP/AVIF/JPEG-XL) comparison exists in this pack — Image is a no-external-comp lane.

## What We Prove

- The primary sparse-stroke route accepts `glyph_a`, `fork_tree`, `loop_spine`, `maze_turns`, and `serpentine`, and rejects all 7 retained mixed and natural-image negatives on the same verification pack.
- Under four perturbation types (dilate_1, salt_pepper_1pct, shift_x1, shift_y1) the worst-case reconstruction IoU is 0.632 and worst-case skeleton F1 is 0.741, both clearing their documented thresholds (0.62 and 0.74 respectively).
- On accepted sparse figures the geometry-sparse-stroke encoder uses a mean of 1,362 bytes (20-bit packed) versus 7,839 bytes for the quadtree-enhanced fallback — a 5.75× byte reduction within this bounded scope.
- The narrower hole-bearing bundle route accepts exactly `glyph_a`, `loop_spine`, and `maze_turns`; `fork_tree` and `serpentine` are correctly outside that subset. Bundle projection IoU = 1.0 and skeleton F1 = 1.0 under all perturbations for all three accepted cases.

## What We Don't Claim

- No photo, texture, gradient, or broad natural-image coverage.
- The hole-bearing route does not cover the full sparse set.
- Bounded acceptance on this pack does not equal general image coverage.
- No external codec comparison (JPEG/WebP/AVIF/JPEG-XL) — baselines are internal only.

## Commercial Readiness

| Field | Value |
|-------|-------|
| Verdict | STAGED |
| Commit SHA | c1ed7abaa560 |
| Source | `validation/results/fresh_falsification_check.json` |

## Tests and Verification

| Code | Check | Verdict |
|---|---|---|
| V_01 | Primary sparse route accepts the five bounded sparse figures. | PASS |
| V_02 | Primary sparse route rejects the mixed and natural-image buckets. | PASS |
| V_03 | Sparse perturbation floors stay above documented thresholds (IoU ≥0.62, skeleton F1 ≥0.74). | PASS |
| V_04 | Secondary hole-bearing route accepts exactly 3/5 positives; rejects out-of-scope positives. | PASS |
| V_05 | Installed package imports and runs without sibling runtime dependencies. | PASS |
| V_06 | Root repo surface ships with Zer0pa Source-Available License v7.0. | PASS |

Run: `pytest -q` — exercises V_01–V_06 via `tests/test_verification.py` and `tests/test_readme_contract.py`.

## Proof Anchors

| Path | State |
|---|---|
| `proofs/manifests/CURRENT_VERIFICATION_PACKET.md` | VERIFIED |
| `proofs/artifacts/fresh_falsification_packet.json` | VERIFIED |
| `validation/results/fresh_falsification_check.json` | VERIFIED |

## Repo Shape

| Field | Value |
|---|---|
| Proof Anchors | 3 |
| Authority Source | `proofs/artifacts/fresh_falsification_packet.json` |
| Runtime Package | `src/zpe_image_codec` |

## Quick Start

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install '.[dev]'
zpe-image-verify --output validation/results/fresh_falsification_check.local.json
pytest -q
```
