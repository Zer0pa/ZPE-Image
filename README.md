# ZPE-Image

## What This Is
ZPE-Image is a bounded image encoder for sparse-stroke figures, with a second narrower route for exactly three hole-bearing sparse forms. Broad natural-image coverage stays out of scope.

| Field | Value |
|-------|-------|
| Architecture | IMAGE_STREAM |
| Encoding | IMAGE_SPARSE_GEOMETRY_V1 |

## Key Metrics
| Metric | Value | Baseline |
|---|---:|---|
| SPARSE_ACCEPTS | 5/5 | bounded |
| REJECT_RATE | 100% | retained negatives |
| SPARSE_WORST_IOU | >=0.62 | perturb floor |
| BUNDLE_ACCEPTS | 3/3 | subset |

> Source: `proofs/artifacts/fresh_falsification_packet.json`, `validation/results/fresh_falsification_check.json`

## What We Prove
- The primary sparse-stroke route accepts `glyph_a`, `fork_tree`, `loop_spine`, `maze_turns`, and `serpentine`, and rejects the retained mixed and natural-image buckets on the same verification pack.
- The narrower hole-bearing route accepts exactly `glyph_a`, `loop_spine`, and `maze_turns`, with `fork_tree` and `serpentine` outside that narrower subset.
- The retained source packet reports `fresh_falsification_ready` and `ready_for_publication_review`.

## What We Don't Claim
- We do not claim photo, texture, gradient, or broad natural-image coverage.
- We do not claim that the narrower hole-bearing route covers the full sparse set.
- We do not claim that bounded acceptance on this pack equals general image coverage.

## Commercial Readiness
This table restates the retained source fields without broadening the bounded claim surface.

| Field | Value |
|-------|-------|
| Verdict | ready_for_publication_review |
| Verification Status | fresh_falsification_ready |
| Commit SHA | c1ed7abaa560 |
| Source | validation/results/fresh_falsification_check.json |

## Tests and Verification
| Code | Check | Verdict |
|---|---|---|
| V_01 | `pytest -q` verifies the retained bounded-route results and README proof anchors. | PASS |

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
