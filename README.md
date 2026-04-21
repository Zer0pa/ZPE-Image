# ZPE-Image

## What This Is
ZPE-Image is a standalone bounded image encoder for sparse-stroke figures, with a second narrower route for hole-bearing sparse forms. It is useful now for clean line-structured black-on-light graphics, and it keeps mixed and natural-image content outside scope. The package installs and runs on its own from a fresh clone.

| Field | Value |
|-------|-------|
| Architecture | IMAGE_STREAM |
| Encoding | IMAGE_SPARSE_GEOMETRY_V1 |

## Key Metrics
| Metric | Value | Baseline |
|---|---:|---|
| SPARSE_ACCEPTS | 5/5 | bounded |
| REJECT_FRONTIER | 7/7 | rejected |
| SPARSE_WORST_IOU | 0.63189 | 0.62 floor |
| BUNDLE_ACCEPTS | 3/3 | subset |

> Source: `proofs/artifacts/fresh_falsification_packet.json`, `validation/results/fresh_falsification_check.json`

## Competitive Benchmarks
| Method | Scope | Result | Source |
|---|---|---|---|
| **ZPE-Image sparse route** | Five sparse-stroke cases | 1362.0 mean bytes | `proofs/artifacts/fresh_falsification_packet.json` |
| Quadtree baseline | Five sparse-stroke cases | 7839.4 mean bytes | `proofs/artifacts/fresh_falsification_packet.json` |
| **ZPE-Image hole-bearing route** | Three hole-bearing cases | 3654.666667 mean bytes | `proofs/artifacts/fresh_falsification_packet.json` |
| Sparse route on same subset | Three hole-bearing cases | 1871.666667 mean bytes | `proofs/artifacts/fresh_falsification_packet.json` |

## What We Prove
- The primary sparse-stroke route accepts the five bounded sparse figures and rejects the mixed and natural-image buckets on the same fresh verification pack.
- A second narrower route exists for exactly three hole-bearing sparse figures: `glyph_a`, `loop_spine`, and `maze_turns`.
- The hole-bearing route is not the default route and does not widen the primary sparse-stroke claim.
- The installed package can replay the documented verification path without any sibling runtime checkout.

## What We Don't Claim
- We do not claim photo, texture, gradient, or broad natural-image coverage.
- We do not claim that the narrower hole-bearing route covers the full sparse set.
- We do not claim that bounded acceptance on this pack equals general image coverage.
- We do not claim any broader cross-product admission beyond this repo's two bounded scopes.

## Commercial Readiness
Fresh falsification works from a clean install. This repo now ships with `Zer0pa Source-Available License v7.0` at the root surface. This release candidate is restamped to the verified source commit below.

| Field | Value |
|-------|-------|
| Verdict | STAGED |
| Commit SHA | c1ed7abaa560 |
| Confidence | 100% |
| Source | validation/results/fresh_falsification_check.json |

## Tests and Verification
| Code | Check | Verdict |
|---|---|---|
| V_01 | `zpe-image-verify --output validation/results/fresh_falsification_check.local.json` replays the sparse and hole-bearing verification pack from the installed package. | PASS |
| V_02 | `pytest -q` verifies the bounded route results and README parser contract. | PASS |
| V_03 | Root repo surface ships with `Zer0pa Source-Available License v7.0`. | PASS |

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
| Modality Lanes | 1 |
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
