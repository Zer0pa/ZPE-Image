# NOVELTY_CARD — ZPE-Image

**Product:** ZPE-Image image codec
**Domain:** Image encoding / sparse-stroke graphics compression
**Value prop:** Achieves ~5.8x byte reduction over quadtree baseline for sparse-stroke and hole-bearing binary figures by encoding images as skeleton polylines and topological row spans rather than pixel blocks.
**Date:** 2026-04-17
**Author:** Zer0pa (Pty) Ltd

---

## Compass-8 Status

**NO**

ZPE-Image does not implement any Compass-8 primitives. The codec operates on 2-D pixel arrays and produces a flat word stream; no directional orientation graph, 8-primitive spatial index, or Compass-8 routing layer is present anywhere in the implementation.

The LICENSE §7.5 novelty schedule confirms this. Code inspection of every module under `src/zpe_image_codec/` corroborates: no references to Compass-8, orientation primitives, or directional decomposition exist in the source tree.

What the codec actually does:
- Converts an input image to a binary mask using **Otsu threshold selection** (`geometry_codec.py:216–227`).
- Routes the masked image through either a **sparse-stroke path** or a **topological path** based on feature thresholds (`geometry_codec.py:506–574`).
- Falls back to an **enhanced quadtree encoder** (`enhanced_codec.py`) for images that pass neither gate.

---

## Standard Techniques Used

The following are standard, well-established techniques used as building blocks. None of these are claimed as novel contributions:

| Technique | Where used | Notes |
|---|---|---|
| Otsu's thresholding | `geometry_codec.py:216–227` (`otsu_threshold`) | Standard global histogram binarisation |
| Zhang-Suen thinning (morphological skeletonisation) | `geometry_codec.py:344–378` (`thin_binary_mask`) | Classic two-pass parallel thinning algorithm |
| Euclidean distance transform | `geometry_codec.py:851–854` (`_distance_map`) | Standard `scipy.ndimage.distance_transform_edt` |
| Connected-component labelling | `geometry_codec.py:283–319` (`connected_components`) | Standard 8-connected flood-fill |
| Shannon entropy | `geometry_codec.py:201–207` (`shannon_entropy_u8`) | Standard pixel histogram entropy |
| Run-length encoding (row spans) | `geometry_codec.py:888–900` (`_row_spans`) | Standard RLE over binary row scanlines |
| Bresenham line rasterisation | `geometry_codec.py:939–966` (`_draw_bool_line`, `_draw_gray_line`) | Standard Bresenham integer line algorithm |
| Quadtree spatial subdivision | `enhanced_codec.py` | Standard quadtree image encoding (fallback path) |

---

## Novel Contributions

The following contributions are scope-limited to this codec. File:line citations are provided for each.

### N1 — Skeleton-polyline direction-run encoding

**Description:** The codec traces the thinned skeleton into ordered polylines (`trace_skeleton_to_polylines`, `geometry_codec.py:411–452`), then encodes each polyline as a compact sequence of (direction, run-length) pairs using a fixed 8-direction step vocabulary (`_polyline_runs`, `geometry_codec.py:857–885`). Each step is a 3-bit direction index drawn from `_DIRS` (`geometry_codec.py:56–66`) packed alongside a 4-bit run count in a single 7-bit payload word. This represents the stroke structure of a binary figure in a form that is far more compact than pixel- or block-level encodings for sparse line art.

**Scope:** Only effective for sparse-stroke binary images. Explicitly gated by feature thresholds (`_try_sparse_route`, `geometry_codec.py:506–542`); non-qualifying images are rejected to the quadtree fallback.

**Key code anchors:**
- `geometry_codec.py:56–66` — `_DIRS` tuple defining the 8-step vocabulary
- `geometry_codec.py:411–452` — `trace_skeleton_to_polylines` — skeleton graph to ordered polylines
- `geometry_codec.py:616–657` — `_encode_sparse_candidate` — polyline-to-word-stream serialisation
- `geometry_codec.py:857–885` — `_polyline_runs` — direction/run compression of a polyline

### N2 — Dual-route geometry dispatcher with feature-gated acceptance

**Description:** The codec implements a dual-route dispatcher (`route_image`, `geometry_codec.py:141–166`) that evaluates both the sparse-stroke and topological routes against a multi-dimensional feature vector and selects the best-accepted route, falling back to quadtree only when neither geometry route qualifies. Acceptance criteria include binary purity, Shannon entropy, contrast, foreground fraction, skeleton-to-foreground ratio, component fill, and reconstruction IoU gating (`_try_sparse_route`, `geometry_codec.py:506–542`). The combination of these feature gates as a codec-level routing signal — rather than a preprocessing label — is novel in scope.

**Key code anchors:**
- `geometry_codec.py:141–166` — `route_image` — dispatcher entry point
- `geometry_codec.py:506–542` — `_try_sparse_route` — feature-gated sparse acceptance
- `geometry_codec.py:545–574` — `_try_topological_route` — feature-gated topological acceptance

### N3 — Topological row-span encoding with hole-count gating

**Description:** For hole-bearing sparse images that pass binary purity and topological feature thresholds, the codec encodes the binary mask as ordered (row, start-column, length) span triples (`_encode_topological_mask`, `geometry_codec.py:660–694`; `_row_spans`, `geometry_codec.py:888–900`). The route is activated only when the hole count is >= 1 (`count_holes`, `geometry_codec.py:489–503`; gate at `geometry_codec.py:573`), ensuring the topological representation is not applied to non-topological forms. This bounded, hole-count-gated span encoding for binary graphics is novel in scope.

**Key code anchors:**
- `geometry_codec.py:489–503` — `count_holes` — topological hole detection
- `geometry_codec.py:545–574` — `_try_topological_route` — topological acceptance + gating
- `geometry_codec.py:660–694` — `_encode_topological_mask` — span serialisation

---

## What Is NOT Claimed

- No claim to novelty over general image codecs (JPEG, PNG, WebP, etc.).
- No claim that the sparse-stroke route works on photographs, textures, gradients, or mixed-content images.
- No claim that direction-run polyline encoding is novel as a concept in isolation (it is novel as a codec routing primitive for binary sparse-stroke figures within this scope).
- No Compass-8 claim of any kind.
