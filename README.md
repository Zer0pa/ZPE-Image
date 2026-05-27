# ZPE-Image

> Product-page mirror for `/encoding/ZPE-Image/`.
> Live public repo: [Zer0pa/ZPE-Image](https://github.com/Zer0pa/ZPE-Image).
> GitHub Markdown cannot reproduce the website typography, CSS, JavaScript, scroll behavior, or live bento layout; this README translates the product page into GitHub-safe Markdown evidence blocks.

## 0. Install / Developer Commands

The product page is the positioning authority. This section is the only retained developer-surface material from the previous root README.

```bash
python3 -m pip install --upgrade pip
python3 -m pip install '.[dev]'
pytest -q
```

## Product Page Mirror

**Product-page title:** ZPE-Image · Structural representation from sparse geometry and strokes · Zer0pa

**Product-page description:** ZPE-Image · bounded sparse-stroke codec · 5.75× internal-baseline byte reduction across 5 accepted sparse figures, 7/7 retained negative rejects · proof packet 2026-04-21 · PyPI v0.1.0 stale · SAL-7.1

### Hero Translation

> 00 · ZPE-IMAGE · STRUCTURAL CODECRESEARCH-READY · EXTERNAL COMPARATORS OPEN Store the shape, not the screenshot. Sparse-stroke encoder · ZPE-Image · zpe-image 0.1.0 stale · github.com/Zer0pa/ZPE-Image Most image codecs sell one number: bytes saved on photographs. ZPE-Image sells two. On five sparse structural figures — glyphs, mazes, flow-graphs, skeletons — it delivers 5.75× byte reduction against the internal baseline. On seven natural and mixed inputs, it returns a refusal at encode time. The codec is built to answer two questions in one packet: what it stores, and what it will never accept. Photographs are not in scope, and that is the point.

## Positioning

| Field | Value |
| --- | --- |
| Section | encoding |
| Product route | /encoding/ZPE-Image/ |
| Live public repository | https://github.com/Zer0pa/ZPE-Image |
| Repo identity used here | ZPE-Image |
| Website display identity | ZPE-Image |
| Verdict | STAGED |
| Posture | ready_for_publication_review |
| Headline metric | SPARSE_ACCEPTS: 5/5. ZPE-Image canonical authority surface; useful now, improving continuously. |
| Honest blocker | We do not claim photo, texture, gradient, or broad natural-image coverage.; We do not claim that the narrower hole-bearing route covers the full sparse set.; We do not claim that bounded acceptance on this pack equals general image coverage. |
| Mechanics asset from product page | IMAGE.gif |

## Key Metrics

| Metric | Value | Baseline |
| --- | --- | --- |
| SPARSE_ACCEPTS | 5/5 | bounded pack |
| REJECT_RATE | 100% | 7 retained negatives |
| SPARSE_WORST_PERTURB_IOU | 0.632 | floor threshold ≥0.62 |
| SPARSE_WORST_PERTURB_SKELETON_F1 | 0.741 | floor threshold ≥0.74 |

## Proof Anchors

| Path | State |
| --- | --- |
| proofs/manifests/CURRENT_VERIFICATION_PACKET.md | VERIFIED |
| proofs/artifacts/fresh_falsification_packet.json | VERIFIED |
| validation/results/fresh_falsification_check.json | VERIFIED |

## What We Prove

- The primary sparse-stroke route accepts `glyph_a`, `fork_tree`, `loop_spine`, `maze_turns`, and `serpentine`, and rejects all 7 retained mixed and natural-image negatives on the same verification pack.
- Under four perturbation types (dilate_1, salt_pepper_1pct, shift_x1, shift_y1) the worst-case reconstruction IoU is 0.632 and worst-case skeleton F1 is 0.741, both clearing their documented thresholds (0.62 and 0.74 respectively).
- On accepted sparse figures the geometry-sparse-stroke encoder uses a mean of 1,362 bytes (20-bit packed) versus 7,839 bytes for the quadtree-enhanced fallback — a 5.75× byte reduction within this bounded scope.
- The narrower hole-bearing bundle route accepts exactly `glyph_a`, `loop_spine`, and `maze_turns`; `fork_tree` and `serpentine` are correctly outside that subset. Bundle projection IoU = 1.0 and skeleton F1 = 1.0 under all perturbations for all three accepted cases.

## What We Do Not Claim

- No photo, texture, gradient, or broad natural-image coverage.
- The hole-bearing route does not cover the full sparse set.
- Bounded acceptance on this pack does not equal general image coverage.
- No external codec comparison (JPEG/WebP/AVIF/JPEG-XL) — baselines are internal only.

## Blockers / Failures

> We do not claim photo, texture, gradient, or broad natural-image coverage.; We do not claim that the narrower hole-bearing route covers the full sparse set.; We do not claim that bounded acceptance on this pack equals general image coverage.

## Verification Surface

| Code | Check | Verdict |
| --- | --- | --- |
| V_01 | Primary sparse route accepts the five bounded sparse figures. | PASS |
| V_02 | Primary sparse route rejects the mixed and natural-image buckets. | PASS |
| V_03 | Sparse perturbation floors stay above documented thresholds (IoU ≥0.62, skeleton F1 ≥0.74). | PASS |
| V_04 | Secondary hole-bearing route accepts exactly 3/5 positives; rejects out-of-scope positives. | PASS |
| V_05 | Installed package imports and runs without sibling runtime dependencies. | PASS |
| V_06 | Root repo surface ships with Zer0pa Source-Available License v7.0. | PASS |

## License

| Field | Value |
| --- | --- |
| License | SAL-7.0 |
| Authority source | proofs/artifacts/fresh_falsification_packet.json |

## Upcoming Workstreams

| Category | Summary |
| --- | --- |
| Active Engineering | External comparator integration — add gzipped-SVG and PNG-of-render-at-fixed-quality comparators to the proof artifact; report both, let reviewer pick. Removes the no-market-reference-frame gap on the current 5.75× CR claim. |
| Operations / External Dependency | Maintain CI gates and license-resolver synchronization with Zer0pa/ZPE-License-Commercial. |

## Related Repos

No related repos are declared on the product page frontmatter.

<details>
<summary>Full Visible Product-Page Bento Translation</summary>

This section preserves the product page cells as Markdown text blocks. It intentionally omits shared site navigation, footer chrome, CSS, and scripts.

### Bento Cell 1

> 00 · ZPE-IMAGE · STRUCTURAL CODECRESEARCH-READY · EXTERNAL COMPARATORS OPEN Store the shape, not the screenshot. Sparse-stroke encoder · ZPE-Image · zpe-image 0.1.0 stale · github.com/Zer0pa/ZPE-Image Most image codecs sell one number: bytes saved on photographs. ZPE-Image sells two. On five sparse structural figures — glyphs, mazes, flow-graphs, skeletons — it delivers 5.75× byte reduction against the internal baseline. On seven natural and mixed inputs, it returns a refusal at encode time. The codec is built to answer two questions in one packet: what it stores, and what it will never accept. Photographs are not in scope, and that is the point.

### Bento Cell 2

> 01 · THE GAPENCODED OR REFUSED No compact codec addresses sparse structural strokes while explicitly refusing everything else.

### Bento Cell 3

> 02 · MARKETSADJACENT FORECASTS Technical illustration / CAD'30 · $12.8B Diagramming software'30 · $1.9B Document archival'30 · $7.3B Vector graphics tools'30 · $4.1B Graphic design'31 · $85.5B Every technical drawing held in a corporate archive touches these markets; ZPE-Image addresses the sparse-stroke slice that compresses without taking in photographs.

### Bento Cell 4

> 03 · VALUE $1.9B 2030 diagramming software; ZPE-Image is the sparse-stroke archive slice inside that category.

### Bento Cell 5

> 04 · INSIGHT Structural drawings need a bounded archive.

### Bento Cell 6

> 05.1 · CURRENT TECHCOMPRESS EVERYTHING JPEG, WebP, AVIF, and JPEG-XL compress photographs. SVG describes vectors. None of them ask whether the input is the right kind of image. A maze, a glyph, a skeleton sits inside formats built for something else.

### Bento Cell 7

> 05.2 · OUR TECHKNOW THE BOUNDARY ZPE-Image encodes structural figures where the line itself carries meaning. It names its accepted scope — 5 of 5 sparse figures — and its categorical refusals — 7 of 7 natural and mixed inputs — inside the same packet. On accepted figures, bytes drop by 5.75× against the internal baseline. On the rest, the codec returns nothing.

### Bento Cell 8

> 05.3 · BENCHMARKSACCEPT / REJECT PACK Bytes5.75× vs quadtree Accept5/5sparse figures Reject7/7natural negatives Floors0.632IoU · F1 0.741 5/5 acceptPASS 7/7 rejectPASS comparatorsOPEN Scope: internal baseline only. JPEG, WebP, AVIF, and JPEG-XL comparator runs are open.

### Bento Cell 9

> 06 · MEASUREMENTACCEPT/REJECT PACKET Every measurement names what the codec accepted — and what it refused.

### Bento Cell 10

> 06.1 · COMPARATIVE PERFORMANCEINTERNAL BASELINE BYTES ZPE-Image5.75× smaller Internal quadtree1.00× baseline JPEG/WebP/AVIFno comparator JPEG-XLnot closed Five sparse figures accepted, seven natural and mixed inputs refused. Perturbation floors: IoU 0.632, skeleton F1 0.741 under four perturbation types. Artifact dated 2026-04-21. External comparators against JPEG, WebP, AVIF, JPEG-XL not yet closed.

### Bento Cell 11

> 07 · KEY METRICSMEASURED RESULTS

### Bento Cell 12

> 07.1 · INTERNAL BYTES 5.75× vs internal baseline · five accepted sparse figures

### Bento Cell 13

> 07.2 · ACCEPTED 5/5 sparse figures accepted · internal sparse pack

### Bento Cell 14

> 07.3 · REJECTED 7/7 natural and mixed inputs refused · at encode time

### Bento Cell 15

> 07.4 · PERTURBATION FLOOR 0.632 IoU 0.632 · skeleton F1 0.741 · four perturbation types

### Bento Cell 16

> 07.5 · PROOF PACKET 04-21 2026-04-21 · PyPI zpe-image 0.1.0 stale

### Bento Cell 17

> 08 · SCOPE ENFORCEMENTACCEPTED STRUCTURE ONLY The archive stays useful because the boundary is enforced.

### Bento Cell 18

> 08.1 · WHAT THE BOUNDARY MEANSACCEPT / REJECT On five accepted sparse figures, the dated artifact reports 5.75× byte reduction against the internal quadtree fallback. Seven natural or mixed inputs are refused at encode time, so photographs and wrong inputs never enter the structural archive in the first place. Under four perturbation types — dilate, salt-and-pepper, x-shift, y-shift — the accepted figures hold an IoU floor of 0.632 and a skeleton F1 floor of 0.741. The structural identity of a stored drawing survives a degraded source.

### Bento Cell 19

> 08.2 · HONEST BLOCKER Honest Blocker · Natural-image coverage is out of scope by design, not a gap. External comparator closure against JPEG, WebP, AVIF, JPEG-XL, gzip, SVG, and PNG is open. The five-figure corpus has not been expanded. PyPI zpe-image 0.1.0 is stale, with a corrected release pending.

### Bento Cell 20

> 09 A CODEC THAT KNOWS WHAT IT REFUSES.

### Bento Cell 21

> 09.1 · THE AMBITIONA BOUNDARY THAT HOLDS The aim is a structural archive for the world's technical drawings — engineering diagrams, glyphs, schematic flows, hand-drawn proofs — that compresses where line structure carries the meaning, supports retrieval by shape, and physically refuses to swallow photographs that would corrupt the corpus and the compression claim along with it.

### Bento Cell 22

> 09.2 · WHAT WORKS NOW Working today: 5.75× byte reduction on five accepted sparse figures, 7 of 7 photograph refusals, perturbation floors held.

### Bento Cell 23

> 09.3 · WHAT'S STILL OPEN Open: external comparator closure against JPEG, WebP, AVIF and JPEG-XL; a current PyPI release; corpus expansion.

### Bento Cell 24

> 09.4 · ARCHIVES · NEAR-TERM (12–24 MO) Engineering archives hold more drawings A technical-illustration archive on a fixed storage budget can keep roughly six times more sparse figures online. The procurement question stops being “what do we delete this quarter” and becomes “what else can we keep accessible.”

### Bento Cell 25

> 09.5 · PROCUREMENT · NEAR-TERM (12–24 MO) Buyers can pick a codec that refuses A procurement team choosing a structural archive codec can specify behavior on out-of-scope inputs. A codec that rejects a photograph at encode time is a different purchase than one that silently produces garbage, and the contract can say so.

### Bento Cell 26

> 09.6 · STANDARDS · MID-TERM (24–48 MO) Drawing standards gain a refusal contract Engineering-drawing and digital-humanities standards bodies can adopt scope-refusal as a spec property, not a warning in documentation. The format itself enforces what it does not encode, so downstream pipelines stop inheriting silent degradation as a category of bug.

### Bento Cell 27

> 09.7 · SEARCH · MID-TERM (24–48 MO) Drawing archives become searchable by shape When the same structural figure encodes to the same compact packet, searching a glyph or schematic archive by skeleton topology stops being a brute pixel-similarity problem. Researchers and engineers find drawings the way they think about them — by structure.

### Bento Cell 28

> 09.8 · IDENTITY · PARADIGM (48 MO+) A drawing has a name, not just a file Once a structural figure resolves to the same packet across draftspeople and tools, a drawing acquires an identity independent of its file. Citation, provenance, and version chains for technical drawings become first-class, the way they already are for text and code.

</details>

---

Source mapping: product route `/encoding/ZPE-Image/` -> live public repo `Zer0pa/ZPE-Image`. README generated from product-page authority plus retained install/dev commands only.
