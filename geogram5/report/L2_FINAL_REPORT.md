# L2 Image Lane Report (Geogram 5)

Date (UTC): 2026-04-20T14:22:09Z
Lane: `L2 image`
Wave: `geogram5`

## Claim Class
- `bounded_geometry_image_codec_with_isolated_bundle_extension`

## Frozen Inherited Truth
- `sparse_stroke` bounded adopter is real and remains sovereign on the shipped release surface.
- `topological_segmentation` alone remains non-adopted.
- `mixed` and `negative_natural` remain hard reject buckets.

## Primary Target: Split Preservation
- Repo: `zpe-image-codec`
- Result: `preserved`
- Live authority branch remains `source.core.imc.IMCEncoder.add_image -> image.geometry_codec.route_image_for_imc`
- Authoritative branch verdict: `PASS`
- Sparse live metrics: positive accept `1.0`, negative reject `1.0`, worst perturb IoU `0.63189`, worst perturb skeleton F1 `0.741313`, helper gap `0.0`
- Sparse accepted positive cases remain `glyph_a`, `fork_tree`, `loop_spine`, `maze_turns`, `serpentine`
- Sparse mean bytes on accepted scope remain candidate `1362.0` vs quadtree `7839.4`

## Split Hygiene Result
- The split defect was repo bootstrap, not lane science. `zpe-image-codec` was still depending on sibling `zpe-core`, but the local repo surfaces were not finding it automatically.
- The fix was limited to split-local bootstrap and script/test path hygiene so the sparse codec can be rerun honestly in the split repo without touching the bounded authority implementation in `zpe-core`.
- Standalone preservation checks now pass in this repo.

## Secondary Target: Bundle Experiment
- Hypothesis: `base = topological segmentation`, `fiber = sparse stroke`, `state = scale/context`
- Experimental result: `bounded_secondary_codec`
- Scope name: `hole_bearing_sparse_bundle`
- Accepted positive cases: `glyph_a`, `loop_spine`, `maze_turns`
- Out-of-scope positive cases: `fork_tree`, `serpentine`
- Reject bucket reject rate: `1.0`
- Scope accept rate: `1.0`
- Worst projection perturbation on accepted scope: IoU `1.0`, skeleton F1 `1.0`
- Helper gap: `0.0`
- Rescue cells: `loop_spine:dilate_1`, `maze_turns:dilate_1`

## What The Bundle Actually Did
- The accepted bundle scope is the hole-bearing sparse subset.
- The bundle decodes the topological base projection, which stayed exact on the accepted scope and its perturbation pack.
- Sparse fiber plus scale/context state rescue the two component-fill stress cells where topological-only would reject: `loop_spine:dilate_1` and `maze_turns:dilate_1`.
- This does **not** overturn the Geogram 4 truth that topological-only is non-adopted. The success is the bundle object, not topological-only promotion.

## Audit-Only Byte Surface
- Accepted bundle mean bytes: `3654.666667`
- Accepted topological mean bytes: `1773.0`
- Accepted sparse mean bytes: `1871.666667`
- Accepted quadtree mean bytes: `8458.666667`
- Reading: the bundle remains smaller than quadtree on its accepted scope, but byte size is not the promotion gate.

## Promotion Discipline
- The bundle codec is **not** promoted to the default release surface.
- Reason: the shipped sparse route remains broader and already authoritative across five positive cases, while the bundle codec is a narrower second codec on three hole-bearing cases.
- Release truth therefore stays:
  - default bounded release surface = `sparse_stroke`
  - isolated secondary codec = `hole_bearing_sparse_bundle`

## What Survived
- The sparse bounded adopter survived the repo split unchanged in substance.
- Reject buckets stayed hard rejects on both the preserved sparse route and the bundle experiment.
- The bundle hypothesis produced a real second codec only in the narrower hole-bearing subset, without contaminating the shipped sparse route.

## What Failed
- Broad image closure is still false.
- `fork_tree` and `serpentine` remain outside the bundle scope.
- `topological_segmentation` alone is still not adoptable and should not be re-described as if the bundle success rehabilitated it.

## Frozen Non-Claims
- No unified general-image codec claim.
- No natural-image codec claim.
- No softening of prior kills: natural-image/proxy audit floors remain inherited from Geogram 4 and unchanged for this lane.

## Final Verdict
- Primary target: `SUCCESS_PRESERVED`
- Secondary target: `SUCCESS_BOUNDED_SECONDARY_CODEC`
- Lane truth after Geogram 5: `sparse_stroke` remains the sovereign shipped bounded codec, and a second isolated bundle codec now exists for the narrower hole-bearing sparse subset.

## Evidence Locations
- `/Users/Zer0pa/ZPE_CANONICAL/zpe-image-codec/geogram5/artifacts/l2_image_geogram5.json`
- `/Users/Zer0pa/ZPE_CANONICAL/zpe-image-codec/geogram5/artifacts/preservation_probe.json`
- `/Users/Zer0pa/ZPE_CANONICAL/zpe-image-codec/geogram5/experimental_bundle_codec.py`
- `/Users/Zer0pa/ZPE_CANONICAL/zpe-image-codec/scripts/geogram5_l2_image_eval.py`
- `/Users/Zer0pa/ZPE_CANONICAL/zpe-image-codec/tests/test_image_bundle_experiment.py`
- `/Users/Zer0pa/ZPE_CANONICAL/zpe-image-codec/tests/test_image_geometry_authority.py`
