# L2 Image Lane Report (Geogram 4)

Date (UTC): 2026-04-20T12:15:39Z
Lane: `L2 image`
Wave: `geogram4`

## Claim Class
- `bounded_geometry_image_codec`

## Sovereign Result
- Status: `SUCCESS_BOUNDED_ADOPTER`
- Promoted family: `sparse_stroke`
- Why: sparse-stroke alone passes the full bounded-adopter matrix on the live authority path; topological-segmentation does not.
- Non-promotion rule: the hybrid router was not promoted as a single image claim because topological-only fails and the lane brief forbids laundering a mixed family into one codec claim.

## Frozen Scope
- Statement: only the sparse-stroke positive bucket is geometry-positive; `mixed` and `negative_natural` remain hard reject buckets.
- Positive bucket: `positive`
- Reject buckets: `mixed`, `negative_natural`
- Fixture counts: `positive=5`, `mixed=3`, `negative_natural=4`
- Accepted positive cases: `glyph_a`, `fork_tree`, `loop_spine`, `maze_turns`, `serpentine`

## Candidate Family Matrix
- `sparse_stroke`: `bounded_adopter_candidate`
- Sparse metrics: positive accept `1.0`, negative reject `1.0`, worst perturb IoU `0.63189`, worst perturb skeleton F1 `0.741313`, max helper gap `0.0`
- Sparse bytes: mean candidate `1362.0` vs quadtree `7839.4`, mean byte gain vs quadtree `20.908335`
- `topological_segmentation`: `fail_partial_scope`
- Topological metrics: positive accept `0.6`, negative reject `1.0`, worst perturb IoU `0.0`, worst perturb skeleton F1 `0.0`, max helper gap `0.0`
- `hybrid`: `bounded_adopter_candidate`
- Hybrid metrics: positive accept `1.0`, negative reject `1.0`, worst perturb IoU `0.738028`, worst perturb skeleton F1 `0.773663`, max helper gap `0.0`
- Hybrid bytes: mean candidate `1302.8` vs quadtree `7839.4`, mean byte gain vs quadtree `19.292924`

## Live Authority Outcome
- Authoritative branch: `source.core.imc.IMCEncoder.add_image -> image.geometry_codec.route_image_for_imc`
- Authoritative branch verdict: `PASS`
- Live route verdict: `bounded_adopter_candidate`
- Live metrics: positive accept `1.0`, negative reject `1.0`, worst perturb IoU `0.63189`, worst perturb skeleton F1 `0.741313`, max helper gap `0.0`
- Live bytes: mean candidate `1362.0` vs quadtree `7839.4`, mean byte gain vs quadtree `20.908335`
- Geometry route leaks on reject buckets: `[]`

## What Survived
- Sparse-stroke geometry is now wired into the authoritative add-image branch and passes the frozen 12-case matrix end to end.
- Every accepted positive case stayed on the geometry route under the perturbation pack used in this lane.
- Reject buckets stayed clean: all `mixed` and `negative_natural` cases fell back to enhanced quadtree rather than leaking through the geometry branch.
- Accepted-scope helper leakage stayed at `0.0`, so the bounded adopter is not being propped up by a side-channel inside scope.

## What Failed
- Topological-segmentation is not a bounded adopter. It drops two positive cases outright (`fork_tree`, `serpentine`) and collapses on dilated perturbations for `loop_spine` and `maze_turns`.
- Broad image closure remains false. The lane solved only the sparse-stroke subset; it did not solve general images or natural scenes.
- Hybrid is evidence, not the published claim. Its pass does not rescue topological-segmentation as an independent family.

## Non-Claims And Prior Kills
- Broad natural-image authority remains killed by prior falsifiers and stays outside this lane claim.
- Natural-image/proxy audit floors remain unchanged: max helper leakage gap `1.0`, max proxy-minus-authority gap `0.892895`, strengthened positive floor `0.379151`, strengthened negative floor `0.0`
- Worst prior natural negative remains `branch_tree__shadow_inversion_blur` with oracle authority gate `1.0` and strengthened authority gate `0.0`
- Contour-theater cases remain explicit rejects: `delta_fan__texture_low_contrast`, `woven_bundle__camouflage_occlusion`, `loop_cluster__camouflage_occlusion`

## Method Integrity Notes
- Sparse-stroke and topological-segmentation were tested separately, then jointly, as required by the brief.
- The promoted sparse result uses family-consistent reference masks for perturbation scoring. Raw Otsu-mask scoring was rejected because it mis-scored accepted geometry decodes against the wrong target.
- Promotion is limited to the family that independently clears the sovereign gate on the live authority path.

## Final Verdict
- Lane truth: `bounded_sparse_stroke_success`
- Publishable statement: IMC now has a bounded L2 image codec for sparse-stroke geometry-decomposable images, with explicit hard rejects outside that subset.
- Rejected statement: there is no honest claim here for a unified general image codec or for topological-segmentation as an independent bounded adopter.

## Evidence Locations
- `/Users/Zer0pa/ZPE/ZPE-IMC/.gpd/research/geogram4/lane2-image/BRIEF.md`
- `/Users/Zer0pa/ZPE/ZPE-IMC/.gpd/research/geogram3/lane2-image/GEOGRAM3_L2_IMAGE_FAMILY_HYPOTHESIS.md`
- `/Users/Zer0pa/ZPE/ZPE-IMC/v0.0/proofs/artifacts/geogram4/lane2-image/l2_image_geogram4.json`
- `/Users/Zer0pa/ZPE/ZPE-IMC/v0.0/proofs/artifacts/geometry-first/wave3/l2_image_wave3.json`
- `/Users/Zer0pa/ZPE/ZPE-IMC/v0.0/proofs/artifacts/geogram/geogram_natural_image_negative.json`
- `/Users/Zer0pa/ZPE/ZPE-IMC/v0.0/proofs/artifacts/geogram/geogram_proxy_audit.json`
- `/Users/Zer0pa/ZPE/ZPE-IMC/v0.0/code/source/image/geometry_codec.py`
- `/Users/Zer0pa/ZPE/ZPE-IMC/v0.0/code/source/image/dual_dispatch.py`
- `/Users/Zer0pa/ZPE/ZPE-IMC/v0.0/code/source/core/imc.py`
- `/Users/Zer0pa/ZPE/ZPE-IMC/v0.0/code/scripts/geogram4_l2_image_eval.py`
- `/Users/Zer0pa/ZPE/ZPE-IMC/v0.0/code/tests/test_image_geometry_authority.py`
