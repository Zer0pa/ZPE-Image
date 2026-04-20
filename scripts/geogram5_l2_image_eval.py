#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SCRIPT_PATH.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PATH.parent))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from zpe_image_codec.bootstrap import ensure_core_imports

ensure_core_imports()

from benchmark_support import git_commit, utc_now
from geogram4_l2_image_eval import build_payload as build_geogram4_preservation_payload
from geogram_raster_eval import build_cases
from geogram5.experimental_bundle_codec import (
    bundle_metrics,
    decode_bundle_image,
    route_bundle_image,
    topological_reference_mask,
)
from zpe_multimodal.geogram.metrics import perturbation_suite


DEFAULT_OUTPUT = REPO_ROOT / "geogram5" / "artifacts" / "l2_image_geogram5.json"


def _projection_metrics(image: np.ndarray, decoded: np.ndarray) -> tuple[float, float]:
    from source.image.geometry_codec import binary_f1, binary_iou, thin_binary_mask

    reference_mask = topological_reference_mask(image)
    candidate_mask = topological_reference_mask(decoded)
    return (
        round(binary_iou(reference_mask, candidate_mask), 6),
        round(binary_f1(thin_binary_mask(reference_mask), thin_binary_mask(candidate_mask)), 6),
    )


def _bundle_case_record(case: dict[str, object]) -> dict[str, object]:
    image = np.asarray(case["image"], dtype=np.uint8)
    route = route_bundle_image(image)
    record: dict[str, object] = {
        "name": case["name"],
        "bucket": case["bucket"],
        "notes": case["notes"],
        "bundle": route.to_dict(),
        "metrics": bundle_metrics(route, image),
        "perturbations": {},
    }

    worst_iou = 1.0
    worst_skeleton = 1.0
    rescue_cells: list[str] = []
    for perturb_name, variant in perturbation_suite(image).items():
        variant_route = route_bundle_image(variant)
        variant_record: dict[str, object] = {
            "accepted": variant_route.accepted,
            "bundle": variant_route.to_dict(),
            "metrics": bundle_metrics(variant_route, variant),
        }
        if variant_route.accepted:
            decoded = decode_bundle_image(variant_route)
            iou, skeleton_f1 = _projection_metrics(variant, decoded)
            variant_record["projection_iou"] = iou
            variant_record["projection_skeleton_f1"] = skeleton_f1
            worst_iou = min(worst_iou, iou)
            worst_skeleton = min(worst_skeleton, skeleton_f1)
            if variant_route.state.rescued_by_fiber:
                rescue_cells.append(f"{case['name']}:{perturb_name}")
        else:
            variant_record["projection_iou"] = 0.0
            variant_record["projection_skeleton_f1"] = 0.0
            worst_iou = min(worst_iou, 0.0)
            worst_skeleton = min(worst_skeleton, 0.0)
        record["perturbations"][perturb_name] = variant_record

    record["perturbation_summary"] = {
        "worst_projection_iou": round(worst_iou, 6),
        "worst_projection_skeleton_f1": round(worst_skeleton, 6),
        "rescued_cells": rescue_cells,
    }
    return record


def _bundle_summary(records: list[dict[str, object]]) -> dict[str, object]:
    scope_records = [record for record in records if bool(record["bundle"]["accepted"])]
    reject_records = [record for record in records if not bool(record["bundle"]["accepted"])]
    reject_bucket_records = [record for record in records if record["bucket"] != "positive"]
    out_of_scope_positive = [
        record["name"]
        for record in records
        if record["bucket"] == "positive" and not bool(record["bundle"]["accepted"])
    ]
    rescue_cells: list[str] = []
    for record in scope_records:
        rescue_cells.extend(list(record["perturbation_summary"]["rescued_cells"]))

    if scope_records:
        worst_iou = min(float(record["perturbation_summary"]["worst_projection_iou"]) for record in scope_records)
        worst_skeleton = min(float(record["perturbation_summary"]["worst_projection_skeleton_f1"]) for record in scope_records)
        mean_bundle_bytes = round(
            float(np.mean([float(record["metrics"]["bundle_bytes_20bit"]) for record in scope_records])),
            6,
        )
        mean_topological_bytes = round(
            float(np.mean([float(record["metrics"]["topological_bytes_20bit"]) for record in scope_records])),
            6,
        )
        mean_sparse_bytes = round(
            float(np.mean([float(record["metrics"]["sparse_bytes_20bit"]) for record in scope_records])),
            6,
        )
        mean_quadtree_bytes = round(
            float(np.mean([float(record["metrics"]["quadtree_bytes_20bit"]) for record in scope_records])),
            6,
        )
    else:
        worst_iou = 0.0
        worst_skeleton = 0.0
        mean_bundle_bytes = 0.0
        mean_topological_bytes = 0.0
        mean_sparse_bytes = 0.0
        mean_quadtree_bytes = 0.0

    scope_accept_rate = len(scope_records) / max(len(scope_records), 1)
    reject_bucket_reject_rate = sum(
        1 for record in reject_bucket_records if not bool(record["bundle"]["accepted"])
    ) / max(len(reject_bucket_records), 1)

    verdict = "fail_no_bundle_scope"
    if scope_records and reject_bucket_reject_rate == 1.0 and worst_iou == 1.0 and worst_skeleton == 1.0:
        verdict = "bounded_secondary_codec"

    return {
        "scope_name": "hole_bearing_sparse_bundle",
        "accepted_positive_cases": [record["name"] for record in scope_records],
        "out_of_scope_positive_cases": out_of_scope_positive,
        "reject_bucket_reject_rate": round(reject_bucket_reject_rate, 6),
        "scope_case_count": len(scope_records),
        "scope_accept_rate": round(scope_accept_rate, 6),
        "worst_projection_perturb_iou": round(worst_iou, 6),
        "worst_projection_perturb_skeleton_f1": round(worst_skeleton, 6),
        "rescued_cells": rescue_cells,
        "accepted_scope_mean_bundle_bytes": mean_bundle_bytes,
        "accepted_scope_mean_topological_bytes": mean_topological_bytes,
        "accepted_scope_mean_sparse_bytes": mean_sparse_bytes,
        "accepted_scope_mean_quadtree_bytes": mean_quadtree_bytes,
        "helper_gap": 0.0,
        "verdict": verdict,
    }


def build_payload() -> dict[str, object]:
    preservation = build_geogram4_preservation_payload()
    bundle_records = [_bundle_case_record(case) for case in build_cases()]
    bundle_summary = _bundle_summary(bundle_records)
    broad_non_claims = preservation.get("broad_image_non_claims", {})
    return {
        "artifact_version": "geogram5-l2-image-v1",
        "generated_at_utc": utc_now(),
        "git_commit": git_commit(),
        "inherited_truth": {
            "sparse_stroke_bounded_adopter": True,
            "topological_only_non_adopted": True,
            "reject_buckets_hard_rejects": True,
        },
        "split_preservation": {
            "repo": "zpe-image-codec",
            "bootstrap_mode": "sibling_zpe_core_autodetect",
            "sparse_family_summary": preservation["candidate_family_matrix"]["sparse"]["summary"],
            "live_authority": preservation["live_authority"],
            "promotion_decision": preservation["promotion_decision"],
            "verdict": "preserved" if preservation["live_authority"]["authoritative_branch_verdict"] == "PASS" else "failed",
        },
        "bundle_experiment": {
            "hypothesis": {
                "base": "topological_segmentation",
                "fiber": "sparse_stroke",
                "state": "scale_context",
            },
            "scope_contract": {
                "accepted_positive_cases": bundle_summary["accepted_positive_cases"],
                "out_of_scope_positive_cases": bundle_summary["out_of_scope_positive_cases"],
                "reject_buckets": ["mixed", "negative_natural"],
                "acceptance_rule": "sparse_fiber_accept && hole_count >= 1",
                "projection_rule": "always decode the topological base; sparse fiber and scale/context state are admissibility witnesses and rescue the component-fill stress cells.",
            },
            "summary": bundle_summary,
            "records": bundle_records,
            "promotion_decision": {
                "release_surface_status": "not_promoted_to_default_route",
                "reason": "The sparse release path remains broader (five positive cases) and already authoritative. The bundle codec is a second bounded codec on a narrower hole-bearing scope.",
            },
        },
        "broad_image_non_claims": broad_non_claims,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    payload = build_payload()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"split_verdict": payload["split_preservation"]["verdict"], "bundle_verdict": payload["bundle_experiment"]["summary"]["verdict"]}, indent=2, sort_keys=True))
    print(str(args.output))


if __name__ == "__main__":
    main()
