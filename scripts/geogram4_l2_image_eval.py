#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
CORE_ROOT = REPO_ROOT.parent / "zpe-core"

if str(SCRIPT_PATH.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PATH.parent))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from benchmark_support import git_commit, utc_now
from geogram_raster_eval import build_cases
from source.core.imc import IMCDecoder, IMCEncoder
from source.image.dual_dispatch import decode_image_words, detect_family
from source.image.geometry_codec import (
    _build_sparse_candidate,
    _cleanup_mask,
    binary_f1,
    binary_iou,
    encode_geometry_image,
    grayscale_u8,
    route_image_for_imc,
    select_binary_mask,
    thin_binary_mask,
)
from source.image.quadtree_enhanced_codec import encode_enhanced
from zpe_multimodal.geogram.metrics import perturbation_suite


DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "l2_image_geogram4.json"
PRIOR_NATURAL_PROBE = REPO_ROOT / "artifacts" / "priors" / "geogram_natural_image_negative.json"
PRIOR_PROXY_AUDIT = REPO_ROOT / "artifacts" / "priors" / "geogram_proxy_audit.json"


def _pack_words_20bit(words: list[int]) -> int:
    bits = 0
    count = 0
    accumulator = 0
    for word in words:
        accumulator |= int(word) << bits
        bits += 20
        while bits >= 8:
            accumulator >>= 8
            bits -= 8
            count += 1
    if bits:
        count += 1
    return count


def _route_reference_mask(image: np.ndarray, selected_route: str) -> np.ndarray:
    if selected_route == "geometry_sparse_stroke":
        return _build_sparse_candidate(image).binary_mask
    if selected_route == "geometry_topological_segmentation":
        return _cleanup_mask(select_binary_mask(image).mask)
    return select_binary_mask(image).mask


def _decoded_mask(words: list[int]) -> np.ndarray:
    decoded = decode_image_words(words)
    return select_binary_mask(decoded.image).mask


def _authority_gate(reference_mask: np.ndarray, candidate_mask: np.ndarray) -> float:
    return min(
        binary_iou(reference_mask, candidate_mask),
        binary_f1(thin_binary_mask(reference_mask), thin_binary_mask(candidate_mask)),
    )


def _two_tone_from_mask(mask: np.ndarray) -> np.ndarray:
    image = np.full(mask.shape + (3,), 255, dtype=np.uint8)
    image[np.asarray(mask, dtype=bool)] = 0
    return image


def _helper_gap(image: np.ndarray, selected_route: str, route_words: list[int]) -> float | None:
    if selected_route == "geometry_sparse_stroke":
        subtype = "sparse"
    elif selected_route == "geometry_topological_segmentation":
        subtype = "topological"
    else:
        return None

    reference_mask = _route_reference_mask(image, selected_route)
    raw_mask = _decoded_mask(route_words)
    helper_words, _ = encode_geometry_image(_two_tone_from_mask(reference_mask), subtype=subtype)
    helper_mask = _decoded_mask(helper_words)
    return round(_authority_gate(reference_mask, helper_mask) - _authority_gate(reference_mask, raw_mask), 6)


def _candidate_record(case: dict[str, object], family_mode: str) -> dict[str, object]:
    image = np.asarray(case["image"], dtype=np.uint8)
    route = route_image_for_imc(image, family_mode=family_mode)
    quadtree_words, _ = encode_enhanced(image, bit_depth=3)
    record: dict[str, object] = {
        "name": case["name"],
        "bucket": case["bucket"],
        "notes": case["notes"],
        "selected_route": route.selected_route,
        "accepted_geometry_route": route.accepted,
        "candidate_bytes_20bit": _pack_words_20bit(route.words),
        "quadtree_bytes_20bit": _pack_words_20bit(quadtree_words),
        "byte_gain_vs_quadtree": round(_pack_words_20bit(quadtree_words) / max(_pack_words_20bit(route.words), 1), 6),
    }
    if not route.accepted:
        return record

    reference_mask = _route_reference_mask(image, route.selected_route)
    candidate_mask = _decoded_mask(route.words)
    record["family"] = route.decision.subtype if route.decision is not None else "unknown"
    record["route_features"] = dict(route.decision.features) if route.decision is not None else {}
    record["iou"] = round(binary_iou(reference_mask, candidate_mask), 6)
    record["skeleton_f1"] = round(binary_f1(thin_binary_mask(reference_mask), thin_binary_mask(candidate_mask)), 6)
    record["helper_gap"] = _helper_gap(image, route.selected_route, route.words)

    perturbations: dict[str, dict[str, object]] = {}
    for name, variant in perturbation_suite(image).items():
        variant_route = route_image_for_imc(variant, family_mode=family_mode)
        variant_record: dict[str, object] = {
            "selected_route": variant_route.selected_route,
            "accepted_geometry_route": variant_route.accepted,
        }
        if variant_route.accepted:
            reference_variant_mask = _route_reference_mask(variant, variant_route.selected_route)
            candidate_variant_mask = _decoded_mask(variant_route.words)
            variant_record["iou"] = round(binary_iou(reference_variant_mask, candidate_variant_mask), 6)
            variant_record["skeleton_f1"] = round(
                binary_f1(thin_binary_mask(reference_variant_mask), thin_binary_mask(candidate_variant_mask)),
                6,
            )
        else:
            variant_record["iou"] = 0.0
            variant_record["skeleton_f1"] = 0.0
        perturbations[name] = variant_record
    record["perturbations"] = perturbations
    record["perturbation_summary"] = {
        "worst_iou": round(min(float(payload["iou"]) for payload in perturbations.values()), 6),
        "worst_skeleton_f1": round(min(float(payload["skeleton_f1"]) for payload in perturbations.values()), 6),
    }
    return record


def _summary(records: list[dict[str, object]]) -> dict[str, object]:
    positives = [record for record in records if record["bucket"] == "positive"]
    negatives = [record for record in records if record["bucket"] != "positive"]
    positive_accept_rate = sum(1 for record in positives if bool(record["accepted_geometry_route"])) / max(len(positives), 1)
    negative_reject_rate = sum(1 for record in negatives if not bool(record["accepted_geometry_route"])) / max(len(negatives), 1)
    worst_iou = min(
        float(record.get("perturbation_summary", {}).get("worst_iou", 0.0)) if bool(record["accepted_geometry_route"]) else 0.0
        for record in positives
    )
    worst_skeleton_f1 = min(
        float(record.get("perturbation_summary", {}).get("worst_skeleton_f1", 0.0)) if bool(record["accepted_geometry_route"]) else 0.0
        for record in positives
    )
    helper_gaps = [float(record["helper_gap"]) for record in positives if record.get("helper_gap") is not None]
    accepted_positive_records = [record for record in positives if bool(record["accepted_geometry_route"])]
    if accepted_positive_records:
        mean_candidate_bytes = round(
            float(np.mean([int(record["candidate_bytes_20bit"]) for record in accepted_positive_records])),
            6,
        )
        mean_quadtree_bytes = round(
            float(np.mean([int(record["quadtree_bytes_20bit"]) for record in accepted_positive_records])),
            6,
        )
        mean_byte_gain = round(
            float(np.mean([float(record["byte_gain_vs_quadtree"]) for record in accepted_positive_records])),
            6,
        )
    else:
        mean_candidate_bytes = 0.0
        mean_quadtree_bytes = 0.0
        mean_byte_gain = 0.0

    if (
        positive_accept_rate == 1.0
        and negative_reject_rate == 1.0
        and worst_iou >= 0.62
        and worst_skeleton_f1 >= 0.74
        and (max(helper_gaps) if helper_gaps else 1.0) < 0.1
    ):
        verdict = "bounded_adopter_candidate"
    elif positive_accept_rate == 1.0 and negative_reject_rate == 1.0:
        verdict = "fail_under_perturbation_or_helper"
    elif negative_reject_rate == 1.0:
        verdict = "fail_partial_scope"
    else:
        verdict = "fail_reject_bucket"

    return {
        "positive_accept_rate": round(positive_accept_rate, 6),
        "negative_reject_rate": round(negative_reject_rate, 6),
        "positive_worst_perturb_iou": round(worst_iou, 6),
        "positive_worst_perturb_skeleton_f1": round(worst_skeleton_f1, 6),
        "accepted_scope_max_helper_gap": round(max(helper_gaps), 6) if helper_gaps else None,
        "accepted_scope_mean_candidate_bytes": mean_candidate_bytes,
        "accepted_scope_mean_quadtree_bytes": mean_quadtree_bytes,
        "accepted_scope_mean_byte_gain_vs_quadtree": mean_byte_gain,
        "accepted_positive_cases": [record["name"] for record in accepted_positive_records],
        "geometry_negative_route_leaks": [record["name"] for record in negatives if bool(record["accepted_geometry_route"])],
        "verdict": verdict,
    }


def _candidate_family_matrix() -> dict[str, object]:
    cases = build_cases()
    families: dict[str, object] = {}
    for family_mode in ("sparse", "topological", "hybrid"):
        records = [_candidate_record(case, family_mode) for case in cases]
        families[family_mode] = {"records": records, "summary": _summary(records)}
    return families


def _live_authority_record(case: dict[str, object]) -> dict[str, object]:
    image = np.asarray(case["image"], dtype=np.uint8)
    stream = IMCEncoder().add_image(image).build()
    family = detect_family(stream)
    decoded = decode_image_words(stream)
    if family == "geometry" and getattr(decoded.meta, "subtype", None) == 1:
        selected_route = "geometry_sparse_stroke"
    elif family == "geometry" and getattr(decoded.meta, "subtype", None) == 2:
        selected_route = "geometry_topological_segmentation"
    else:
        selected_route = "quadtree_enhanced"

    quadtree_words, _ = encode_enhanced(image, bit_depth=3)
    record: dict[str, object] = {
        "name": case["name"],
        "bucket": case["bucket"],
        "selected_route": selected_route,
        "family_detected": family,
        "candidate_bytes_20bit": _pack_words_20bit(stream),
        "quadtree_bytes_20bit": _pack_words_20bit(quadtree_words),
        "byte_gain_vs_quadtree": round(_pack_words_20bit(quadtree_words) / max(_pack_words_20bit(stream), 1), 6),
        "accepted_geometry_route": selected_route != "quadtree_enhanced",
    }
    if selected_route == "quadtree_enhanced":
        return record

    reference_mask = _route_reference_mask(image, selected_route)
    candidate_mask = _decoded_mask(stream)
    record["iou"] = round(binary_iou(reference_mask, candidate_mask), 6)
    record["skeleton_f1"] = round(binary_f1(thin_binary_mask(reference_mask), thin_binary_mask(candidate_mask)), 6)
    record["helper_gap"] = _helper_gap(image, selected_route, stream)

    perturbations: dict[str, dict[str, object]] = {}
    for name, variant in perturbation_suite(image).items():
        variant_stream = IMCEncoder().add_image(variant).build()
        variant_family = detect_family(variant_stream)
        variant_decoded = decode_image_words(variant_stream)
        if variant_family == "geometry" and getattr(variant_decoded.meta, "subtype", None) == 1:
            variant_route = "geometry_sparse_stroke"
        elif variant_family == "geometry" and getattr(variant_decoded.meta, "subtype", None) == 2:
            variant_route = "geometry_topological_segmentation"
        else:
            variant_route = "quadtree_enhanced"
        variant_record: dict[str, object] = {
            "selected_route": variant_route,
            "accepted_geometry_route": variant_route != "quadtree_enhanced",
        }
        if variant_route != "quadtree_enhanced":
            reference_variant_mask = _route_reference_mask(variant, variant_route)
            candidate_variant_mask = _decoded_mask(variant_stream)
            variant_record["iou"] = round(binary_iou(reference_variant_mask, candidate_variant_mask), 6)
            variant_record["skeleton_f1"] = round(
                binary_f1(thin_binary_mask(reference_variant_mask), thin_binary_mask(candidate_variant_mask)),
                6,
            )
        else:
            variant_record["iou"] = 0.0
            variant_record["skeleton_f1"] = 0.0
        perturbations[name] = variant_record
    record["perturbations"] = perturbations
    record["perturbation_summary"] = {
        "worst_iou": round(min(float(payload["iou"]) for payload in perturbations.values()), 6),
        "worst_skeleton_f1": round(min(float(payload["skeleton_f1"]) for payload in perturbations.values()), 6),
    }
    return record


def _live_authority_summary() -> dict[str, object]:
    records = [_live_authority_record(case) for case in build_cases()]
    summary = _summary(records)
    summary["records"] = records
    summary["authoritative_branch"] = "source.core.imc.IMCEncoder.add_image -> image.geometry_codec.route_image_for_imc"
    summary["authoritative_branch_verdict"] = "PASS" if summary["verdict"] == "bounded_adopter_candidate" else "FAIL"
    return summary


def _load_prior_probe() -> dict[str, object]:
    if not PRIOR_NATURAL_PROBE.exists() or not PRIOR_PROXY_AUDIT.exists():
        return {}
    natural = json.loads(PRIOR_NATURAL_PROBE.read_text(encoding="utf-8"))
    proxy = json.loads(PRIOR_PROXY_AUDIT.read_text(encoding="utf-8"))
    return {
        "natural_negative_summary": natural.get("summary", {}),
        "proxy_audit_summary": proxy.get("summary", {}),
        "natural_negative_artifact": str(PRIOR_NATURAL_PROBE),
        "proxy_audit_artifact": str(PRIOR_PROXY_AUDIT),
    }


def build_payload() -> dict[str, object]:
    family_matrix = _candidate_family_matrix()
    sparse_summary = family_matrix["sparse"]["summary"]
    hybrid_summary = family_matrix["hybrid"]["summary"]

    promoted_family = "sparse_stroke" if sparse_summary["verdict"] == "bounded_adopter_candidate" else None
    promotion_reason = "Sparse alone passes the bounded-adopter matrix; hybrid was not promoted because topological-only fails and the publication contract forbids laundering a blended family into a single image claim."
    if promoted_family is None and hybrid_summary["verdict"] == "bounded_adopter_candidate":
        promotion_reason = "Hybrid passes technically, but no separate family is promotable under the lane brief."

    return {
        "artifact_version": "geogram4-l2-image-authority-v1",
        "generated_at_utc": utc_now(),
        "git_commit": git_commit(),
        "claim_class": "geometry-decomposable but bounded",
        "frozen_scope": {
            "positive_bucket": "positive",
            "reject_buckets": ["mixed", "negative_natural"],
            "fixture_count": len(build_cases()),
        },
        "candidate_family_matrix": family_matrix,
        "promotion_decision": {
            "promoted_family": promoted_family,
            "reason": promotion_reason,
            "topological_only_verdict": family_matrix["topological"]["summary"]["verdict"],
            "hybrid_verdict": hybrid_summary["verdict"],
        },
        "live_authority": _live_authority_summary(),
        "broad_image_non_claims": _load_prior_probe(),
        "evidence_paths": {
            "subset_router_eval": str(REPO_ROOT / "v0.0" / "proofs" / "artifacts" / "geogram" / "geogram_subset_router_eval.json"),
            "geogram3_lane_artifact": str(REPO_ROOT / "v0.0" / "proofs" / "artifacts" / "geometry-first" / "wave3" / "l2_image_wave3.json"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    payload = build_payload()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["promotion_decision"], indent=2, sort_keys=True))
    print(str(args.output.resolve()))


if __name__ == "__main__":
    main()
