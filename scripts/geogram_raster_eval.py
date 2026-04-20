#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Callable

import numpy as np

try:
    from scipy import ndimage
except Exception:  # pragma: no cover - dependency guard
    ndimage = None  # type: ignore[assignment]

SCRIPT_PATH = Path(__file__).resolve()
CODE_ROOT = SCRIPT_PATH.parents[1]
if str(SCRIPT_PATH.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PATH.parent))
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from benchmark_support import git_commit, median_latency_ms, utc_now, write_json_artifact
from zpe_multimodal.geogram.router import RouterConfig, route_raster_image, subset_boundary


CANVAS = 64
ARTIFACT_PREFIX = "geogram/"


def _blank(background: int = 255) -> np.ndarray:
    canvas = np.full((CANVAS, CANVAS, 3), background, dtype=np.uint8)
    return canvas


def _draw_line(image: np.ndarray, start: tuple[int, int], end: tuple[int, int], color: tuple[int, int, int], width: int = 1) -> None:
    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    error = dx + dy
    while True:
        _paint_disk(image, x0, y0, radius=max(0, width // 2), color=color)
        if x0 == x1 and y0 == y1:
            break
        doubled = 2 * error
        if doubled >= dy:
            error += dy
            x0 += sx
        if doubled <= dx:
            error += dx
            y0 += sy


def _paint_disk(image: np.ndarray, cx: int, cy: int, radius: int, color: tuple[int, int, int]) -> None:
    for y in range(cy - radius, cy + radius + 1):
        for x in range(cx - radius, cx + radius + 1):
            if 0 <= y < image.shape[0] and 0 <= x < image.shape[1] and (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                image[y, x] = np.array(color, dtype=np.uint8)


def _polyline(image: np.ndarray, points: list[tuple[int, int]], color: tuple[int, int, int] = (0, 0, 0), width: int = 1) -> None:
    for start, end in zip(points, points[1:]):
        _draw_line(image, start, end, color=color, width=width)


def _circle_outline(image: np.ndarray, center: tuple[int, int], radius: int, color: tuple[int, int, int] = (0, 0, 0), width: int = 1) -> None:
    points: list[tuple[int, int]] = []
    for idx in range(72):
        angle = (2.0 * np.pi * idx) / 72.0
        x = int(round(center[0] + radius * np.cos(angle)))
        y = int(round(center[1] + radius * np.sin(angle)))
        points.append((x, y))
    points.append(points[0])
    _polyline(image, points, color=color, width=width)


def _fill_circle(image: np.ndarray, center: tuple[int, int], radius: int, color: tuple[int, int, int]) -> None:
    _paint_disk(image, center[0], center[1], radius, color)


def _checkerboard(tile: int = 4) -> np.ndarray:
    ys, xs = np.indices((CANVAS, CANVAS))
    board = ((ys // tile + xs // tile) % 2) * 255
    image = np.repeat(board.astype(np.uint8)[:, :, None], 3, axis=2)
    return image


def _gradient() -> np.ndarray:
    image = np.zeros((CANVAS, CANVAS, 3), dtype=np.uint8)
    x_ramp = np.tile(np.linspace(20, 230, CANVAS, dtype=np.uint8), (CANVAS, 1))
    y_ramp = np.tile(np.linspace(240, 30, CANVAS, dtype=np.uint8), (CANVAS, 1)).T
    image[:, :, 0] = x_ramp
    image[:, :, 1] = y_ramp
    image[:, :, 2] = ((x_ramp.astype(np.uint16) + y_ramp.astype(np.uint16)) // 2).astype(np.uint8)
    return image


def _texture_blobs() -> np.ndarray:
    rng = np.random.default_rng(5)
    image = rng.integers(0, 255, size=(CANVAS, CANVAS, 3), dtype=np.uint8)
    if ndimage is not None:
        image = ndimage.gaussian_filter(image.astype(np.float32), sigma=(2.2, 2.2, 0))
        image = np.clip(image, 0, 255).astype(np.uint8)
    for _ in range(6):
        center = int(rng.integers(8, CANVAS - 8)), int(rng.integers(8, CANVAS - 8))
        radius = int(rng.integers(5, 12))
        color = tuple(int(value) for value in rng.integers(20, 235, size=3))
        _fill_circle(image, center, radius, color)
    return image


def _color_field_noise() -> np.ndarray:
    rng = np.random.default_rng(11)
    image = _blank(0)
    palette = [(220, 40, 40), (40, 180, 220), (230, 220, 60), (30, 30, 30)]
    stripe = CANVAS // len(palette)
    for idx, color in enumerate(palette):
        image[:, idx * stripe : (idx + 1) * stripe] = np.array(color, dtype=np.uint8)
    noise = rng.integers(-30, 31, size=image.shape)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return image


def _glyph_a() -> np.ndarray:
    image = _blank()
    _polyline(image, [(10, 55), (31, 8), (54, 55)], width=2)
    _polyline(image, [(18, 36), (45, 36)], width=2)
    return image


def _fork_tree() -> np.ndarray:
    image = _blank()
    _polyline(image, [(32, 56), (32, 28), (20, 14)], width=2)
    _polyline(image, [(32, 28), (46, 12)], width=2)
    _polyline(image, [(32, 40), (14, 30)], width=2)
    _polyline(image, [(32, 40), (50, 34)], width=2)
    return image


def _loop_spine() -> np.ndarray:
    image = _blank()
    _circle_outline(image, (24, 26), 14, width=2)
    _polyline(image, [(24, 40), (24, 58), (42, 58)], width=2)
    return image


def _maze_turns() -> np.ndarray:
    image = _blank()
    _polyline(image, [(8, 8), (8, 52), (24, 52), (24, 20), (40, 20), (40, 44), (56, 44)], width=2)
    _polyline(image, [(16, 16), (16, 32), (32, 32), (32, 12), (52, 12)], width=2)
    return image


def _serpentine() -> np.ndarray:
    image = _blank()
    _polyline(image, [(6, 20), (18, 8), (30, 20), (42, 8), (54, 20), (42, 32), (30, 44), (18, 32), (6, 44)], width=2)
    return image


def _filled_disk_with_spine() -> np.ndarray:
    image = _blank()
    _fill_circle(image, (24, 28), 14, (0, 0, 0))
    _polyline(image, [(24, 28), (52, 52)], width=2)
    return image


def _gradient_with_symbol() -> np.ndarray:
    image = _gradient()
    _polyline(image, [(10, 54), (30, 10), (54, 50)], color=(0, 0, 0), width=2)
    _polyline(image, [(18, 36), (44, 36)], color=(250, 250, 250), width=2)
    return image


def _stroke_over_texture() -> np.ndarray:
    image = _texture_blobs()
    _polyline(image, [(8, 50), (20, 14), (36, 46), (54, 18)], color=(255, 255, 255), width=2)
    return image


def build_cases() -> list[dict[str, object]]:
    return [
        {"name": "glyph_a", "bucket": "positive", "image": _glyph_a(), "notes": "letterform with strong figure-ground and thin strokes"},
        {"name": "fork_tree", "bucket": "positive", "image": _fork_tree(), "notes": "branch-heavy local skeleton"},
        {"name": "loop_spine", "bucket": "positive", "image": _loop_spine(), "notes": "closed loop plus stem"},
        {"name": "maze_turns", "bucket": "positive", "image": _maze_turns(), "notes": "orthogonal morphology-heavy routing"},
        {"name": "serpentine", "bucket": "positive", "image": _serpentine(), "notes": "diagonal turn-rich single figure"},
        {"name": "filled_disk_with_spine", "bucket": "mixed", "image": _filled_disk_with_spine(), "notes": "mixed filled mass plus stroke"},
        {"name": "gradient_with_symbol", "bucket": "mixed", "image": _gradient_with_symbol(), "notes": "stroke over non-binary background"},
        {"name": "stroke_over_texture", "bucket": "mixed", "image": _stroke_over_texture(), "notes": "figure mixed with textured nuisance"},
        {"name": "smooth_gradient", "bucket": "negative_natural", "image": _gradient(), "notes": "smooth natural-style color field"},
        {"name": "texture_blobs", "bucket": "negative_natural", "image": _texture_blobs(), "notes": "blob/texture image without sparse stroke figure"},
        {"name": "checkerboard_texture", "bucket": "negative_natural", "image": _checkerboard(), "notes": "high-frequency texture"},
        {"name": "color_field_noise", "bucket": "negative_natural", "image": _color_field_noise(), "notes": "multitone block field with nuisance noise"},
    ]


def _case_record(case: dict[str, object], router_config: RouterConfig) -> dict[str, object]:
    image = np.asarray(case["image"], dtype=np.uint8)
    decision = route_raster_image(image, router_config=router_config, include_perturbations=True)
    return {
        "name": case["name"],
        "bucket": case["bucket"],
        "notes": case["notes"],
        "input_shape": list(image.shape),
        "router": decision.to_dict(),
        "encode_latency_ms": median_latency_ms(lambda: route_raster_image(image, router_config=router_config, include_perturbations=False), runs=5),
    }


def _worst_cell_matrix(records: list[dict[str, object]]) -> dict[str, object]:
    matrix: dict[str, dict[str, dict[str, float | int]]] = {}
    for record in records:
        bucket = str(record["bucket"])
        perturbations = dict(record["router"]["perturbations"])  # type: ignore[index]
        bucket_matrix = matrix.setdefault(bucket, {})
        for name, payload in perturbations.items():
            cell = bucket_matrix.setdefault(name, {"worst_iou": 1.0, "worst_skeleton_f1": 1.0, "reject_count": 0, "count": 0})
            cell["worst_iou"] = min(float(cell["worst_iou"]), float(payload["iou"]))
            cell["worst_skeleton_f1"] = min(float(cell["worst_skeleton_f1"]), float(payload["skeleton_f1"]))
            cell["reject_count"] = int(cell["reject_count"]) + (0 if bool(payload["accept_p8"]) else 1)
            cell["count"] = int(cell["count"]) + 1
    return matrix


def _summary(records: list[dict[str, object]], router_config: RouterConfig) -> dict[str, object]:
    positives = [record for record in records if record["bucket"] == "positive"]
    negatives = [record for record in records if record["bucket"] != "positive"]
    positive_accepts = sum(1 for record in positives if bool(record["router"]["accept_p8"]))  # type: ignore[index]
    negative_rejects = sum(1 for record in negatives if not bool(record["router"]["accept_p8"]))  # type: ignore[index]
    positive_accept_rate = positive_accepts / max(len(positives), 1)
    negative_reject_rate = negative_rejects / max(len(negatives), 1)

    positive_worst_iou = min(float(record["router"]["perturbation_summary"].get("worst_iou", 0.0)) for record in positives)  # type: ignore[index]
    positive_worst_skeleton = min(float(record["router"]["perturbation_summary"].get("worst_skeleton_f1", 0.0)) for record in positives)  # type: ignore[index]
    kill_test_pass = all(not bool(record["router"]["accept_p8"]) for record in negatives)  # type: ignore[index]

    if (
        positive_accept_rate >= 0.6
        and negative_reject_rate >= 1.0
        and positive_worst_iou >= router_config.min_worst_perturb_iou
        and positive_worst_skeleton >= router_config.min_worst_perturb_skeleton_f1
    ):
        verdict = "credible_positive"
    elif positive_accept_rate < 0.4 or negative_reject_rate < 0.75:
        verdict = "credible_negative"
    else:
        verdict = "unresolved"

    authentic_negatives: list[dict[str, object]] = []
    for record in negatives:
        findings = list(record["router"]["authentic_negative_findings"])  # type: ignore[index]
        authentic_negatives.append({"name": record["name"], "bucket": record["bucket"], "findings": findings})

    return {
        "positive_accept_rate": round(positive_accept_rate, 6),
        "negative_reject_rate": round(negative_reject_rate, 6),
        "positive_case_count": len(positives),
        "negative_case_count": len(negatives),
        "positive_worst_perturb_iou": round(positive_worst_iou, 6),
        "positive_worst_perturb_skeleton_f1": round(positive_worst_skeleton, 6),
        "kill_test_pass": kill_test_pass,
        "verdict": verdict,
        "authentic_negatives": authentic_negatives,
    }


def _comparison_payload(records: list[dict[str, object]], router_config: RouterConfig) -> dict[str, object]:
    cases: list[dict[str, object]] = []
    for record in records:
        router = record["router"]  # type: ignore[assignment]
        cases.append(
            {
                "name": record["name"],
                "bucket": record["bucket"],
                "selected_route": router["selected_route"],
                "accept_p8": router["accept_p8"],
                "p8_20bit_bytes": router["p8_metrics"]["packed_20bit_bytes"],
                "quadtree_20bit_bytes": router["quadtree_metrics"]["packed_20bit_bytes"],
                "svg_raw_bytes": router["svg_metrics"]["raw_utf8_bytes"],
                "svg_gzip_bytes": router["svg_metrics"]["gzip_bytes"],
                "p8_iou": router["p8_metrics"]["iou"],
                "quadtree_iou": router["quadtree_metrics"]["iou"],
                "p8_skeleton_f1": router["p8_metrics"]["skeleton_f1"],
                "quadtree_skeleton_f1": router["quadtree_metrics"]["skeleton_f1"],
                "byte_gain_vs_quadtree": router["p8_metrics"]["byte_gain_vs_quadtree"],
                "svg_beats_p8": router["svg_metrics"]["svg_beats_p8"],
                "authentic_negative_findings": router["authentic_negative_findings"],
            }
        )
    grouped: dict[str, list[dict[str, object]]] = {}
    for case in cases:
        grouped.setdefault(str(case["bucket"]), []).append(case)
    aggregates: dict[str, object] = {}
    for bucket, bucket_cases in grouped.items():
        aggregates[bucket] = {
            "count": len(bucket_cases),
            "mean_p8_bytes": round(float(np.mean([case["p8_20bit_bytes"] for case in bucket_cases])), 6),
            "mean_quadtree_bytes": round(float(np.mean([case["quadtree_20bit_bytes"] for case in bucket_cases])), 6),
            "mean_svg_bytes": round(float(np.mean([case["svg_raw_bytes"] for case in bucket_cases])), 6),
            "mean_byte_gain_vs_quadtree": round(float(np.mean([case["byte_gain_vs_quadtree"] for case in bucket_cases])), 6),
            "worst_p8_iou": round(float(min(case["p8_iou"] for case in bucket_cases)), 6),
            "worst_p8_skeleton_f1": round(float(min(case["p8_skeleton_f1"] for case in bucket_cases)), 6),
        }
    return {
        "generated_at_utc": utc_now(),
        "git_commit": git_commit(),
        "comparison_scope": "bounded geogram P8 candidate against live quadtree authority and shared SVG transport for locally generated raster fixtures",
        "subset_boundary": subset_boundary(router_config),
        "cases": cases,
        "aggregates": aggregates,
        "notes": {
            "sovereign_doctrine": "P8 primitive-basis truth is sovereign; shared SVG transport is reported only as a size baseline, not as success.",
            "authority": "Quadtree in core/imc.py:add_image remains the live authority path unless the bounded router evidence says otherwise.",
        },
    }


def main() -> None:
    router_config = RouterConfig()
    cases = build_cases()
    records = [_case_record(case, router_config) for case in cases]
    summary = _summary(records, router_config)
    subset_payload = {
        "generated_at_utc": utc_now(),
        "git_commit": git_commit(),
        "scope": "bounded stroke-dominant geogram router evaluation with explicit local kill tests",
        "subset_boundary": subset_boundary(router_config),
        "summary": summary,
        "worst_cell_matrix": _worst_cell_matrix(records),
        "cases": records,
        "notes": {
            "authoritative_path": "Live image authority is the enhanced quadtree path in core/imc.py:add_image.",
            "claim_boundary": "No broad natural-image success claim is made. Acceptance is limited to locally generated stroke-dominant morphology-heavy figures.",
        },
    }
    comparison_payload = _comparison_payload(records, router_config)

    subset_artifact = write_json_artifact(f"{ARTIFACT_PREFIX}geogram_subset_router_eval.json", subset_payload)
    comparison_artifact = write_json_artifact(f"{ARTIFACT_PREFIX}geogram_p8_vs_quadtree_vs_svg.json", comparison_payload)
    print(
        json.dumps(
            {
                "subset_artifact": str(subset_artifact),
                "comparison_artifact": str(comparison_artifact),
                "summary": summary,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
