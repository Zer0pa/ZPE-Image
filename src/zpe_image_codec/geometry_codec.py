from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

try:
    from scipy import ndimage
except Exception:  # pragma: no cover - dependency guard
    ndimage = None  # type: ignore[assignment]

from .codec_constants import DEFAULT_VERSION, Mode, packed_word_bytes
from .enhanced_codec import decode_enhanced, encode_enhanced


GEOMETRY_FAMILY_MASK = 0x0C00
GEOMETRY_FAMILY_VALUE = 0x0C00

VERSION_META = 0
VERSION_SPARSE = 1
VERSION_TOPOLOGICAL = 2

META_WIDTH_HI = 0
META_WIDTH_LO = 1
META_HEIGHT_HI = 2
META_HEIGHT_LO = 3
META_FLAGS = 4
META_FG_GRAY = 5
META_BG_GRAY = 6
META_END = 7

SPARSE_MOVE_X_HI = 0
SPARSE_MOVE_X_LO = 1
SPARSE_MOVE_Y_HI = 2
SPARSE_MOVE_Y_LO = 3
SPARSE_SET_WIDTH = 4
SPARSE_DRAW = 5
SPARSE_END_PATH = 6
SPARSE_END_STREAM = 7

TOPO_ROW_HI = 0
TOPO_ROW_LO = 1
TOPO_START_HI = 2
TOPO_START_LO = 3
TOPO_LEN_HI = 4
TOPO_LEN_LO = 5
TOPO_NEXT_SPAN = 6
TOPO_END_STREAM = 7

SUBTYPE_SPARSE = 1
SUBTYPE_TOPOLOGICAL = 2

FLAG_FOREGROUND_LIGHT = 0x04

_DIRS: tuple[tuple[int, int], ...] = (
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
)
_DIR_TO_INDEX = {direction: index for index, direction in enumerate(_DIRS)}
_NEIGHBOR_OFFSETS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


@dataclass(frozen=True)
class BinarySelection:
    mask: np.ndarray
    threshold: int
    foreground_mode: str
    foreground_fraction: float
    contrast: float
    foreground_mean: float
    background_mean: float
    binary_purity: float
    gray_entropy: float


@dataclass(frozen=True)
class ConnectedComponentSummary:
    count: int
    area_mean: float
    area_median: float
    bbox_fill_mean: float
    bbox_fill_max: float
    tiny_fraction: float


@dataclass(frozen=True)
class GeometryMeta:
    width: int
    height: int
    subtype: int
    foreground_gray: int
    background_gray: int
    foreground_mode: str


@dataclass(frozen=True)
class GeometryDecision:
    subtype: str
    accept: bool
    features: dict[str, float]
    rejection_reasons: tuple[str, ...]


@dataclass(frozen=True)
class GeometryRoute:
    selected_route: str
    family_mode: str
    words: list[int]
    accepted: bool
    decision: GeometryDecision | None


@dataclass(frozen=True)
class SparseCandidate:
    selection: BinarySelection
    component_summary: ConnectedComponentSummary
    binary_mask: np.ndarray
    skeleton: np.ndarray
    polylines: list[list[tuple[int, int]]]
    polyline_widths: list[float]
    reconstructed_mask: np.ndarray
    feature_vector: dict[str, float]


def route_image(
    image: np.ndarray,
    *,
    bit_depth: int = 3,
    family_mode: str = "winner",
) -> GeometryRoute:
    arr = _rgb_u8(image)
    mode = "sparse" if family_mode == "winner" else family_mode
    if mode not in {"sparse", "topological", "hybrid"}:
        raise ValueError(f"unsupported image family mode: {family_mode}")

    sparse_decision, sparse_words = _try_sparse_route(arr)
    topological_decision, topological_words = _try_topological_route(arr)

    if mode == "sparse" and sparse_decision.accept:
        return GeometryRoute("geometry_sparse_stroke", mode, sparse_words, True, sparse_decision)
    if mode == "topological" and topological_decision.accept:
        return GeometryRoute("geometry_topological_segmentation", mode, topological_words, True, topological_decision)
    if mode == "hybrid":
        if topological_decision.accept and _topological_priority(topological_decision.features):
            return GeometryRoute("geometry_topological_segmentation", mode, topological_words, True, topological_decision)
        if sparse_decision.accept:
            return GeometryRoute("geometry_sparse_stroke", mode, sparse_words, True, sparse_decision)

    enhanced_words, _ = encode_enhanced(arr, bit_depth=bit_depth)
    return GeometryRoute("quadtree_enhanced", mode, enhanced_words, False, None)


def decode_geometry_image(words: Sequence[int]) -> tuple[np.ndarray, GeometryMeta]:
    payloads = [_payload_from_word(word) for word in words if _is_geometry_word(word)]
    if not payloads:
        raise ValueError("no geometry image payloads found")

    meta = _decode_meta(payloads)
    if meta.subtype == SUBTYPE_SPARSE:
        return _decode_sparse_payloads(payloads, meta), meta
    if meta.subtype == SUBTYPE_TOPOLOGICAL:
        return _decode_topological_payloads(payloads, meta), meta
    raise ValueError(f"unsupported geometry subtype in metadata: {meta.subtype}")


def detect_geometry_subtype(words: Sequence[int]) -> str:
    _, meta = decode_geometry_image(words)
    if meta.subtype == SUBTYPE_SPARSE:
        return "sparse"
    if meta.subtype == SUBTYPE_TOPOLOGICAL:
        return "topological"
    return "unknown"


def grayscale_u8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        return arr.astype(np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("image must have shape (H, W) or (H, W, 3)")
    gray = np.rint(0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2])
    return np.clip(gray, 0, 255).astype(np.uint8)


def shannon_entropy_u8(gray: np.ndarray) -> float:
    hist = np.bincount(gray.reshape(-1), minlength=256).astype(np.float64)
    total = float(hist.sum())
    if total <= 0.0:
        return 0.0
    probs = hist[hist > 0.0] / total
    return float(-np.sum(probs * np.log2(probs)))


def binary_purity(gray: np.ndarray) -> float:
    distance = np.minimum(gray.astype(np.float64), 255.0 - gray.astype(np.float64))
    normalized = np.clip(distance / 127.5, 0.0, 1.0)
    return float(1.0 - normalized.mean())


def otsu_threshold(gray: np.ndarray) -> int:
    hist = np.bincount(gray.reshape(-1), minlength=256).astype(np.float64)
    total = hist.sum()
    if total <= 0.0:
        return 127
    cumulative = np.cumsum(hist)
    cumulative_mean = np.cumsum(hist * np.arange(256, dtype=np.float64))
    global_mean = cumulative_mean[-1]
    denominator = cumulative * (total - cumulative)
    numerator = (global_mean * cumulative - cumulative_mean) ** 2
    score = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0.0)
    return int(np.argmax(score))


def select_binary_mask(image: np.ndarray) -> BinarySelection:
    gray = grayscale_u8(image)
    threshold = otsu_threshold(gray)
    purity = binary_purity(gray)
    entropy = shannon_entropy_u8(gray)

    candidates: list[BinarySelection] = []
    for foreground_mode, mask in (("dark", gray <= threshold), ("light", gray > threshold)):
        foreground_fraction = float(mask.mean())
        if foreground_fraction <= 0.0 or foreground_fraction >= 1.0:
            continue
        fg = gray[mask]
        bg = gray[~mask]
        if fg.size == 0 or bg.size == 0:
            continue
        fg_mean = float(fg.mean())
        bg_mean = float(bg.mean())
        contrast = abs(bg_mean - fg_mean)
        candidates.append(
            BinarySelection(
                mask=mask.astype(bool),
                threshold=threshold,
                foreground_mode=foreground_mode,
                foreground_fraction=foreground_fraction,
                contrast=contrast,
                foreground_mean=fg_mean,
                background_mean=bg_mean,
                binary_purity=purity,
                gray_entropy=entropy,
            )
        )

    if not candidates:
        fallback = gray <= threshold
        return BinarySelection(
            mask=fallback.astype(bool),
            threshold=threshold,
            foreground_mode="dark",
            foreground_fraction=float(fallback.mean()),
            contrast=0.0,
            foreground_mean=float(gray[fallback].mean()) if np.any(fallback) else 0.0,
            background_mean=float(gray[~fallback].mean()) if np.any(~fallback) else 0.0,
            binary_purity=purity,
            gray_entropy=entropy,
        )

    def score(entry: BinarySelection) -> float:
        occupancy_bias = abs(entry.foreground_fraction - 0.22)
        return entry.contrast - 110.0 * occupancy_bias

    return max(candidates, key=score)


def connected_components(mask: np.ndarray) -> ConnectedComponentSummary:
    binary = np.asarray(mask, dtype=bool)
    if not np.any(binary):
        return ConnectedComponentSummary(0, 0.0, 0.0, 0.0, 0.0, 0.0)

    if ndimage is not None:
        structure = np.ones((3, 3), dtype=np.int8)
        labels, count = ndimage.label(binary, structure=structure)
    else:  # pragma: no cover - exercised only without scipy
        labels, count = _fallback_label(binary)

    areas: list[int] = []
    fill_ratios: list[float] = []
    tiny = 0
    for label in range(1, int(count) + 1):
        rows, cols = np.where(labels == label)
        if rows.size == 0:
            continue
        area = int(rows.size)
        if area <= 4:
            tiny += 1
        bbox_area = int((rows.max() - rows.min() + 1) * (cols.max() - cols.min() + 1))
        fill = float(area / max(bbox_area, 1))
        areas.append(area)
        fill_ratios.append(fill)

    if not areas:
        return ConnectedComponentSummary(0, 0.0, 0.0, 0.0, 0.0, 0.0)

    return ConnectedComponentSummary(
        count=len(areas),
        area_mean=float(np.mean(areas)),
        area_median=float(np.median(areas)),
        bbox_fill_mean=float(np.mean(fill_ratios)),
        bbox_fill_max=float(np.max(fill_ratios)),
        tiny_fraction=float(tiny / len(areas)),
    )


def binary_iou(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=bool)
    cand = np.asarray(candidate, dtype=bool)
    union = np.logical_or(ref, cand).sum()
    if union == 0:
        return 1.0
    intersection = np.logical_and(ref, cand).sum()
    return float(intersection / union)


def binary_f1(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=bool)
    cand = np.asarray(candidate, dtype=bool)
    tp = float(np.logical_and(ref, cand).sum())
    fp = float(np.logical_and(~ref, cand).sum())
    fn = float(np.logical_and(ref, ~cand).sum())
    denom = 2.0 * tp + fp + fn
    if denom == 0.0:
        return 1.0
    return float((2.0 * tp) / denom)


def thin_binary_mask(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask, dtype=np.uint8).copy()
    if binary.ndim != 2:
        raise ValueError("thin_binary_mask expects a 2D mask")
    if binary.shape[0] < 3 or binary.shape[1] < 3:
        return binary.astype(bool)

    changed = True
    while changed:
        changed = False
        for phase in (0, 1):
            to_delete: list[tuple[int, int]] = []
            for row in range(1, binary.shape[0] - 1):
                for col in range(1, binary.shape[1] - 1):
                    if binary[row, col] == 0:
                        continue
                    neighbors = _neighbors(binary, row, col)
                    count = int(sum(neighbors))
                    if count < 2 or count > 6:
                        continue
                    if _transitions(neighbors) != 1:
                        continue
                    p2, _p3, p4, _p5, p6, _p7, p8, _p9 = neighbors
                    if phase == 0:
                        if p2 * p4 * p6 != 0 or p4 * p6 * p8 != 0:
                            continue
                    else:
                        if p2 * p4 * p8 != 0 or p2 * p6 * p8 != 0:
                            continue
                    to_delete.append((row, col))
            if to_delete:
                changed = True
                for row, col in to_delete:
                    binary[row, col] = 0
    return binary.astype(bool)


def prune_spurs(skeleton: np.ndarray, rounds: int = 1) -> np.ndarray:
    current = np.asarray(skeleton, dtype=bool).copy()
    for _ in range(max(0, rounds)):
        degrees = skeleton_degrees(current)
        removable = np.logical_and(current, degrees == 1)
        if not np.any(removable):
            break
        current[removable] = False
    return current


def skeleton_degrees(skeleton: np.ndarray) -> np.ndarray:
    binary = np.asarray(skeleton, dtype=bool)
    degrees = np.zeros(binary.shape, dtype=np.uint8)
    for dy, dx in _NEIGHBOR_OFFSETS:
        shifted = np.zeros_like(binary, dtype=bool)
        src_y0 = max(0, -dy)
        src_y1 = binary.shape[0] - max(0, dy)
        src_x0 = max(0, -dx)
        src_x1 = binary.shape[1] - max(0, dx)
        dst_y0 = max(0, dy)
        dst_y1 = dst_y0 + (src_y1 - src_y0)
        dst_x0 = max(0, dx)
        dst_x1 = dst_x0 + (src_x1 - src_x0)
        shifted[dst_y0:dst_y1, dst_x0:dst_x1] = binary[src_y0:src_y1, src_x0:src_x1]
        degrees += shifted.astype(np.uint8)
    degrees[~binary] = 0
    return degrees


def trace_skeleton_to_polylines(skeleton: np.ndarray) -> list[list[tuple[int, int]]]:
    binary = np.asarray(skeleton, dtype=bool)
    if not np.any(binary):
        return []

    pixels = {tuple(map(int, point)) for point in np.argwhere(binary)}
    neighbor_map: dict[tuple[int, int], list[tuple[int, int]]] = {
        pixel: [neighbor for neighbor in _neighbor_points(pixel) if neighbor in pixels]
        for pixel in pixels
    }
    nodes = {pixel for pixel, neighbors in neighbor_map.items() if len(neighbors) != 2}
    visited_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    polylines: list[list[tuple[int, int]]] = []

    for node in sorted(nodes):
        for neighbor in neighbor_map[node]:
            edge = _ordered_edge(node, neighbor)
            if edge in visited_edges:
                continue
            polyline = _walk_path(node, neighbor, neighbor_map, nodes, visited_edges)
            if len(polyline) >= 2:
                polylines.append(polyline)

    for start in sorted(pixels):
        for neighbor in neighbor_map[start]:
            edge = _ordered_edge(start, neighbor)
            if edge in visited_edges:
                continue
            polyline = _walk_cycle(start, neighbor, neighbor_map, visited_edges)
            if len(polyline) >= 2:
                polylines.append(polyline)

    normalized: list[list[tuple[int, int]]] = []
    for polyline in polylines:
        deduped: list[tuple[int, int]] = []
        for point in polyline:
            if not deduped or deduped[-1] != point:
                deduped.append(point)
        if len(deduped) >= 2:
            normalized.append(deduped)
    normalized.sort(key=lambda polyline: (-len(polyline), polyline[0][1], polyline[0][0]))
    return normalized


def estimate_polyline_widths(polylines: Sequence[list[tuple[int, int]]], distance: np.ndarray) -> list[float]:
    widths: list[float] = []
    for polyline in polylines:
        samples: list[float] = []
        for x, y in polyline:
            if 0 <= y < distance.shape[0] and 0 <= x < distance.shape[1]:
                local = float(distance[y, x])
                if local > 0.0:
                    samples.append(local)
        if not samples:
            widths.append(1.0)
            continue
        widths.append(max(1.0, float(np.median(samples) * 2.0)))
    return widths


def rasterize_polylines(
    polylines: Sequence[list[tuple[int, int]]],
    shape: tuple[int, int],
    widths: Sequence[float] | None = None,
) -> np.ndarray:
    canvas = np.zeros(shape, dtype=bool)
    width_values = list(widths) if widths is not None else []
    for index, polyline in enumerate(polylines):
        if len(polyline) < 2:
            continue
        radius = 0
        if index < len(width_values):
            radius = max(0, int(np.ceil((float(width_values[index]) - 1.0) / 2.0)))
        for start, end in zip(polyline, polyline[1:]):
            _draw_bool_line(canvas, start, end, radius=radius)
    return canvas


def count_holes(mask: np.ndarray) -> int:
    binary = np.asarray(mask, dtype=bool)
    if not np.any(binary):
        return 0
    if ndimage is not None:
        inverted = np.logical_not(binary)
        labels, count = ndimage.label(inverted, structure=np.ones((3, 3), dtype=np.uint8))
    else:  # pragma: no cover - exercised only without scipy
        labels, count = _fallback_label(np.logical_not(binary))
    border_labels: set[int] = set()
    border_labels.update(int(value) for value in labels[0, :] if value > 0)
    border_labels.update(int(value) for value in labels[-1, :] if value > 0)
    border_labels.update(int(value) for value in labels[:, 0] if value > 0)
    border_labels.update(int(value) for value in labels[:, -1] if value > 0)
    return len({label for label in range(1, int(count) + 1) if label not in border_labels})


def _try_sparse_route(image: np.ndarray) -> tuple[GeometryDecision, list[int]]:
    candidate = _build_sparse_candidate(image)
    words, _ = _encode_sparse_candidate(candidate)
    reasons: list[str] = []

    enhanced_words, _ = encode_enhanced(image, bit_depth=3)
    enhanced_image, _ = decode_enhanced(enhanced_words)
    enhanced_mask = select_binary_mask(enhanced_image).mask
    byte_gain = packed_word_bytes(enhanced_words) / max(packed_word_bytes(words), 1)
    p8_iou = binary_iou(candidate.binary_mask, candidate.reconstructed_mask)
    enhanced_iou = binary_iou(candidate.binary_mask, enhanced_mask)
    p8_skeleton_f1 = binary_f1(candidate.skeleton, thin_binary_mask(candidate.reconstructed_mask))

    features = dict(candidate.feature_vector)
    features["byte_gain_vs_quadtree"] = round(float(byte_gain), 6)
    features["p8_iou"] = round(float(p8_iou), 6)
    features["p8_skeleton_f1"] = round(float(p8_skeleton_f1), 6)
    features["quadtree_iou"] = round(float(enhanced_iou), 6)

    _reject(features["binary_purity"] < 0.84, reasons, "binary_purity_too_low")
    _reject(features["gray_entropy"] > 4.6, reasons, "gray_entropy_too_high")
    _reject(features["contrast"] < 72.0, reasons, "contrast_too_low")
    _reject(
        features["foreground_fraction"] < 0.01 or features["foreground_fraction"] > 0.48,
        reasons,
        "foreground_fraction_out_of_bounds",
    )
    _reject(features["skeleton_foreground_ratio"] < 0.13, reasons, "skeleton_ratio_too_low")
    _reject(features["component_bbox_fill_mean"] > 0.46, reasons, "component_fill_too_high")
    _reject(features["component_tiny_fraction"] > 0.55, reasons, "tiny_component_fraction_too_high")
    _reject(features["constructor_iou"] < 0.62, reasons, "constructor_iou_too_low")
    _reject(features["p8_iou"] < 0.62, reasons, "p8_iou_too_low")
    _reject(features["p8_skeleton_f1"] < 0.74, reasons, "p8_skeleton_f1_too_low")
    _reject(features["byte_gain_vs_quadtree"] < 1.20, reasons, "quadtree_byte_advantage_not_overcome")
    _reject(features["quadtree_iou"] - features["p8_iou"] > 0.40, reasons, "p8_fidelity_gap_vs_quadtree_too_large")

    return GeometryDecision("sparse_stroke", not reasons, features, tuple(reasons)), words


def _try_topological_route(image: np.ndarray) -> tuple[GeometryDecision, list[int]]:
    selection = select_binary_mask(image)
    mask = _cleanup_mask(selection.mask)
    components = connected_components(mask)
    holes = float(count_holes(mask))
    words, _ = _encode_topological_mask(mask, selection)
    features = {
        "binary_purity": round(float(selection.binary_purity), 6),
        "gray_entropy": round(float(selection.gray_entropy), 6),
        "contrast": round(float(selection.contrast), 6),
        "foreground_fraction": round(float(selection.foreground_fraction), 6),
        "component_bbox_fill_mean": round(float(components.bbox_fill_mean), 6),
        "component_tiny_fraction": round(float(components.tiny_fraction), 6),
        "component_count": round(float(components.count), 6),
        "hole_count": round(holes, 6),
        "packed_20bit_bytes": float(packed_word_bytes(words)),
    }
    reasons: list[str] = []
    _reject(features["binary_purity"] < 0.84, reasons, "binary_purity_too_low")
    _reject(features["gray_entropy"] > 4.6, reasons, "gray_entropy_too_high")
    _reject(features["contrast"] < 72.0, reasons, "contrast_too_low")
    _reject(
        features["foreground_fraction"] < 0.01 or features["foreground_fraction"] > 0.30,
        reasons,
        "foreground_fraction_out_of_bounds",
    )
    _reject(features["component_bbox_fill_mean"] > 0.30, reasons, "component_fill_too_high_for_topology")
    _reject(features["component_tiny_fraction"] > 0.20, reasons, "tiny_component_fraction_too_high")
    _reject(features["hole_count"] < 1.0, reasons, "insufficient_hole_structure")
    return GeometryDecision("topological_segmentation", not reasons, features, tuple(reasons)), words


def _topological_priority(features: dict[str, float]) -> bool:
    return float(features.get("hole_count", 0.0)) >= 1.0


def _build_sparse_candidate(image: np.ndarray) -> SparseCandidate:
    selection = select_binary_mask(image)
    mask = _cleanup_mask(selection.mask)
    component_summary = connected_components(mask)
    skeleton = thin_binary_mask(mask)
    skeleton = prune_spurs(skeleton, rounds=1)
    polylines = trace_skeleton_to_polylines(skeleton)
    distance = _distance_map(mask)
    widths = estimate_polyline_widths(polylines, distance)
    reconstructed_mask = rasterize_polylines(polylines, mask.shape, widths)
    feature_vector = {
        "foreground_fraction": float(selection.foreground_fraction),
        "contrast": float(selection.contrast),
        "binary_purity": float(selection.binary_purity),
        "gray_entropy": float(selection.gray_entropy),
        "component_count": float(component_summary.count),
        "component_bbox_fill_mean": float(component_summary.bbox_fill_mean),
        "component_bbox_fill_max": float(component_summary.bbox_fill_max),
        "component_tiny_fraction": float(component_summary.tiny_fraction),
        "skeleton_foreground_ratio": float(skeleton.sum() / max(mask.sum(), 1)),
        "polyline_count": float(len(polylines)),
        "constructor_iou": binary_iou(mask, reconstructed_mask),
    }
    return SparseCandidate(
        selection,
        component_summary,
        mask,
        skeleton,
        polylines,
        widths,
        reconstructed_mask,
        feature_vector,
    )


def _encode_sparse_candidate(candidate: SparseCandidate) -> tuple[list[int], GeometryMeta]:
    selection = candidate.selection
    fg_gray = _quantize_gray(selection.foreground_mean)
    bg_gray = _quantize_gray(selection.background_mean)
    flags = SUBTYPE_SPARSE
    if selection.foreground_mode == "light":
        flags |= FLAG_FOREGROUND_LIGHT

    words = _encode_meta(
        width=candidate.binary_mask.shape[1],
        height=candidate.binary_mask.shape[0],
        flags=flags,
        foreground_gray=fg_gray,
        background_gray=bg_gray,
    )
    for polyline, width in zip(candidate.polylines, candidate.polyline_widths):
        if len(polyline) < 2:
            continue
        x0, y0 = polyline[0]
        words.extend(
            [
                _geometry_word(VERSION_SPARSE, SPARSE_MOVE_X_HI, _hi7(x0)),
                _geometry_word(VERSION_SPARSE, SPARSE_MOVE_X_LO, _lo7(x0)),
                _geometry_word(VERSION_SPARSE, SPARSE_MOVE_Y_HI, _hi7(y0)),
                _geometry_word(VERSION_SPARSE, SPARSE_MOVE_Y_LO, _lo7(y0)),
                _geometry_word(VERSION_SPARSE, SPARSE_SET_WIDTH, int(round(min(max(width, 1.0), 127.0)))),
            ]
        )
        for direction, run in _polyline_runs(polyline):
            packed = ((min(run, 16) - 1) << 3) | (direction & 0x7)
            words.append(_geometry_word(VERSION_SPARSE, SPARSE_DRAW, packed))
        words.append(_geometry_word(VERSION_SPARSE, SPARSE_END_PATH, 0))
    words.append(_geometry_word(VERSION_SPARSE, SPARSE_END_STREAM, 0))
    meta = GeometryMeta(
        width=candidate.binary_mask.shape[1],
        height=candidate.binary_mask.shape[0],
        subtype=SUBTYPE_SPARSE,
        foreground_gray=fg_gray,
        background_gray=bg_gray,
        foreground_mode=selection.foreground_mode,
    )
    return words, meta


def _encode_topological_mask(mask: np.ndarray, selection: BinarySelection) -> tuple[list[int], GeometryMeta]:
    fg_gray = _quantize_gray(selection.foreground_mean)
    bg_gray = _quantize_gray(selection.background_mean)
    flags = SUBTYPE_TOPOLOGICAL
    if selection.foreground_mode == "light":
        flags |= FLAG_FOREGROUND_LIGHT
    words = _encode_meta(
        width=mask.shape[1],
        height=mask.shape[0],
        flags=flags,
        foreground_gray=fg_gray,
        background_gray=bg_gray,
    )
    for row, start, length in _row_spans(mask):
        words.extend(
            [
                _geometry_word(VERSION_TOPOLOGICAL, TOPO_ROW_HI, _hi7(row)),
                _geometry_word(VERSION_TOPOLOGICAL, TOPO_ROW_LO, _lo7(row)),
                _geometry_word(VERSION_TOPOLOGICAL, TOPO_START_HI, _hi7(start)),
                _geometry_word(VERSION_TOPOLOGICAL, TOPO_START_LO, _lo7(start)),
                _geometry_word(VERSION_TOPOLOGICAL, TOPO_LEN_HI, _hi7(length)),
                _geometry_word(VERSION_TOPOLOGICAL, TOPO_LEN_LO, _lo7(length)),
                _geometry_word(VERSION_TOPOLOGICAL, TOPO_NEXT_SPAN, 0),
            ]
        )
    words.append(_geometry_word(VERSION_TOPOLOGICAL, TOPO_END_STREAM, 0))
    meta = GeometryMeta(
        width=mask.shape[1],
        height=mask.shape[0],
        subtype=SUBTYPE_TOPOLOGICAL,
        foreground_gray=fg_gray,
        background_gray=bg_gray,
        foreground_mode=selection.foreground_mode,
    )
    return words, meta


def _decode_meta(payloads: Sequence[tuple[int, int, int]]) -> GeometryMeta:
    width_hi = width_lo = height_hi = height_lo = 0
    flags = 0
    fg_gray = 0
    bg_gray = 127
    for version, opcode, value in payloads:
        if version != VERSION_META:
            continue
        if opcode == META_WIDTH_HI:
            width_hi = value
        elif opcode == META_WIDTH_LO:
            width_lo = value
        elif opcode == META_HEIGHT_HI:
            height_hi = value
        elif opcode == META_HEIGHT_LO:
            height_lo = value
        elif opcode == META_FLAGS:
            flags = value
        elif opcode == META_FG_GRAY:
            fg_gray = value
        elif opcode == META_BG_GRAY:
            bg_gray = value
        elif opcode == META_END:
            break
    subtype = flags & 0x03
    foreground_mode = "light" if (flags & FLAG_FOREGROUND_LIGHT) else "dark"
    return GeometryMeta(
        width=_from_hi_lo(width_hi, width_lo),
        height=_from_hi_lo(height_hi, height_lo),
        subtype=subtype,
        foreground_gray=fg_gray,
        background_gray=bg_gray,
        foreground_mode=foreground_mode,
    )


def _decode_sparse_payloads(payloads: Sequence[tuple[int, int, int]], meta: GeometryMeta) -> np.ndarray:
    canvas = np.full((meta.height, meta.width), _dequantize_gray(meta.background_gray), dtype=np.uint8)
    current_x = 0
    current_y = 0
    current_width = 1

    for version, opcode, value in payloads:
        if version != VERSION_SPARSE:
            continue
        if opcode == SPARSE_MOVE_X_HI:
            current_x = _from_hi_lo(value, current_x & 0x7F)
        elif opcode == SPARSE_MOVE_X_LO:
            current_x = _from_hi_lo(current_x >> 7, value)
        elif opcode == SPARSE_MOVE_Y_HI:
            current_y = _from_hi_lo(value, current_y & 0x7F)
        elif opcode == SPARSE_MOVE_Y_LO:
            current_y = _from_hi_lo(current_y >> 7, value)
        elif opcode == SPARSE_SET_WIDTH:
            current_width = max(1, int(value))
        elif opcode == SPARSE_DRAW:
            run = ((value >> 3) & 0x0F) + 1
            direction = value & 0x07
            dx, dy = _DIRS[direction]
            for _ in range(run):
                next_x = current_x + dx
                next_y = current_y + dy
                _draw_gray_line(
                    canvas,
                    (current_x, current_y),
                    (next_x, next_y),
                    current_width,
                    _dequantize_gray(meta.foreground_gray),
                )
                current_x, current_y = next_x, next_y
        elif opcode == SPARSE_END_STREAM:
            break
    return np.repeat(canvas[:, :, None], 3, axis=2)


def _decode_topological_payloads(payloads: Sequence[tuple[int, int, int]], meta: GeometryMeta) -> np.ndarray:
    canvas = np.full((meta.height, meta.width), _dequantize_gray(meta.background_gray), dtype=np.uint8)
    row = start = length = 0
    for version, opcode, value in payloads:
        if version != VERSION_TOPOLOGICAL:
            continue
        if opcode == TOPO_ROW_HI:
            row = _from_hi_lo(value, row & 0x7F)
        elif opcode == TOPO_ROW_LO:
            row = _from_hi_lo(row >> 7, value)
        elif opcode == TOPO_START_HI:
            start = _from_hi_lo(value, start & 0x7F)
        elif opcode == TOPO_START_LO:
            start = _from_hi_lo(start >> 7, value)
        elif opcode == TOPO_LEN_HI:
            length = _from_hi_lo(value, length & 0x7F)
        elif opcode == TOPO_LEN_LO:
            length = _from_hi_lo(length >> 7, value)
        elif opcode == TOPO_NEXT_SPAN:
            if 0 <= row < meta.height:
                x0 = max(0, start)
                x1 = min(meta.width, start + max(length, 0))
                if x0 < x1:
                    canvas[row, x0:x1] = _dequantize_gray(meta.foreground_gray)
        elif opcode == TOPO_END_STREAM:
            break
    return np.repeat(canvas[:, :, None], 3, axis=2)


def _encode_meta(*, width: int, height: int, flags: int, foreground_gray: int, background_gray: int) -> list[int]:
    return [
        _geometry_word(VERSION_META, META_WIDTH_HI, _hi7(width)),
        _geometry_word(VERSION_META, META_WIDTH_LO, _lo7(width)),
        _geometry_word(VERSION_META, META_HEIGHT_HI, _hi7(height)),
        _geometry_word(VERSION_META, META_HEIGHT_LO, _lo7(height)),
        _geometry_word(VERSION_META, META_FLAGS, flags & 0x7F),
        _geometry_word(VERSION_META, META_FG_GRAY, foreground_gray),
        _geometry_word(VERSION_META, META_BG_GRAY, background_gray),
        _geometry_word(VERSION_META, META_END, 0),
    ]


def _geometry_word(version: int, opcode: int, value: int) -> int:
    payload = GEOMETRY_FAMILY_VALUE | ((opcode & 0x7) << 7) | (value & 0x7F)
    return (Mode.EXTENSION.value << 18) | ((version & 0x3) << 16) | payload


def _payload_from_word(word: int) -> tuple[int, int, int]:
    value = int(word)
    version = (value >> 16) & 0x3
    payload = value & 0xFFFF
    opcode = (payload >> 7) & 0x7
    literal = payload & 0x7F
    return version, opcode, literal


def _is_geometry_word(word: int) -> bool:
    value = int(word)
    mode = (value >> 18) & 0x3
    payload = value & 0xFFFF
    return mode == Mode.EXTENSION.value and (payload & GEOMETRY_FAMILY_MASK) == GEOMETRY_FAMILY_VALUE


def _cleanup_mask(mask: np.ndarray) -> np.ndarray:
    cleaned = np.asarray(mask, dtype=bool)
    if ndimage is not None:
        structure = np.ones((3, 3), dtype=bool)
        cleaned = ndimage.binary_closing(cleaned, structure=structure)
        labels, count = ndimage.label(cleaned, structure=structure)
        keep = np.zeros_like(cleaned, dtype=bool)
        for label in range(1, int(count) + 1):
            component = labels == label
            if int(component.sum()) >= 6:
                keep |= component
        if np.any(keep):
            cleaned = keep
    return cleaned


def _distance_map(mask: np.ndarray) -> np.ndarray:
    if ndimage is None:  # pragma: no cover - exercised only without scipy
        return np.where(mask, 1.0, 0.0).astype(np.float64)
    return ndimage.distance_transform_edt(np.asarray(mask, dtype=bool))


def _polyline_runs(polyline: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    steps: list[int] = []
    for start, end in zip(polyline, polyline[1:]):
        dx = int(end[0] - start[0])
        dy = int(end[1] - start[1])
        step = (
            0 if dx == 0 else (1 if dx > 0 else -1),
            0 if dy == 0 else (1 if dy > 0 else -1),
        )
        count = max(abs(dx), abs(dy))
        if step == (0, 0):
            continue
        direction = _DIR_TO_INDEX[step]
        for _ in range(count):
            steps.append(direction)
    if not steps:
        return []
    runs: list[tuple[int, int]] = []
    current = steps[0]
    length = 1
    for direction in steps[1:]:
        if direction == current and length < 16:
            length += 1
            continue
        runs.append((current, length))
        current = direction
        length = 1
    runs.append((current, length))
    return runs


def _row_spans(mask: np.ndarray) -> list[tuple[int, int, int]]:
    binary = np.asarray(mask, dtype=bool)
    spans: list[tuple[int, int, int]] = []
    for row in range(binary.shape[0]):
        col = 0
        while col < binary.shape[1]:
            if not binary[row, col]:
                col += 1
                continue
            start = col
            while col < binary.shape[1] and binary[row, col]:
                col += 1
            spans.append((row, start, col - start))
    return spans


def _quantize_gray(value: float) -> int:
    clipped = min(max(int(round(value)), 0), 255)
    return clipped >> 1


def _dequantize_gray(value: int) -> int:
    return min(max(int(value) << 1, 0), 255)


def _hi7(value: int) -> int:
    return (int(value) >> 7) & 0x7F


def _lo7(value: int) -> int:
    return int(value) & 0x7F


def _from_hi_lo(high: int, low: int) -> int:
    return ((int(high) & 0x7F) << 7) | (int(low) & 0x7F)


def _rgb_u8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("image_array must have shape (H, W, 3) or (H, W)")
    return arr.astype(np.uint8)


def _reject(condition: bool, reasons: list[str], reason: str) -> None:
    if condition:
        reasons.append(reason)


def _draw_bool_line(canvas: np.ndarray, start: tuple[int, int], end: tuple[int, int], *, radius: int) -> None:
    x0, y0 = int(start[0]), int(start[1])
    x1, y1 = int(end[0]), int(end[1])
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    error = dx + dy
    while True:
        _paint_bool_disc(canvas, x0, y0, radius)
        if x0 == x1 and y0 == y1:
            break
        doubled = 2 * error
        if doubled >= dy:
            error += dy
            x0 += sx
        if doubled <= dx:
            error += dx
            y0 += sy


def _paint_bool_disc(canvas: np.ndarray, x: int, y: int, radius: int) -> None:
    for yy in range(y - radius, y + radius + 1):
        for xx in range(x - radius, x + radius + 1):
            if not (0 <= yy < canvas.shape[0] and 0 <= xx < canvas.shape[1]):
                continue
            if (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2:
                canvas[yy, xx] = True


def _draw_gray_line(canvas: np.ndarray, start: tuple[int, int], end: tuple[int, int], width: int, value: int) -> None:
    radius = max(0, int(np.ceil((float(width) - 1.0) / 2.0)))
    x0, y0 = int(start[0]), int(start[1])
    x1, y1 = int(end[0]), int(end[1])
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    error = dx + dy
    while True:
        for yy in range(y0 - radius, y0 + radius + 1):
            for xx in range(x0 - radius, x0 + radius + 1):
                if not (0 <= yy < canvas.shape[0] and 0 <= xx < canvas.shape[1]):
                    continue
                if (xx - x0) ** 2 + (yy - y0) ** 2 <= radius ** 2:
                    canvas[yy, xx] = value
        if x0 == x1 and y0 == y1:
            break
        doubled = 2 * error
        if doubled >= dy:
            error += dy
            x0 += sx
        if doubled <= dx:
            error += dx
            y0 += sy


def _fallback_label(mask: np.ndarray) -> tuple[np.ndarray, int]:
    labels = np.zeros(mask.shape, dtype=np.int32)
    next_label = 0
    offsets = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]
    for row in range(mask.shape[0]):
        for col in range(mask.shape[1]):
            if not mask[row, col] or labels[row, col] != 0:
                continue
            next_label += 1
            stack = [(row, col)]
            labels[row, col] = next_label
            while stack:
                cur_row, cur_col = stack.pop()
                for dy, dx in offsets:
                    nxt_row = cur_row + dy
                    nxt_col = cur_col + dx
                    if not (0 <= nxt_row < mask.shape[0] and 0 <= nxt_col < mask.shape[1]):
                        continue
                    if not mask[nxt_row, nxt_col] or labels[nxt_row, nxt_col] != 0:
                        continue
                    labels[nxt_row, nxt_col] = next_label
                    stack.append((nxt_row, nxt_col))
    return labels, next_label


def _walk_path(
    start: tuple[int, int],
    neighbor: tuple[int, int],
    neighbor_map: dict[tuple[int, int], list[tuple[int, int]]],
    nodes: set[tuple[int, int]],
    visited_edges: set[tuple[tuple[int, int], tuple[int, int]]],
) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = [start, neighbor]
    visited_edges.add(_ordered_edge(start, neighbor))
    previous = start
    current = neighbor
    while current not in nodes:
        next_candidates = [candidate for candidate in neighbor_map[current] if candidate != previous]
        if not next_candidates:
            break
        nxt = next_candidates[0]
        edge = _ordered_edge(current, nxt)
        if edge in visited_edges:
            break
        visited_edges.add(edge)
        path.append(nxt)
        previous, current = current, nxt
    return [_to_xy(point) for point in path]


def _walk_cycle(
    start: tuple[int, int],
    neighbor: tuple[int, int],
    neighbor_map: dict[tuple[int, int], list[tuple[int, int]]],
    visited_edges: set[tuple[tuple[int, int], tuple[int, int]]],
) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = [start, neighbor]
    visited_edges.add(_ordered_edge(start, neighbor))
    previous = start
    current = neighbor
    while True:
        next_candidates = [candidate for candidate in neighbor_map[current] if candidate != previous]
        if not next_candidates:
            break
        nxt = next_candidates[0]
        edge = _ordered_edge(current, nxt)
        if edge in visited_edges:
            if nxt == start:
                path.append(start)
            break
        visited_edges.add(edge)
        path.append(nxt)
        previous, current = current, nxt
    return [_to_xy(point) for point in path]


def _neighbor_points(point: tuple[int, int]) -> list[tuple[int, int]]:
    row, col = point
    return [(row + dy, col + dx) for dy, dx in _NEIGHBOR_OFFSETS]


def _ordered_edge(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    return (a, b) if a <= b else (b, a)


def _to_xy(point: tuple[int, int]) -> tuple[int, int]:
    row, col = point
    return int(col), int(row)


def _neighbors(binary: np.ndarray, row: int, col: int) -> tuple[int, int, int, int, int, int, int, int]:
    return (
        int(binary[row - 1, col]),
        int(binary[row - 1, col + 1]),
        int(binary[row, col + 1]),
        int(binary[row + 1, col + 1]),
        int(binary[row + 1, col]),
        int(binary[row + 1, col - 1]),
        int(binary[row, col - 1]),
        int(binary[row - 1, col - 1]),
    )


def _transitions(neighbors: Sequence[int]) -> int:
    total = 0
    wrapped = list(neighbors[1:]) + [neighbors[0]]
    for current, nxt in zip(neighbors, wrapped):
        if current == 0 and nxt == 1:
            total += 1
    return total


__all__ = [
    "BinarySelection",
    "GeometryDecision",
    "GeometryMeta",
    "GeometryRoute",
    "binary_f1",
    "binary_iou",
    "count_holes",
    "decode_geometry_image",
    "detect_geometry_subtype",
    "grayscale_u8",
    "route_image",
    "select_binary_mask",
    "thin_binary_mask",
]
