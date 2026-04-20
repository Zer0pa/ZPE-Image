from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from zpe_image_codec.bootstrap import ensure_core_imports

ensure_core_imports()

from source.image.geometry_codec import (
    _cleanup_mask,
    _try_sparse_route,
    _try_topological_route,
    binary_f1,
    binary_iou,
    count_holes,
    decode_geometry_image,
    select_binary_mask,
    thin_binary_mask,
)
from source.image.quadtree_enhanced_codec import encode_enhanced


_STATE_MARKER = 0x0E00


@dataclass(frozen=True)
class BundleState:
    hole_count: int
    component_fill_mean: float
    foreground_fraction: float
    rescued_by_fiber: bool
    topological_primary_accept: bool


@dataclass(frozen=True)
class BundleRoute:
    accepted: bool
    scope_name: str
    decode_route: str
    base_words: tuple[int, ...]
    fiber_words: tuple[int, ...]
    state_words: tuple[int, ...]
    state: BundleState
    rejection_reasons: tuple[str, ...]

    @property
    def bundle_words(self) -> tuple[int, ...]:
        return self.state_words + self.base_words + self.fiber_words

    def to_dict(self) -> dict[str, object]:
        payload = {
            "accepted": self.accepted,
            "scope_name": self.scope_name,
            "decode_route": self.decode_route,
            "base_word_count": len(self.base_words),
            "fiber_word_count": len(self.fiber_words),
            "state_word_count": len(self.state_words),
            "rejection_reasons": list(self.rejection_reasons),
            "state": asdict(self.state),
        }
        payload["bundle_word_count"] = payload["base_word_count"] + payload["fiber_word_count"] + payload["state_word_count"]
        return payload


def _pack_words_20bit(words: tuple[int, ...] | list[int]) -> int:
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


def _state_words(state: BundleState) -> tuple[int, ...]:
    component_fill = min(int(round(state.component_fill_mean * 1000.0)), 0xFF)
    foreground_fraction = min(int(round(state.foreground_fraction * 1000.0)), 0xFF)
    return (
        _STATE_MARKER | (int(state.hole_count) & 0xFF),
        _STATE_MARKER | 0x0100 | component_fill,
        _STATE_MARKER | 0x0200 | foreground_fraction,
        _STATE_MARKER | 0x0300 | (1 if state.rescued_by_fiber else 0),
    )


def topological_reference_mask(image: np.ndarray) -> np.ndarray:
    return _cleanup_mask(select_binary_mask(np.asarray(image, dtype=np.uint8)).mask)


def route_bundle_image(image: np.ndarray) -> BundleRoute:
    arr = np.asarray(image, dtype=np.uint8)
    sparse_decision, sparse_words = _try_sparse_route(arr)
    topological_decision, topological_words = _try_topological_route(arr)
    reference_mask = topological_reference_mask(arr)
    hole_count = int(count_holes(reference_mask))

    state = BundleState(
        hole_count=hole_count,
        component_fill_mean=float(topological_decision.features.get("component_bbox_fill_mean", 0.0)),
        foreground_fraction=float(topological_decision.features.get("foreground_fraction", 0.0)),
        rescued_by_fiber=bool(not topological_decision.accept and sparse_decision.accept and hole_count >= 1),
        topological_primary_accept=bool(topological_decision.accept),
    )

    reasons: list[str] = []
    if not sparse_decision.accept:
        reasons.append("sparse_fiber_not_admissible")
    if hole_count < 1:
        reasons.append("hole_bearing_base_not_present")

    accepted = not reasons
    return BundleRoute(
        accepted=accepted,
        scope_name="hole_bearing_sparse_bundle" if accepted else "reject",
        decode_route="topological_base_projection" if accepted else "reject",
        base_words=tuple(int(word) for word in topological_words),
        fiber_words=tuple(int(word) for word in sparse_words),
        state_words=_state_words(state),
        state=state,
        rejection_reasons=tuple(reasons),
    )


def decode_bundle_image(route: BundleRoute) -> np.ndarray:
    if not route.accepted:
        raise ValueError("bundle route is not accepted")
    image, _meta = decode_geometry_image(route.base_words)
    return image


def bundle_metrics(route: BundleRoute, image: np.ndarray) -> dict[str, object]:
    if not route.accepted:
        return {
            "accepted": False,
            "bundle_bytes_20bit": 0,
            "topological_bytes_20bit": _pack_words_20bit(route.base_words),
            "sparse_bytes_20bit": _pack_words_20bit(route.fiber_words),
            "quadtree_bytes_20bit": _pack_words_20bit(encode_enhanced(np.asarray(image, dtype=np.uint8), bit_depth=3)[0]),
            "rejection_reasons": list(route.rejection_reasons),
        }

    decoded = decode_bundle_image(route)
    reference_mask = topological_reference_mask(np.asarray(image, dtype=np.uint8))
    candidate_mask = topological_reference_mask(decoded)
    quadtree_words, _ = encode_enhanced(np.asarray(image, dtype=np.uint8), bit_depth=3)
    return {
        "accepted": True,
        "bundle_bytes_20bit": _pack_words_20bit(route.bundle_words),
        "topological_bytes_20bit": _pack_words_20bit(route.base_words),
        "sparse_bytes_20bit": _pack_words_20bit(route.fiber_words),
        "quadtree_bytes_20bit": _pack_words_20bit(quadtree_words),
        "projection_iou": round(binary_iou(reference_mask, candidate_mask), 6),
        "projection_skeleton_f1": round(
            binary_f1(thin_binary_mask(reference_mask), thin_binary_mask(candidate_mask)),
            6,
        ),
        "state": asdict(route.state),
    }
