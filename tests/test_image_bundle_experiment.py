from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tests.common import configure_env

configure_env()

from geogram5.experimental_bundle_codec import decode_bundle_image, route_bundle_image, topological_reference_mask
from scripts.geogram_raster_eval import build_cases
from source.image.geometry_codec import binary_f1, binary_iou, thin_binary_mask
from zpe_multimodal.geogram.metrics import perturbation_suite


def _case_by_name(name: str) -> dict[str, object]:
    for case in build_cases():
        if case["name"] == name:
            return case
    raise KeyError(name)


def test_bundle_scope_is_the_hole_bearing_subset() -> None:
    accepted: list[str] = []
    for name in ("glyph_a", "fork_tree", "loop_spine", "maze_turns", "serpentine"):
        image = np.asarray(_case_by_name(name)["image"], dtype=np.uint8)
        route = route_bundle_image(image)
        if route.accepted:
            accepted.append(name)
    assert accepted == ["glyph_a", "loop_spine", "maze_turns"]


def test_bundle_rejects_mixed_and_natural_cases() -> None:
    for name in (
        "filled_disk_with_spine",
        "gradient_with_symbol",
        "stroke_over_texture",
        "smooth_gradient",
        "texture_blobs",
        "checkerboard_texture",
        "color_field_noise",
    ):
        image = np.asarray(_case_by_name(name)["image"], dtype=np.uint8)
        route = route_bundle_image(image)
        assert route.accepted is False


def test_bundle_rescues_dilated_hole_bearing_cells_without_projection_loss() -> None:
    for name in ("loop_spine", "maze_turns"):
        image = np.asarray(_case_by_name(name)["image"], dtype=np.uint8)
        variant = perturbation_suite(image)["dilate_1"]
        route = route_bundle_image(variant)
        decoded = decode_bundle_image(route)
        reference_mask = topological_reference_mask(variant)
        decoded_mask = topological_reference_mask(decoded)
        assert route.accepted is True
        assert route.state.rescued_by_fiber is True
        assert binary_iou(reference_mask, decoded_mask) == 1.0
        assert binary_f1(thin_binary_mask(reference_mask), thin_binary_mask(decoded_mask)) == 1.0
