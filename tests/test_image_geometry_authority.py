from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.common import configure_env

configure_env()

from scripts.geogram_raster_eval import build_cases
from source.core.imc import IMCDecoder, IMCEncoder
from source.image.dual_dispatch import decode_image_words, detect_family
from source.image.geometry_codec import binary_iou, route_image_for_imc, select_binary_mask


def _case_by_name(name: str) -> dict[str, object]:
    for case in build_cases():
        if case["name"] == name:
            return case
    raise KeyError(name)


def test_sparse_geometry_route_accepts_glyph_a() -> None:
    case = _case_by_name("glyph_a")
    image = np.asarray(case["image"], dtype=np.uint8)
    route = route_image_for_imc(image, family_mode="sparse")
    assert route.selected_route == "geometry_sparse_stroke"
    assert route.accepted is True
    assert detect_family(route.words) == "geometry"

    decoded = decode_image_words(route.words)
    reference_mask = select_binary_mask(image).mask
    decoded_mask = select_binary_mask(decoded.image).mask
    assert decoded.mode == "geometry"
    assert float(binary_iou(reference_mask, decoded_mask)) >= 0.62


def test_sparse_geometry_route_rejects_filled_disk_with_spine() -> None:
    case = _case_by_name("filled_disk_with_spine")
    image = np.asarray(case["image"], dtype=np.uint8)
    route = route_image_for_imc(image, family_mode="sparse")
    assert route.selected_route == "quadtree_enhanced"
    assert route.accepted is False
    assert detect_family(route.words) == "enhanced"


def test_topological_geometry_route_has_distinct_subtype() -> None:
    case = _case_by_name("loop_spine")
    image = np.asarray(case["image"], dtype=np.uint8)
    route = route_image_for_imc(image, family_mode="topological")
    assert route.selected_route == "geometry_topological_segmentation"
    decoded = decode_image_words(route.words)
    assert decoded.mode == "geometry"
    assert getattr(decoded.meta, "subtype", None) == 2


def test_live_imc_image_authority_uses_geometry_only_inside_scope() -> None:
    positive = np.asarray(_case_by_name("glyph_a")["image"], dtype=np.uint8)
    mixed = np.asarray(_case_by_name("filled_disk_with_spine")["image"], dtype=np.uint8)

    positive_stream = IMCEncoder().add_image(positive).build()
    mixed_stream = IMCEncoder().add_image(mixed).build()

    assert detect_family(positive_stream) == "geometry"
    assert detect_family(mixed_stream) == "enhanced"

    positive_decoded = IMCDecoder().decode(positive_stream)
    mixed_decoded = IMCDecoder().decode(mixed_stream)

    assert positive_decoded.image_blocks[0].shape == positive.shape
    assert mixed_decoded.image_blocks[0].shape == mixed.shape
