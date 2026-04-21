from __future__ import annotations

from zpe_image_codec import case_by_name, route_bundle_image, route_image
from zpe_image_codec.verify import run_verification


def test_standalone_routes_cover_the_documented_scope() -> None:
    glyph = case_by_name("glyph_a")["image"]
    mixed = case_by_name("filled_disk_with_spine")["image"]

    sparse_route = route_image(glyph, family_mode="sparse")
    reject_route = route_image(mixed, family_mode="sparse")
    bundle_route = route_bundle_image(glyph)

    assert sparse_route.accepted is True
    assert reject_route.accepted is False
    assert bundle_route.accepted is True


def test_fresh_verification_matches_the_bounded_claims() -> None:
    payload = run_verification()

    assert payload["verification_status"] == "fresh_falsification_ready"
    assert payload["primary_route"]["accepted_positive_cases"] == [
        "glyph_a",
        "fork_tree",
        "loop_spine",
        "maze_turns",
        "serpentine",
    ]
    assert payload["primary_route"]["negative_reject_rate"] == 1.0
    assert payload["primary_route"]["worst_perturb_iou"] >= 0.62
    assert payload["secondary_route"]["accepted_positive_cases"] == [
        "glyph_a",
        "loop_spine",
        "maze_turns",
    ]
    assert payload["secondary_route"]["out_of_scope_positive_cases"] == [
        "fork_tree",
        "serpentine",
    ]
    assert payload["secondary_route"]["reject_bucket_reject_rate"] == 1.0
