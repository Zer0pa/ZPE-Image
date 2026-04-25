from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
VALIDATION_PACKET = ROOT / "validation" / "results" / "fresh_falsification_check.json"
PROOF_PACKET = ROOT / "proofs" / "artifacts" / "fresh_falsification_packet.json"
PROOF_MANIFEST = ROOT / "proofs" / "manifests" / "CURRENT_VERIFICATION_PACKET.md"


def test_readme_uses_canonical_heading_order() -> None:
    headings = [line.strip() for line in README.read_text().splitlines() if line.startswith("## ")]
    assert headings == [
        "## What This Is",
        "## Key Metrics",
        "## What We Prove",
        "## What We Don't Claim",
        "## Commercial Readiness",
        "## Tests and Verification",
        "## Proof Anchors",
        "## Repo Shape",
        "## Quick Start",
    ]


def test_readme_metadata_and_quick_start() -> None:
    text = README.read_text()
    assert "| Architecture | IMAGE_STREAM |" in text
    assert "| Encoding | IMAGE_SPARSE_GEOMETRY_V1 |" in text
    assert "| Proof Anchors | 3 |" in text
    assert "| Authority Source | `proofs/artifacts/fresh_falsification_packet.json` |" in text
    assert "| Runtime Package | `src/zpe_image_codec` |" in text
    assert "python3 -m pip install '.[dev]'" in text
    assert "zpe-image-verify --output" in text


def test_readme_proof_anchors_match_validation_packet() -> None:
    text = README.read_text()
    payload = json.loads(VALIDATION_PACKET.read_text())

    for anchor in (PROOF_MANIFEST, PROOF_PACKET, VALIDATION_PACKET):
        assert anchor.exists()
        relative_path = anchor.relative_to(ROOT).as_posix()
        assert f"`{relative_path}`" in text

    assert "| Verdict | STAGED |" in text
    assert f"| Posture | `{payload['publication_status']}` |" in text
    assert f"| Verification Status | `{payload['verification_status']}` |" in text
    assert f"| Commit SHA | {payload['git_commit']} |" in text
    assert "| Source | `validation/results/fresh_falsification_check.json` |" in text
