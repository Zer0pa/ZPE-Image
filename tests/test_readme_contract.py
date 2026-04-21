from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"


def test_readme_uses_canonical_heading_order() -> None:
    headings = [line.strip() for line in README.read_text().splitlines() if line.startswith("## ")]
    required = [
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
    if "## Competitive Benchmarks" in headings:
        expected = required[:2] + ["## Competitive Benchmarks"] + required[2:]
    else:
        expected = required
    assert headings == expected


def test_readme_metadata_and_quick_start() -> None:
    text = README.read_text()
    assert "| Architecture | IMAGE_STREAM |" in text
    assert "| Encoding | IMAGE_SPARSE_GEOMETRY_V1 |" in text
    assert "python3 -m pip install '.[dev]'" in text
    assert "zpe-image-verify --output" in text
