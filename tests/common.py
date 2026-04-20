from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Iterable, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
FIXTURES = ROOT / "fixtures"
NOTES = ROOT / "notes"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zpe_image_codec.bootstrap import ensure_core_imports

ensure_core_imports()


def configure_env() -> None:
    os.environ.setdefault("STROKEGRAM_ENABLE_DIAGRAM", "1")
    os.environ.setdefault("STROKEGRAM_ENABLE_MUSIC", "1")
    os.environ.setdefault("STROKEGRAM_ENABLE_VOICE", "1")


def mean_point_distance(
    points_a: Sequence[Tuple[float, float]],
    points_b: Sequence[Tuple[float, float]],
) -> float:
    if not points_a and not points_b:
        return 0.0
    if not points_a or not points_b:
        return float("inf")
    a = np.array(points_a, dtype=np.float64)
    b = np.array(points_b, dtype=np.float64)
    dists = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2))
    return float(np.mean(np.min(dists, axis=1)))


def flatten_points(polylines: Iterable[Sequence[Tuple[float, float]]]) -> list[Tuple[float, float]]:
    out: list[Tuple[float, float]] = []
    for poly in polylines:
        for pt in poly:
            out.append((float(pt[0]), float(pt[1])))
    return out
