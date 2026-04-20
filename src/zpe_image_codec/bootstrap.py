from __future__ import annotations

import os
import sys
from importlib.util import find_spec
from pathlib import Path


def _core_imports_available() -> bool:
    return find_spec("source") is not None and find_spec("zpe_multimodal") is not None


def _candidate_core_roots() -> list[Path]:
    repo_root = Path(__file__).resolve().parents[2]
    candidates: list[Path] = []

    env_root = os.environ.get("ZPE_CORE_ROOT")
    if env_root:
        candidates.append(Path(env_root))

    candidates.append(repo_root.parent / "zpe-core")
    return candidates


def _is_core_root(path: Path) -> bool:
    return (path / "source").is_dir() and (path / "zpe_multimodal").is_dir()


def ensure_core_imports() -> Path | None:
    if _core_imports_available():
        return None

    for candidate in _candidate_core_roots():
        if not _is_core_root(candidate):
            continue
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
        if _core_imports_available():
            return candidate

    raise ModuleNotFoundError(
        "zpe-core import surface not found. Install zpe-core or set ZPE_CORE_ROOT."
    )
