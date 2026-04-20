from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_package_bootstrap_finds_sibling_zpe_core() -> None:
    import zpe_image_codec

    assert callable(zpe_image_codec.route_image_for_imc)
    assert callable(zpe_image_codec.decode_image_words)
