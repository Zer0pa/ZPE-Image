from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .enhanced_codec import IMAGE_FAMILY_VALUE as ENHANCED_FAMILY, decode_enhanced
from .geometry_codec import GEOMETRY_FAMILY_VALUE as GEOMETRY_FAMILY, decode_geometry_image


FAMILY_MASK = 0x0C00


@dataclass(frozen=True)
class DecodeResult:
    mode: str
    image: Any
    meta: Any


def _iter_image_payloads(words: Sequence[int]) -> list[int]:
    payloads: list[int] = []
    for word in words:
        try:
            value = int(word)
        except (TypeError, ValueError):
            continue
        mode = (value >> 18) & 0x3
        version = (value >> 16) & 0x3
        if mode != 0x2 or version not in (0, 1, 2, 3):
            continue
        payloads.append(value & 0xFFFF)
    return payloads


def detect_family(words: Sequence[int]) -> str:
    payloads = _iter_image_payloads(words)
    if not payloads:
        raise ValueError("no recognizable image family marker found")

    family = payloads[0] & FAMILY_MASK
    if family not in (ENHANCED_FAMILY, GEOMETRY_FAMILY):
        raise ValueError("no recognizable image family marker found")
    for payload in payloads[1:]:
        if (payload & FAMILY_MASK) != family:
            raise ValueError("mixed image family markers found")
    return "geometry" if family == GEOMETRY_FAMILY else "enhanced"


def decode_image_words(words: Sequence[int]) -> DecodeResult:
    mode = detect_family(words)
    if mode == "geometry":
        image, meta = decode_geometry_image(words)
        return DecodeResult(mode=mode, image=image, meta=meta)
    image, meta = decode_enhanced(words)
    return DecodeResult(mode=mode, image=image, meta=meta)
