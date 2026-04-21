from __future__ import annotations

from enum import IntEnum
from typing import Iterable


class Mode(IntEnum):
    NORMAL = 0
    ESCAPE = 1
    EXTENSION = 2
    RESERVED = 3


DEFAULT_VERSION = 0
WORD_BITS = 20
WORD_MASK = (1 << WORD_BITS) - 1
PAYLOAD_16_MASK = 0xFFFF


def packed_word_bytes(words: Iterable[int]) -> int:
    accumulator = 0
    bits = 0
    count = 0
    for word in words:
        accumulator |= int(word) << bits
        bits += WORD_BITS
        while bits >= 8:
            count += 1
            accumulator >>= 8
            bits -= 8
    if bits:
        count += 1
    return count
