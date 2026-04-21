from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from .codec_constants import DEFAULT_VERSION, Mode


IMAGE_FAMILY_MASK = 0x0C00
IMAGE_FAMILY_VALUE = 0x0400
DATA_FLAG = 0x0200

CMD_TL = 0
CMD_TR = 1
CMD_BL = 2
CMD_BR = 3
CMD_PAINT = 4
CMD_SET_COLOR = 5
CMD_BACKTRACK = 6
CMD_META = 7

META_BEGIN = 0
META_END = 1

M_WIDTH_HI = 0
M_WIDTH_LO = 1
M_HEIGHT_HI = 2
M_HEIGHT_LO = 3
M_ROOT_HI = 4
M_ROOT_LO = 5
M_BIT_DEPTH = 6
M_THRESH_X10 = 7

C_R = 0
C_G = 1
C_B = 2

MAX_RUN = 63


@dataclass(frozen=True)
class EnhancedMeta:
    width: int
    height: int
    root: int
    bit_depth: int
    threshold_x10: int


def _ext_word(payload: int) -> int:
    return (Mode.EXTENSION.value << 18) | (DEFAULT_VERSION << 16) | (payload & 0xFFFF)


def _cmd_word(cmd: int, arg: int = 1) -> int:
    arg = max(0, min(int(arg), MAX_RUN))
    payload = IMAGE_FAMILY_VALUE | ((int(cmd) & 0x7) << 6) | (arg & 0x3F)
    return _ext_word(payload)


def _data_word(kind: int, value: int) -> int:
    payload = IMAGE_FAMILY_VALUE | DATA_FLAG | ((int(kind) & 0x7) << 6) | (int(value) & 0x3F)
    return _ext_word(payload)


def _is_family(payload: int) -> bool:
    return (payload & IMAGE_FAMILY_MASK) == IMAGE_FAMILY_VALUE


def _next_pow2(n: int) -> int:
    out = 1
    while out < n:
        out <<= 1
    return out


def _quant_level(value_u8: int, bit_depth: int) -> int:
    maxq = (1 << bit_depth) - 1
    return int(round((int(value_u8) / 255.0) * maxq))


def _dequant_level(level: int, bit_depth: int) -> int:
    maxq = (1 << bit_depth) - 1
    if maxq <= 0:
        return 0
    return int(round((int(level) / maxq) * 255.0))


def _quantize(image: np.ndarray, bit_depth: int) -> np.ndarray:
    arr = image.astype(np.uint8)
    maxq = (1 << bit_depth) - 1
    if maxq <= 0:
        return np.zeros_like(arr, dtype=np.uint8)
    levels = np.rint((arr.astype(np.float64) / 255.0) * maxq).astype(np.int32)
    out = np.zeros_like(arr, dtype=np.uint8)
    out[:, :, 0] = np.vectorize(_dequant_level)(levels[:, :, 0], bit_depth)
    out[:, :, 1] = np.vectorize(_dequant_level)(levels[:, :, 1], bit_depth)
    out[:, :, 2] = np.vectorize(_dequant_level)(levels[:, :, 2], bit_depth)
    return out


def _pack_u12(value: int) -> tuple[int, int]:
    v = max(0, min(int(value), 0xFFF))
    return ((v >> 6) & 0x3F, v & 0x3F)


def _unpack_u12(hi: int, lo: int) -> int:
    return ((int(hi) & 0x3F) << 6) | (int(lo) & 0x3F)


def encode_enhanced(image: np.ndarray, threshold: float = 5.0, bit_depth: int = 3) -> tuple[List[int], EnhancedMeta]:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("expected image shape (H, W, 3)")
    if not (1 <= bit_depth <= 6):
        raise ValueError("bit_depth supported range is [1,6]")

    h, w, _ = image.shape
    root = _next_pow2(max(h, w))

    q = _quantize(image, bit_depth)
    padded = np.zeros((root, root, 3), dtype=np.uint8)
    padded[:h, :w, :] = q

    t10 = int(round(float(threshold) * 10.0))
    meta = EnhancedMeta(width=w, height=h, root=root, bit_depth=bit_depth, threshold_x10=t10)

    words: List[int] = []
    words.append(_cmd_word(CMD_META, META_BEGIN))
    w_hi, w_lo = _pack_u12(w)
    h_hi, h_lo = _pack_u12(h)
    r_hi, r_lo = _pack_u12(root)

    words.extend(
        [
            _data_word(M_WIDTH_HI, w_hi),
            _data_word(M_WIDTH_LO, w_lo),
            _data_word(M_HEIGHT_HI, h_hi),
            _data_word(M_HEIGHT_LO, h_lo),
            _data_word(M_ROOT_HI, r_hi),
            _data_word(M_ROOT_LO, r_lo),
            _data_word(M_BIT_DEPTH, bit_depth),
            _data_word(M_THRESH_X10, t10),
        ]
    )
    words.append(_cmd_word(CMD_META, META_END))

    current = (-1, -1, -1)

    def variance(region: np.ndarray) -> float:
        return float(np.var(region.reshape(-1, 3), axis=0).mean())

    def emit_color(rgb: tuple[int, int, int]) -> None:
        nonlocal current
        levels = tuple(_quant_level(c, bit_depth) for c in rgb)
        if levels == current:
            return
        words.append(_cmd_word(CMD_SET_COLOR, 1))
        words.append(_data_word(C_R, levels[0]))
        words.append(_data_word(C_G, levels[1]))
        words.append(_data_word(C_B, levels[2]))
        current = levels

    def emit_paint() -> None:
        words.append(_cmd_word(CMD_PAINT, 1))

    def rec(x: int, y: int, size: int) -> None:
        region = padded[y : y + size, x : x + size, :]
        if size == 1 or variance(region) <= threshold:
            mean_rgb = tuple(int(np.round(region[:, :, i].mean())) for i in range(3))
            emit_color(mean_rgb)
            emit_paint()
            return

        half = size // 2
        for cmd, nx, ny in (
            (CMD_TL, x, y),
            (CMD_TR, x + half, y),
            (CMD_BL, x, y + half),
            (CMD_BR, x + half, y + half),
        ):
            words.append(_cmd_word(cmd, 1))
            rec(nx, ny, half)
            words.append(_cmd_word(CMD_BACKTRACK, 1))

    rec(0, 0, root)
    return words, meta


def decode_enhanced(words: Sequence[int]) -> tuple[np.ndarray, EnhancedMeta]:
    meta_open = False
    meta_vals = {
        M_WIDTH_HI: 0,
        M_WIDTH_LO: 0,
        M_HEIGHT_HI: 0,
        M_HEIGHT_LO: 0,
        M_ROOT_HI: 0,
        M_ROOT_LO: 0,
        M_BIT_DEPTH: 3,
        M_THRESH_X10: 50,
    }

    commands: list[tuple[int, int]] = []
    color_data: list[tuple[int, int]] = []

    for word_value in words:
        mode = (int(word_value) >> 18) & 0x3
        version = (int(word_value) >> 16) & 0x3
        payload = int(word_value) & 0xFFFF

        if mode != Mode.EXTENSION.value or version != DEFAULT_VERSION or not _is_family(payload):
            continue

        if payload & DATA_FLAG:
            kind = (payload >> 6) & 0x7
            val = payload & 0x3F
            if meta_open:
                meta_vals[kind] = val
            else:
                color_data.append((kind, val))
            continue

        cmd = (payload >> 6) & 0x7
        arg = payload & 0x3F

        if cmd == CMD_META and arg == META_BEGIN:
            meta_open = True
            continue
        if cmd == CMD_META and arg == META_END:
            meta_open = False
            continue

        commands.append((cmd, max(1, arg)))

    width = _unpack_u12(meta_vals[M_WIDTH_HI], meta_vals[M_WIDTH_LO])
    height = _unpack_u12(meta_vals[M_HEIGHT_HI], meta_vals[M_HEIGHT_LO])
    root = _unpack_u12(meta_vals[M_ROOT_HI], meta_vals[M_ROOT_LO])
    bit_depth = max(1, min(int(meta_vals[M_BIT_DEPTH]), 6))
    thresh_x10 = int(meta_vals[M_THRESH_X10])

    meta = EnhancedMeta(width=width, height=height, root=root, bit_depth=bit_depth, threshold_x10=thresh_x10)

    if root <= 0:
        raise ValueError("invalid root in metadata")

    canvas = np.zeros((root, root, 3), dtype=np.uint8)
    stack: list[tuple[int, int, int]] = [(0, 0, root)]
    current = (0, 0, 0)
    color_idx = 0

    for cmd, run in commands:
        for _ in range(run):
            if cmd in (CMD_TL, CMD_TR, CMD_BL, CMD_BR):
                x, y, size = stack[-1]
                half = max(1, size // 2)
                if cmd == CMD_TL:
                    stack.append((x, y, half))
                elif cmd == CMD_TR:
                    stack.append((x + half, y, half))
                elif cmd == CMD_BL:
                    stack.append((x, y + half, half))
                else:
                    stack.append((x + half, y + half, half))
                continue
            if cmd == CMD_BACKTRACK:
                if len(stack) > 1:
                    stack.pop()
                continue
            if cmd == CMD_SET_COLOR:
                triplet = {kind: val for kind, val in color_data[color_idx : color_idx + 3]}
                color_idx += 3
                current = (
                    _dequant_level(triplet.get(C_R, 0), bit_depth),
                    _dequant_level(triplet.get(C_G, 0), bit_depth),
                    _dequant_level(triplet.get(C_B, 0), bit_depth),
                )
                continue
            if cmd == CMD_PAINT:
                x, y, size = stack[-1]
                canvas[y : y + size, x : x + size] = np.array(current, dtype=np.uint8)

    return canvas[:height, :width], meta
