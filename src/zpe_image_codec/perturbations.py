from __future__ import annotations

from typing import Mapping

import numpy as np

try:
    from scipy import ndimage
except Exception:  # pragma: no cover - dependency guard
    ndimage = None  # type: ignore[assignment]

from .geometry_codec import select_binary_mask


def summarize_perturbations(results: Mapping[str, Mapping[str, float]]) -> dict[str, float | str]:
    if not results:
        return {
            "count": 0,
            "worst_iou": 0.0,
            "worst_skeleton_f1": 0.0,
            "worst_case": "none",
            "mean_iou": 0.0,
            "mean_skeleton_f1": 0.0,
        }
    names = list(results)
    worst_case = min(names, key=lambda name: (results[name].get("iou", 0.0), results[name].get("skeleton_f1", 0.0)))
    return {
        "count": len(results),
        "worst_iou": float(min(results[name].get("iou", 0.0) for name in names)),
        "worst_skeleton_f1": float(min(results[name].get("skeleton_f1", 0.0) for name in names)),
        "worst_case": worst_case,
        "mean_iou": float(np.mean([results[name].get("iou", 0.0) for name in names])),
        "mean_skeleton_f1": float(np.mean([results[name].get("skeleton_f1", 0.0) for name in names])),
    }


def binary_palette_colors(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    foreground = arr[np.asarray(mask, dtype=bool)]
    background = arr[~np.asarray(mask, dtype=bool)]
    fg = np.rint(foreground.reshape(-1, 3).mean(axis=0)).astype(np.uint8) if foreground.size else np.zeros(3, dtype=np.uint8)
    bg = np.rint(background.reshape(-1, 3).mean(axis=0)).astype(np.uint8) if background.size else np.full(3, 255, dtype=np.uint8)
    return fg, bg


def compose_two_tone(mask: np.ndarray, foreground: np.ndarray, background: np.ndarray) -> np.ndarray:
    canvas = np.zeros(mask.shape + (3,), dtype=np.uint8)
    canvas[:, :] = np.asarray(background, dtype=np.uint8)
    canvas[np.asarray(mask, dtype=bool)] = np.asarray(foreground, dtype=np.uint8)
    return canvas


def translate_image(image: np.ndarray, *, dx: int = 0, dy: int = 0) -> np.ndarray:
    arr = np.asarray(image)
    out = np.zeros_like(arr)
    height, width = arr.shape[:2]
    src_x0 = max(0, -dx)
    src_x1 = min(width, width - dx) if dx >= 0 else width
    dst_x0 = max(0, dx)
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    src_y0 = max(0, -dy)
    src_y1 = min(height, height - dy) if dy >= 0 else height
    dst_y0 = max(0, dy)
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    out[dst_y0:dst_y1, dst_x0:dst_x1] = arr[src_y0:src_y1, src_x0:src_x1]
    return out


def perturbation_suite(image: np.ndarray) -> dict[str, np.ndarray]:
    selection = select_binary_mask(image)
    foreground, background = binary_palette_colors(image, selection.mask)
    variants: dict[str, np.ndarray] = {
        "shift_x1": translate_image(image, dx=1, dy=0),
        "shift_y1": translate_image(image, dx=0, dy=1),
    }

    rng = np.random.default_rng(17)
    noisy = np.asarray(image).copy()
    flip_count = max(1, int(noisy.shape[0] * noisy.shape[1] * 0.01))
    ys = rng.integers(0, noisy.shape[0], size=flip_count)
    xs = rng.integers(0, noisy.shape[1], size=flip_count)
    noisy[ys, xs] = np.where(rng.integers(0, 2, size=(flip_count, 1)) == 0, foreground, background)
    variants["salt_pepper_1pct"] = noisy.astype(np.uint8)

    if ndimage is not None:
        structure = np.ones((3, 3), dtype=bool)
        variants["dilate_1"] = compose_two_tone(ndimage.binary_dilation(selection.mask, structure=structure), foreground, background)
    else:  # pragma: no cover - exercised only without scipy
        variants["dilate_1"] = compose_two_tone(selection.mask, foreground, background)
    return variants
