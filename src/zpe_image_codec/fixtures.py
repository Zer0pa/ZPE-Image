from __future__ import annotations

import numpy as np

try:
    from scipy import ndimage
except Exception:  # pragma: no cover - dependency guard
    ndimage = None  # type: ignore[assignment]


CANVAS = 64


def _blank(background: int = 255) -> np.ndarray:
    return np.full((CANVAS, CANVAS, 3), background, dtype=np.uint8)


def _paint_disk(image: np.ndarray, cx: int, cy: int, radius: int, color: tuple[int, int, int]) -> None:
    for y in range(cy - radius, cy + radius + 1):
        for x in range(cx - radius, cx + radius + 1):
            if 0 <= y < image.shape[0] and 0 <= x < image.shape[1] and (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                image[y, x] = np.array(color, dtype=np.uint8)


def _draw_line(
    image: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    width: int = 1,
) -> None:
    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    error = dx + dy
    while True:
        _paint_disk(image, x0, y0, radius=max(0, width // 2), color=color)
        if x0 == x1 and y0 == y1:
            break
        doubled = 2 * error
        if doubled >= dy:
            error += dy
            x0 += sx
        if doubled <= dx:
            error += dx
            y0 += sy


def _polyline(
    image: np.ndarray,
    points: list[tuple[int, int]],
    color: tuple[int, int, int] = (0, 0, 0),
    width: int = 1,
) -> None:
    for start, end in zip(points, points[1:]):
        _draw_line(image, start, end, color=color, width=width)


def _circle_outline(
    image: np.ndarray,
    center: tuple[int, int],
    radius: int,
    color: tuple[int, int, int] = (0, 0, 0),
    width: int = 1,
) -> None:
    points: list[tuple[int, int]] = []
    for idx in range(72):
        angle = (2.0 * np.pi * idx) / 72.0
        x = int(round(center[0] + radius * np.cos(angle)))
        y = int(round(center[1] + radius * np.sin(angle)))
        points.append((x, y))
    points.append(points[0])
    _polyline(image, points, color=color, width=width)


def _fill_circle(image: np.ndarray, center: tuple[int, int], radius: int, color: tuple[int, int, int]) -> None:
    _paint_disk(image, center[0], center[1], radius, color)


def _checkerboard(tile: int = 4) -> np.ndarray:
    ys, xs = np.indices((CANVAS, CANVAS))
    board = ((ys // tile + xs // tile) % 2) * 255
    return np.repeat(board.astype(np.uint8)[:, :, None], 3, axis=2)


def _gradient() -> np.ndarray:
    image = np.zeros((CANVAS, CANVAS, 3), dtype=np.uint8)
    x_ramp = np.tile(np.linspace(20, 230, CANVAS, dtype=np.uint8), (CANVAS, 1))
    y_ramp = np.tile(np.linspace(240, 30, CANVAS, dtype=np.uint8), (CANVAS, 1)).T
    image[:, :, 0] = x_ramp
    image[:, :, 1] = y_ramp
    image[:, :, 2] = ((x_ramp.astype(np.uint16) + y_ramp.astype(np.uint16)) // 2).astype(np.uint8)
    return image


def _texture_blobs() -> np.ndarray:
    rng = np.random.default_rng(5)
    image = rng.integers(0, 255, size=(CANVAS, CANVAS, 3), dtype=np.uint8)
    if ndimage is not None:
        image = ndimage.gaussian_filter(image.astype(np.float32), sigma=(2.2, 2.2, 0))
        image = np.clip(image, 0, 255).astype(np.uint8)
    for _ in range(6):
        center = int(rng.integers(8, CANVAS - 8)), int(rng.integers(8, CANVAS - 8))
        radius = int(rng.integers(5, 12))
        color = tuple(int(value) for value in rng.integers(20, 235, size=3))
        _fill_circle(image, center, radius, color)
    return image


def _color_field_noise() -> np.ndarray:
    rng = np.random.default_rng(11)
    image = _blank(0)
    palette = [(220, 40, 40), (40, 180, 220), (230, 220, 60), (30, 30, 30)]
    stripe = CANVAS // len(palette)
    for idx, color in enumerate(palette):
        image[:, idx * stripe : (idx + 1) * stripe] = np.array(color, dtype=np.uint8)
    noise = rng.integers(-30, 31, size=image.shape)
    return np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _glyph_a() -> np.ndarray:
    image = _blank()
    _polyline(image, [(10, 55), (31, 8), (54, 55)], width=2)
    _polyline(image, [(18, 36), (45, 36)], width=2)
    return image


def _fork_tree() -> np.ndarray:
    image = _blank()
    _polyline(image, [(32, 56), (32, 28), (20, 14)], width=2)
    _polyline(image, [(32, 28), (46, 12)], width=2)
    _polyline(image, [(32, 40), (14, 30)], width=2)
    _polyline(image, [(32, 40), (50, 34)], width=2)
    return image


def _loop_spine() -> np.ndarray:
    image = _blank()
    _circle_outline(image, (24, 26), 14, width=2)
    _polyline(image, [(24, 40), (24, 58), (42, 58)], width=2)
    return image


def _maze_turns() -> np.ndarray:
    image = _blank()
    _polyline(image, [(8, 8), (8, 52), (24, 52), (24, 20), (40, 20), (40, 44), (56, 44)], width=2)
    _polyline(image, [(16, 16), (16, 32), (32, 32), (32, 12), (52, 12)], width=2)
    return image


def _serpentine() -> np.ndarray:
    image = _blank()
    _polyline(image, [(6, 20), (18, 8), (30, 20), (42, 8), (54, 20), (42, 32), (30, 44), (18, 32), (6, 44)], width=2)
    return image


def _filled_disk_with_spine() -> np.ndarray:
    image = _blank()
    _fill_circle(image, (24, 28), 14, (0, 0, 0))
    _polyline(image, [(24, 28), (52, 52)], width=2)
    return image


def _gradient_with_symbol() -> np.ndarray:
    image = _gradient()
    _polyline(image, [(10, 54), (30, 10), (54, 50)], color=(0, 0, 0), width=2)
    _polyline(image, [(18, 36), (44, 36)], color=(250, 250, 250), width=2)
    return image


def _stroke_over_texture() -> np.ndarray:
    image = _texture_blobs()
    _polyline(image, [(8, 50), (20, 14), (36, 46), (54, 18)], color=(255, 255, 255), width=2)
    return image


def build_cases() -> list[dict[str, object]]:
    return [
        {"name": "glyph_a", "bucket": "positive", "image": _glyph_a(), "notes": "letterform with strong figure-ground and thin strokes"},
        {"name": "fork_tree", "bucket": "positive", "image": _fork_tree(), "notes": "branch-heavy local skeleton"},
        {"name": "loop_spine", "bucket": "positive", "image": _loop_spine(), "notes": "closed loop plus stem"},
        {"name": "maze_turns", "bucket": "positive", "image": _maze_turns(), "notes": "orthogonal turn-heavy routing"},
        {"name": "serpentine", "bucket": "positive", "image": _serpentine(), "notes": "diagonal turn-rich single figure"},
        {"name": "filled_disk_with_spine", "bucket": "mixed", "image": _filled_disk_with_spine(), "notes": "filled mass plus stroke"},
        {"name": "gradient_with_symbol", "bucket": "mixed", "image": _gradient_with_symbol(), "notes": "stroke over non-binary background"},
        {"name": "stroke_over_texture", "bucket": "mixed", "image": _stroke_over_texture(), "notes": "figure mixed with textured nuisance"},
        {"name": "smooth_gradient", "bucket": "negative_natural", "image": _gradient(), "notes": "smooth natural-style color field"},
        {"name": "texture_blobs", "bucket": "negative_natural", "image": _texture_blobs(), "notes": "blob texture without sparse figure"},
        {"name": "checkerboard_texture", "bucket": "negative_natural", "image": _checkerboard(), "notes": "high-frequency texture"},
        {"name": "color_field_noise", "bucket": "negative_natural", "image": _color_field_noise(), "notes": "multitone block field with nuisance noise"},
    ]


def case_by_name(name: str) -> dict[str, object]:
    for case in build_cases():
        if case["name"] == name:
            return case
    raise KeyError(name)
