"""Where each output pixel comes from in the source frame.

Every motion effect (9:16 crop, face-follow, zoom punch, shake) is a move/scale of the
source image. We combine them into ONE affine transform per frame and resample once, so
quality is lost only once and the maths can guarantee no black edges:

    the visible window in source pixels is always clamped to stay inside the frame.

Coordinates: (cx, cy) is the window centre as a fraction of the source frame.
"""

from __future__ import annotations

import cv2
import numpy as np

from mog.timeline.models import Crop


def crop_center(crop: Crop, t_local: float) -> tuple[float, float]:
    if crop.path:
        ts = [p[0] for p in crop.path]
        return (float(np.interp(t_local, ts, [p[1] for p in crop.path])),
                float(np.interp(t_local, ts, [p[2] for p in crop.path])))
    return crop.x, crop.y


def window_affine(src_w: int, src_h: int, out_w: int, out_h: int, cx: float, cy: float, zoom: float,
                  offset_px: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """2x3 matrix mapping source pixels to output pixels.

    scale = cover * zoom, where `cover` is the smallest scale at which the source fills the
    output (so zoom 1 never shows bars). offset_px (in OUTPUT pixels) is for shake; the
    window is clamped after applying it, so it can never reveal an edge.
    """
    cover = max(out_w / src_w, out_h / src_h)
    s = cover * max(zoom, 1.0)
    win_w, win_h = out_w / s, out_h / s
    # Window centre in source pixels, then clamp the window inside the source
    x = cx * src_w - offset_px[0] / s
    y = cy * src_h - offset_px[1] / s
    x = float(np.clip(x, win_w / 2, src_w - win_w / 2))
    y = float(np.clip(y, win_h / 2, src_h - win_h / 2))
    x0, y0 = x - win_w / 2, y - win_h / 2
    # Pixel centres: output pixel i covers source (x0 + (i + 0.5)/s - 0.5)
    tx = -s * x0 + 0.5 * s - 0.5
    ty = -s * y0 + 0.5 * s - 0.5
    return np.array([[s, 0.0, tx], [0.0, s, ty]], dtype=np.float64)


def apply_affine(src: np.ndarray, m: np.ndarray, out_w: int, out_h: int, quality: str = "high") -> np.ndarray:
    h, w = src.shape[:2]
    if w == out_w and h == out_h and np.allclose(m, [[1, 0, 0], [0, 1, 0]]):
        return src
    flags = cv2.INTER_CUBIC if quality == "high" else cv2.INTER_LINEAR
    # REPLICATE is a safety net only; the clamp above keeps the window inside the source.
    return cv2.warpAffine(src, m, (out_w, out_h), flags=flags, borderMode=cv2.BORDER_REPLICATE)
