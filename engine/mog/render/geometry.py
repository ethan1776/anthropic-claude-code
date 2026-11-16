"""Where each output pixel comes from in the source frame.

Every motion effect (9:16 crop, face-follow, zoom punch, shake) is a move/scale/rotate of
the source image. We combine them into ONE affine transform per frame and resample once,
so quality is lost only once and the maths can guarantee no black edges:

    the (possibly rotated) visible window is scaled up just enough to fit inside the
    source, then its centre is clamped so it never pokes outside.

Coordinates: (cx, cy) is the window centre as a fraction of the source frame.
"""

from __future__ import annotations

import math

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
                  offset_px: tuple[float, float] = (0.0, 0.0), angle_deg: float = 0.0) -> np.ndarray:
    """2x3 matrix mapping source pixels to output pixels.

    zoom 1 = the smallest scale at which the (unrotated) source fills the output.
    offset_px: shake offset in OUTPUT pixels. angle_deg: rotation (shake).
    The result never samples outside the source.
    """
    cover = max(out_w / src_w, out_h / src_h)
    s = cover * max(zoom, 1.0)
    th = math.radians(angle_deg)
    c, si = abs(math.cos(th)), abs(math.sin(th))
    # Half-size, in output pixels, of the axis-aligned box around the rotated output frame
    bw = (out_w * c + out_h * si) / 2
    bh = (out_w * si + out_h * c) / 2
    # Overscale: grow s until that box fits in the source
    s = max(s, 2 * bw / src_w, 2 * bh / src_h)
    hx, hy = bw / s, bh / s  # box half-size in source pixels
    # Shake moves the picture by offset_px on screen => the window moves the other way
    ox = (offset_px[0] * math.cos(th) + offset_px[1] * math.sin(th)) / s
    oy = (-offset_px[0] * math.sin(th) + offset_px[1] * math.cos(th)) / s
    x = float(np.clip(cx * src_w - ox, hx, src_w - hx))
    y = float(np.clip(cy * src_h - oy, hy, src_h - hy))
    # dst = s * R * (src - centre) + out_centre, in edge coordinates; convert to pixel centres
    cos_t, sin_t = math.cos(th), math.sin(th)
    a = np.array([[s * cos_t, -s * sin_t], [s * sin_t, s * cos_t]])
    t = a @ np.array([0.5 - x, 0.5 - y]) + np.array([out_w / 2 - 0.5, out_h / 2 - 0.5])
    return np.hstack([a, t[:, None]])


def apply_affine(src: np.ndarray, m: np.ndarray, out_w: int, out_h: int, quality: str = "high") -> np.ndarray:
    h, w = src.shape[:2]
    if w == out_w and h == out_h and np.allclose(m, [[1, 0, 0], [0, 1, 0]]):
        return src
    flags = cv2.INTER_CUBIC if quality == "high" else cv2.INTER_LINEAR
    # REPLICATE is a safety net only; the maths above keeps the window inside the source.
    return cv2.warpAffine(src, m, (out_w, out_h), flags=flags, borderMode=cv2.BORDER_REPLICATE)
