"""Per-frame pixel effects done in Python/OpenCV (vectorised, multi-threaded C++ under the hood).

Measured on 1080x1920: doing vignette and glow here costs a few ms per frame, while as
FFmpeg filters they were the slowest part of the encode chain (the vignette filter is
single-threaded). Frames are BGR uint8.
"""

from __future__ import annotations

import cv2
import numpy as np


class Vignette:
    """Static darkening towards the corners: frame * mask, with the mask computed once."""

    def __init__(self, width: int, height: int, strength: float):
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        d = np.sqrt(((xx + 0.5) / width - 0.5) ** 2 + ((yy + 0.5) / height - 0.5) ** 2) / np.sqrt(0.5)
        t = np.clip((d - 0.3) / 0.7, 0, 1)
        fall = t * t * (3 - 2 * t)                     # smooth falloff, 1.0 at the corners
        mask = 1.0 - 0.85 * strength * fall
        # 3-channel float mask for cv2.multiply (scale 1 => frame * mask, rounded, saturated)
        self.mask = cv2.merge([mask, mask, mask]).astype(np.float32)

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        return cv2.multiply(frame, self.mask, dtype=cv2.CV_8U)


class Glow:
    """Bloom: take the highlights, blur them wide (at quarter resolution, which looks the
    same for a big blur and costs 1/16th), and screen-blend them back on top."""

    def __init__(self, width: int, height: int, amount: float, threshold: float = 0.55):
        self.w, self.h = width, height
        self.qw, self.qh = max(2, width // 4), max(2, height // 4)
        self.sigma = width * 0.018 / 4
        x = np.arange(256, dtype=np.float32) / 255
        hi = np.clip((x - threshold) / (1 - threshold), 0, 1)        # keep only the bright part
        self.curve = np.clip(hi * 255 * amount + 0.5, 0, 255).astype(np.uint8)

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        small = cv2.resize(frame, (self.qw, self.qh), interpolation=cv2.INTER_AREA)
        small = cv2.LUT(small, self.curve)
        small = cv2.GaussianBlur(small, (0, 0), self.sigma)
        glow = cv2.resize(small, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        # screen blend: 1 - (1 - a)(1 - b)  ==  a + b - a*b
        inv = cv2.multiply(cv2.bitwise_not(frame), cv2.bitwise_not(glow), scale=1 / 255)
        return cv2.bitwise_not(inv)


def exposure_table(stops: float) -> np.ndarray | None:
    """256-entry table that brightens/darkens by `stops`, done in (approximately) linear light
    so it behaves like a camera exposure change. Brightening rolls highlights off smoothly
    instead of clipping them to flat white. None for 0 stops."""
    if abs(stops) < 1e-6:
        return None
    x = np.arange(256, dtype=np.float64) / 255
    lin = x ** 2.2 * (2.0 ** stops)
    if stops > 0:
        knee = 0.6                                                # soft shoulder above 60%
        over = np.maximum(lin - knee, 0)
        lin = np.where(lin > knee, knee + (1 - knee) * (1 - np.exp(-over / (1 - knee))), lin)
    out = np.clip(lin, 0, 1) ** (1 / 2.2)
    return np.clip(out * 255 + 0.5, 0, 255).astype(np.uint8)
