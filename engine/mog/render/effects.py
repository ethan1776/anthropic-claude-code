"""Effect envelopes: for each output frame, how much of each effect applies.

Everything is computed from the timeline alone and on whole output frames, so an effect
at segment time t starts exactly on output frame round((timeline_in + t) * fps): that is
what keeps flashes and punches within one frame of the beat.

    zoom punch : instant scale-in to `strength`, eased back out over `decay` (ease-out cubic)
    shake      : smooth pseudo-random offset + small rotation, decaying over `decay`
    flash      : full strength on the beat frame, fading out over `frames`
    rgb split  : red/blue channels pushed apart, closing over `frames`
    motion blur: on during [t_in, t_out]
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field

import numpy as np

from mog.timeline.models import Flash, MotionBlur, RgbSplit, Segment, Shake, ZoomPunch

REF_WIDTH = 1080.0  # pixel amounts in the timeline are given at 1080 px width


@dataclass
class FrameFx:
    zoom: float = 1.0
    dx: float = 0.0          # shake offset, output pixels
    dy: float = 0.0
    angle: float = 0.0       # shake rotation, degrees
    flash: float = 0.0       # 0..1 blend towards flash_color
    flash_color: tuple[int, int, int] = (255, 255, 255)  # BGR
    rgb_split: float = 0.0   # output pixels
    blur: float = 0.0        # motion blur strength 0..1
    anchor_center: bool = False
    notes: list[str] = field(default_factory=list)


def _ease_out_cubic(u: float) -> float:
    return 1 - (1 - u) ** 3


def _hex_bgr(h: str) -> tuple[int, int, int]:
    return int(h[5:7], 16), int(h[3:5], 16), int(h[1:3], 16)


class _ShakeNoise:
    """Smooth 'random' wobble: three sines per axis at unrelated frequencies with seeded phases.
    Deterministic: the same effect always shakes the same way."""

    def __init__(self, seed: int):
        rng = np.random.default_rng(seed)
        self.freq = np.array([11.3, 17.9, 23.7])
        self.amp = np.array([0.55, 0.3, 0.15])
        self.px, self.py, self.pr = rng.uniform(0, 2 * math.pi, (3, 3))

    def __call__(self, t: float) -> tuple[float, float, float]:
        w = 2 * math.pi * self.freq * t
        return (float(np.sum(self.amp * np.sin(w + self.px))),
                float(np.sum(self.amp * np.sin(w * 0.93 + self.py))),
                float(np.sum(self.amp * np.sin(w * 0.71 + self.pr))))


def segment_effects(seg: Segment, frames: np.ndarray, fps: int, out_w: int) -> list[FrameFx]:
    """FrameFx for each output frame index in `frames` (absolute indices)."""
    px = out_w / REF_WIDTH
    out = [FrameFx() for _ in frames]
    for idx, e in enumerate(seg.effects):
        if isinstance(e, MotionBlur):
            a = int(round((seg.timeline_in + e.t_in) * fps))
            b = int(round((seg.timeline_in + e.t_out) * fps))
            for fx, n in zip(out, frames):
                if a <= n < b:
                    fx.blur = max(fx.blur, e.strength)
            continue
        start = int(round((seg.timeline_in + e.t) * fps))   # the beat frame
        if isinstance(e, ZoomPunch):
            dur = max(1, int(round(e.decay * fps)))
            for fx, n in zip(out, frames):
                k = n - start
                if 0 <= k < dur:
                    fx.zoom *= 1 + (e.strength - 1) * (1 - _ease_out_cubic(k / dur))
                    fx.anchor_center = fx.anchor_center or e.anchor == "center"
        elif isinstance(e, Shake):
            dur = max(1, int(round(e.decay * fps)))
            # zlib.crc32, not hash(): Python randomises hash() per process, which would
            # make every render shake differently
            noise = _ShakeNoise(zlib.crc32(f"{seg.id}|{idx}|{e.t:.4f}".encode()))
            for fx, n in zip(out, frames):
                k = n - start
                if 0 <= k < dur:
                    env = (1 - k / dur) ** 2
                    nx, ny, nr = noise(k / fps)
                    amp = e.intensity * px * env
                    fx.dx += amp * nx
                    fx.dy += amp * ny
                    fx.angle += 1.5 * (e.intensity / 18.0) * env * nr
        elif isinstance(e, Flash):
            color = _hex_bgr(e.color)
            for fx, n in zip(out, frames):
                k = n - start
                if 0 <= k < e.frames:
                    a = e.strength * (1 - k / e.frames)
                    if a > fx.flash:
                        fx.flash, fx.flash_color = a, color
        elif isinstance(e, RgbSplit):
            for fx, n in zip(out, frames):
                k = n - start
                if 0 <= k < e.frames:
                    fx.rgb_split = max(fx.rgb_split, e.offset * px * (1 - k / e.frames))
    return out


# ------------------------------------------------------------------ per-frame pixel ops

def apply_flash(frame: np.ndarray, a: float, color: tuple[int, int, int]) -> np.ndarray:
    import cv2

    if a <= 0:
        return frame
    solid = np.empty_like(frame)
    solid[:] = color
    return cv2.addWeighted(frame, 1 - a, solid, a, 0)


def apply_rgb_split(frame: np.ndarray, offset: float) -> np.ndarray:
    """Shift red right and blue left by `offset` px (edges repeat, so nothing goes black)."""
    o = int(round(offset))
    if o <= 0:
        return frame
    out = frame.copy()
    b, r = frame[:, :, 0], frame[:, :, 2]
    out[:, :-o, 0] = b[:, o:]            # blue moves left
    out[:, -o:, 0] = b[:, -1:]
    out[:, o:, 2] = r[:, :-o]            # red moves right
    out[:, :o, 2] = r[:, :1]
    return out
