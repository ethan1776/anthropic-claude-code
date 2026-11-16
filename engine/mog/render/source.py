"""Frame source with smooth in-between frames (for slow motion and motion blur).

Slow motion shows source moments *between* recorded frames. Repeating the nearest frame
stutters, so we synthesise the in-between frame with optical flow:

  1. Estimate how every pixel moves from frame k to k+1 and back (OpenCV DIS optical flow;
     computed once per frame pair and cached, since slow-mo reuses a pair many times).
  2. For a moment a (0..1) between them, warp frame k forward by a and frame k+1 backward
     by (1 - a) (Super-SloMo-style flow approximation), and blend.

Quality: "fast" = flow at <=480 px (good for previews), "high" = flow at <=960 px with a
finer search. Both run on CPU. (RIFE on a GPU would be sharper still; this machine has no
GPU, so it isn't wired in.)
"""

from __future__ import annotations

from collections import OrderedDict
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from mog.ffmpeg.frames import FrameReader

FLOW_MAX_DIM = {"fast": 480, "high": 960}
FLOW_PRESET = {"fast": cv2.DISOPTICAL_FLOW_PRESET_FAST, "high": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM}


class FrameSource:
    def __init__(self, path: Path, fps: Fraction, frames: int, start_frame: int, width: int, height: int,
                 size: tuple[int, int], quality: str = "fast"):
        self.fps = float(fps)
        self.n_frames = frames
        # Start a few frames early: motion blur looks slightly backwards in time
        self.reader = FrameReader(path, fps, max(0, start_frame - 3), width, height, out_size=size)
        self.w, self.h = size
        self.quality = quality if quality in FLOW_MAX_DIM else "fast"
        self._frames: OrderedDict[int, np.ndarray] = OrderedDict()
        self._flows: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self._dis = cv2.DISOpticalFlow_create(FLOW_PRESET[self.quality])
        scale = min(1.0, FLOW_MAX_DIM[self.quality] / max(self.w, self.h))
        self._fw, self._fh = max(8, int(self.w * scale)), max(8, int(self.h * scale))
        gx, gy = np.meshgrid(np.arange(self.w, dtype=np.float32), np.arange(self.h, dtype=np.float32))
        self._gx, self._gy = gx, gy

    # -------------------------------------------------------------- whole frames
    def frame(self, k: int) -> np.ndarray:
        k = int(min(max(k, 0), self.n_frames - 1))
        if k in self._frames:
            return self._frames[k]
        f = self.reader.frame_at(k)
        self._frames[k] = f
        while len(self._frames) > 8:
            self._frames.popitem(last=False)
        return f

    # -------------------------------------------------------------- in-between frames
    def _flow(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        if k in self._flows:
            return self._flows[k]
        a = cv2.cvtColor(cv2.resize(self.frame(k), (self._fw, self._fh), interpolation=cv2.INTER_AREA),
                         cv2.COLOR_BGR2GRAY)
        b = cv2.cvtColor(cv2.resize(self.frame(k + 1), (self._fw, self._fh), interpolation=cv2.INTER_AREA),
                         cv2.COLOR_BGR2GRAY)
        f01 = self._dis.calc(a, b, None)
        f10 = self._dis.calc(b, a, None)
        sx, sy = self.w / self._fw, self.h / self._fh

        def up(f):
            f = cv2.resize(f, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
            f[..., 0] *= sx
            f[..., 1] *= sy
            return f

        self._flows[k] = (up(f01), up(f10))
        while len(self._flows) > 2:
            self._flows.popitem(last=False)
        return self._flows[k]

    def sample(self, t: float, smooth: bool = True) -> np.ndarray:
        """The picture at source time t (seconds). smooth=False: the frame showing at t."""
        pos = t * self.fps
        k = int(np.floor(pos + 1e-6))
        a = pos - k
        if not smooth or a < 0.02 or k + 1 >= self.n_frames:
            return self.frame(k)
        if a > 0.98:
            return self.frame(k + 1)
        f01, f10 = self._flow(k)
        # Flow from the in-between moment back to each real frame (Super SloMo approximation)
        a32 = np.float32(a)
        ft0 = -(1 - a32) * a32 * f01 + a32 * a32 * f10
        ft1 = (1 - a32) * (1 - a32) * f01 - a32 * (1 - a32) * f10
        w0 = cv2.remap(self.frame(k), self._gx + ft0[..., 0], self._gy + ft0[..., 1],
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        w1 = cv2.remap(self.frame(k + 1), self._gx + ft1[..., 0], self._gy + ft1[..., 1],
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return cv2.addWeighted(w0, 1 - a, w1, a, 0)

    def blurred(self, t: float, span: float, strength: float) -> np.ndarray:
        """Motion blur: average of in-between pictures spread over `span` source seconds
        centred on t (a virtual camera shutter), mixed with the sharp frame by strength."""
        n = int(np.clip(np.ceil(span * self.fps * 3), 3, 12))
        acc = np.zeros((self.h, self.w, 3), np.float32)
        for u in np.linspace(-span / 2, span / 2, n):
            acc += self.sample(max(t + u, 0.0), smooth=True)
        blur = acc / n
        sharp = self.sample(t, smooth=True).astype(np.float32)
        return np.clip(sharp + (blur - sharp) * strength + 0.5, 0, 255).astype(np.uint8)

    def close(self) -> None:
        self.reader.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
