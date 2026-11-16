"""Frame source with smooth in-between frames (for slow motion and motion blur).

Slow motion shows source moments *between* recorded frames. Repeating the nearest frame
stutters, so we synthesise the in-between frame with optical flow:

  1. Estimate how every pixel moves from frame k to k+1 and back (OpenCV DIS optical flow;
     computed once per frame pair and cached, since slow-mo reuses a pair many times).
  2. For a moment a (0..1) between them, warp frame k forward by a and frame k+1 backward
     by (1 - a) (Super-SloMo-style flow approximation), and blend.

Quality: "fast" = flow at <=480 px (good for previews), "high" = flow at <=720 px with a
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

FLOW_MAX_DIM = {"fast": 480, "high": 720}
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
        self._grid = np.dstack([gx, gy])  # (h, w, 2) pixel coordinates, for remap

    # -------------------------------------------------------------- whole frames
    def frame(self, k: int) -> np.ndarray:
        k = int(min(max(k, 0), self.n_frames - 1))
        if k in self._frames:
            return self._frames[k]
        f = self.reader.frame_at(k)
        self._frames[k] = f
        while len(self._frames) > 16:
            self._frames.popitem(last=False)
        return f

    # -------------------------------------------------------------- in-between frames
    def _flow(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Forward/backward flow between frames k and k+1, at the (small) flow resolution,
        already scaled to full-resolution pixel units."""
        if k in self._flows:
            return self._flows[k]
        a = cv2.cvtColor(cv2.resize(self.frame(k), (self._fw, self._fh), interpolation=cv2.INTER_AREA),
                         cv2.COLOR_BGR2GRAY)
        b = cv2.cvtColor(cv2.resize(self.frame(k + 1), (self._fw, self._fh), interpolation=cv2.INTER_AREA),
                         cv2.COLOR_BGR2GRAY)
        scale = np.array([self.w / self._fw, self.h / self._fh], np.float32)
        f01 = self._dis.calc(a, b, None) * scale
        f10 = self._dis.calc(b, a, None) * scale
        self._flows[k] = (f01, f10)
        while len(self._flows) > 3:
            self._flows.popitem(last=False)
        return self._flows[k]

    def _warp(self, img: np.ndarray, flow_small: np.ndarray) -> np.ndarray:
        """Sample img at (x, y) + flow, with the flow given at the small resolution."""
        up = cv2.resize(flow_small, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        m1, m2 = cv2.convertMaps(cv2.add(self._grid, up), None, cv2.CV_16SC2)  # fixed point: faster remap
        return cv2.remap(img, m1, m2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

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
        # Flow from the in-between moment back to each real frame (Super SloMo approximation),
        # combined at the small flow resolution, then upscaled once per warp
        ft0 = cv2.addWeighted(f01, -(1 - a) * a, f10, a * a, 0)
        ft1 = cv2.addWeighted(f01, (1 - a) * (1 - a), f10, -a * (1 - a), 0)
        w0 = self._warp(self.frame(k), ft0)
        w1 = self._warp(self.frame(k + 1), ft1)
        return cv2.addWeighted(w0, 1 - a, w1, a, 0)

    def blurred(self, t: float, span: float, strength: float) -> np.ndarray:
        """Motion blur: average what a camera shutter open for `span` source-seconds around t
        would have seen, mixed with the sharp frame by `strength`.

        Fast sections (the usual case) already cover several recorded frames per output frame:
        we average those real frames (classic frame blending: cheap). Only when fewer than 3
        real frames fall inside the shutter do we add optical-flow in-between pictures."""
        a, b = t - span / 2, t + span / 2
        ks = list(range(max(int(np.ceil(a * self.fps - 1e-6)), 0),
                        min(int(np.floor(b * self.fps + 1e-6)), self.n_frames - 1) + 1))
        # Fetch in time order: the reader only moves forward
        if len(ks) >= 3:
            pics = [self.frame(k) for k in ks]
        else:
            pics = [self.sample(max(u, 0.0), smooth=True) for u in np.linspace(a, b, 4)]
        sharp = self.sample(t, smooth=True)
        acc = np.zeros((self.h, self.w, 3), np.float32)
        for pic in pics:
            cv2.accumulate(pic, acc)
        blur = acc * (1.0 / len(pics))
        return cv2.addWeighted(sharp.astype(np.float32), 1 - strength, blur, strength, 0.5,
                               dtype=cv2.CV_32F).astype(np.uint8)

    def close(self) -> None:
        self.reader.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
