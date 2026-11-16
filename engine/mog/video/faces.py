"""Face detection with OpenCV's YuNet (a small, fast CNN; ~5 ms per frame on CPU).

Chosen over MediaPipe (the spec's suggestion) because MediaPipe needs system OpenGL/EGL
libraries even on CPU, while YuNet runs on the OpenCV we already ship. YuNet returns a box,
5 landmarks (eyes, nose tip, mouth corners) and a confidence per face.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from mog.errors import MogError
from mog.paths import MODELS

MODEL = MODELS / "face_detection_yunet_2023mar.onnx"
DETECT_MAX_DIM = 480


@dataclass
class Face:
    cx: float      # box centre, fraction of frame width
    cy: float      # fraction of frame height
    w: float       # box width, fraction of frame width
    h: float       # fraction of frame height
    score: float
    yaw: float     # ~0 = looking at camera, +-1 = full profile (from landmarks)

    @property
    def area(self) -> float:
        return self.w * self.h


class FaceDetector:
    def __init__(self, min_score: float = 0.7):
        if not MODEL.exists():
            raise MogError(f"Face model missing: {MODEL}")
        self._det = cv2.FaceDetectorYN.create(str(MODEL), "", (320, 320), min_score, 0.3, 50)
        self._size: tuple[int, int] | None = None

    def detect(self, bgr: np.ndarray) -> list[Face]:
        h, w = bgr.shape[:2]
        s = min(1.0, DETECT_MAX_DIM / max(w, h))
        img = cv2.resize(bgr, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else bgr
        ih, iw = img.shape[:2]
        if self._size != (iw, ih):
            self._det.setInputSize((iw, ih))
            self._size = (iw, ih)
        _, faces = self._det.detect(img)
        out = []
        if faces is None:
            return out
        for f in faces:
            x, y, fw, fh = f[0:4]
            re_x, le_x, nose_x = f[4], f[6], f[8]  # right eye, left eye, nose tip (x)
            eye_mid = (re_x + le_x) / 2
            eye_dist = max(abs(le_x - re_x), 1e-3)
            yaw = float(np.clip((nose_x - eye_mid) / eye_dist * 2.0, -1, 1))
            out.append(Face(cx=float((x + fw / 2) / iw), cy=float((y + fh / 2) / ih), w=float(fw / iw),
                            h=float(fh / ih), score=float(f[14]), yaw=yaw))
        return out
