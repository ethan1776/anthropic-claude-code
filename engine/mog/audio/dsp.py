"""Small DSP helpers shared by the analysis modules."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfiltfilt

SR = 22050          # analysis sample rate
HOP = 128           # analysis hop in samples -> 5.8 ms time resolution


def bandpass(y: np.ndarray, sr: int, lo: float | None, hi: float | None, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth filter (sosfiltfilt => no time delay, so onset times stay exact)."""
    nyq = sr / 2
    if lo and hi:
        sos = butter(order, [lo / nyq, min(hi / nyq, 0.99)], btype="band", output="sos")
    elif lo:
        sos = butter(order, lo / nyq, btype="high", output="sos")
    elif hi:
        sos = butter(order, hi / nyq, btype="low", output="sos")
    else:
        return y
    return sosfiltfilt(sos, y).astype(np.float32)


def frame_rms(y: np.ndarray, hop: int = HOP, win: int | None = None) -> np.ndarray:
    """RMS per hop, using a centered window (frame i is centered on sample i*hop)."""
    win = win or 2 * hop
    pad = win // 2
    yp = np.pad(y.astype(np.float64) ** 2, (pad, pad))
    c = np.concatenate([[0.0], np.cumsum(yp)])
    n = len(y) // hop + 1
    starts = np.arange(n) * hop
    ends = np.minimum(starts + win, len(yp))
    return np.sqrt((c[ends] - c[starts]) / win).astype(np.float32)


def smooth(x: np.ndarray, seconds: float, frame_rate: float) -> np.ndarray:
    size = max(1, int(round(seconds * frame_rate)))
    return uniform_filter1d(x, size=size, mode="nearest")


def normalize(x: np.ndarray, pct: float = 100.0) -> np.ndarray:
    ref = np.percentile(x, pct) if pct < 100 else np.max(x)
    if ref <= 1e-12:
        return np.zeros_like(x)
    return np.clip(x / ref, 0.0, 1.0)


def resample_curve(x: np.ndarray, src_rate: float, dst_hop: float) -> list[float]:
    """Resample a frame-rate curve to a coarser uniform hop (for compact JSON)."""
    duration = len(x) / src_rate
    t = np.arange(0.0, duration, dst_hop)
    vals = np.interp(t * src_rate, np.arange(len(x)), x)
    return [round(float(v), 4) for v in vals]
