"""Speed-curve maths: which source moment to show at each output moment.

A segment's `speed_curve` is a list of [t, speed] keyframes (segment-local t). Between two
keyframes the speed eases from one value to the next. Before the first / after the last
keyframe it holds.

Source time is the integral of speed:   src(t) = src_in + integral_0^t speed(u) du
(At speed 2 for 1 second, 2 seconds of footage go by; at speed 0.3, 0.3 seconds do.)
We integrate numerically on a fine grid (16 steps per output frame), which is exact
enough (<0.1 ms) and works for every easing shape.
"""

from __future__ import annotations

import numpy as np

from mog.timeline.models import Ease, Segment


def ease(u: np.ndarray, kind: Ease) -> np.ndarray:
    u = np.clip(u, 0.0, 1.0)
    if kind == "linear":
        return u
    if kind == "smoothstep":
        return u * u * (3 - 2 * u)
    if kind == "cubic":  # ease-in-out cubic: a sharper "snap" than smoothstep
        return np.where(u < 0.5, 4 * u**3, 1 - (-2 * u + 2) ** 3 / 2)
    raise ValueError(f"unknown ease {kind}")


def speed_at(curve: list[tuple[float, float]], t: np.ndarray, kind: Ease = "smoothstep") -> np.ndarray:
    t = np.asarray(t, dtype=float)
    ks = np.array([k[0] for k in curve], dtype=float)
    vs = np.array([k[1] for k in curve], dtype=float)
    if len(ks) == 1:
        return np.full_like(t, vs[0])
    i = np.clip(np.searchsorted(ks, t, side="right") - 1, 0, len(ks) - 2)
    t0, t1 = ks[i], ks[i + 1]
    span = np.where(t1 > t0, t1 - t0, 1.0)
    u = np.where(t1 > t0, (t - t0) / span, 1.0)
    out = vs[i] + (vs[i + 1] - vs[i]) * ease(u, kind)
    out = np.where(t <= ks[0], vs[0], out)
    return np.where(t >= ks[-1], vs[-1], out)


def source_offsets(curve: list[tuple[float, float]], times: np.ndarray, kind: Ease = "smoothstep",
                   step: float = 1 / 960) -> np.ndarray:
    """Seconds of source footage consumed from segment start up to each time in `times`."""
    times = np.asarray(times, dtype=float)
    if len(curve) == 1:
        return times * curve[0][1]
    end = float(times.max()) if times.size else 0.0
    grid = np.arange(0.0, end + 2 * step, step)
    v = speed_at(curve, grid, kind)
    cum = np.concatenate([[0.0], np.cumsum((v[1:] + v[:-1]) * 0.5 * step)])  # trapezoid rule
    return np.interp(times, grid, cum)


def segment_source_times(seg: Segment, fps: int) -> tuple[np.ndarray, np.ndarray]:
    """For each output frame of the segment: (output frame indices, source times in seconds).

    Frame n of the output covers [n/fps, (n+1)/fps). A segment owns the frames from
    round(timeline_in*fps) to round(timeline_out*fps), so a cut lands within half a frame
    of its exact time.
    """
    first = int(round(seg.timeline_in * fps))
    last = int(round(seg.timeline_out * fps))
    frames = np.arange(first, last)
    local = frames / fps - seg.timeline_in
    local = np.maximum(local, 0.0)
    return frames, seg.src_in + source_offsets(seg.speed_curve, local, seg.ease)


def footage_needed(seg: Segment) -> float:
    """Seconds of source footage the segment consumes (used to pick long-enough shots)."""
    return float(source_offsets(seg.speed_curve, np.array([seg.duration]), seg.ease)[0])
