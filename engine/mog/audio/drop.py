"""Drop detection.

"The drop" = the downbeat where bass energy jumps the most compared with the bars just
before it. We score every downbeat:

    score(d) = mean(energy over the 2 bars after d) - mean(energy over the 2 bars before d)

where energy mixes the 40-150 Hz band (70%) with full-band loudness (30%). The best
downbeat wins. Confidence combines
  * contrast   - how much quieter the build-up was (0 = no change, 1 = silence -> full)
  * uniqueness - how far ahead the winner is of the next-best candidate at least 4 bars away
                 (a song with three identical drops is still fine; we take the first strong one)
"""

from __future__ import annotations

import numpy as np

from mog.audio.models import Drop

WINDOW_BARS = 2


def detect_drop(
    downbeats: np.ndarray, bar_len: float, low: np.ndarray, full: np.ndarray, fps: float
) -> Drop | None:
    if len(downbeats) < 3:
        return None
    mix = 0.7 * low + 0.3 * full
    win = int(WINDOW_BARS * bar_len * fps)

    def mean_range(a: int, b: int) -> float:
        a, b = max(a, 0), min(b, len(mix))
        return float(mix[a:b].mean()) if b > a else 0.0

    cands = []
    for d in downbeats:
        f = int(round(d * fps))
        if f < int(0.75 * bar_len * fps) or f + win // 2 >= len(mix):
            continue  # need at least ~1 bar of "before" and 1 bar of "after"
        before = mean_range(f - win, f)
        after = mean_range(f, f + win)
        cands.append((after - before, float(d), before, after))
    if not cands:
        return None
    scores = np.array([c[0] for c in cands])
    best_i = int(np.argmax(scores))
    best, t, before, after = cands[best_i]
    if best <= 0:
        return Drop(t=round(t, 4), confidence=0.0)

    # Among near-equal winners (repeated drops), prefer the earliest one.
    for c in cands:
        if c[0] >= 0.92 * best:
            best, t, before, after = c
            break

    contrast = (after - before) / max(after, 1e-6)
    others = [c[0] for c in cands if abs(c[1] - t) >= 4 * bar_len and c[1] < t]
    others += [c[0] for c in cands if abs(c[1] - t) >= 4 * bar_len and c[1] > t and c[0] < 0.92 * best]
    second = max(others) if others else 0.0
    uniqueness = 1.0 - max(second, 0.0) / best
    conf = float(np.clip(0.6 * contrast + 0.4 * uniqueness, 0.0, 1.0))
    return Drop(t=round(t, 4), confidence=round(conf, 3))
