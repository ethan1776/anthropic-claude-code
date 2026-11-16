"""Drop detection.

"The drop" = the beat where bass energy jumps the most compared with the bars just
before it. We score every beat (not only detected downbeats, which may be wrong; a
confident drop is then used to *fix* the downbeats, since drops land on beat 1):

    score(d) = mean(energy over the 2 bars after d) - mean(energy over the 2 bars before d)

where energy mixes the 40-150 Hz band (70%) with full-band loudness (30%). The best
downbeat wins. Confidence combines
  * contrast   - how much quieter the build-up was (0 = no change, 1 = silence -> full)
  * standout   - how far the winner is above a typical bar line (repeated drops are fine;
                 among near-equal candidates we take the first one)
"""

from __future__ import annotations

import numpy as np

from mog.audio.models import Drop

WINDOW_BARS = 2


def detect_drop(
    candidates: np.ndarray, bar_len: float, low: np.ndarray, full: np.ndarray, fps: float,
    low_raw: np.ndarray | None = None,
) -> Drop | None:
    """candidates: times the drop may land on (the beat grid).
    low/full: smoothed, normalized energy curves. low_raw: unsmoothed bass energy, used to
    pin the drop to the exact beat (the smoothed curves are too blurry for that)."""
    if len(candidates) < 3:
        return None
    mix = 0.7 * low + 0.3 * full
    win = int(WINDOW_BARS * bar_len * fps)

    def mean_range(a: int, b: int) -> float:
        a, b = max(a, 0), min(b, len(mix))
        return float(mix[a:b].mean()) if b > a else 0.0

    cands = []
    for d in candidates:
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

    # Repeated drops: if an EARLIER candidate at least 4 bars away scores nearly as high,
    # take the first drop of the song (then the best beat right around it).
    early = [c for c in cands if c[0] >= 0.92 * best and c[1] <= t - 4 * bar_len]
    if early:
        first = early[0][1]
        best, t, before, after = max((c for c in cands if abs(c[1] - first) <= bar_len), key=lambda c: c[0])

    if low_raw is not None:
        t = _refine(t, candidates, bar_len / 4, low_raw, fps)

    contrast = (after - before) / max(after, 1e-6)
    # Standout: how far the drop is above a *typical* bar line (75th percentile of all
    # candidates). Songs often repeat their drop 2-3 times; that shouldn't lower confidence,
    # and with ~30+ bar lines a few repeats don't move the 75th percentile.
    typical = float(np.percentile(np.maximum(scores, 0.0), 75))
    standout = 1.0 - typical / best
    conf = float(np.clip(0.5 * contrast + 0.5 * standout, 0.0, 1.0))
    return Drop(t=round(t, 4), confidence=round(conf, 3))


def _refine(t: float, candidates: np.ndarray, beat: float, low_raw: np.ndarray, fps: float) -> float:
    """Pick the beat within +-2 beats of t with the sharpest bass jump: unsmoothed bass level
    in the beat after minus the beat before."""
    w = max(1, int(beat * fps))
    best_t, best_s = t, -np.inf
    for c in candidates[np.abs(candidates - t) <= 2.01 * beat]:
        f = int(round(c * fps))
        if f - w < 0 or f + w > len(low_raw):
            continue
        s = float(low_raw[f:f + w].mean() - low_raw[f - w:f].mean())
        if s > best_s:
            best_t, best_s = float(c), s
    return best_t
