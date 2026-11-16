"""Beat-grid cleanup: snap to the producer's tempo grid, keep real jumps.

Why: phonk is made in a DAW at one fixed tempo, so its true beats form a perfectly
straight grid. Beat trackers sometimes slip onto the off-beat or a 16th for a few bars
(found on real tracks: 14 beats slipped by 100 ms). Cuts in that stretch would land
off-beat.

But a straight grid is not always right. Audio from screen recordings of edits, DJ blends
or "sped up" versions can be *spliced*: at the cut, the beat jumps. There the tracker is
right and the straight grid is wrong.

So, per stretch of beats that disagrees with the straight grid, we let the drums decide:
whichever grid lands on more (and stronger) drum hits wins. Kept jumps are reported as
"grid breaks", and downbeats are chosen separately for each piece between breaks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mog.audio.beats import BEATS_PER_BAR
from mog.audio.models import Onset

ON_GRID_S = 0.025     # a beat within 25 ms of the straight grid counts as "on grid"
HIT_WINDOW_S = 0.020  # a drum hit within 20 ms of a beat counts as supporting it
KEEP_TRACKED_RATIO = 1.15  # an alternative grid must land on 15% more drum support to win


@dataclass
class Grid:
    beats: np.ndarray
    period: float
    breaks: list[float] = field(default_factory=list)  # times where the beat jumps
    # True when the grid shifted by half a beat lands on almost as many drums: the track's
    # drums sit on both the beats and the "and"s, so on-beat vs off-beat needs a human ear.
    offbeat_ambiguous: bool = False


def fit_constant_grid(beats: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Find the straight grid t0 + k*period that the MOST beats agree with (within 25 ms).

    "Most beats agree" (a vote) rather than least squares on all beats: if a third of the
    song is spliced half a beat later, least squares would land between the two grids and
    fit neither. Steps: try periods within +-0.5% of the median beat gap; for each, find the
    phase most beats share; keep the best; then refine with least squares on the agreeing
    beats only. Returns (period, t0, on-grid mask).
    """
    beats = np.asarray(beats, float)
    if len(beats) < 4:
        p = float(np.median(np.diff(beats))) if len(beats) > 1 else 0.5
        return p, float(beats[0]) if len(beats) else 0.0, np.ones(len(beats), bool)
    med = float(np.median(np.diff(beats)))
    best = (-1, med, float(beats[0]))
    for period in med * np.linspace(0.995, 1.005, 201):
        phase = np.mod(beats - beats[0], period)
        # Vote: count beats within ON_GRID_S of each candidate phase (circular)
        for cand in phase[:: max(1, len(phase) // 64)]:
            d = np.abs(phase - cand)
            d = np.minimum(d, period - d)
            votes = int(np.sum(d < ON_GRID_S))
            if votes > best[0]:
                best = (votes, float(period), float(beats[0] + cand))
    _, period, t0 = best
    mask = np.zeros(len(beats), bool)
    for _ in range(4):  # least-squares refinement on the agreeing beats
        k = np.round((beats - t0) / period)
        new = np.abs(beats - (t0 + k * period)) < ON_GRID_S
        if new.sum() < 4 or np.array_equal(new, mask):
            break
        mask = new
        A = np.vstack([k[mask], np.ones(mask.sum())]).T
        period, t0 = (float(v) for v in np.linalg.lstsq(A, beats[mask], rcond=None)[0])
    k = np.round((beats - t0) / period)
    mask = np.abs(beats - (t0 + k * period)) < ON_GRID_S
    return period, t0, mask


def _support(times: np.ndarray, onsets: list[Onset]) -> float:
    """Average drum support per beat: strength of the best hit within 20 ms (kick/snare x1.5)."""
    if len(times) == 0 or not onsets:
        return 0.0
    ot = np.array([o.t for o in onsets])
    w = np.array([o.strength * (1.5 if o.kind in ("kick", "snare") else 1.0) for o in onsets])
    total = 0.0
    for t in times:
        m = np.abs(ot - t) < HIT_WINDOW_S
        if m.any():
            total += float(w[m].max())
    return total / len(times)


def _straight(t0: float, period: float, a: float, b: float) -> np.ndarray:
    """Straight-grid beat times strictly inside (a, b)."""
    k0 = int(np.floor((a - t0) / period)) + 1
    t = t0 + period * np.arange(k0, k0 + int((b - a) / period) + 2)
    return t[(t > a + 0.5 * ON_GRID_S) & (t < b - 0.5 * ON_GRID_S)]


def regularize(beats: np.ndarray, onsets: list[Onset], depth: int = 0) -> Grid:
    beats = np.asarray(beats, float)
    if len(beats) < 8:
        return Grid(beats, float(np.median(np.diff(beats))) if len(beats) > 1 else 0.5)
    period, t0, on = fit_constant_grid(beats)

    # Off-beat check: trackers can lock onto the "and"s for a whole song. Compare the drum
    # support of the straight grid with the same grid shifted by half a beat.
    ambiguous = False
    if depth == 0:
        span = _straight(t0, period, beats[0] - period, beats[-1] + period)
        shifted = _straight(t0 + period / 2, period, beats[0] - period, beats[-1] + period)
        s0, s_half = _support(span, onsets), _support(shifted, onsets)
        if s_half > KEEP_TRACKED_RATIO * s0:
            t0 += period / 2
            on = np.zeros(len(beats), bool)  # every tracked beat is now "off grid"
        else:
            ambiguous = s_half > 0.85 * s0

    out: list[float] = []
    breaks: list[float] = []
    i, n = 0, len(beats)
    while i < n:
        if on[i]:
            # Between two on-grid beats, fill any beats the tracker skipped
            # (> 1.75 periods: a missing beat leaves a 2-period gap; a half-beat splice 1.5)
            if out and beats[i] - out[-1] > 1.75 * period:
                out.extend(_straight(t0, period, out[-1], beats[i]))
            out.append(float(beats[i]))
            i += 1
            continue
        # A run of off-grid beats: [i, j)
        j = i
        while j < n and not on[j]:
            j += 1
        # Bounds: the neighbouring kept beats, or half a beat past the music at the song edges
        a = out[-1] if out else beats[i] - 0.5 * period
        b = beats[j] if j < n else beats[j - 1] + 0.5 * period
        tracked = beats[i:j]
        straight = _straight(t0, period, a, b)
        if _support(tracked, onsets) > KEEP_TRACKED_RATIO * _support(straight, onsets) and len(tracked) >= 2:
            # The music really jumped here. One level deep, clean up the run on its own grid.
            if depth == 0 and len(tracked) >= 16:
                sub = regularize(tracked, onsets, depth=1)
                tracked, sub_breaks = sub.beats, sub.breaks
                breaks.extend(sub_breaks)
            if out:
                breaks.append(float(tracked[0]))
            out.extend(float(t) for t in tracked)
            if j < n:
                breaks.append(float(beats[j]))
        else:
            out.extend(float(t) for t in straight)
        i = j
    res = np.array(sorted(set(round(t, 5) for t in out)))
    # Remove accidental near-duplicates at run boundaries
    keep = np.concatenate([[True], np.diff(res) > 0.5 * period])
    return Grid(res[keep], period, sorted(set(round(b, 4) for b in breaks)), offbeat_ambiguous=ambiguous)


def pick_downbeats(
    beats: np.ndarray,
    breaks: list[float],
    evidence: np.ndarray,
    hint_downbeats: np.ndarray | None = None,
    anchor: float | None = None,
) -> tuple[np.ndarray, float]:
    """Choose beat 1 of each bar, separately for each piece between grid breaks.

    hint_downbeats: downbeats from a tracker that predicts them (beat_this). If given, each
    piece takes the phase most of those hints agree with; otherwise the phase with the
    highest mean `evidence`.
    anchor: a time known to be beat 1 (a confident drop). Its piece is phased to match.
    Returns (downbeats, confidence 0..1 for the longest piece).
    """
    if len(beats) == 0:
        return np.zeros(0), 0.0
    cuts = [0] + [int(np.searchsorted(beats, b - 1e-4)) for b in breaks] + [len(beats)]
    cuts = sorted(set(c for c in cuts if 0 <= c <= len(beats)))
    downs: list[float] = []
    longest, conf_longest = 0, 0.0
    prev_phase_next = 0
    for s, e in zip(cuts[:-1], cuts[1:]):
        if e <= s:
            continue
        seg = np.arange(s, e)
        phase, conf = None, 0.0
        if anchor is not None and beats[s] - 0.05 <= anchor < (beats[e - 1] + 0.05):
            k = int(np.argmin(np.abs(beats - anchor)))
            phase, conf = (k - s) % BEATS_PER_BAR, 1.0
        if phase is None and hint_downbeats is not None and len(hint_downbeats):
            idx = [int(np.argmin(np.abs(beats - d))) for d in hint_downbeats
                   if s <= int(np.argmin(np.abs(beats - d))) < e and np.min(np.abs(beats - d)) < 0.05]
            if idx:
                votes = np.bincount([(k - s) % BEATS_PER_BAR for k in idx], minlength=BEATS_PER_BAR)
                phase, conf = int(np.argmax(votes)), float(votes.max() / votes.sum())
        if phase is None:
            if len(seg) >= 2 * BEATS_PER_BAR:
                scores = np.array([evidence[s + p:e:BEATS_PER_BAR].mean() for p in range(BEATS_PER_BAR)])
                order = np.argsort(scores)[::-1]
                spread = scores.max() - scores.min()
                phase = int(order[0])
                conf = float((scores[order[0]] - scores[order[1]]) / spread) if spread > 1e-9 else 0.0
            else:
                phase = prev_phase_next  # too short to judge: continue the previous bar count
        downs.extend(beats[s + phase:e:BEATS_PER_BAR])
        last = s + phase + ((e - 1 - (s + phase)) // BEATS_PER_BAR) * BEATS_PER_BAR
        prev_phase_next = (last + BEATS_PER_BAR - e) % BEATS_PER_BAR
        if e - s > longest:
            longest, conf_longest = e - s, conf
    return np.array(downs), conf_longest
