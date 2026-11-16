"""Pick the part of the song the edit will use."""

from __future__ import annotations

import numpy as np

from mog.audio.models import AudioAnalysis, Section
from mog.errors import MogError

ALLOWED_LENGTHS = (10, 15, 20, 30)


def pick_section(an: AudioAnalysis, target: float, lead_bars: int | None = None) -> Section:
    """Choose a window of about `target` seconds that contains the drop.

    * Starts 2-4 bars before the drop (see `lead_bars_for`), so the
      viewer gets a short build-up, then the drop hits.
    * Start and end are snapped to downbeats, so the length is a whole number of bars and
      may differ from the target by up to half a bar.
    """
    if target <= 0:
        raise MogError("Target length must be positive.")
    warnings: list[str] = []
    downs = np.asarray(an.downbeats, dtype=float)
    bar = an.bar_length

    if an.duration <= target + 0.5 * bar or len(downs) < 2:
        if an.duration < target:
            warnings.append(
                f"The song is only {an.duration:.1f}s long, shorter than the {target:.0f}s target; using the whole song."
            )
        start = float(downs[0]) if len(downs) and downs[0] < 0.5 * bar else 0.0
        return Section(start=round(start, 4), end=round(an.duration, 4),
                       drop=an.drop.t if an.drop else None, lead_bars=0, warnings=warnings)

    n_bars = max(1, int(round(target / bar)))
    if lead_bars is None:
        lead_bars = lead_bars_for(target)

    if an.drop is None:
        warnings.append("No clear drop found; using the loudest part of the song.")
        drop_t = _loudest_downbeat(an, downs, n_bars)
    else:
        drop_t = an.drop.t
        if an.drop.confidence < 0.3:
            warnings.append(f"The drop detection is unsure (confidence {an.drop.confidence:.2f}). Check the section.")

    di = int(np.argmin(np.abs(downs - drop_t)))
    si = max(di - lead_bars, 0)
    ei = si + n_bars

    # Use a virtual downbeat grid that extends past the last detected downbeat, so the end
    # can snap even near the end of the song.
    def grid(i: int) -> float:
        if i < len(downs):
            return float(downs[i])
        return float(downs[-1] + (i - len(downs) + 1) * bar)

    # Shift the window left if it runs past the end of the song (keep the drop inside).
    while grid(ei) > an.duration + 1e-6 and si > 0 and si > di - n_bars + 1:
        si -= 1
        ei -= 1
    end = min(grid(ei), an.duration)
    start = grid(si)
    if end - start < 0.75 * target:
        warnings.append(f"Only {end - start:.1f}s of song available around the drop.")
    return Section(start=round(start, 4), end=round(end, 4), drop=round(drop_t, 4),
                   lead_bars=di - si, warnings=warnings)


def lead_bars_for(target: float) -> int:
    """Bars of build-up before the drop: 2 for short edits, up to 4 for 30 s ones."""
    if target <= 15:
        return 2
    if target <= 20:
        return 3
    return 4


def _loudest_downbeat(an: AudioAnalysis, downs: np.ndarray, n_bars: int) -> float:
    best_t, best_e = float(downs[0]), -1.0
    win = n_bars * an.bar_length
    for d in downs:
        ts = np.arange(d, min(d + win, an.duration), an.energy.hop)
        e = float(np.mean([an.energy.at(t) for t in ts])) if len(ts) else 0.0
        if e > best_e:
            best_t, best_e = float(d), e
    return best_t
