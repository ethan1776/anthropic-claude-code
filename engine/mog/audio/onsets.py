"""Onset detection split by drum type.

How it works (no ML, just filters):
  1. Split the audio into frequency bands with zero-phase filters (no timing delay).
  2. For each band, take the loudness envelope (RMS every 5.8 ms), convert to log scale,
     and look at how fast it *rises*. A drum hit = a sudden rise = a peak in that curve.
  3. Peak-pick each band's "rise" curve.
       low band  40-150 Hz    -> kicks / 808 hits
       mid band  150-4000 Hz  -> snares, claps... and cowbells
       high band 6-11 kHz     -> hi-hats, shakers
  4. Mid-band hits are split into cowbell vs snare by how *tonal* they are: a cowbell is
     two clean pitched tones (energy in a few frequency bins), a snare/clap is noise (spread out).
  5. Remove "echoes": a kick's click leaks into the mid band and a snare's noise leaks into
     the high band. If a weaker band fires at the same moment as a stronger hit, it's
     usually the same hit, so we drop it.
"""

from __future__ import annotations

import librosa
import numpy as np

from mog.audio.dsp import HOP, bandpass, frame_rms
from mog.audio.models import Onset

BANDS = {
    # Low band is a plain low-pass: a narrow 40-150 Hz band-pass "rings" for tens of ms and,
    # being zero-phase, smears a kick's energy *before* the real hit.
    "low": (None, 150.0),
    "mid": (150.0, 4000.0),
    "high": (6000.0, 11000.0),
}
COINCIDE_S = 0.030
# Tonality (see _tonality_at) above which a mid-band hit is a cowbell. Measured: clean or
# heavily distorted cowbells 0.66-0.98, snares/claps 0.13-0.17.
TONAL_THRESH = 0.5
# Stricter when a kick lands at the same moment: a kick's click on top of a still-ringing
# cowbell measures ~0.57.
TONAL_THRESH_ON_KICK = 0.8  # hits closer than 30 ms are treated as the same moment


def band_flux(y: np.ndarray, sr: int, lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (rise, env): positive rate of change of log band energy (peaks = hits) and
    the band's RMS envelope, one value per HOP."""
    yb = bandpass(y, sr, lo, hi)
    env = frame_rms(yb, HOP, win=2 * HOP)
    log_env = np.log1p(env * 1000.0)
    # Compare with ~17 ms ago rather than the previous frame: robust to the frame window
    # smearing an attack across two frames.
    lag = 3
    rise = np.zeros_like(log_env)
    rise[lag:] = log_env[lag:] - log_env[:-lag]
    rise = np.maximum(rise, 0.0)
    # The rise curve peaks ~half-way up the attack; shift so the peak sits at the attack start.
    return np.roll(rise, -1), env


# A hit must reach at least this fraction of the band's loud level (-20 dB). Without this
# gate, faint leakage from other instruments (a cowbell's tail in the kick band) produces
# big *relative* jumps in near-silent passages.
LEVEL_GATE = 0.1


def _peaks(
    flux: np.ndarray, env: np.ndarray, fps: float, min_gap_s: float, rel_thresh: float
) -> tuple[np.ndarray, np.ndarray]:
    ref = np.percentile(flux, 99.5)
    if ref <= 1e-9:
        return np.array([], int), np.array([])
    norm = flux / ref
    idx = librosa.util.peak_pick(
        norm, pre_max=3, post_max=3, pre_avg=int(0.1 * fps), post_avg=int(0.1 * fps), delta=rel_thresh, wait=1
    )
    idx = idx[norm[idx] >= rel_thresh * 1.5]
    # Within min_gap keep the STRONGEST peak, not the first one. (Keeping the first would let
    # a weak MP3 "pre-echo" just before a hit win over the hit itself.)
    gap = max(1, int(min_gap_s * fps))
    kept: list[int] = []
    for i in sorted(idx, key=lambda i: -norm[i]):
        if all(abs(i - k) >= gap for k in kept):
            kept.append(int(i))
    idx = np.array(sorted(kept), dtype=int)
    # Level after the attack (max over the next ~25 ms) vs. the band's loud level
    loud = np.percentile(env, 99.0)
    after = np.array([env[i:i + int(0.025 * fps) + 1].max() for i in idx]) if len(idx) else np.array([])
    idx = idx[after >= LEVEL_GATE * loud] if len(idx) else idx
    return idx, np.clip(norm[idx], 0.0, 1.0)


def _tonality_at(y: np.ndarray, sr: int, t: float) -> float:
    """How tonal the sound that *starts* at t is: share of its 400-8000 Hz energy that sits
    in the 8 strongest frequency bins. Cowbell (2 pitched tones) ~0.95, clap/snare noise
    ~0.15, a kick's click on top of a ringing cowbell ~0.6.

    We take the spectrum 5-45 ms after t and subtract the spectrum 5-45 ms before t, so a
    cowbell still ringing from the previous note doesn't make the next drum hit look tonal.
    Only 400-8000 Hz is used, so a kick/808 at the same moment (tonal, but low) can't hide
    or fake a cowbell.
    """
    n = int(0.040 * sr)
    a = int((t + 0.005) * sr)
    b = int((t - 0.005) * sr) - n
    if a + n > len(y):
        return 0.0
    win = np.hanning(n)
    after = np.abs(np.fft.rfft(y[a:a + n] * win)) ** 2
    before = np.abs(np.fft.rfft(y[b:b + n] * win)) ** 2 if b >= 0 else np.zeros_like(after)
    new = np.maximum(after - before, 0.0)
    freqs = np.fft.rfftfreq(n, 1 / sr)
    m = (freqs >= 400) & (freqs <= 8000)
    band = np.sort(new[m])[::-1]
    total = band.sum()
    return float(band[:8].sum() / total) if total > 1e-12 else 0.0


def _sharp_attacks(env: np.ndarray, idx: np.ndarray, fps: float, margin: float = 0.1, dip: float = 1.0) -> np.ndarray:
    """True for low-band peaks that are real hits, False for "bass swells".

    Two bass notes of slightly different pitch (e.g. a kick tail and an 808) drift in and
    out of phase: the level briefly cancels out (a deep dip), then recovers. The recovery
    looks like a hit to a "rising energy" detector, but the level never goes *above* where
    it was ~100 ms earlier. We reject a peak only if BOTH are true:
      * there was a deep dip (> `dip` in log units, ~-9 dB) just before it, and
      * the level after it does not exceed the level before the dip.
    A real kick under a heavily compressed mix may not exceed the previous kick's tail, but
    it doesn't come out of a cancellation dip either, so it is kept.
    """
    L = np.log1p(env * 1000.0)
    short, long_ = 3, int(0.1 * fps)
    out = np.ones(len(idx), dtype=bool)
    for j, i in enumerate(idx):
        top = L[min(i + 3, len(L) - 1)]
        recent = L[max(i - long_, 0):max(i - short, 1)]
        if recent.size == 0:
            continue
        local_min = L[max(i - short, 0):i + 1].min()
        if recent.max() - local_min > dip and top <= recent.max() + margin:
            out[j] = False
    return out


KICK_MAX_CENTROID_HZ = 130.0


def _low_centroid(y: np.ndarray, sr: int, t: float) -> float:
    """Centre frequency (20-250 Hz) of the bass energy that *appears* at t.

    80 ms after t minus 80 ms before t. Kicks/808s: ~45-100 Hz. A clap's 190 Hz body (or the
    distortion products it makes with a ringing kick) can leak into the kick band: ~180 Hz.
    """
    n = int(0.08 * sr)
    a = int(t * sr)
    b = a - n
    if a + n > len(y):
        return 0.0
    w = np.hanning(n)
    after = np.abs(np.fft.rfft(y[a:a + n] * w)) ** 2
    before = np.abs(np.fft.rfft(y[b:b + n] * w)) ** 2 if b >= 0 else np.zeros_like(after)
    new = np.maximum(after - before, 0.0)
    f = np.fft.rfftfreq(n, 1 / sr)
    m = (f >= 20) & (f <= 250)
    tot = new[m].sum()
    return float((new[m] * f[m]).sum() / tot) if tot > 1e-12 else 0.0


def _snap_to_attack(idx: np.ndarray, rise: np.ndarray, fps: float, before_s: float, after_s: float) -> np.ndarray:
    """Move each hit to the sharpest broadband attack within a small window."""
    out = idx.copy()
    b, a = int(before_s * fps), int(after_s * fps)
    for j, i in enumerate(idx):
        lo, hi = max(0, i - b), min(len(rise), i + a + 1)
        out[j] = lo + int(np.argmax(rise[lo:hi]))
    return out


def detect_onsets(y: np.ndarray, sr: int) -> tuple[list[Onset], dict[str, np.ndarray]]:
    """Return (onsets sorted by time, per-band flux curves for reuse by other modules)."""
    fps = sr / HOP
    flux, env = {}, {}
    for name, (lo, hi) in BANDS.items():
        flux[name], env[name] = band_flux(y, sr, lo, hi)

    k_idx, k_str = _peaks(flux["low"], env["low"], fps, min_gap_s=0.09, rel_thresh=0.12)
    keep = _sharp_attacks(env["low"], k_idx, fps)
    k_idx, k_str = k_idx[keep], k_str[keep]
    m_idx, m_str = _peaks(flux["mid"], env["mid"], fps, min_gap_s=0.06, rel_thresh=0.12)
    h_idx, h_str = _peaks(flux["high"], env["high"], fps, min_gap_s=0.05, rel_thresh=0.10)

    # Low-band timing is blurry (bass waves are long); the kick's click is sharp in the full
    # band, so pin each kick to it.
    full_rise, _ = band_flux(y, sr, None, None)
    k_idx = _snap_to_attack(k_idx, full_rise, fps, before_s=0.035, after_s=0.035)
    keep = np.array([_low_centroid(y, sr, i / fps) <= KICK_MAX_CENTROID_HZ for i in k_idx], dtype=bool)
    k_idx, k_str = k_idx[keep], k_str[keep]
    kicks = [Onset(t=float(i / fps), strength=float(s), kind="kick") for i, s in zip(k_idx, k_str)]
    kick_t = np.array([o.t for o in kicks])

    mids: list[Onset] = []
    for i, s in zip(m_idx, m_str):
        t = float(i / fps)
        tonality = _tonality_at(y, sr, t)
        near_kick = bool(kick_t.size) and float(np.min(np.abs(kick_t - t))) < COINCIDE_S
        tonal = tonality >= (TONAL_THRESH_ON_KICK if near_kick else TONAL_THRESH)
        if near_kick and not tonal:
            # Probably the kick's click. Keep only if the mid hit is clearly noisy *and* strong
            # (a snare layered on a kick).
            if tonality > 0.35 or s < 0.5:
                continue
        mids.append(Onset(t=t, strength=float(s), kind="cowbell" if tonal else "snare"))

    snare_t = np.array([o.t for o in mids if o.kind == "snare"])
    snare_s = np.array([o.strength for o in mids if o.kind == "snare"])
    hats: list[Onset] = []
    for i, s in zip(h_idx, h_str):
        t = float(i / fps)
        if snare_t.size:
            j = int(np.argmin(np.abs(snare_t - t)))
            if abs(snare_t[j] - t) < COINCIDE_S and snare_s[j] >= s * 0.6:
                continue  # it's the snare's noise tail
        hats.append(Onset(t=t, strength=float(s), kind="hihat"))

    onsets = sorted(kicks + mids + hats, key=lambda o: (o.t, o.kind))
    for o in onsets:
        o.t = round(o.t, 4)
        o.strength = round(o.strength, 3)
    return onsets, flux
