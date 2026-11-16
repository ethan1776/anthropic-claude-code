"""Beat and downbeat tracking.

Primary tracker: beat_this (a neural network from JKU Linz, state of the art for beats
AND downbeats). It needs PyTorch and a model checkpoint download, so it is optional.

Fallback: librosa's dynamic-programming beat tracker, plus our own downbeat finder.
librosa does not estimate downbeats, so we pick which of the 4 possible "phases" of
the beat grid is beat 1 (see `_pick_downbeat_phase`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import librosa
import numpy as np

from mog.audio.dsp import HOP
from mog.errors import MogError

log = logging.getLogger(__name__)

# Phonk sits roughly between 100 and 180 BPM when counted at the "cut" level. Tracks with a
# half-time feel (70-90 BPM) are counted double, which is what an editor wants for cuts.
MIN_BPM, MAX_BPM = 100.0, 180.0
BEATS_PER_BAR = 4


@dataclass
class BeatResult:
    beats: np.ndarray
    downbeats: np.ndarray
    bpm: float
    source: str
    warnings: list[str] = field(default_factory=list)


def track_beats(
    y: np.ndarray,
    sr: int,
    onset_env: np.ndarray,
    low_flux: np.ndarray,
    prefer: str = "auto",
) -> BeatResult:
    """Run the best available tracker.

    onset_env / low_flux are frame-rate curves (hop = HOP) computed by the caller; the
    librosa path uses them, the beat_this path ignores them.
    """
    warnings: list[str] = []
    if prefer in ("auto", "beat_this"):
        try:
            res = _track_beat_this(y, sr)
            if len(res.beats) >= 8:
                return res
            warnings.append("beat_this found too few beats; used librosa instead.")
        except Exception as exc:  # noqa: BLE001 - any failure means "use the fallback"
            if prefer == "beat_this":
                raise
            log.info("beat_this unavailable (%s); falling back to librosa", exc)
    res = _track_librosa(y, sr, onset_env, low_flux)
    res.warnings = warnings + res.warnings
    return res


# --------------------------------------------------------------------------- beat_this

_B2B = None


def _track_beat_this(y: np.ndarray, sr: int) -> BeatResult:
    global _B2B
    from beat_this.inference import Audio2Beats  # optional dependency

    if _B2B is None:
        _B2B = Audio2Beats(checkpoint_path="final0", device="cpu", dbn=False)
    beats, downbeats = _B2B(y, sr)
    beats = np.asarray(beats, dtype=float)
    downbeats = np.asarray(downbeats, dtype=float)
    bpm = 60.0 / float(np.median(np.diff(beats))) if len(beats) > 1 else 0.0
    beats, downbeats, bpm = _fold_tempo(beats, downbeats, bpm)
    return BeatResult(beats, downbeats, bpm, "beat_this")


def _fold_tempo(beats: np.ndarray, downbeats: np.ndarray, bpm: float):
    """Bring a half-time result into the MIN_BPM..MAX_BPM range by inserting midpoints."""
    if 0 < bpm < MIN_BPM and len(beats) > 1:
        mids = (beats[:-1] + beats[1:]) / 2
        beats = np.sort(np.concatenate([beats, mids]))
        bpm *= 2
        # downbeats stay where they are; bars now hold 8 grid beats, which is fine for cuts
    return beats, downbeats, bpm


# --------------------------------------------------------------------------- librosa

def _track_librosa(y: np.ndarray, sr: int, onset_env: np.ndarray, low_flux: np.ndarray) -> BeatResult:
    fps = sr / HOP
    # 1. Tempo, restricted to the phonk range so we don't lock onto half/double time.
    tempo = estimate_tempo(onset_env, fps)
    # 2. Dynamic-programming beat tracker locked to that tempo.
    _, frames = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sr, hop_length=HOP, bpm=tempo, tightness=200, trim=False, units="frames"
    )
    frames = np.asarray(frames, dtype=int)
    warnings: list[str] = []
    if len(frames) < 8:
        raise MogError("Could not find a steady beat in this track. Is it music with drums?")

    # 3. Fill gaps: the tracker can skip beats in quiet passages. Re-grid any gap that is
    #    clearly a multiple of the beat period.
    period = fps * 60.0 / tempo
    frames = _fill_gaps(frames, period)

    # 4. Sub-frame refinement: nudge each beat to the strongest onset within +-2 frames
    #    (+-12 ms). Keeps the grid but lands exactly on the transient when there is one.
    beats_f = _refine(frames, onset_env, radius=2)
    beats = beats_f / fps

    # 5. Downbeats.
    phase, phase_conf = _pick_downbeat_phase(beats, y, sr, low_flux, fps)
    if phase_conf < 0.15:
        warnings.append("Downbeat detection is uncertain for this track; cuts may land on beat 3 instead of 1.")
    downbeats = beats[phase::BEATS_PER_BAR]
    bpm = 60.0 / float(np.median(np.diff(beats)))
    return BeatResult(beats, downbeats, bpm, "librosa", warnings)


def estimate_tempo(onset_env: np.ndarray, fps: float, lo: float = MIN_BPM, hi: float = MAX_BPM) -> float:
    """Comb-filter tempo estimate.

    Autocorrelate the onset curve (how similar the song is to itself shifted by `lag`), then
    for each candidate BPM add up the autocorrelation at 1, 2, ... 8 beat periods. The true
    beat period lines up with beats, half-bars, bars and two-bar phrases all at once, so it
    beats "syncopation tempos" (e.g. 4/3 x the real BPM from a 3-step cowbell rhythm) that
    only match at a few multiples. librosa's default estimator falls for those on phonk.
    """
    e = onset_env - onset_env.mean()
    n = len(e)
    ac = np.fft.irfft(np.abs(np.fft.rfft(e, 2 * n)) ** 2)[:n]
    ac /= ac[0] + 1e-12
    bpms = np.arange(lo, hi, 0.05)
    lags = np.arange(n)
    periods = fps * 60.0 / bpms
    score = sum(np.interp(k * periods, lags, ac) for k in range(1, 9))
    return float(bpms[int(np.argmax(score))])


def _fill_gaps(frames: np.ndarray, period: float) -> np.ndarray:
    out = [int(frames[0])]
    for f in frames[1:]:
        prev = out[-1]
        gap = f - prev
        n = int(round(gap / period))
        if n >= 2 and abs(gap - n * period) < 0.25 * period:
            out.extend(int(round(prev + gap * k / n)) for k in range(1, n))
        out.append(int(f))
    return np.asarray(out)


def _refine(frames: np.ndarray, env: np.ndarray, radius: int) -> np.ndarray:
    """Return fractional frame positions, snapped to local onset peaks with parabolic interpolation."""
    out = np.empty(len(frames), dtype=float)
    n = len(env)
    for i, f in enumerate(frames):
        lo, hi = max(0, f - radius), min(n, f + radius + 1)
        k = lo + int(np.argmax(env[lo:hi]))
        # Only move if the peak is meaningfully stronger than where we were.
        if env[k] <= env[min(f, n - 1)] * 1.05:
            k = min(f, n - 1)
        # Parabolic interpolation around the peak for sub-frame precision.
        if 0 < k < n - 1:
            a, b, c = env[k - 1], env[k], env[k + 1]
            denom = a - 2 * b + c
            delta = 0.5 * (a - c) / denom if denom != 0 else 0.0
            out[i] = k + float(np.clip(delta, -0.5, 0.5))
        else:
            out[i] = float(k)
    return out


def _pick_downbeat_phase(beats: np.ndarray, y: np.ndarray, sr: int, low_flux: np.ndarray, fps: float):
    """Decide which beat of every 4 is beat 1.

    Evidence for "this beat starts a bar", summed per phase:
      * kick / 808 strength on the beat (phonk almost always hits beat 1)
      * harmonic change (chroma novelty): chords and 808 notes change on bar lines
      * loudness change (RMS novelty): sections start on bar lines, the drop above all
    Kick evidence alone can't separate beat 1 from beat 3 (both usually have a kick), which
    is why the harmonic and structural cues matter.
    Returns (phase, confidence 0..1).
    """
    if len(beats) < 2 * BEATS_PER_BAR:
        return 0, 0.0
    frames = np.clip(np.round(beats * fps).astype(int), 0, len(low_flux) - 1)
    kick = _peak_near(low_flux, frames, 2)

    # Beat-synchronous chroma and RMS (one column per beat interval)
    hop = HOP * 4
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop)
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    bf = np.clip(np.round(beats * sr / hop).astype(int), 0, chroma.shape[1] - 1)
    chroma_sync = librosa.util.sync(chroma, bf, aggregate=np.median)
    rms_sync = librosa.util.sync(rms[None, :], bf, aggregate=np.mean)[0]
    # sync() returns one column before the first beat; drop it so column i = beat i..i+1
    chroma_sync, rms_sync = chroma_sync[:, 1:], rms_sync[1:]
    m = min(len(beats), chroma_sync.shape[1])
    chroma_nov = np.zeros(len(beats))
    rms_nov = np.zeros(len(beats))
    # Compare the bar starting at beat i with the bar ending at beat i (4 beats each side)
    for i in range(BEATS_PER_BAR, m - BEATS_PER_BAR):
        a = chroma_sync[:, i - BEATS_PER_BAR:i].mean(1)
        b = chroma_sync[:, i:i + BEATS_PER_BAR].mean(1)
        chroma_nov[i] = 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
        ra = rms_sync[i - BEATS_PER_BAR:i].mean()
        rb = rms_sync[i:i + BEATS_PER_BAR].mean()
        rms_nov[i] = abs(np.log((rb + 1e-6) / (ra + 1e-6)))

    def z(x):
        s = x.std()
        return (x - x.mean()) / s if s > 1e-9 else np.zeros_like(x)

    evidence = z(kick) + 0.7 * z(chroma_nov) + 1.0 * z(rms_nov)
    scores = np.array([evidence[p::BEATS_PER_BAR].mean() for p in range(BEATS_PER_BAR)])
    order = np.argsort(scores)[::-1]
    best, second = scores[order[0]], scores[order[1]]
    spread = scores.max() - scores.min()
    conf = float(np.clip((best - second) / (spread + 1e-9), 0.0, 1.0)) if spread > 0 else 0.0
    return int(order[0]), conf


def _peak_near(x: np.ndarray, frames: np.ndarray, radius: int) -> np.ndarray:
    n = len(x)
    return np.array([x[max(0, f - radius):min(n, f + radius + 1)].max() for f in frames])


def align_to_onsets(
    beats: np.ndarray, downbeats: np.ndarray, onset_times: np.ndarray, snap_s: float = 0.020, search_s: float = 0.035
) -> tuple[np.ndarray, np.ndarray]:
    """Pull the beat grid onto the actual drum hits.

    Beat trackers work on smeared spectral curves and land ~10 ms late (librosa) or are
    quantized to 20 ms frames (beat_this). Our band onsets are time-domain and precise. So:
      1. measure the typical (median) offset between beats and their nearest hits, and shift
         the whole grid by it;
      2. snap each beat to a hit within +-20 ms if there is one (keeps the grid on beats with
         no drum hit).
    Downbeats are moved with the beat they belong to.
    """
    if len(beats) == 0 or len(onset_times) == 0:
        return beats, downbeats
    on = np.sort(onset_times)

    def nearest(t: np.ndarray) -> np.ndarray:
        i = np.clip(np.searchsorted(on, t), 1, len(on) - 1)
        left, right = on[i - 1], on[i]
        return np.where(np.abs(t - left) <= np.abs(t - right), left, right)

    near = nearest(beats)
    diffs = near - beats
    close = np.abs(diffs) < search_s
    if close.sum() >= max(4, 0.2 * len(beats)):
        beats = beats + float(np.median(diffs[close]))
    near = nearest(beats)
    snapped = np.where(np.abs(near - beats) < snap_s, near, beats)

    # Map each downbeat to the index of its closest (original) beat, then take the snapped value
    if len(downbeats):
        j = np.clip(np.searchsorted(beats, downbeats), 1, len(beats) - 1)
        j = np.where(np.abs(beats[j - 1] - downbeats) <= np.abs(beats[j] - downbeats), j - 1, j)
        downbeats = snapped[j]
    return snapped, downbeats


def trim_to_music(beats: np.ndarray, downbeats: np.ndarray, onset_times: np.ndarray, period: float):
    """Drop grid beats before the first / after the last drum hit (silence, reverb tails)."""
    if len(onset_times) == 0:
        return beats, downbeats
    lo, hi = onset_times.min() - 0.5 * period, onset_times.max() + 0.5 * period
    keep_b = (beats >= lo) & (beats <= hi)
    keep_d = (downbeats >= lo) & (downbeats <= hi)
    return beats[keep_b], downbeats[keep_d]
