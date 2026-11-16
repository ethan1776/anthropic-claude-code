"""analyze_audio(path) -> AudioAnalysis: the single entry point for Phase 1."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

from mog.audio.beats import align_to_onsets, downbeat_evidence, track_beats, trim_to_music
from mog.audio.grid import pick_downbeats, regularize
from mog.audio.dsp import HOP, SR, bandpass, frame_rms, normalize, resample_curve, smooth
from mog.audio.drop import detect_drop
from mog.audio.models import AudioAnalysis, Bar, Curve
from mog.audio.onsets import detect_onsets
from mog.errors import MogError
from mog.ffmpeg.audio_io import decode_audio

CURVE_HOP = 0.05  # seconds between samples of the stored energy curves
MIN_DURATION = 4.0
DROP_ANCHOR_CONF = 0.5  # a drop at least this confident decides where beat 1 is


def analyze_audio(path: str | Path, tracker: str = "auto", offbeat: bool = False) -> AudioAnalysis:
    path = Path(path)
    y = decode_audio(path, sr=SR)
    return analyze_signal(y, SR, source=path.name, tracker=tracker, offbeat=offbeat)


def analyze_signal(
    y: np.ndarray, sr: int, source: str = "signal", tracker: str = "auto", offbeat: bool = False
) -> AudioAnalysis:
    """offbeat=True shifts the whole beat grid by half a beat (manual fix for tracks whose
    drums sit on both beats and off-beats, when the clicks sound between the beats)."""
    duration = len(y) / sr
    if duration < MIN_DURATION:
        raise MogError(f"The audio is only {duration:.1f}s long; it needs at least {MIN_DURATION:.0f}s.")
    if np.max(np.abs(y)) < 1e-4:
        raise MogError("The audio is silent.")
    fps = sr / HOP

    # Onsets per drum type (+ the band "rise" curves, reused below)
    onsets, flux = detect_onsets(y, sr)

    # Generic onset envelope for the beat tracker (mel spectral flux, 5.8 ms hop)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP, n_fft=1024, aggregate=np.median)
    beats = track_beats(y, sr, onset_env, flux["low"], prefer=tracker)
    onset_times = np.array([o.t for o in onsets])
    hint_downbeats = beats.downbeats if len(beats.downbeats) else None

    # 1. Shift the tracker's grid onto the real drum hits (removes tracker latency)
    aligned, _ = align_to_onsets(beats.beats, np.zeros(0), onset_times)
    # 2. Snap to the producer's straight tempo grid, keeping genuine jumps (splices)
    grid = regularize(aligned, onsets)
    # 3. Snap each grid beat onto a drum hit within 20 ms, then drop beats in silence
    grid_beats = grid.beats + (grid.period / 2 if offbeat else 0.0)
    final, _ = align_to_onsets(grid_beats, np.zeros(0), onset_times)
    final, _ = trim_to_music(final, np.zeros(0), onset_times, grid.period)
    breaks = [b for b in grid.breaks if final.size and final[0] < b <= final[-1]]
    # 4. Energy curves and the drop (searched on every beat)
    full_rms = smooth(frame_rms(y, HOP, win=4 * HOP), 0.5, fps)
    low_band = bandpass(y, sr, 40, 150)
    low_rms = smooth(frame_rms(low_band, HOP, win=4 * HOP), 0.5, fps)
    full_n = normalize(full_rms, 99.5)
    low_n = normalize(low_rms, 99.5)
    low_raw = normalize(frame_rms(low_band, HOP, win=4 * HOP), 99.5)
    drop = detect_drop(final, 4 * grid.period, low_n, full_n, fps, low_raw=low_raw)

    # 5. Downbeats, chosen per piece between jumps. A confident drop is beat 1 of its bar.
    evidence = downbeat_evidence(final, y, sr, flux["low"], fps)
    anchor = drop.t if drop is not None and drop.confidence >= DROP_ANCHOR_CONF else None
    downbeats_arr, db_conf = pick_downbeats(final, breaks, evidence, hint_downbeats, anchor=anchor)
    beats.beats, beats.downbeats = final, downbeats_arr
    beats.bpm = 60.0 / grid.period
    if beats.source == "librosa" and db_conf < 0.15 and anchor is None:
        beats.warnings.append("Downbeat detection is uncertain for this track; cuts may land on beat 3 instead of 1.")
    if grid.offbeat_ambiguous and not offbeat:
        beats.warnings.append(
            "This track's drums sit on both the beats and the off-beats; check the click track. "
            "If the clicks sound like they're between the beats, re-run with --offbeat."
        )
    if breaks:
        beats.warnings.append(
            "The beat jumps at " + ", ".join(f"{b:.1f}s" for b in breaks[:6])
            + (" ..." if len(breaks) > 6 else "") + " (the audio looks spliced or changes tempo there)."
        )

    downbeats = beats.downbeats
    bar_len = 4 * grid.period

    # Per-bar energy, and a low/mid/high label relative to this song's quietest and loudest bar
    raw = []
    for i, start in enumerate(downbeats):
        end = float(downbeats[i + 1]) if i + 1 < len(downbeats) else min(float(start) + bar_len, duration)
        a = min(int(start * fps), len(full_n) - 1)
        b = min(max(int(end * fps), a + 1), len(full_n))
        raw.append((float(start), end, float(full_n[a:b].mean())))
    energies = np.array([e for _, _, e in raw]) if raw else np.zeros(0)
    lo_e, hi_e = (energies.min(), energies.max()) if len(energies) else (0.0, 1.0)
    bars = []
    for i, (start, end, e) in enumerate(raw):
        rel = (e - lo_e) / (hi_e - lo_e) if hi_e > lo_e else 0.5
        bars.append(Bar(index=i, start=round(start, 4), end=round(end, 4), energy=round(e, 4),
                        level="high" if rel >= 0.66 else "mid" if rel >= 0.33 else "low"))

    warnings = list(beats.warnings)
    if beats.source == "librosa":
        warnings.insert(0, "Using the librosa beat tracker (beat_this not installed); downbeats are estimated.")

    return AudioAnalysis(
        source=source,
        duration=round(duration, 4),
        bpm=round(beats.bpm, 2),
        beat_tracker=beats.source,
        beats=[round(float(t), 4) for t in beats.beats],
        downbeats=[round(float(t), 4) for t in downbeats],
        onsets=onsets,
        energy=Curve(hop=CURVE_HOP, values=resample_curve(full_n, fps, CURVE_HOP)),
        low_energy=Curve(hop=CURVE_HOP, values=resample_curve(low_n, fps, CURVE_HOP)),
        drop=drop,
        bars=bars,
        grid_breaks=[round(b, 4) for b in breaks],
        warnings=warnings,
    )
