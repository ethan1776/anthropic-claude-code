"""analyze_audio(path) -> AudioAnalysis: the single entry point for Phase 1."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

from mog.audio.beats import align_to_onsets, track_beats, trim_to_music
from mog.audio.dsp import HOP, SR, bandpass, frame_rms, normalize, resample_curve, smooth
from mog.audio.drop import detect_drop
from mog.audio.models import AudioAnalysis, Bar, Curve
from mog.audio.onsets import detect_onsets
from mog.errors import MogError
from mog.ffmpeg.audio_io import decode_audio

CURVE_HOP = 0.05  # seconds between samples of the stored energy curves
MIN_DURATION = 4.0


def analyze_audio(path: str | Path, tracker: str = "auto") -> AudioAnalysis:
    path = Path(path)
    y = decode_audio(path, sr=SR)
    return analyze_signal(y, SR, source=path.name, tracker=tracker)


def analyze_signal(y: np.ndarray, sr: int, source: str = "signal", tracker: str = "auto") -> AudioAnalysis:
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
    beats.beats, beats.downbeats = align_to_onsets(beats.beats, beats.downbeats, onset_times)
    beats.beats, beats.downbeats = trim_to_music(beats.beats, beats.downbeats, onset_times, 60.0 / beats.bpm)

    # Energy curves
    full_rms = smooth(frame_rms(y, HOP, win=4 * HOP), 0.5, fps)
    low_rms = smooth(frame_rms(bandpass(y, sr, 40, 150), HOP, win=4 * HOP), 0.5, fps)
    full_n = normalize(full_rms, 99.5)
    low_n = normalize(low_rms, 99.5)

    downbeats = beats.downbeats
    bar_len = float(np.median(np.diff(downbeats))) if len(downbeats) > 1 else 4 * 60 / beats.bpm
    drop = detect_drop(downbeats, bar_len, low_n, full_n, fps)

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
        warnings=warnings,
    )
