"""Synthetic phonk-style test tracks with exact ground truth.

Real songs don't come with an answer key, so for unit tests we *build* songs where we
know precisely when every kick, snare, hat, cowbell, beat, downbeat and the drop happen.

Structure of a generated track:
    [offset of silence] [intro: hats + cowbell melody + pad, no kick/808]
    [DROP] [main: kick + 808, clap on 2 & 4, hats, cowbell melody]
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mog.audio.dsp import bandpass

# The "cowbell melody" is a 16-step pattern of semitone offsets (None = rest)
COWBELL_PATTERN = [0, None, None, 3, None, None, 7, None, 5, None, None, 3, None, 0, None, None]


@dataclass
class SynthTruth:
    bpm: float
    beats: list[float]
    downbeats: list[float]
    drop: float
    kicks: list[float] = field(default_factory=list)
    snares: list[float] = field(default_factory=list)
    hats: list[float] = field(default_factory=list)
    cowbells: list[float] = field(default_factory=list)


def _env(n: int, sr: int, decay: float) -> np.ndarray:
    t = np.arange(n) / sr
    return np.exp(-t / decay)


def kick(sr: int) -> np.ndarray:
    n = int(0.35 * sr)
    t = np.arange(n) / sr
    freq = 45 + 110 * np.exp(-t / 0.03)  # pitch sweep 155 -> 45 Hz
    phase = 2 * np.pi * np.cumsum(freq) / sr
    body = np.sin(phase) * _env(n, sr, 0.12)
    click = np.random.default_rng(1).standard_normal(n) * _env(n, sr, 0.003) * 0.3
    return np.tanh(1.5 * (body + click)).astype(np.float32)


def bass808(sr: int, freq: float, length: float) -> np.ndarray:
    n = int(length * sr)
    t = np.arange(n) / sr
    tone = np.sin(2 * np.pi * freq * t) * _env(n, sr, length * 0.8)
    fade = np.minimum(1.0, (n - np.arange(n)) / (0.01 * sr))  # avoid a click at the end
    return (np.tanh(2.5 * tone) * 0.55 * fade).astype(np.float32)


def clap(sr: int, rng: np.random.Generator) -> np.ndarray:
    n = int(0.2 * sr)
    noise = rng.standard_normal(n).astype(np.float32)
    noise = bandpass(noise, sr, 900, 6000) * _env(n, sr, 0.06)
    body = np.sin(2 * np.pi * 190 * np.arange(n) / sr) * _env(n, sr, 0.03) * 0.4
    return (0.9 * noise / (np.max(np.abs(noise)) + 1e-9) + body).astype(np.float32) * 0.7


def hat(sr: int, rng: np.random.Generator) -> np.ndarray:
    n = int(0.05 * sr)
    noise = bandpass(rng.standard_normal(n).astype(np.float32), sr, 7000, 10500)
    return (noise / (np.max(np.abs(noise)) + 1e-9) * _env(n, sr, 0.012) * 0.35).astype(np.float32)


def cowbell(sr: int, semitone: int) -> np.ndarray:
    """TR-808 style cowbell: two square-ish tones (540 & 800 Hz ratio), band-limited."""
    n = int(0.22 * sr)
    t = np.arange(n) / sr
    k = 2 ** (semitone / 12)
    tone = np.sign(np.sin(2 * np.pi * 540 * k * t)) + np.sign(np.sin(2 * np.pi * 800 * k * t))
    tone = bandpass(tone.astype(np.float32), sr, 400, 3000)
    env = np.where(t < 0.02, _env(n, sr, 0.05), _env(n, sr, 0.05)[int(0.02 * sr)] * _env(n, sr, 0.15))
    return (tone / (np.max(np.abs(tone)) + 1e-9) * env * 0.4).astype(np.float32)


def pad(sr: int, length: float) -> np.ndarray:
    n = int(length * sr)
    t = np.arange(n) / sr
    chord = sum(np.sin(2 * np.pi * f * t) for f in (220.0, 261.6, 329.6))
    return (chord / 3 * 0.06).astype(np.float32)


def make_track(
    bpm: float = 145.0,
    intro_bars: int = 8,
    main_bars: int = 16,
    offset: float = 0.37,
    sr: int = 22050,
    seed: int = 0,
    with_drop: bool = True,
) -> tuple[np.ndarray, SynthTruth]:
    rng = np.random.default_rng(seed)
    beat = 60.0 / bpm
    step = beat / 4
    total_bars = intro_bars + main_bars
    length = offset + total_bars * 4 * beat + 1.5
    y = np.zeros(int(length * sr), dtype=np.float32)

    def place(sample: np.ndarray, t: float, gain: float = 1.0) -> None:
        a = int(round(t * sr))
        b = min(a + len(sample), len(y))
        y[a:b] += sample[: b - a] * gain

    truth = SynthTruth(bpm=bpm, beats=[], downbeats=[], drop=offset + intro_bars * 4 * beat)
    k, cl, ht = kick(sr), clap(sr, rng), hat(sr, rng)
    bells = {s: cowbell(sr, s) for s in set(p for p in COWBELL_PATTERN if p is not None)}
    bass_notes = [43.65, 43.65, 49.0, 41.2]  # F1, F1, G1, E1 - changes every bar

    place(pad(sr, intro_bars * 4 * beat), offset)
    for bar in range(total_bars):
        t_bar = offset + bar * 4 * beat
        main = bar >= intro_bars or not with_drop
        for b in range(4):
            truth.beats.append(t_bar + b * beat)
        truth.downbeats.append(t_bar)
        for s in range(16):
            t = t_bar + s * step
            # hi-hats on 8ths (intro: quieter)
            if s % 2 == 0:
                place(ht, t, 1.0 if main else 0.6)
                truth.hats.append(t)
            # cowbell melody, every bar
            note = COWBELL_PATTERN[s]
            if note is not None:
                place(bells[note], t, 1.0 if main else 0.7)
                truth.cowbells.append(t)
            if not main:
                continue
            # kick on 1, 3 and the "and" of 3 (s=0, 8, 10); 808 under kick on beat 1
            if s in (0, 8, 10):
                place(k, t, 0.9)
                truth.kicks.append(t)
            if s == 0:
                place(bass808(sr, bass_notes[bar % 4], 2 * beat), t)
            # clap on 2 and 4
            if s in (4, 12):
                place(cl, t, 0.8)
                truth.snares.append(t)
    if not with_drop:
        truth.drop = float("nan")
    y = y / (np.max(np.abs(y)) + 1e-9) * 0.9
    return y.astype(np.float32), truth


def dirty_master(y: np.ndarray, sr: int, seed: int = 1, reverb_db: float = -10.0, drive: float = 3.0,
                 noise: float = 0.02) -> np.ndarray:
    """Make a clean synthetic track sound more like a real phonk master: reverb, heavy
    saturation/clipping and a noise floor."""
    from scipy.signal import fftconvolve

    rng = np.random.default_rng(seed)
    n = int(1.2 * sr)
    ir = rng.standard_normal(n) * np.exp(-np.arange(n) / sr / 0.35)
    ir[0] = np.sqrt(np.sum(ir[1:] ** 2) / 10 ** (reverb_db / 10))  # dry level vs. reverb tail
    out = fftconvolve(y, ir)[: len(y)]
    out = np.tanh(drive * out / np.max(np.abs(out)))
    out = out + noise * rng.standard_normal(len(out))
    return (out / np.max(np.abs(out)) * 0.95).astype(np.float32)
