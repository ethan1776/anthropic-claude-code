"""Debug outputs so a human can check the analysis by eye and by ear.

* debug PNG: waveform + beats, downbeats, each drum type, energy curves, the drop and the
  chosen section. Top panel = whole song, bottom panel = 4 bars either side of the drop.
* click track: the original song with clicks mixed in on the detected beats (high click on
  downbeats, lower click on other beats) and a short "ping" at the drop. If the clicks sound
  like they're in time with the drums, the beat detection is right.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from mog.audio.models import AudioAnalysis, Section  # noqa: E402

KIND_STYLE = {
    "kick": ("#ff4d4d", 0.85, "kick"),
    "snare": ("#4da6ff", 0.70, "snare / clap"),
    "cowbell": ("#ffd24d", 0.55, "cowbell"),
    "hihat": ("#b3b3b3", 0.40, "hi-hat"),
}


def _plot_panel(ax, y, sr, an: AudioAnalysis, section: Section | None, t0: float, t1: float, fine: bool) -> None:
    a, b = int(t0 * sr), int(min(t1, len(y) / sr) * sr)
    seg = y[a:b]
    # Draw the waveform as a min/max envelope (fast even for whole songs)
    cols = 4000
    step = max(1, len(seg) // cols)
    n = len(seg) // step
    if n > 0:
        blocks = seg[: n * step].reshape(n, step)
        tt = t0 + (np.arange(n) * step) / sr
        ax.fill_between(tt, blocks.min(1), blocks.max(1), color="#555", lw=0)

    if section is not None:
        ax.axvspan(max(section.start, t0), min(section.end, t1), color="#2a7", alpha=0.12, label="chosen section")

    for t in an.beats:
        if t0 <= t <= t1:
            ax.axvline(t, color="#888", lw=0.5 if not fine else 0.8, alpha=0.6)
    for t in an.downbeats:
        if t0 <= t <= t1:
            ax.axvline(t, color="#fff", lw=1.0 if not fine else 1.6, alpha=0.9)

    for kind, (color, level, label) in KIND_STYLE.items():
        ts = [o.t for o in an.onsets if o.kind == kind and t0 <= o.t <= t1]
        ss = [o.strength for o in an.onsets if o.kind == kind and t0 <= o.t <= t1]
        if ts:
            ax.scatter(ts, [level] * len(ts), s=[8 + 40 * s for s in ss], color=color, label=label, zorder=3,
                       marker="v" if kind == "kick" else "o")

    def curve(c, color, label):
        t = np.arange(len(c.values)) * c.hop
        m = (t >= t0) & (t <= t1)
        ax.plot(t[m], np.array(c.values)[m] * 0.95 - 1.0, color=color, lw=1.2, label=label)

    curve(an.energy, "#c9f", "energy (RMS)")
    curve(an.low_energy, "#f80", "bass energy")

    if an.drop and t0 <= an.drop.t <= t1:
        ax.axvline(an.drop.t, color="#ff2d55", lw=2.5, label=f"DROP (conf {an.drop.confidence:.2f})")

    ax.set_xlim(t0, t1)
    ax.set_ylim(-1.05, 1.0)
    ax.set_facecolor("#111")
    ax.tick_params(colors="#ccc")
    ax.set_yticks([])
    ax.set_xlabel("seconds", color="#ccc")


def write_debug_png(y: np.ndarray, sr: int, an: AudioAnalysis, out: str | Path, section: Section | None = None) -> Path:
    out = Path(out)
    fig, axes = plt.subplots(2, 1, figsize=(24, 10), facecolor="#0b0b0b",
                             gridspec_kw={"height_ratios": [1, 1.2]})
    title = (f"{an.source} | {an.bpm:.1f} BPM | tracker: {an.beat_tracker} | "
             f"{len(an.beats)} beats, {len(an.downbeats)} downbeats | "
             f"drop: {an.drop.t:.3f}s (conf {an.drop.confidence:.2f})" if an.drop else f"{an.source} | no drop found")
    fig.suptitle(title, color="#eee", fontsize=14)

    _plot_panel(axes[0], y, sr, an, section, 0.0, an.duration, fine=False)
    axes[0].set_title("Whole song  (white = downbeat, grey = beat; bottom: energy curves)", color="#aaa", loc="left")

    center = an.drop.t if an.drop else an.duration / 2
    bar = an.bar_length
    _plot_panel(axes[1], y, sr, an, section, max(0.0, center - 4 * bar), min(an.duration, center + 4 * bar), fine=True)
    axes[1].set_title("Zoom: 4 bars either side of the drop  (row markers: kick / snare / cowbell / hi-hat)",
                      color="#aaa", loc="left")
    leg = axes[1].legend(loc="upper right", fontsize=9, facecolor="#222", edgecolor="#444", ncol=2)
    for text in leg.get_texts():
        text.set_color("#ddd")
    fig.tight_layout()
    fig.savefig(out, dpi=80, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def _click(sr: int, freq: float, length: float = 0.03, gain: float = 0.5) -> np.ndarray:
    t = np.arange(int(length * sr)) / sr
    return (np.sin(2 * np.pi * freq * t) * np.exp(-t / (length / 4)) * gain).astype(np.float32)


def write_click_track(y: np.ndarray, sr: int, an: AudioAnalysis, out: str | Path, section: Section | None = None) -> Path:
    """Song + clicks. If `section` is given, only that part of the song is written."""
    out = Path(out)
    mix = y.astype(np.float32).copy() * 0.7
    if mix.ndim == 2:
        mix = mix.mean(axis=1)
    hi, lo = _click(sr, 1760, gain=0.6), _click(sr, 880, gain=0.45)
    ping = _click(sr, 2640, length=0.25, gain=0.5)
    downs = set(round(t, 4) for t in an.downbeats)

    def add(sample: np.ndarray, t: float) -> None:
        a = int(round(t * sr))
        if 0 <= a < len(mix):
            b = min(a + len(sample), len(mix))
            mix[a:b] += sample[: b - a]

    for t in an.beats:
        add(hi if round(t, 4) in downs else lo, t)
    if an.drop:
        add(ping, an.drop.t)
    if section is not None:
        mix = mix[int(section.start * sr): int(section.end * sr)]
    peak = np.max(np.abs(mix))
    if peak > 0.99:
        mix = mix / peak * 0.99
    sf.write(out, mix, sr, subtype="PCM_16")
    return out
