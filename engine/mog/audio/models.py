"""Data models for audio analysis. All times are in seconds from the start of the track."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

OnsetKind = Literal["kick", "snare", "hihat", "cowbell"]


class Onset(BaseModel):
    t: float
    strength: float = Field(ge=0.0, le=1.0, description="0..1, relative to the loudest hits of this kind")
    kind: OnsetKind


class Curve(BaseModel):
    """A uniformly sampled curve. values[i] is the value at time i * hop."""

    hop: float
    values: list[float]

    def at(self, t: float) -> float:
        if not self.values:
            return 0.0
        i = int(round(t / self.hop))
        return self.values[min(max(i, 0), len(self.values) - 1)]


class Drop(BaseModel):
    t: float
    confidence: float = Field(ge=0.0, le=1.0)


class Bar(BaseModel):
    index: int
    start: float
    end: float
    energy: float = Field(description="Mean smoothed RMS energy in the bar, 0..1")
    level: Literal["low", "mid", "high"] = Field(description="Relative to this song's quietest/loudest bar")


class AudioAnalysis(BaseModel):
    version: int = 1
    source: str
    duration: float
    bpm: float
    beat_tracker: Literal["beat_this", "librosa"]
    beats: list[float]
    downbeats: list[float]
    onsets: list[Onset]
    energy: Curve = Field(description="Full-band RMS, smoothed ~0.5 s, normalized 0..1")
    low_energy: Curve = Field(description="40-150 Hz RMS, smoothed, normalized 0..1")
    drop: Drop | None
    bars: list[Bar]
    grid_breaks: list[float] = Field(default=[], description="Times where the beat grid jumps (spliced audio)")
    warnings: list[str] = []

    def onsets_of(self, kind: OnsetKind) -> list[Onset]:
        return [o for o in self.onsets if o.kind == kind]

    @property
    def bar_length(self) -> float:
        if len(self.downbeats) >= 2:
            import numpy as np

            return float(np.median(np.diff(self.downbeats)))
        return 4 * 60.0 / self.bpm


class Section(BaseModel):
    """The window of the song the edit will use."""

    start: float
    end: float
    drop: float | None
    lead_bars: int = Field(description="Bars between section start and the drop")
    warnings: list[str] = []

    @property
    def duration(self) -> float:
        return self.end - self.start
