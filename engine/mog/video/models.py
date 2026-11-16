"""Results of video analysis (cached as JSON per clip). Coordinates are fractions of the
full source frame; times are seconds from the start of the clip."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Shot(BaseModel):
    index: int
    start: float
    end: float
    face_score: float = Field(description="0..1: a large, centred face present most of the time")
    frontal: float = Field(description="0..1: face turned towards camera (1) vs profile (0)")
    motion_score: float = Field(description="0..1: how much the picture moves")
    stare_score: float = Field(description="0..1: big frontal face, little motion = slow-mo 'mog' shot")
    action_score: float = Field(description="0..1: lots of motion = good for fast sections")
    quality: float = Field(description="0..1: sharp, well exposed, enough resolution")
    brightness: float = Field(description="Mean brightness of the footage (or the face), 0..1")
    exposure: float = Field(description="Suggested exposure correction in stops")
    face_frac: float = Field(description="Fraction of the shot where a face was found")
    motion: float = Field(description="Median motion, frame-widths per second")
    crop_path: list[tuple[float, float, float]] = Field(description="[[t, x, y], ...] smoothed 9:16 crop centre")
    face_path: list[tuple[float, float, float, float]] | None = Field(
        None, description="[[t, x, y, size], ...] smoothed main face (None if no face)")

    @property
    def duration(self) -> float:
        return self.end - self.start


class VideoAnalysis(BaseModel):
    version: int
    source: str
    duration: float
    fps: float
    width: int
    height: int
    content: tuple[float, float, float, float] = Field(description="Usable footage area [x0, y0, x1, y1]")
    shots: list[Shot]
    warnings: list[str] = []
