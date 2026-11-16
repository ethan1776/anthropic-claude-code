"""The timeline: the single source of truth between the edit planner and the renderer.

The planner only writes this, the renderer only reads it, and the UI edits it. It is
plain JSON (see `Timeline.model_json_schema()`, exported to TypeScript for the web app).

Time conventions
----------------
* `segments[].timeline_in/out`, `text[].t_in/t_out`: seconds on the OUTPUT timeline.
* Inside a segment (`speed_curve` keyframes, `effects[].t`, `crop.path`): seconds from
  the START OF THAT SEGMENT. A segment can then be moved or nudged in the editor without
  rewriting everything inside it.
* `src_in`: seconds into the source clip where the segment starts reading.
* `audio.start/end`: seconds into the music file.
"""

from __future__ import annotations

import re
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

HEX_COLOR = re.compile(r"^#[0-9A-Fa-f]{6}$")
Ease = Literal["linear", "smoothstep", "cubic"]


class _Strict(BaseModel):
    # extra="forbid": typos in JSON are errors, not silently ignored.
    # serialization_defaults_required: in JSON the engine *writes*, defaulted fields are always present.
    model_config = ConfigDict(extra="forbid", json_schema_serialization_defaults_required=True)


class Output(_Strict):
    width: int = Field(1080, ge=16, le=4096, multiple_of=2)
    height: int = Field(1920, ge=16, le=4096, multiple_of=2)
    fps: Literal[24, 25, 30, 50, 60] = 60
    duration: float = Field(gt=0, le=600)


class Audio(_Strict):
    file: str
    start: float = Field(0.0, ge=0)
    end: float
    fade_in: float = Field(0.01, ge=0, le=5, description="Seconds. Default 10 ms avoids a click.")
    fade_out: float = Field(0.4, ge=0, le=10)

    @model_validator(mode="after")
    def _order(self):
        if self.end <= self.start:
            raise ValueError("audio.end must be after audio.start")
        return self


class Grade(_Strict):
    lut: str | None = Field(None, description="Built-in LUT name (e.g. 'bw_crush.cube') or a path to a .cube file")
    lut_strength: float = Field(1.0, ge=0, le=1, description="Mix between the original (0) and the LUT (1)")
    contrast: float = Field(1.0, ge=0.5, le=2.5)
    brightness: float = Field(0.0, ge=-0.3, le=0.3)
    gamma: float = Field(1.0, ge=0.5, le=2.0)
    saturation: float = Field(1.0, ge=0.0, le=3.0, description="0 = black & white")
    black_crush: float = Field(0.0, ge=0.0, le=0.4, description="Darkest X of the range goes to pure black")
    grain: float = Field(0.0, ge=0.0, le=1.0)
    vignette: float = Field(0.0, ge=0.0, le=1.0)
    glow: float = Field(0.0, ge=0.0, le=1.0)


class Crop(_Strict):
    """Which part of the source fills the 9:16 frame.

    x, y: centre of the crop window as a fraction of the source frame (0.5, 0.5 = centre).
    zoom: 1.0 = the smallest zoom that fills the output frame (no black bars); 2.0 = 2x closer.
    path: optional [[t, x, y], ...] keyframes (segment-local t), e.g. a face track. When
    present it overrides x/y. The window is always clamped inside the source, so no black
    edges can appear.
    """

    mode: Literal["center", "manual", "face_track"] = "center"
    x: float = Field(0.5, ge=0, le=1)
    y: float = Field(0.5, ge=0, le=1)
    zoom: float = Field(1.0, ge=1.0, le=4.0)
    path: list[tuple[float, float, float]] | None = None
    bounds: tuple[float, float, float, float] = Field(
        (0.0, 0.0, 1.0, 1.0),
        description="[x0, y0, x1, y1]: the part of the source that is real footage (excludes letterbox "
                    "bars, phone status bars). The crop never shows anything outside it.")

    @model_validator(mode="after")
    def _path(self):
        x0, y0, x1, y1 = self.bounds
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1) or x1 - x0 < 0.05 or y1 - y0 < 0.05:
            raise ValueError("crop.bounds must be [x0, y0, x1, y1] inside 0..1 with x0 < x1 and y0 < y1")
        if self.mode == "face_track" and not self.path:
            raise ValueError("crop.mode 'face_track' needs a 'path' of [t, x, y] keyframes")
        if self.path:
            ts = [p[0] for p in self.path]
            if ts != sorted(ts):
                raise ValueError("crop.path keyframes must be in time order")
            for _, x, y in self.path:
                if not (0 <= x <= 1 and 0 <= y <= 1):
                    raise ValueError("crop.path x/y must be between 0 and 1")
        return self


# ----------------------------------------------------------------------------- effects
# All `t` values are seconds from the start of the segment.

class ZoomPunch(_Strict):
    type: Literal["zoom_punch"] = "zoom_punch"
    t: float = Field(ge=0)
    strength: float = Field(1.2, ge=1.0, le=2.0, description="Peak scale, e.g. 1.2 = 120%")
    decay: float = Field(0.3, gt=0, le=3)
    anchor: Literal["center", "face"] = "center"


class Shake(_Strict):
    type: Literal["shake"] = "shake"
    t: float = Field(ge=0)
    intensity: float = Field(18, gt=0, le=200, description="Peak offset in pixels at 1080 px width")
    decay: float = Field(0.35, gt=0, le=3)


class Flash(_Strict):
    type: Literal["flash"] = "flash"
    t: float = Field(ge=0)
    frames: int = Field(2, ge=1, le=6)
    color: str = "#FFFFFF"
    strength: float = Field(1.0, gt=0, le=1)

    @model_validator(mode="after")
    def _color(self):
        if not HEX_COLOR.match(self.color):
            raise ValueError("flash.color must look like #RRGGBB")
        return self


class RgbSplit(_Strict):
    type: Literal["rgb_split"] = "rgb_split"
    t: float = Field(ge=0)
    frames: int = Field(3, ge=1, le=12)
    offset: float = Field(8, gt=0, le=60, description="Channel shift in pixels at 1080 px width")


class MotionBlur(_Strict):
    type: Literal["motion_blur"] = "motion_blur"
    t_in: float = Field(ge=0)
    t_out: float
    strength: float = Field(0.6, gt=0, le=1)


Effect = Annotated[Union[ZoomPunch, Shake, Flash, RgbSplit, MotionBlur], Field(discriminator="type")]


class Segment(_Strict):
    id: str
    source: str = Field(description="Clip file name, relative to the project media folder")
    src_in: float = Field(ge=0)
    timeline_in: float = Field(ge=0)
    timeline_out: float
    speed_curve: list[tuple[float, float]] = Field(
        default_factory=lambda: [(0.0, 1.0)],
        description="[[t, speed], ...]: playback speed keyframes (segment-local t). 1 = normal, 0.3 = slow-mo.",
    )
    ease: Ease = "smoothstep"
    exposure: float = Field(0.0, ge=-3.0, le=3.0,
                            description="Brightness correction in stops before the grade (+1 = twice as bright)")
    crop: Crop = Field(default_factory=Crop)
    effects: list[Effect] = Field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.timeline_out - self.timeline_in

    @model_validator(mode="after")
    def _check(self):
        if self.timeline_out <= self.timeline_in:
            raise ValueError(f"segment {self.id}: timeline_out must be after timeline_in")
        if not self.speed_curve:
            raise ValueError(f"segment {self.id}: speed_curve needs at least one keyframe")
        ts = [k[0] for k in self.speed_curve]
        if ts != sorted(ts):
            raise ValueError(f"segment {self.id}: speed_curve keyframes must be in time order")
        for t, v in self.speed_curve:
            if not 0.05 <= v <= 10:
                raise ValueError(f"segment {self.id}: speed {v} out of range (0.05 to 10)")
            if t < 0:
                raise ValueError(f"segment {self.id}: speed_curve times are segment-local and can't be negative")
        for e in self.effects:
            t = e.t_in if isinstance(e, MotionBlur) else e.t
            if t > self.duration + 1e-6:
                raise ValueError(f"segment {self.id}: {e.type} at t={t} is after the segment ends ({self.duration:.3f}s)")
        return self


class TextOverlay(_Strict):
    content: str = Field(min_length=1, max_length=120)
    t_in: float = Field(ge=0)
    t_out: float
    font: str = "Cinzel"
    size: float = Field(0.06, gt=0, le=0.3, description="Text height as a fraction of output height")
    y: float = Field(0.72, ge=0, le=1, description="Vertical position of the text centre (0 = top)")
    color: str = "#FFFFFF"
    anim: Literal["pop", "fade", "none"] = "pop"


class Timeline(_Strict):
    version: Literal[1] = 1
    seed: int | None = Field(None, description="Planner seed that produced this timeline (for re-roll)")
    output: Output
    audio: Audio
    grade: Grade = Field(default_factory=Grade)
    segments: list[Segment] = Field(min_length=1)
    text: list[TextOverlay] = Field(default_factory=list)

    @model_validator(mode="after")
    def _tiling(self):
        """Segments must cover the whole output, back to back, with no gaps or overlaps."""
        tol = 0.5 / self.output.fps  # half a frame
        segs = self.segments
        if abs(segs[0].timeline_in) > tol:
            raise ValueError(f"the first segment must start at 0 (starts at {segs[0].timeline_in})")
        for a, b in zip(segs, segs[1:]):
            if abs(b.timeline_in - a.timeline_out) > tol:
                kind = "gap" if b.timeline_in > a.timeline_out else "overlap"
                raise ValueError(f"{kind} between segments {a.id} and {b.id} "
                                 f"({a.timeline_out:.3f}s vs {b.timeline_in:.3f}s)")
        if abs(segs[-1].timeline_out - self.output.duration) > tol:
            raise ValueError(f"the last segment must end at output.duration ({self.output.duration}s), "
                             f"ends at {segs[-1].timeline_out}s")
        ids = [s.id for s in segs]
        if len(set(ids)) != len(ids):
            raise ValueError("segment ids must be unique")
        if self.audio.end - self.audio.start < self.output.duration - tol:
            raise ValueError("the audio section is shorter than the video")
        for tx in self.text:
            if tx.t_out <= tx.t_in or tx.t_out > self.output.duration + tol:
                raise ValueError(f"text '{tx.content}' has invalid timing")
        return self
