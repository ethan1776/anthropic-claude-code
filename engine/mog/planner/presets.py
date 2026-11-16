"""Style presets: plain JSON files in /presets (one per style). No style logic lives in code;
the planner reads these numbers. Add a style by adding a JSON file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from mog.errors import MogError
from mog.paths import PRESETS
from mog.timeline.models import Grade

PRESETS_DIR = PRESETS


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CutRules(_Strict):
    pre_drop: Literal["bar", "half_bar", "beat"] = Field("bar", description="Cut spacing before the drop")
    post_drop: Literal["bar", "half_bar", "beat", "half_beat"] = Field("beat", description="Cut spacing after the drop")
    calm_post_drop: Literal["bar", "half_bar", "beat"] = Field(
        "half_bar", description="Cut spacing after the drop in lower-energy bars")


class SpeedRules(_Strict):
    pre_drop: float = Field(1.0, ge=0.5, le=2.0)
    build_ramp: float = Field(2.0, ge=1.0, le=4.0, description="Speed reached at the end of the last bar before the drop (1 = off)")
    stare_speed: float = Field(0.35, ge=0.15, le=1.0, description="Slow-mo speed of the post-drop stare")
    stare_bars: float = Field(1.5, ge=0.5, le=4.0)
    velocity_ramps: bool = True
    ramp_every_bars: int = Field(1, ge=1, le=4)
    ramp_fast: float = Field(2.5, ge=1.2, le=4.0)
    ramp_slow: float = Field(0.35, ge=0.15, le=0.8)


class EffectRules(_Strict):
    drop_flash_frames: int = Field(3, ge=0, le=6)
    drop_punch: float = Field(1.25, ge=1.0, le=1.6, description="Zoom punch strength on the drop (1 = off)")
    drop_shake: float = Field(26, ge=0, le=80, description="Shake intensity on the drop (0 = off)")
    punch_on: Literal["none", "downbeat", "kick"] = "none"
    punch_strength: float = Field(1.15, ge=1.0, le=1.5)
    punch_prob: float = Field(0.5, ge=0, le=1)
    flash_on_cuts: float = Field(0.2, ge=0, le=1, description="Chance of a flash on a cut that lands on a snare/downbeat")
    flash_frames: int = Field(2, ge=1, le=6)
    flash_color: str = "#FFFFFF"
    shake_on_bass: float = Field(0.0, ge=0, le=1, description="Chance per post-drop bar of a shake on its bass hit")
    shake_intensity: float = Field(16, ge=0, le=80)
    rgb_on: Literal["none", "snare", "cowbell", "downbeat"] = "none"
    rgb_prob: float = Field(0.4, ge=0, le=1)
    motion_blur: float = Field(0.0, ge=0, le=1, description="Blur strength during fast sections (0 = off)")


class Preset(_Strict):
    id: str
    name: str
    description: str
    grade: Grade
    cuts: CutRules = Field(default_factory=CutRules)
    speed: SpeedRules = Field(default_factory=SpeedRules)
    effects: EffectRules = Field(default_factory=EffectRules)
    auto_exposure: bool = True
    text_font: str = "Cinzel"
    text_anim: Literal["pop", "fade", "none"] = "pop"
    preview: str | None = Field(None, description="Path of a looping preview clip for the style picker")


def list_presets(directory: Path = PRESETS_DIR) -> list[Preset]:
    return [load_preset(p.stem, directory) for p in sorted(directory.glob("*.json"))]


def load_preset(name: str, directory: Path = PRESETS_DIR) -> Preset:
    path = Path(name) if name.endswith(".json") else directory / f"{name}.json"
    if not path.exists():
        names = ", ".join(sorted(p.stem for p in directory.glob("*.json")))
        raise MogError(f"Unknown preset '{name}'. Available: {names}")
    try:
        return Preset.model_validate(json.loads(path.read_text()))
    except (ValidationError, json.JSONDecodeError) as e:
        raise MogError(f"Preset file {path.name} is invalid: {e}") from None
