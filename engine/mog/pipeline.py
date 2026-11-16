"""The whole job: music + clips + style -> finished edit. Used by the CLI and the worker.

Stages (names are shown in the UI's progress bar):
  "Finding the beat"   audio analysis + section picking
  "Scouting shots"     video analysis of each clip (cached per clip)
  "Directing"          the planner writes the timeline
  "Rendering"          the renderer executes it
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from mog.audio import analyze_audio, pick_section
from mog.audio.models import AudioAnalysis, Section
from mog.errors import MogError
from mog.planner.planner import PlanOptions, plan_edit
from mog.planner.presets import load_preset
from mog.render.renderer import RenderResult, render
from mog.timeline.models import Timeline
from mog.video.analyze import analyze_video

Progress = Callable[[str, float], None]

# Overall progress weights per stage (rendering dominates)
STAGES = {"Finding the beat": (0.0, 0.08), "Scouting shots": (0.08, 0.25), "Directing": (0.25, 0.28),
          "Rendering": (0.28, 1.0)}


@dataclass
class EditResult:
    timeline: Timeline
    render: RenderResult | None
    audio: AudioAnalysis
    section: Section
    warnings: list[str] = field(default_factory=list)
    seconds: float = 0.0


def stage_media(clips: list[Path], audio: Path, into: Path) -> tuple[list[str], str]:
    """Put links to all inputs in one folder (timelines name files relative to one media dir)."""
    into.mkdir(parents=True, exist_ok=True)
    names = []
    for i, c in enumerate(clips):
        name = f"{i + 1:02d}_{c.name}"
        _link(c, into / name)
        names.append(name)
    aname = f"audio{audio.suffix.lower()}"
    _link(audio, into / aname)
    return names, aname


def _link(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        shutil.copy2(src, dst)


def make_edit(clips: list[str | Path], audio: str | Path, preset: str = "mog", length: float = 15.0,
              seed: int = 1, intensity: float = 0.6, text: str | None = None, fps: int = 30,
              out: str | Path | None = None, preview: bool = False, section_start: float | None = None,
              progress: Progress | None = None, media_dir: Path | None = None) -> EditResult:
    """Plan (and, if `out` is given, render) an edit. `section_start` overrides the auto section."""
    t0 = time.time()
    report = progress or (lambda s, f: None)

    def stage(name: str, frac: float) -> None:
        a, b = STAGES[name]
        report(name, a + (b - a) * min(max(frac, 0.0), 1.0))

    clips = [Path(c) for c in clips]
    audio = Path(audio)
    if not clips:
        raise MogError("Add at least one video clip.")
    for p in clips + [audio]:
        if not p.exists():
            raise MogError(f"File not found: {p}")
    pr = load_preset(preset)

    stage("Finding the beat", 0.0)
    an = analyze_audio(audio)
    sec = pick_section(an, length)
    if section_start is not None:
        sec = _manual_section(an, section_start, length)
    stage("Finding the beat", 1.0)

    own_media = media_dir is None
    media = media_dir or Path(tempfile.mkdtemp(prefix="mog-media-"))
    try:
        names, aname = stage_media(clips, audio, media)
        analyses = []
        for i, name in enumerate(names):
            stage("Scouting shots", i / len(names))
            va = analyze_video(media / name)
            analyses.append(va)
        stage("Scouting shots", 1.0)

        stage("Directing", 0.0)
        tl, warnings = plan_edit(an, sec, analyses, pr,
                                 PlanOptions(seed=seed, intensity=intensity, fps=fps, text=text, audio_file=aname))
        stage("Directing", 1.0)

        res = None
        if out is not None:
            res = render(tl, out, media, preview=preview,
                         progress=lambda s, f: stage("Rendering", f if s == "Rendering" else (1.0 if s == "Encoding" else 0)))
            warnings += res.warnings
        report("Done", 1.0)
        return EditResult(tl, res, an, sec, list(dict.fromkeys(warnings)), time.time() - t0)
    finally:
        if own_media:
            shutil.rmtree(media, ignore_errors=True)


def _manual_section(an: AudioAnalysis, start: float, length: float) -> Section:
    """User-dragged section: snap start and end to the nearest downbeats."""
    import numpy as np

    downs = np.array(an.downbeats)
    if len(downs) == 0:
        raise MogError("This song has no detected bars to snap to.")
    s = float(downs[np.argmin(np.abs(downs - start))])
    e_target = s + length
    after = downs[downs > s + length * 0.5]
    e = float(after[np.argmin(np.abs(after - e_target))]) if len(after) else min(an.duration, e_target)
    e = min(e, an.duration)
    drop = an.drop.t if an.drop and s <= an.drop.t < e else None
    lead = int(round((drop - s) / an.bar_length)) if drop is not None else 0
    return Section(start=round(s, 4), end=round(e, 4), drop=drop, lead_bars=lead,
                   warnings=[] if drop is not None else ["The chosen section doesn't contain the drop."])
