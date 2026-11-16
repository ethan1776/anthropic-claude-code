"""Editor operations on a timeline (the UI's "tap a segment" actions).

They live here, not in the browser, because they need the same knowledge the planner has:
how much footage a speed curve consumes, which shots are long enough, where the face is.
Each op returns a NEW valid Timeline (or raises MogError with a clear message).
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from mog.audio.models import AudioAnalysis
from mog.errors import MogError
from mog.planner.presets import Preset
from mog.timeline.models import Timeline
from mog.timeline.speed import footage_needed, source_offsets
from mog.video.models import Shot, VideoAnalysis

SpeedMode = Literal["normal", "slow", "fast", "ramp"]
EffectType = Literal["flash", "zoom_punch", "shake", "rgb_split", "motion_blur"]


def _seg_index(tl: Timeline, sid: str) -> int:
    for i, s in enumerate(tl.segments):
        if s.id == sid:
            return i
    raise MogError(f"Segment {sid} not found.")


def _dump(tl: Timeline) -> dict:
    return tl.model_dump()


def _crop_for(seg: dict, shot: Shot, content, src_in: float) -> dict:
    dur = seg["timeline_out"] - seg["timeline_in"]
    taus = np.arange(0.0, dur + 1e-9, 0.1)
    src = src_in + source_offsets([tuple(k) for k in seg["speed_curve"]], taus, seg.get("ease", "smoothstep"))
    cp = np.array(shot.crop_path)
    xs = np.interp(src - shot.start, cp[:, 0], cp[:, 1])
    ys = np.interp(src - shot.start, cp[:, 0], cp[:, 2])
    return {"mode": "face_track" if shot.face_path else "manual", "zoom": 1.0, "bounds": list(content),
            "path": [[round(float(t), 3), round(float(x), 4), round(float(y), 4)] for t, x, y in zip(taus, xs, ys)]}


def _need(seg: dict) -> float:
    from mog.timeline.models import Segment

    return footage_needed(Segment.model_validate(seg))


def swap_shot(tl: Timeline, sid: str, clips: list[VideoAnalysis], source: str | None = None,
              shot_index: int | None = None) -> Timeline:
    """Put a different shot in a segment. With no source given, pick the next best shot that
    isn't already used nearby (cycles through alternatives on repeated taps)."""
    d = _dump(tl)
    i = _seg_index(tl, sid)
    seg = d["segments"][i]
    need = _need(seg) + 0.05
    by_src = {c.source: c for c in clips}
    if source is not None:
        clip = by_src.get(source)
        if clip is None:
            raise MogError("That clip isn't in this project.")
        shots = [s for s in clip.shots if shot_index is None or s.index == shot_index]
        cands = [(clip, s) for s in shots if s.duration >= need]
        if not cands:
            raise MogError("That shot is too short for this segment at this speed.")
    else:
        cands = [(c, s) for c in clips for s in c.shots if s.duration >= need and s.quality >= 0.08]
        if not cands:
            raise MogError("No other shot is long enough for this segment.")
        current = (seg["source"], seg["src_in"])
        # order: best overall first; skip the current one; prefer clips not used by neighbours
        neigh = {d["segments"][j]["source"] for j in (i - 1, i + 1) if 0 <= j < len(d["segments"])}
        cands.sort(key=lambda cs: (cs[0].source in neigh, -(cs[1].stare_score + cs[1].action_score + cs[1].face_score)))
        idx = next((k for k, (c, s) in enumerate(cands) if c.source == current[0] and s.start <= current[1] < s.end), -1)
        cands = cands[idx + 1:] + cands[:idx + 1]
    clip, shot = cands[0]
    src_in = round(shot.start + max(0.0, (shot.duration - need) / 2), 4)
    seg.update(source=clip.source, src_in=src_in, exposure=shot.exposure,
               crop=_crop_for(seg, shot, clip.content, src_in))
    return Timeline.model_validate(d)


def set_speed(tl: Timeline, sid: str, mode: SpeedMode, clips: list[VideoAnalysis], preset: Preset) -> Timeline:
    d = _dump(tl)
    i = _seg_index(tl, sid)
    seg = d["segments"][i]
    dur = seg["timeline_out"] - seg["timeline_in"]
    sp = preset.speed
    if mode == "normal":
        curve, ease = [[0.0, 1.0]], "smoothstep"
    elif mode == "slow":
        curve, ease = [[0.0, sp.stare_speed]], "smoothstep"
    elif mode == "fast":
        curve, ease = [[0.0, 2.0]], "smoothstep"
    else:  # ramp: fast, snap to slow at the middle beat
        snap = round(dur / 2, 4)
        curve = [[0.0, sp.ramp_fast], [round(snap - min(0.06, snap * 0.3), 4), sp.ramp_fast], [snap, sp.ramp_slow],
                 [round(dur, 4), sp.ramp_slow]]
        ease = "cubic"
    seg["speed_curve"], seg["ease"] = curve, ease
    # keep footage valid: if the current shot is now too short, swap to one that fits
    need = _need(seg) + 0.05
    clip = next((c for c in clips if c.source == seg["source"]), None)
    shot = next((s for s in clip.shots if s.start - 1e-6 <= seg["src_in"] < s.end), None) if clip else None
    if clip is None or shot is None:
        raise MogError("This segment's clip is no longer in the project.")
    if seg["src_in"] + need > shot.end:
        if shot.duration >= need:
            seg["src_in"] = round(shot.end - need, 4)
        else:
            return swap_shot(Timeline.model_validate(_with_crop(seg, shot, clip, d)), sid, clips)
    seg["crop"] = _crop_for(seg, shot, clip.content, seg["src_in"])
    return Timeline.model_validate(d)


def _with_crop(seg, shot, clip, d):
    seg["crop"] = _crop_for(seg, shot, clip.content, seg["src_in"])
    return d


def toggle_effect(tl: Timeline, sid: str, effect: EffectType, on: bool, preset: Preset) -> Timeline:
    d = _dump(tl)
    seg = d["segments"][_seg_index(tl, sid)]
    dur = seg["timeline_out"] - seg["timeline_in"]
    seg["effects"] = [e for e in seg["effects"] if e["type"] != effect]
    if on:
        e = preset.effects
        if effect == "flash":
            seg["effects"].append({"type": "flash", "t": 0.0, "frames": e.flash_frames, "color": e.flash_color})
        elif effect == "zoom_punch":
            seg["effects"].append({"type": "zoom_punch", "t": 0.0, "strength": max(e.punch_strength, 1.15),
                                   "decay": round(min(0.3, dur), 3), "anchor": "face"})
        elif effect == "shake":
            seg["effects"].append({"type": "shake", "t": 0.0, "intensity": max(e.shake_intensity, 16),
                                   "decay": round(min(0.35, dur), 3)})
        elif effect == "rgb_split":
            seg["effects"].append({"type": "rgb_split", "t": 0.0, "frames": 3, "offset": 10})
        else:
            seg["effects"].append({"type": "motion_blur", "t_in": 0.0, "t_out": round(dur, 4), "strength": 0.7})
    return Timeline.model_validate(d)


def nudge_cut(tl: Timeline, sid: str, edge: Literal["in", "out"], beats: int, audio: AudioAnalysis,
              clips: list[VideoAnalysis]) -> Timeline:
    """Move the cut at the start ('in') or end ('out') of a segment by whole beats. The
    neighbouring segment grows/shrinks to match; cuts stay on the beat grid."""
    d = _dump(tl)
    i = _seg_index(tl, sid)
    j = i - 1 if edge == "in" else i + 1
    if not 0 <= j < len(d["segments"]):
        raise MogError("There's no cut on that side of this segment.")
    start = d["audio"]["start"]
    grid = np.array([b - start for b in audio.beats if start <= b <= d["audio"]["end"]])
    left, right = (d["segments"][j], d["segments"][i]) if edge == "in" else (d["segments"][i], d["segments"][j])
    cut = left["timeline_out"]
    k = int(np.argmin(np.abs(grid - cut))) + beats
    if not 0 <= k < len(grid):
        raise MogError("Can't move the cut past the song section.")
    new = round(float(grid[k]), 4)
    if new <= left["timeline_in"] + 0.05 or new >= right["timeline_out"] - 0.05:
        raise MogError("That would make a segment too short.")
    # The right-hand segment now starts later/earlier on the timeline: shift its source start
    # by the same amount of FOOTAGE so the picture stays in sync with where it was.
    delta_t = new - cut
    shift = float(source_offsets([tuple(k) for k in right["speed_curve"]], np.array([abs(delta_t)]),
                                 right.get("ease", "smoothstep"))[0]) * np.sign(delta_t)
    right["src_in"] = round(max(0.0, right["src_in"] + shift), 4)
    left["timeline_out"], right["timeline_in"] = new, new
    # Effects/speed keyframes are segment-local: drop those that fall outside the new lengths
    for s in (left, right):
        dur = s["timeline_out"] - s["timeline_in"]
        s["effects"] = [e for e in s["effects"] if (e.get("t", e.get("t_in", 0)) <= dur)]
        s["speed_curve"] = [k for k in s["speed_curve"] if k[0] <= dur + 1e-6] or [[0.0, s["speed_curve"][0][1]]]
    for s in (left, right):
        clip = next(c for c in clips if c.source == s["source"])
        shot = next((sh for sh in clip.shots if sh.start - 1e-6 <= s["src_in"] < sh.end), clip.shots[-1])
        if s["src_in"] + _need(s) > shot.end + 1e-3:
            s["src_in"] = round(max(shot.start, shot.end - _need(s) - 0.02), 4)
        s["crop"] = _crop_for(s, shot, clip.content, s["src_in"])
    return Timeline.model_validate(d)


def set_text(tl: Timeline, content: str | None, preset: Preset, drop: float | None) -> Timeline:
    d = _dump(tl)
    if not content:
        d["text"] = []
        return Timeline.model_validate(d)
    D = d["output"]["duration"]
    old = d["text"][0] if d["text"] else None
    t_in = old["t_in"] if old else round(min(max((drop or 0) + 1.5, 0.5), max(D - 1.0, 0.1)), 4)
    t_out = old["t_out"] if old else round(min(D - 0.15, t_in + 5.0), 4)
    d["text"] = [{"content": content[:120], "t_in": t_in, "t_out": t_out, "font": preset.text_font,
                  "anim": preset.text_anim, "y": 0.70, "size": 0.05}]
    return Timeline.model_validate(d)
