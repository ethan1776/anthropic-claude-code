"""Edit planner rules, on the synthetic track + hand-built clip analyses (no video needed)."""

import numpy as np
import pytest

from conftest import analyzed
from mog.audio.section import pick_section
from mog.planner.planner import PlanOptions, plan_edit
from mog.planner.presets import list_presets, load_preset
from mog.timeline.speed import footage_needed
from mog.video.models import Shot, VideoAnalysis


def shot(i, a, b, face=0.5, motion=0.3, stare=None, action=None, quality=0.8):
    stare = stare if stare is not None else face * (1 - motion) ** 1.5
    action = action if action is not None else motion
    return Shot(index=i, start=a, end=b, face_score=face, frontal=0.8, motion_score=motion, stare_score=stare,
                action_score=action, quality=quality, brightness=0.4, exposure=0.3, face_frac=1.0 if face else 0,
                motion=motion, crop_path=[(0.0, 0.5, 0.5), (b - a, 0.55, 0.5)],
                face_path=[(0.0, 0.5, 0.4, 0.2), (b - a, 0.5, 0.4, 0.2)] if face else None)


def clip(name, shots, dur):
    return VideoAnalysis(version=1, source=name, duration=dur, fps=30, width=1920, height=1080,
                         content=(0.0, 0.05, 1.0, 0.95), shots=shots)


@pytest.fixture(scope="module")
def music():
    an, truth, _ = analyzed(145, 0.37)
    return an, pick_section(an, 15)


def library():
    return [
        clip("stare.mp4", [shot(0, 0, 6, face=0.9, motion=0.05, stare=0.95)], 6),
        clip("action.mp4", [shot(0, 0, 5, face=0.2, motion=0.9), shot(1, 5, 12, face=0.1, motion=0.8)], 12),
        clip("mid.mp4", [shot(0, 0, 4, face=0.6, motion=0.4), shot(1, 4, 9, face=0.5, motion=0.3),
                         shot(2, 9, 15, face=0.4, motion=0.5)], 15),
    ]


def plan(music, preset="mog", **kw):
    an, sec = music
    return plan_edit(an, sec, kw.pop("clips", library()), load_preset(preset), PlanOptions(**kw))


@pytest.mark.parametrize("preset", [p.id for p in list_presets()])
def test_every_preset_plans_a_valid_edit(music, preset):
    tl, _ = plan(music, preset)
    an, sec = music
    grid = np.array([b - sec.start for b in an.beats] + [0.0, sec.end - sec.start])
    for s in tl.segments:
        # every cut lands on a beat (never off-beat)
        assert np.min(np.abs(grid - s.timeline_in)) < 1e-3, (preset, s.id, s.timeline_in)
    assert tl.output.duration == pytest.approx(sec.end - sec.start, abs=1e-3)


def test_the_drop_is_the_strongest_moment(music):
    tl, _ = plan(music)
    an, sec = music
    drop = sec.drop - sec.start
    seg = next(s for s in tl.segments if abs(s.timeline_in - drop) < 1e-3)
    types = {e.type for e in seg.effects}
    assert {"flash", "zoom_punch", "shake"} <= types
    assert seg.source == "stare.mp4"                       # the best stare shot
    assert max(v for _, v in seg.speed_curve) < 0.5        # in slow motion
    assert seg.duration >= 1.5 * 4 * 60 / 145 - 1e-3       # held 1.5-2 bars
    punches = [e.strength for s in tl.segments for e in s.effects if e.type == "zoom_punch"]
    drop_punch = next(e.strength for e in seg.effects if e.type == "zoom_punch")
    assert drop_punch == max(punches)


def test_velocity_ramps_snap_on_downbeats(music):
    tl, _ = plan(music, "velocity")
    an, sec = music
    downs = np.array([d - sec.start for d in an.downbeats])
    ramps = [s for s in tl.segments if len(s.speed_curve) == 4 and s.speed_curve[0][1] > 2]
    assert ramps
    for s in ramps:
        snap = s.timeline_in + s.speed_curve[2][0]          # where the speed reaches slow
        assert np.min(np.abs(downs - snap)) < 1e-3
        assert s.speed_curve[2][1] < 0.5


def test_effect_density_rules(music):
    for seed in range(1, 6):
        tl, _ = plan(music, "drift_phonk", seed=seed, intensity=1.0)
        beat = 60 / 145
        flashes = sorted(s.timeline_in + e.t for s in tl.segments for e in s.effects if e.type == "flash")
        assert all(b - a > 0.85 * beat for a, b in zip(flashes, flashes[1:]))       # <= 1 flash per beat
        shakes = [s.timeline_in + e.t for s in tl.segments for e in s.effects if e.type == "shake"]
        bars = [int(t // (4 * beat)) for t in shakes]
        assert len(bars) == len(set(bars))                                           # <= 1 shake per bar
        cut_fx = [next((e.type for e in s.effects if getattr(e, "t", -1) == 0.0 and e.type in
                        ("flash", "rgb_split")), "none") for s in tl.segments]
        for a, b, c in zip(cut_fx, cut_fx[1:], cut_fx[2:]):
            assert not (a == b == c != "none")                                       # variety


def test_no_footage_reused_when_there_is_enough(music):
    tl, warnings = plan(music)
    ranges = {}
    for s in tl.segments:
        need = footage_needed(s)
        for a, b in ranges.get(s.source, []):
            assert s.src_in >= b - 1e-6 or s.src_in + need <= a + 1e-6, f"{s.id} overlaps"
        ranges.setdefault(s.source, []).append((s.src_in, s.src_in + need))
    assert not any("reused" in w for w in warnings)


def test_footage_stays_inside_shots(music):
    lib = {c.source: c for c in library()}
    tl, _ = plan(music)
    for s in tl.segments:
        end = s.src_in + footage_needed(s)
        assert any(sh.start - 1e-6 <= s.src_in and end <= sh.end + 1e-6 for sh in lib[s.source].shots), s.id


def test_too_little_footage_reuses_with_new_crop_and_warns(music):
    tiny = [clip("one.mp4", [shot(0, 0, 3, face=0.8, motion=0.2)], 3)]
    tl, warnings = plan(music, clips=tiny)
    assert any("reused" in w for w in warnings)
    assert any(s.crop.zoom > 1.1 for s in tl.segments)


def test_same_seed_same_edit_other_seed_different(music):
    a, _ = plan(music, seed=7)
    b, _ = plan(music, seed=7)
    c, _ = plan(music, seed=8)
    assert a.model_dump() == b.model_dump()
    assert a.model_dump() != c.model_dump()
    # clip order doesn't matter
    d, _ = plan(music, seed=7, clips=list(reversed(library())))
    assert d.model_dump() == a.model_dump()


def test_clips_are_balanced(music):
    tl, _ = plan(music)
    counts = {}
    for s in tl.segments:
        counts[s.source] = counts.get(s.source, 0) + 1
    assert len(counts) == 3 and max(counts.values()) <= 0.6 * len(tl.segments)


def test_text_pops_after_the_drop(music):
    tl, _ = plan(music, text="he doesn't chase.")
    an, sec = music
    assert tl.text[0].t_in > sec.drop - sec.start
    assert tl.text[0].font == "Cinzel"


def test_intensity_scales_effects(music):
    lo, _ = plan(music, "mog", intensity=0.0, seed=3)
    hi, _ = plan(music, "mog", intensity=1.0, seed=3)
    count = lambda tl: sum(len(s.effects) for s in tl.segments)  # noqa: E731
    assert count(hi) > count(lo)
