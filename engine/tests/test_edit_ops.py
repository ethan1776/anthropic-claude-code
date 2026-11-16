"""Editor operations keep the timeline valid, on the beat grid, and inside the footage."""

import numpy as np
import pytest

from mog.errors import MogError
from mog.planner import edit_ops as ops
from mog.planner.presets import load_preset
from mog.timeline.speed import footage_needed
from test_planner import library, plan


@pytest.fixture(scope="module")
def music():
    from conftest import analyzed
    from mog.audio.section import pick_section

    an, _, _ = analyzed(145, 0.37)
    return an, pick_section(an, 15)


def fits(tl):
    lib = {c.source: c for c in library()}
    for s in tl.segments:
        end = s.src_in + footage_needed(s)
        assert any(sh.start - 1e-6 <= s.src_in and end <= sh.end + 1e-3 for sh in lib[s.source].shots), s.id


def test_swap_cycles_through_alternatives(music):
    tl, _ = plan(music)
    sid = tl.segments[3].id
    seen = set()
    cur = tl
    for _ in range(4):
        cur = ops.swap_shot(cur, sid, library())
        s = next(x for x in cur.segments if x.id == sid)
        seen.add((s.source, round(s.src_in, 2)))
        fits(cur)
    assert len(seen) >= 3


def test_swap_to_specific_shot_and_too_short(music):
    tl, _ = plan(music)
    sid = tl.segments[2].id  # the slow-mo drop: needs little footage
    out = ops.swap_shot(tl, sid, library(), source="mid.mp4", shot_index=2)
    assert next(s for s in out.segments if s.id == sid).source == "mid.mp4"
    tiny = library()
    tiny[2].shots[0] = tiny[2].shots[0].model_copy(update={"end": 0.2})
    with pytest.raises(MogError, match="too short"):
        ops.swap_shot(tl, tl.segments[0].id, tiny, source="mid.mp4", shot_index=0)


@pytest.mark.parametrize("mode", ["normal", "slow", "fast", "ramp"])
def test_speed_modes_keep_footage_valid(music, mode):
    tl, _ = plan(music)
    for s in tl.segments[:6]:
        tl = ops.set_speed(tl, s.id, mode, library(), load_preset("mog"))
    fits(tl)


def test_toggle_effects(music):
    tl, _ = plan(music, "clean_sigma")
    sid = tl.segments[5].id
    on = ops.toggle_effect(tl, sid, "shake", True, load_preset("clean_sigma"))
    assert any(e.type == "shake" for e in next(s for s in on.segments if s.id == sid).effects)
    off = ops.toggle_effect(on, sid, "shake", False, load_preset("clean_sigma"))
    assert not any(e.type == "shake" for e in next(s for s in off.segments if s.id == sid).effects)


def test_nudge_moves_cut_one_beat_and_stays_on_grid(music):
    an, sec = music
    tl, _ = plan(music)
    i = 6
    sid = tl.segments[i].id
    before = tl.segments[i].timeline_in
    out = ops.nudge_cut(tl, sid, "in", 1, an, library())
    after = next(s for s in out.segments if s.id == sid).timeline_in
    grid = np.array([b - sec.start for b in an.beats])
    assert after > before and np.min(np.abs(grid - after)) < 1e-3
    assert out.segments[i - 1].timeline_out == after          # neighbour follows
    fits(out)
    with pytest.raises(MogError):
        ops.nudge_cut(tl, tl.segments[0].id, "in", -1, an, library())   # nothing before the first cut


def test_text_add_change_remove(music):
    tl, _ = plan(music)
    t1 = ops.set_text(tl, "he doesn't chase.", load_preset("mog"), 3.3)
    assert t1.text[0].content == "he doesn't chase." and t1.text[0].t_in > 3.3
    t2 = ops.set_text(t1, "new line", load_preset("mog"), 3.3)
    assert t2.text[0].t_in == t1.text[0].t_in and t2.text[0].content == "new line"
    assert ops.set_text(t2, None, load_preset("mog"), 3.3).text == []
