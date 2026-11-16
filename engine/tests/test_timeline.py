"""Timeline schema validation and speed-curve maths (fast, no media)."""

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from mog.timeline.models import Timeline
from mog.timeline.speed import footage_needed, segment_source_times, source_offsets, speed_at

SAMPLE = Path(__file__).parents[2] / "samples" / "phase2" / "sample_timeline.json"


def base(**over):
    d = {
        "output": {"width": 1080, "height": 1920, "fps": 60, "duration": 2.0},
        "audio": {"file": "a.wav", "start": 0, "end": 2.0},
        "segments": [
            {"id": "a", "source": "x.mp4", "src_in": 0, "timeline_in": 0, "timeline_out": 1.0},
            {"id": "b", "source": "y.mp4", "src_in": 0, "timeline_in": 1.0, "timeline_out": 2.0},
        ],
    }
    d.update(over)
    return d


def test_sample_timeline_is_valid_and_round_trips():
    tl = Timeline.model_validate_json(SAMPLE.read_text())
    again = Timeline.model_validate_json(tl.model_dump_json())
    assert again == tl


@pytest.mark.parametrize("mutate,msg", [
    (lambda d: d["segments"][1].update(timeline_in=1.1), "gap between segments"),
    (lambda d: d["segments"][1].update(timeline_in=0.9), "overlap between segments"),
    (lambda d: d["segments"][0].update(timeline_in=0.2), "must start at 0"),
    (lambda d: d["segments"][1].update(timeline_out=1.8), "must end at output.duration"),
    (lambda d: d["segments"][0].update(speed_curve=[[0, 20]]), "out of range"),
    (lambda d: d["segments"][0].update(speed_curve=[[0.5, 1], [0.2, 2]]), "time order"),
    (lambda d: d["segments"][0].update(bogus=1), "Extra inputs are not permitted"),
    (lambda d: d["segments"][0].update(crop={"mode": "face_track"}), "needs a 'path'"),
    (lambda d: d["segments"][0].update(effects=[{"type": "flash", "t": 1.5}]), "after the segment ends"),
    (lambda d: d["segments"][0].update(effects=[{"type": "flash", "t": 0.1, "color": "white"}]), "#RRGGBB"),
    (lambda d: d["segments"][0].update(effects=[{"type": "explode", "t": 0.1}]), "does not match any"),
    (lambda d: d["audio"].update(end=1.0), "audio section is shorter"),
    (lambda d: d["segments"][1].update(id="a"), "unique"),
    (lambda d: d["output"].update(fps=48), "Input should be"),
])
def test_invalid_timelines_are_rejected(mutate, msg):
    d = base()
    mutate(d)
    with pytest.raises(ValidationError, match=msg):
        Timeline.model_validate(d)


def test_half_frame_tolerance_on_tiling():
    d = base()
    d["segments"][1]["timeline_in"] = 1.0 + 0.4 / 60  # under half a frame: fine
    Timeline.model_validate(d)


def test_constant_speed():
    t = np.array([0.0, 0.5, 1.0])
    assert np.allclose(source_offsets([(0.0, 2.0)], t), [0, 1, 2])


def test_linear_ramp_matches_closed_form():
    # speed goes 2 -> 0.5 linearly over 1 s: footage = area under the line = (2 + 0.5) / 2
    curve = [(0.0, 2.0), (1.0, 0.5)]
    assert source_offsets(curve, np.array([1.0]), "linear")[0] == pytest.approx(1.25, abs=1e-4)
    # holds after the last keyframe
    assert source_offsets(curve, np.array([2.0]), "linear")[0] == pytest.approx(1.75, abs=1e-4)


@pytest.mark.parametrize("ease", ["linear", "smoothstep", "cubic"])
def test_symmetric_eases_consume_the_same_footage(ease):
    # Every ease here is symmetric, so a full ramp averages to the midpoint speed
    assert source_offsets([(0, 3.0), (1, 1.0)], np.array([1.0]), ease)[0] == pytest.approx(2.0, abs=1e-3)


def test_speed_at_holds_and_eases():
    curve = [(0.2, 2.0), (0.6, 0.4)]
    v = speed_at(curve, np.array([0.0, 0.2, 0.4, 0.6, 1.0]), "smoothstep")
    assert v[0] == 2.0 and v[1] == 2.0 and v[3] == pytest.approx(0.4) and v[4] == pytest.approx(0.4)
    assert v[2] == pytest.approx(1.2)  # smoothstep(0.5) = 0.5


def test_velocity_ramp_footage_needed():
    tl = Timeline.model_validate(base())
    seg = tl.segments[0].model_copy(update={"speed_curve": [(0, 2.5), (0.55, 2.5), (0.62, 0.3), (1.0, 0.3)]})
    # 0.55 s at 2.5x + 0.07 s ramp (averages 1.4x) + 0.38 s at 0.3x = 1.375 + 0.098 + 0.114
    assert footage_needed(seg) == pytest.approx(1.587, abs=1e-3)


def test_frames_owned_by_segment_round_to_nearest():
    d = base()
    d["segments"][0]["timeline_out"] = 1.004  # 60.24 frames -> frame 60 belongs to segment b
    d["segments"][1]["timeline_in"] = 1.004
    tl = Timeline.model_validate(d)
    fa, _ = segment_source_times(tl.segments[0], 60)
    fb, _ = segment_source_times(tl.segments[1], 60)
    assert fa[-1] == 59 and fb[0] == 60 and fb[-1] == 119


def test_schema_exports_are_up_to_date():
    from mog.timeline.export import json_schema, typescript

    web = Path(__file__).parents[2] / "web" / "lib"
    assert (web / "timeline.ts").read_text() == typescript(), "run `python -m mog schema`"
    assert json.loads((web / "timeline.schema.json").read_text()) == json_schema(), "run `python -m mog schema`"
