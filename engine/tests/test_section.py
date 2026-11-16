"""Section picker tests on hand-built analyses (no audio needed, so these are fast)."""

import numpy as np
import pytest

from mog.audio.models import AudioAnalysis, Curve, Drop
from mog.audio.section import pick_section


def fake_analysis(bpm=140.0, duration=120.0, drop=40.0, offset=0.5, conf=0.9) -> AudioAnalysis:
    beat = 60 / bpm
    beats = list(np.arange(offset, duration, beat))
    downbeats = beats[::4]
    n = int(duration / 0.05)
    return AudioAnalysis(
        source="fake", duration=duration, bpm=bpm, beat_tracker="librosa",
        beats=beats, downbeats=downbeats, onsets=[],
        energy=Curve(hop=0.05, values=[0.5] * n), low_energy=Curve(hop=0.05, values=[0.5] * n),
        drop=Drop(t=drop, confidence=conf) if drop is not None else None, bars=[],
    )


def is_downbeat(an, t):
    return min(abs(np.array(an.downbeats) - t)) < 1e-3 or abs(t - an.duration) < 1e-3


@pytest.mark.parametrize("target", [10, 15, 20, 30])
def test_contains_drop_snapped_to_downbeats(target):
    an = fake_analysis()
    drop = an.downbeats[17]
    an.drop = Drop(t=drop, confidence=0.9)
    sec = pick_section(an, target)
    assert sec.start < drop < sec.end
    assert is_downbeat(an, sec.start) and is_downbeat(an, sec.end)
    assert 2 <= sec.lead_bars <= 4
    assert sec.start == pytest.approx(drop - sec.lead_bars * an.bar_length, abs=2e-3)
    assert abs(sec.duration - target) <= an.bar_length / 2 + 1e-6


def test_lead_bars_grow_with_length():
    an = fake_analysis()
    an.drop = Drop(t=an.downbeats[17], confidence=0.9)
    assert pick_section(an, 10).lead_bars == 2
    assert pick_section(an, 15).lead_bars == 2
    assert pick_section(an, 20).lead_bars == 3
    assert pick_section(an, 30).lead_bars == 4


def test_drop_near_song_start():
    an = fake_analysis()
    an.drop = Drop(t=an.downbeats[1], confidence=0.9)
    sec = pick_section(an, 15)
    assert sec.start == pytest.approx(an.downbeats[0])
    assert sec.lead_bars == 1
    assert sec.start < an.drop.t < sec.end


def test_drop_near_song_end_shifts_window_left():
    an = fake_analysis(duration=60.0)
    an.drop = Drop(t=an.downbeats[-2], confidence=0.9)
    sec = pick_section(an, 20)
    assert sec.end <= an.duration + 1e-6
    assert sec.start < an.drop.t < sec.end
    assert sec.duration >= 15


def test_song_shorter_than_target():
    an = fake_analysis(duration=8.0, drop=None)
    sec = pick_section(an, 15)
    assert sec.end == pytest.approx(8.0)
    assert any("shorter" in w for w in sec.warnings)


def test_no_drop_uses_loudest_part_and_warns():
    an = fake_analysis(drop=None)
    vals = [0.2] * len(an.energy.values)
    loud_from = int(an.downbeats[20] / 0.05)
    for i in range(loud_from, min(loud_from + 300, len(vals))):
        vals[i] = 1.0
    an.energy = Curve(hop=0.05, values=vals)
    sec = pick_section(an, 15)
    assert any("No clear drop" in w for w in sec.warnings)
    assert sec.drop == pytest.approx(an.downbeats[20])


def test_low_confidence_warns():
    an = fake_analysis(drop=None)
    an.drop = Drop(t=an.downbeats[10], confidence=0.1)
    assert any("unsure" in w for w in pick_section(an, 15).warnings)
