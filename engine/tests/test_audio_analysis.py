"""Accuracy tests on synthetic phonk tracks with exact ground truth."""

import numpy as np
import pytest

from conftest import FRAME_60, SR, analyzed, match
from mog.audio.analyze import analyze_signal
from mog.audio.beats import estimate_tempo
from mog.errors import MogError
from mog.testing.synth import make_track


def test_bpm(track):
    an, truth, _ = track
    assert an.bpm == pytest.approx(truth.bpm, rel=0.01)


def test_beats_within_one_frame(track):
    an, truth, _ = track
    recall, precision, mae = match(an.beats, truth.beats, FRAME_60)
    assert recall >= 0.98
    assert precision >= 0.97
    assert mae < 0.006  # average error under 6 ms


def test_downbeats(track):
    an, truth, _ = track
    recall, precision, _ = match(an.downbeats, truth.downbeats, FRAME_60)
    assert recall >= 0.95
    assert precision >= 0.9


@pytest.mark.parametrize(
    "kind,attr,min_recall,min_precision",
    [("kick", "kicks", 0.95, 0.95), ("snare", "snares", 0.95, 0.95),
     ("hihat", "hats", 0.95, 0.95), ("cowbell", "cowbells", 0.95, 0.95)],
)
def test_onsets_by_kind(track, kind, attr, min_recall, min_precision):
    an, truth, _ = track
    det = [o.t for o in an.onsets_of(kind)]
    recall, precision, _ = match(det, getattr(truth, attr), FRAME_60)
    assert recall >= min_recall, f"{kind} recall {recall:.2f}"
    assert precision >= min_precision, f"{kind} precision {precision:.2f}"


def test_onset_strengths_in_range(track):
    an, _, _ = track
    assert all(0.0 <= o.strength <= 1.0 for o in an.onsets)
    assert an.onsets == sorted(an.onsets, key=lambda o: (o.t, o.kind))


def test_drop_found_on_the_frame(track):
    an, truth, _ = track
    assert an.drop is not None
    assert abs(an.drop.t - truth.drop) < FRAME_60
    assert an.drop.confidence >= 0.7


def test_energy_curve_rises_at_drop(track):
    an, truth, _ = track
    before = an.energy.at(truth.drop - 2.0)
    after = an.energy.at(truth.drop + 1.0)
    assert after > before * 2
    assert all(0.0 <= v <= 1.0 for v in an.energy.values)


def test_bars_cover_downbeats(track):
    an, _, _ = track
    assert len(an.bars) == len(an.downbeats)
    levels = {b.level for b in an.bars}
    assert "high" in levels and "low" in levels  # intro is quiet, main part is loud


def test_robust_to_noise_and_level():
    """Same track buried in noise at -20 dB and turned down 12 dB."""
    y, truth = make_track(bpm=140, offset=0.8, sr=SR, seed=3)
    rng = np.random.default_rng(7)
    noise = rng.standard_normal(len(y)).astype(np.float32) * 0.1 * np.sqrt(np.mean(y**2))  # -20 dB
    an = analyze_signal((y + noise) * 0.25, SR, tracker="librosa")
    assert an.bpm == pytest.approx(140, rel=0.01)
    recall, precision, _ = match(an.beats, truth.beats, FRAME_60)
    assert recall >= 0.95 and precision >= 0.95
    assert abs(an.drop.t - truth.drop) < FRAME_60


def test_no_drop_track_has_low_confidence():
    an, _, _ = analyzed(145, 0.37, with_drop=False)
    assert an.drop is None or an.drop.confidence < 0.5


def test_comb_tempo_ignores_syncopation():
    """A syncopated 3-step cowbell rhythm makes naive estimators report 4/3 x the tempo."""
    for bpm in (108, 130, 145):
        _, truth, y = analyzed(bpm, 0.37 if bpm == 145 else (0.6 if bpm == 108 else 1.1))
        import librosa

        env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=128, n_fft=1024, aggregate=np.median)
        assert estimate_tempo(env, SR / 128) == pytest.approx(bpm, rel=0.005)


def test_deterministic():
    y, _ = make_track(bpm=150, sr=SR)
    a = analyze_signal(y, SR, tracker="librosa")
    b = analyze_signal(y, SR, tracker="librosa")
    assert a.model_dump() == b.model_dump()


def test_rejects_short_and_silent_audio():
    with pytest.raises(MogError, match="at least"):
        analyze_signal(np.zeros(SR * 2, np.float32) + 0.1, SR)
    with pytest.raises(MogError, match="silent"):
        analyze_signal(np.zeros(SR * 10, np.float32), SR)


@pytest.mark.parametrize("bpm", [115, 140, 155])
def test_dirty_master(bpm):
    """Reverb (-10 dB), hard saturation and a noise floor, like a real phonk master.

    Beats, downbeats and the drop must stay frame-accurate. Drum labels are allowed to
    degrade a bit (distortion smears the differences between drum types)."""
    from mog.testing.synth import dirty_master

    y, truth = make_track(bpm=bpm, offset=0.9, sr=SR, seed=5)
    an = analyze_signal(dirty_master(y, SR), SR, tracker="librosa")
    assert an.bpm == pytest.approx(bpm, rel=0.01)
    r, p, _ = match(an.beats, truth.beats, FRAME_60)
    assert r >= 0.97 and p >= 0.95
    r, p, _ = match(an.downbeats, truth.downbeats, FRAME_60)
    assert r >= 0.95 and p >= 0.9
    assert abs(an.drop.t - truth.drop) < FRAME_60
    for kind, attr, floor in [("kick", "kicks", 0.8), ("snare", "snares", 0.9),
                              ("hihat", "hats", 0.9), ("cowbell", "cowbells", 0.75)]:
        r, p, _ = match([o.t for o in an.onsets_of(kind)], getattr(truth, attr), FRAME_60)
        assert r >= floor and p >= 0.8, f"{kind}: recall {r:.2f} precision {p:.2f}"
