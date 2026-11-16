import numpy as np

from mog.audio.beats import _fill_gaps, align_to_onsets, trim_to_music


def test_align_fixes_latency_and_quantization():
    """A tracker that is 15 ms late and quantized to 20 ms frames (like beat_this's 50 fps
    output) gets pulled back onto the real drum hits."""
    true = np.arange(1.0, 30.0, 60 / 145)
    late = np.round((true + 0.015) / 0.02) * 0.02
    hits = np.sort(np.concatenate([true[::2], true[1::4] + 0.0005]))  # hits on most beats
    beats, downs = align_to_onsets(late, late[::4], hits)
    assert np.max(np.abs(beats - true)) < 0.012
    covered = np.isin(np.round(true, 4), np.round(hits, 4))
    assert np.max(np.abs(beats[covered] - true[covered])) < 0.001
    assert np.allclose(downs, beats[::4])


def test_align_without_hits_is_noop():
    b = np.arange(0, 10, 0.5)
    out, d = align_to_onsets(b, b[::4], np.array([]))
    assert np.array_equal(out, b)


def test_fill_gaps_regrids_skipped_beats():
    period = 10.0
    frames = np.array([0, 10, 20, 50, 60])  # beats at 30 and 40 were skipped
    assert list(_fill_gaps(frames, period)) == [0, 10, 20, 30, 40, 50, 60]


def test_trim_to_music():
    beats = np.arange(0, 20, 0.5)
    b, d = trim_to_music(beats, beats[::4], np.array([2.0, 10.0]), 0.5)
    assert b.min() >= 1.75 and b.max() <= 10.25
