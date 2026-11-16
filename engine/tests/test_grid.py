"""Beat-grid cleanup: slips get fixed, real splices are kept, off-beat locks get corrected."""

import numpy as np

from mog.audio.grid import pick_downbeats, regularize
from mog.audio.models import Onset

P = 60 / 140  # beat period


def hits(times, kind="kick", strength=1.0):
    return [Onset(t=float(t), strength=strength, kind=kind) for t in times]


def test_slipped_run_is_snapped_back():
    true = 1.0 + P * np.arange(64)
    tracked = true.copy()
    tracked[20:34] += 0.1  # tracker slipped by 100 ms for 14 beats
    g = regularize(tracked, hits(true))
    assert len(g.beats) == len(true)
    assert np.max(np.abs(g.beats - true)) < 1e-4
    assert g.breaks == []


def test_genuine_splice_is_kept():
    true = 1.0 + P * np.arange(64)
    music = true.copy()
    music[30:] += P / 2  # the audio was cut: from beat 30 the beat sits half a beat later
    g = regularize(music, hits(music))
    assert np.max(np.abs(g.beats - music)) < 1e-4
    assert len(g.breaks) == 1 and abs(g.breaks[0] - music[30]) < 1e-3


def test_skipped_beats_are_filled():
    true = 1.0 + P * np.arange(40)
    tracked = np.delete(true, [10, 11, 25])
    g = regularize(tracked, hits(true))
    assert np.max(np.abs(g.beats - true)) < 1e-4


def test_whole_song_locked_on_offbeat_is_shifted():
    true = 1.0 + P * np.arange(64)
    tracked = true + P / 2  # tracker followed the "and"s all song
    drums = hits(true) + hits(true[::2] + P / 2, kind="hihat", strength=0.3)
    g = regularize(tracked, drums)
    # Every output beat is on the true grid; at most the edge beat differs (a half-beat
    # shift leaves the first/last beat exactly on the boundary of the tracked span)
    assert all(np.min(np.abs(true - b)) < 1e-4 for b in g.beats)
    assert len(g.beats) >= len(true) - 1
    assert not g.offbeat_ambiguous


def test_ambiguous_offbeat_is_flagged():
    true = 1.0 + P * np.arange(64)
    drums = hits(true, "snare") + hits(true + P / 2, "kick", 0.95)
    g = regularize(true, drums)
    assert g.offbeat_ambiguous


def test_downbeats_chosen_per_piece():
    beats = 1.0 + P * np.arange(32)
    evidence = np.zeros(32)
    evidence[0:16:4] = 1.0       # first piece: bars start at beat 0
    evidence[16 + 2::4] = 1.0    # after the splice: bars start 2 beats later
    downs, conf = pick_downbeats(beats, [beats[16]], evidence)
    assert list(np.round(downs, 6)) == list(np.round(np.concatenate([beats[0:16:4], beats[18::4]]), 6))
    assert conf > 0.5


def test_downbeats_follow_tracker_hints():
    beats = 1.0 + P * np.arange(32)
    downs, conf = pick_downbeats(beats, [], np.zeros(32), hint_downbeats=beats[1::4] + 0.01)
    assert np.allclose(downs, beats[1::4]) and conf == 1.0


def test_drop_anchor_overrides_evidence():
    beats = 1.0 + P * np.arange(32)
    evidence = np.zeros(32)
    evidence[0::4] = 1.0  # weak evidence says bars start at beat 0
    downs, conf = pick_downbeats(beats, [], evidence, anchor=beats[10])  # drop on beat 10
    assert np.allclose(downs, beats[2::4]) and conf == 1.0
