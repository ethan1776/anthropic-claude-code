"""Phase 8 edge cases: odd inputs must give a usable edit or a clear message, never a crash."""

import os
import subprocess
import tempfile

import numpy as np
import pytest
import soundfile as sf

from mog.errors import MogError
from mog.pipeline import make_edit
from mog.testing.clips import make_demo_clips
from mog.testing.synth import make_track


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    return make_demo_clips(tmp_path_factory.mktemp("clips"))


def test_song_shorter_than_target(tmp_path, clips):
    y, _ = make_track(bpm=140, intro_bars=2, main_bars=3, sr=44100)   # ~9 s song
    sf.write(tmp_path / "short.wav", y, 44100)
    res = make_edit(clips[:2], tmp_path / "short.wav", preset="mog", length=15)
    assert res.timeline.output.duration <= res.audio.duration + 1e-3
    assert any("shorter than the 15s target" in w for w in res.warnings)


def test_no_faces_and_tiny_clip(tmp_path, clips):
    y, _ = make_track(bpm=150, sr=44100)
    sf.write(tmp_path / "s.wav", y, 44100)
    tiny = tmp_path / "tiny.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=640x360:r=30:d=0.8",
                    "-pix_fmt", "yuv420p", str(tiny)], check=True)
    res = make_edit([tiny], tmp_path / "s.wav", preset="velocity", length=10)
    w = " ".join(res.warnings)
    assert "No faces found" in w and ("reused" in w or "slowed" in w)
    assert res.timeline.segments  # still a complete, valid edit


def test_bad_inputs_give_clear_errors(tmp_path, clips):
    y, _ = make_track(bpm=150, sr=44100)
    sf.write(tmp_path / "s.wav", y, 44100)
    with pytest.raises(MogError, match="at least one video clip"):
        make_edit([], tmp_path / "s.wav")
    with pytest.raises(MogError, match="File not found"):
        make_edit([tmp_path / "nope.mp4"], tmp_path / "s.wav")
    sf.write(tmp_path / "blip.wav", np.zeros(44100 * 2, np.float32) + 0.1, 44100)
    with pytest.raises(MogError, match="needs at least"):
        make_edit(clips[:1], tmp_path / "blip.wav")
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"\x00" * 1000)
    with pytest.raises(MogError, match="video"):
        make_edit([junk], tmp_path / "s.wav")


def test_no_temp_files_left_behind(tmp_path, clips):
    y, _ = make_track(bpm=150, sr=44100)
    sf.write(tmp_path / "s.wav", y, 44100)
    before = set(os.listdir(tempfile.gettempdir()))
    out = tmp_path / "out" / "e.mp4"
    make_edit(clips[:2], tmp_path / "s.wav", preset="clean_sigma", length=10, out=out, preview=True)
    after = set(os.listdir(tempfile.gettempdir()))
    assert not [n for n in after - before if n.startswith("mog-")]       # staging dir removed
    assert sorted(p.name for p in out.parent.iterdir()) == ["e.mp4"]     # no .part files
