"""End to end: synthetic music + synthetic clips -> planned and rendered edit."""

import json

import soundfile as sf

from mog.__main__ import main
from mog.ffmpeg.probe import probe_video
from mog.pipeline import make_edit
from mog.testing.clips import make_demo_clips
from mog.testing.synth import make_track


def test_make_edit_end_to_end(tmp_path):
    clips = make_demo_clips(tmp_path / "clips")
    y, truth = make_track(bpm=145, sr=44100)
    sf.write(tmp_path / "song.wav", y, 44100)
    seen = []
    res = make_edit(clips, tmp_path / "song.wav", preset="velocity", length=10, seed=3, text="test caption",
                    out=tmp_path / "edit.mp4", preview=True, progress=lambda s, f: seen.append((s, f)))
    info = probe_video(res.render.path, count_frames=True)
    assert (info.width, info.height) == (540, 960) and info.has_audio
    assert info.frames == int(round(res.timeline.output.duration * 30))
    stages = [s for s, _ in seen]
    for name in ("Finding the beat", "Scouting shots", "Directing", "Rendering", "Done"):
        assert name in stages
    fracs = [f for _, f in seen]
    assert fracs == sorted(fracs) and fracs[-1] == 1.0       # progress only goes up
    assert abs((res.section.drop or 0) - truth.drop) < 1 / 30


def test_cli_make_plan_only(tmp_path, capsys):
    clips = make_demo_clips(tmp_path / "clips")
    y, _ = make_track(bpm=130, sr=44100)
    sf.write(tmp_path / "song.wav", y, 44100)
    out = tmp_path / "e.mp4"
    assert main(["make", "--clips", *map(str, clips), "--audio", str(tmp_path / "song.wav"), "--preset", "clean_sigma",
                 "--length", "10", "--plan-only", "--out", str(out)]) == 0
    tl = json.loads(out.with_suffix(".timeline.json").read_text())
    assert tl["segments"] and not out.exists()
    assert main(["make", "--clips", str(clips[0]), "--audio", str(tmp_path / "song.wav"), "--preset", "nope",
                 "--plan-only", "--out", str(out)]) == 2
    assert "Unknown preset" in capsys.readouterr().err
