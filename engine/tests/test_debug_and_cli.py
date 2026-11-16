"""End-to-end: real files through FFmpeg, the CLI, and the debug outputs."""

import json
import subprocess

import numpy as np
import soundfile as sf

from mog.__main__ import main
from mog.ffmpeg.audio_io import decode_audio
from mog.testing.synth import make_track


def _mp3(tmp_path, bpm=145):
    wav = tmp_path / "t.wav"
    mp3 = tmp_path / "t.mp3"
    y, truth = make_track(bpm=bpm, sr=44100)
    sf.write(wav, y, 44100, subtype="PCM_16")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(wav), "-b:a", "192k", str(mp3)], check=True)
    return mp3, truth


def test_decode_mp3_and_video_audio(tmp_path):
    mp3, _ = _mp3(tmp_path)
    y = decode_audio(mp3, sr=22050)
    assert y.dtype == np.float32 and len(y) > 22050 * 30
    # audio inside a video container (users may upload a video as the music source)
    mp4 = tmp_path / "v.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=black:s=64x64:d=5",
                    "-i", str(mp3), "-shortest", "-c:v", "libx264", "-c:a", "aac", str(mp4)], check=True)
    assert len(decode_audio(mp4, sr=22050)) > 22050 * 4


def test_debug_command_writes_outputs(tmp_path, capsys):
    mp3, truth = _mp3(tmp_path)
    out = tmp_path / "dbg"
    assert main(["debug", str(mp3), "--out", str(out), "--length", "15", "--tracker", "librosa"]) == 0
    png = out / "t.debug.png"
    assert png.exists() and png.stat().st_size > 50_000
    clicks, _ = sf.read(out / "t.clicks.wav")
    assert abs(len(clicks) / 44100 - len(make_track(sr=44100)[0]) / 44100) < 0.1
    sec_clicks, _ = sf.read(out / "t.section-clicks.wav")
    data = json.loads((out / "t.analysis.json").read_text())
    sec = data["section"]
    assert abs(len(sec_clicks) / 44100 - (sec["end"] - sec["start"])) < 0.05
    assert abs(data["analysis"]["drop"]["t"] - truth.drop) < 1 / 60
    assert "drop:" in capsys.readouterr().out


def test_cli_reports_errors_cleanly(tmp_path, capsys):
    assert main(["analyze", str(tmp_path / "missing.mp3")]) == 2
    assert "not found" in capsys.readouterr().err
    bad = tmp_path / "bad.mp3"
    bad.write_bytes(b"this is not audio")
    assert main(["analyze", str(bad)]) == 2
    assert "Could not read audio" in capsys.readouterr().err
