"""Looping 3-second style previews for the style picker (web/public/presets/<id>.mp4).

Made by the real engine from synthetic footage and the synthetic track (no rights issues):
a short edit per preset, then the 3 seconds around the drop, small and silent.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import soundfile as sf

from mog.ffmpeg.audio_io import ffmpeg_bin
from mog.pipeline import make_edit
from mog.planner.presets import list_presets
from mog.testing.clips import make_demo_clips
from mog.testing.synth import make_track


def make_previews(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        clips = make_demo_clips(td / "clips")
        y, truth = make_track(bpm=145, sr=44100)
        sf.write(td / "song.wav", y, 44100)
        made = []
        for p in list_presets():
            res = make_edit(clips, td / "song.wav", preset=p.id, length=10, seed=2, out=td / f"{p.id}.mp4",
                            preview=True)
            drop = (res.section.drop or res.section.start) - res.section.start
            start = max(0.0, drop - 1.0)
            dst = out_dir / f"{p.id}.mp4"
            subprocess.run([ffmpeg_bin(), "-v", "error", "-y", "-ss", f"{start:.3f}", "-t", "3", "-i",
                            str(res.render.path), "-an", "-vf", "scale=270:480", "-c:v", "libx264", "-crf", "28",
                            "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)],
                           check=True)
            made.append(dst)
        return made
