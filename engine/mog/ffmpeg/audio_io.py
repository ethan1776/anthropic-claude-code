"""Decode any audio (or the audio track of a video) to a mono float32 NumPy array.

We always go through FFmpeg so MP3 / WAV / M4A / MOV / MP4 all behave identically
and we never depend on which codecs libsndfile happens to support.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np

from mog.errors import MogError


def ffmpeg_bin() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise MogError("FFmpeg is not installed or not on PATH. Install it (e.g. `apt install ffmpeg`).")
    return path


def decode_audio(path: str | Path, sr: int = 22050, mono: bool = True) -> np.ndarray:
    """Return audio as float32 in [-1, 1]. Shape (n,) if mono else (n, 2)."""
    path = Path(path)
    if not path.exists():
        raise MogError(f"Audio file not found: {path}")
    channels = 1 if mono else 2
    cmd = [
        ffmpeg_bin(), "-v", "error", "-nostdin", "-i", str(path),
        "-vn", "-ac", str(channels), "-ar", str(sr), "-f", "f32le", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        msg = proc.stderr.decode(errors="replace").strip().splitlines()
        detail = msg[-1] if msg else "unknown error"
        raise MogError(f"Could not read audio from {path.name}: {detail}")
    y = np.frombuffer(proc.stdout, dtype=np.float32)
    if y.size == 0:
        raise MogError(f"{path.name} has no audio track (or it is empty).")
    if not mono:
        y = y.reshape(-1, 2)
    return y.copy()
