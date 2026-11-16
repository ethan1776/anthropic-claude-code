"""Stream raw frames out of / into FFmpeg through pipes.

FrameReader: decode a normalized clip from a given frame index, one BGR NumPy frame at a time.
FrameWriter: feed BGR frames into an FFmpeg process (which grades, adds audio and encodes).
"""

from __future__ import annotations

import subprocess
import threading
from collections import deque
from fractions import Fraction
from pathlib import Path

import numpy as np

from mog.errors import MogError
from mog.ffmpeg.audio_io import ffmpeg_bin


class FrameReader:
    """Sequential reader starting at an exact frame index.

    The clip must be CFR (see ingest): frame k sits at k / fps. We seek half a frame
    *before* frame k, so floating-point rounding can never make FFmpeg skip frame k.
    """

    def __init__(self, path: Path, fps: Fraction, start_frame: int, width: int, height: int,
                 out_size: tuple[int, int] | None = None):
        self.fps = fps
        self.index = start_frame  # index of the frame the next read() returns
        self.w, self.h = out_size or (width, height)
        ss = max(0.0, (start_frame - 0.5) / float(fps))
        vf = f"scale={self.w}:{self.h}:flags=area:in_color_matrix=bt709:in_range=tv:out_range=full,format=bgr24"
        cmd = [ffmpeg_bin(), "-v", "error", "-nostdin", "-ss", f"{ss:.6f}", "-i", str(path), "-an",
               "-vf", vf, "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8)
        self._size = self.w * self.h * 3
        self._last: np.ndarray | None = None

    def read(self) -> np.ndarray | None:
        assert self._proc.stdout is not None
        buf = self._proc.stdout.read(self._size)
        if len(buf) < self._size:
            return None
        self.index += 1
        self._last = np.frombuffer(buf, np.uint8).reshape(self.h, self.w, 3)
        return self._last

    def frame_at(self, k: int) -> np.ndarray:
        """Return frame k (k must not go backwards). Reuses the last frame if k repeats;
        holds the final frame if the clip runs out a frame early (rounding at clip end)."""
        if self._last is not None and k == self.index - 1:
            return self._last
        if k < self.index - 1:
            raise MogError("internal: FrameReader can only move forward")
        frame = self._last
        while self.index <= k:
            nxt = self.read()
            if nxt is None:
                if frame is None:
                    raise MogError("Ran out of video frames while reading a clip.")
                return frame
            frame = nxt
        assert frame is not None
        return frame

    def close(self) -> None:
        if self._proc.poll() is None:
            self._proc.kill()
        self._proc.wait()
        for s in (self._proc.stdout, self._proc.stderr):
            if s:
                s.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class FrameWriter:
    """Feeds frames to `ffmpeg -f rawvideo -i pipe:0 <extra args>`."""

    def __init__(self, width: int, height: int, fps: int, args_after_input: list[str]):
        cmd = [ffmpeg_bin(), "-v", "error", "-nostdin", "-y",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0",
               *args_after_input]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self._err: deque[bytes] = deque(maxlen=50)
        # Drain stderr in the background so FFmpeg never blocks on a full pipe
        self._t = threading.Thread(target=self._drain, daemon=True)
        self._t.start()
        self.shape = (height, width, 3)

    def _drain(self):
        assert self._proc.stderr is not None
        for line in self._proc.stderr:
            self._err.append(line)

    def write(self, frame: np.ndarray) -> None:
        if frame.shape != self.shape or frame.dtype != np.uint8:
            raise MogError(f"internal: frame shape {frame.shape} != {self.shape}")
        try:
            assert self._proc.stdin is not None
            self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError:
            raise MogError("The video encoder stopped unexpectedly: " + self.error_text()) from None

    def error_text(self) -> str:
        self._t.join(timeout=2)
        return b"".join(self._err).decode(errors="replace").strip()[-800:] or "no details"

    def close(self) -> None:
        if self._proc.stdin and not self._proc.stdin.closed:
            try:
                self._proc.stdin.close()
            except BrokenPipeError:
                pass
        code = self._proc.wait()
        self._t.join(timeout=5)
        if code != 0:
            raise MogError("Video encoding failed: " + self.error_text())

    def abort(self) -> None:
        if self._proc.poll() is None:
            self._proc.kill()
        self._proc.wait()
