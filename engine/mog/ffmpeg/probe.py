"""ffprobe wrapper: what's in a media file."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from mog.errors import MogError


def ffprobe_bin() -> str:
    path = shutil.which("ffprobe")
    if not path:
        raise MogError("ffprobe is not installed or not on PATH (it comes with FFmpeg).")
    return path


@dataclass
class VideoInfo:
    path: Path
    width: int            # as displayed (rotation applied)
    height: int
    fps: Fraction         # average frame rate (what was actually recorded)
    nominal_fps: Fraction | None  # the container's declared rate (r_frame_rate)
    duration: float
    rotation: int         # degrees, from phone metadata
    pix_fmt: str
    color_transfer: str   # e.g. "bt709", "arib-std-b67" (HLG), "smpte2084" (PQ / HDR10)
    has_audio: bool
    frames: int | None = None

    @property
    def is_hdr(self) -> bool:
        return self.color_transfer in ("arib-std-b67", "smpte2084")


def _fraction(s: str | None) -> Fraction | None:
    if not s or s in ("0/0", "0"):
        return None
    try:
        f = Fraction(s)
    except (ValueError, ZeroDivisionError):
        return None
    return f if f > 0 else None


def probe_video(path: str | Path, count_frames: bool = False) -> VideoInfo:
    path = Path(path)
    if not path.exists():
        raise MogError(f"Video file not found: {path}")
    cmd = [ffprobe_bin(), "-v", "error", "-show_streams", "-show_format", "-of", "json"]
    if count_frames:
        cmd += ["-count_packets"]
    proc = subprocess.run(cmd + [str(path)], capture_output=True)
    if proc.returncode != 0:
        raise MogError(f"Could not read {path.name}: it doesn't look like a video file.")
    data = json.loads(proc.stdout or b"{}")
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"
                  and not s.get("disposition", {}).get("attached_pic")), None)
    if video is None:
        raise MogError(f"{path.name} has no video track.")
    rotation = 0
    for sd in video.get("side_data_list", []) or []:
        if "rotation" in sd:
            rotation = int(round(float(sd["rotation"]))) % 360
    if not rotation and "rotate" in video.get("tags", {}):
        rotation = int(video["tags"]["rotate"]) % 360
    w, h = int(video["width"]), int(video["height"])
    if rotation in (90, 270):
        w, h = h, w
    fps = _fraction(video.get("avg_frame_rate")) or _fraction(video.get("r_frame_rate"))
    if fps is None:
        raise MogError(f"Could not work out the frame rate of {path.name}.")
    duration = float(video.get("duration") or data.get("format", {}).get("duration") or 0)
    if duration <= 0:
        raise MogError(f"{path.name} has no duration (is it a still image?).")
    frames = int(video["nb_read_packets"]) if count_frames and "nb_read_packets" in video else None
    return VideoInfo(
        path=path, width=w, height=h, fps=fps, nominal_fps=_fraction(video.get("r_frame_rate")),
        duration=duration, rotation=rotation,
        pix_fmt=video.get("pix_fmt", ""), color_transfer=video.get("color_transfer", "") or "",
        has_audio=any(s.get("codec_type") == "audio" for s in streams), frames=frames,
    )
