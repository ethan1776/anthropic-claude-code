"""Ingest: convert every uploaded clip into a predictable intermediate file.

Phone and camera footage is messy: variable frame rate, rotation flags, HDR, odd sizes,
4K. The renderer needs to ask for "frame 137" and get exactly that frame, fast. So each
clip is converted once (and cached) to:

  * constant frame rate (CFR) at the clip's own standard rate (23.976 ... 60 fps).
    Frame k is at exactly k / fps seconds; that's what makes frame-accurate cuts possible.
  * rotation applied (portrait phone video comes out portrait)
  * HDR (iPhone HLG / HDR10) tone-mapped to normal SDR Rec.709
  * no bigger than needed: enough for a 1080x1920 crop with 1.5x zoom headroom
  * H.264, a keyframe every second (fast seeking), high quality (CRF 14), no audio
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from mog.errors import MogError
from mog.ffmpeg.audio_io import ffmpeg_bin
from mog.ffmpeg.probe import VideoInfo, probe_video

INGEST_VERSION = 1
STANDARD_RATES = [Fraction(24000, 1001), Fraction(24), Fraction(25), Fraction(30000, 1001), Fraction(30),
                  Fraction(50), Fraction(60000, 1001), Fraction(60)]
MAX_OUT_W, MAX_OUT_H = 1080, 1920
ZOOM_HEADROOM = 1.5


@dataclass
class Clip:
    """A normalized clip, ready for the renderer."""

    source: str        # original file name (what timelines refer to)
    warnings: list[str]
    path: Path         # the normalized intermediate
    width: int
    height: int
    fps: Fraction
    frames: int

    @property
    def duration(self) -> float:
        return float(self.frames / self.fps)


def cache_dir() -> Path:
    d = Path(os.environ.get("MOG_CACHE", Path.home() / ".cache" / "mog")) / "ingest"
    d.mkdir(parents=True, exist_ok=True)
    return d


def standard_fps(fps: Fraction, nominal: Fraction | None = None) -> Fraction:
    """Pick the clip's frame rate from the standard ones.

    Prefer the declared (nominal) rate when it is standard: a variable-frame-rate phone clip
    declared as 30 fps but averaging 26 (it dropped frames in low light) is still a 30 fps
    recording. Otherwise snap the measured average to the nearest standard rate."""
    if nominal is not None:
        # closest standard rate (29.97 and 30 are only 0.1% apart, so "first within tolerance" is wrong)
        r = min(STANDARD_RATES, key=lambda r: abs(float(nominal) - float(r)))
        if abs(float(nominal) - float(r)) / float(r) < 0.0005:
            return r
    if fps > 60:
        return Fraction(60)  # 120/240 fps slow-mo phone clips: keep 60 real frames per second
    return min(STANDARD_RATES, key=lambda r: abs(float(r) - float(fps)))


def target_size(w: int, h: int) -> tuple[int, int]:
    """Shrink only if the clip is much bigger than a 1080x1920 crop (with zoom headroom) needs."""
    cover = max(MAX_OUT_W / w, MAX_OUT_H / h)  # scale at which the clip just fills 1080x1920
    s = min(1.0, cover * ZOOM_HEADROOM)
    tw, th = int(round(w * s / 2)) * 2, int(round(h * s / 2)) * 2
    return max(tw, 2), max(th, 2)


def _key(path: Path) -> str:
    st = path.stat()
    raw = f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}|v{INGEST_VERSION}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _video_filter(info: VideoInfo, tw: int, th: int, fps: Fraction) -> str:
    chain = []
    if info.is_hdr:
        # HDR -> SDR: linearize, tone-map highlights (Hable curve), convert to Rec.709
        chain += ["zscale=t=linear:npl=100", "format=gbrpf32le", "zscale=p=bt709",
                  "tonemap=tonemap=hable:desat=0", "zscale=t=bt709:m=bt709:r=tv"]
    chain += [
        f"fps={fps.numerator}/{fps.denominator}",
        f"scale={tw}:{th}:flags=lanczos:out_color_matrix=bt709:out_range=tv",
        "setsar=1",
        "format=yuv420p",
    ]
    return ",".join(chain)


def ingest(path: str | Path, force: bool = False) -> Clip:
    path = Path(path)
    info = probe_video(path)
    fps = standard_fps(info.fps, info.nominal_fps)
    tw, th = target_size(info.width, info.height)
    out = cache_dir() / f"{path.stem[:40]}-{_key(path)}.mp4"
    if force or not out.exists():
        tmp = out.with_suffix(".part.mp4")
        cmd = [
            ffmpeg_bin(), "-v", "error", "-nostdin", "-y", "-i", str(path), "-an", "-sn", "-dn",
            "-vf", _video_filter(info, tw, th, fps),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "14",
            "-g", str(round(float(fps))), "-keyint_min", str(round(float(fps))), "-sc_threshold", "0",
            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
            "-video_track_timescale", str(fps.numerator * 10 if fps.denominator == 1 else fps.numerator),
            "-movflags", "+faststart", str(tmp),
        ]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0:
            tmp.unlink(missing_ok=True)
            err = proc.stderr.decode(errors="replace").strip().splitlines()
            raise MogError(f"Could not convert {path.name}: {err[-1] if err else 'unknown FFmpeg error'}")
        tmp.replace(out)
    norm = probe_video(out, count_frames=True)
    if norm.frames is None or norm.frames < 1:
        raise MogError(f"{path.name} produced no video frames.")
    warnings = []
    cover = max(MAX_OUT_W / info.width, MAX_OUT_H / info.height)
    if cover > 2.0:
        warnings.append(f"{path.name} is low resolution ({info.width}x{info.height}); it has to be enlarged "
                        f"{cover:.1f}x to fill a 1080x1920 frame and will look soft.")
    return Clip(source=path.name, warnings=warnings, path=out, width=norm.width, height=norm.height, fps=fps,
                frames=norm.frames)
