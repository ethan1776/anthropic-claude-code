"""Synthetic test clips.

* barcode clips: every frame shows its own frame number as 12 black/white bars, so tests
  can check that the renderer shows exactly the frame it should.
* demo clips: colourful moving FFmpeg test sources in the shapes real uploads come in
  (landscape 1080p, vertical 60 fps, 24 fps 4:3, rotated 29.97 fps phone video), for sample renders.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from mog.ffmpeg.audio_io import ffmpeg_bin
from mog.ffmpeg.frames import FrameWriter

BITS = 12


def barcode_frame(k: int, w: int, h: int, tint: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """Frame number k as 12 vertical bars (MSB left), on the middle band of the frame."""
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (40, 40, 40)
    bar_w = w // BITS
    for b in range(BITS):
        if (k >> (BITS - 1 - b)) & 1:
            img[h // 4: 3 * h // 4, b * bar_w:(b + 1) * bar_w] = tint
    return img


def read_barcode(img: np.ndarray, region: tuple[int, int, int, int] | None = None) -> int:
    """Decode a frame number from a frame (or from a sub-region x, y, w, h of it)."""
    if region:
        x, y, w, h = region
        img = img[y:y + h, x:x + w]
    h, w = img.shape[:2]
    bar_w = w / BITS
    row = img[h // 2 - h // 8: h // 2 + h // 8].mean(axis=(0, 2))
    k = 0
    for b in range(BITS):
        seg = row[int(b * bar_w + bar_w * 0.3): int((b + 1) * bar_w - bar_w * 0.3)]
        k = (k << 1) | int(seg.mean() > 128)
    return k


def make_barcode_clip(path: Path, frames: int, fps: int = 30, w: int = 640, h: int = 360,
                      rotation: int | None = None, offset: int = 0) -> Path:
    """Frame k shows the number k + offset (use different offsets to tell clips apart)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    target = path if rotation is None else path.with_suffix(".norot.mp4")
    wr = FrameWriter(w, h, fps, ["-c:v", "libx264", "-preset", "veryfast", "-crf", "10",
                                 "-pix_fmt", "yuv420p", str(target)])
    for k in range(frames):
        wr.write(barcode_frame(k + offset, w, h))
    wr.close()
    if rotation is not None:
        subprocess.run([ffmpeg_bin(), "-v", "error", "-y", "-display_rotation", str(rotation), "-i", str(target),
                        "-c", "copy", str(path)], check=True)
        target.unlink()
    return path


DEMO_CLIPS = {
    # name: (lavfi source, extra output args)
    "street.mp4": ("testsrc2=s=1920x1080:r=30:d=8", []),
    "neon.mp4": ("gradients=s=1080x1920:r=60:d=8:speed=0.02:c0=0x1a0033:c1=0x00e5ff:c2=0xff2d95:c3=0x111111:n=4", []),
    "cells.mp4": ("life=s=320x240:r=24:mold=10:ratio=0.12:death_color=#1a0a2e:life_color=#ffd166,"
                  "scale=960:720:flags=neighbor", ["-t", "6"]),
}


def make_demo_clips(out_dir: Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, (src, extra) in DEMO_CLIPS.items():
        p = out_dir / name
        if not p.exists():
            subprocess.run([ffmpeg_bin(), "-v", "error", "-y", "-f", "lavfi", "-i", src, *extra,
                            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", str(p)],
                           check=True)
        paths.append(p)
    # A "phone" clip: stored landscape with a 90-degree rotation flag, 29.97 fps
    phone = out_dir / "phone.mov"
    if not phone.exists():
        tmp = out_dir / "phone.tmp.mp4"
        subprocess.run([ffmpeg_bin(), "-v", "error", "-y", "-f", "lavfi",
                        "-i", "testsrc2=s=1920x1080:r=30000/1001:d=6,hue=H=2*PI*t/6:s=1.4",
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", str(tmp)],
                       check=True)
        subprocess.run([ffmpeg_bin(), "-v", "error", "-y", "-display_rotation", "90", "-i", str(tmp),
                        "-c", "copy", str(phone)], check=True)
        tmp.unlink()
    paths.append(phone)
    return paths
