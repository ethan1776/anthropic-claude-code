"""Render a Timeline to an MP4. Reads ONLY the timeline JSON (plus the media it names).

Pipeline per output frame:
  1. which segment owns this frame, and which source moment it shows (speed curve)
  2. read that source frame (FrameReader, frame-exact)
  3. crop/zoom it into the 9:16 output with one affine warp (geometry.py)
  4. pipe it to FFmpeg, which grades, adds grain/vignette, mixes in the music and encodes

Phase 2 supports: cuts, speed curves (nearest-frame sampling), crop/zoom/face-track paths,
grade (LUT, contrast, saturation, black crush, glow), grain, vignette, audio fades.
Per-segment effects and text overlays are Phase 3; a timeline that uses them is rejected
with a clear message instead of rendering without them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from mog.errors import MogError
from mog.ffmpeg.frames import FrameReader, FrameWriter
from mog.ffmpeg.probe import probe_video
from mog.render.finish import EncodeSettings, build_graph, encoder_args
from mog.render.geometry import apply_affine, crop_center, window_affine
from mog.render.pixel_fx import Glow, Vignette, exposure_table
from mog.timeline.models import Timeline
from mog.timeline.speed import segment_source_times
from mog.video.ingest import Clip, ingest

Progress = Callable[[str, float], None]


@dataclass
class RenderResult:
    path: Path
    width: int
    height: int
    fps: int
    frames: int
    seconds: float
    warnings: list[str] = field(default_factory=list)


def _check_supported(tl: Timeline) -> None:
    used = sorted({e.type for s in tl.segments for e in s.effects})
    if used:
        raise MogError(f"This timeline uses effects ({', '.join(used)}) that the renderer adds in Phase 3.")
    if tl.text:
        raise MogError("This timeline has text overlays, which the renderer adds in Phase 3.")


def _even(x: float) -> int:
    return max(2, int(round(x / 2)) * 2)


def render(tl: Timeline, out_path: str | Path, media_dir: str | Path, preview: bool = False,
           progress: Progress | None = None) -> RenderResult:
    t_start = time.time()
    out_path = Path(out_path)
    media_dir = Path(media_dir)
    report = progress or (lambda stage, frac: None)
    _check_supported(tl)
    warnings: list[str] = []

    # ---- inputs
    missing = sorted({s.source for s in tl.segments if not (media_dir / s.source).exists()})
    if missing:
        raise MogError(f"Missing clip(s) in {media_dir}: {', '.join(missing)}")
    audio_path = media_dir / tl.audio.file
    if not audio_path.exists():
        raise MogError(f"Missing audio file in {media_dir}: {tl.audio.file}")

    clips: dict[str, Clip] = {}
    sources = list(dict.fromkeys(s.source for s in tl.segments))
    for i, name in enumerate(sources):
        report("Preparing clips", i / len(sources))
        clips[name] = ingest(media_dir / name)
        warnings.extend(clips[name].warnings)

    # ---- output size
    fps = tl.output.fps
    W, H = tl.output.width, tl.output.height
    if preview:
        W, H = _even(W / 2), _even(H / 2)
    settings = EncodeSettings.preview() if preview else EncodeSettings.export()
    quality = "fast" if preview else "high"
    total = int(round(tl.output.duration * fps))

    # ---- plan every frame up front, so footage problems fail before any encoding starts
    plans = []
    for seg in tl.segments:
        clip = clips[seg.source]
        frames, src_t = segment_source_times(seg, fps)
        if len(frames) == 0:
            continue
        need = float(src_t.max())
        if need > clip.duration + 1.0 / float(clip.fps):
            raise MogError(f"Segment {seg.id} needs {seg.source} up to {need:.2f}s, "
                           f"but the clip is only {clip.duration:.2f}s long.")
        src_idx = np.minimum(np.floor(src_t * float(clip.fps) + 1e-6).astype(int), clip.frames - 1)
        plans.append((seg, clip, frames, src_idx))

    tmp = out_path.with_name(out_path.stem + ".part" + out_path.suffix)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    graph = build_graph(tl.grade, W, H, tl.output.duration, tl.audio, fps, [media_dir])
    writer = FrameWriter(W, H, fps, encoder_args(graph, audio_path, tl.audio, tl.output.duration, fps,
                                                 tmp, settings))
    # Glow before vignette (the glow shouldn't be darkened twice at the edges); the LUT runs
    # afterwards in FFmpeg. Glow on pre-LUT colours is fine: it picks up the bright areas.
    post = []
    if tl.grade.glow > 0:
        post.append(Glow(W, H, tl.grade.glow))
    if tl.grade.vignette > 0:
        post.append(Vignette(W, H, tl.grade.vignette))
    written = 0
    try:
        for seg, clip, frames, src_idx in plans:
            # Decode smaller than the clip when the output is small (preview), but keep
            # 1.5x headroom so zooming in later stays sharp.
            cover = max(W / clip.width, H / clip.height)
            d = min(1.0, cover * 1.5)
            dw, dh = _even(clip.width * d), _even(clip.height * d)
            expo = exposure_table(seg.exposure)
            with FrameReader(clip.path, clip.fps, int(src_idx[0]), clip.width, clip.height,
                             out_size=(dw, dh)) as reader:
                for n, k in zip(frames, src_idx):
                    if n != written:
                        raise MogError(f"internal: frame order broken at segment {seg.id} ({n} != {written})")
                    src = reader.frame_at(int(k))
                    t_local = n / fps - seg.timeline_in
                    cx, cy = crop_center(seg.crop, t_local)
                    m = window_affine(dw, dh, W, H, cx, cy, seg.crop.zoom)
                    frame = apply_affine(src, m, W, H, quality)
                    if expo is not None:
                        frame = cv2.LUT(frame, expo)
                    for fx in post:
                        frame = fx(frame)
                    writer.write(frame)
                    written += 1
                    if written % 15 == 0:
                        report("Rendering", written / total)
        if written != total:
            raise MogError(f"internal: wrote {written} frames, expected {total}")
        report("Encoding", 1.0)
        writer.close()
    except BaseException:
        writer.abort()
        tmp.unlink(missing_ok=True)
        raise

    info = probe_video(tmp, count_frames=True)
    if (info.width, info.height) != (W, H) or info.frames != total or not info.has_audio:
        tmp.unlink(missing_ok=True)
        raise MogError(f"Output check failed: got {info.width}x{info.height}, {info.frames} frames, "
                       f"audio={info.has_audio}; expected {W}x{H}, {total} frames, audio.")
    tmp.replace(out_path)
    return RenderResult(out_path, W, H, fps, total, time.time() - t_start, warnings)
