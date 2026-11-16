"""Auto-reframe a clip to 9:16 using its analysis, and draw a debug sheet.

reframe_timeline(): one timeline segment per shot, each following that shot's smoothed
crop path (face, or motion when there is no face), confined to the content area, with the
suggested exposure. It is a plain Timeline, so the normal renderer draws it.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import soundfile as sf

from mog.render.geometry import window_affine
from mog.timeline.models import Timeline
from mog.video.models import VideoAnalysis


def silent_wav(path: Path, seconds: float) -> Path:
    if not path.exists():
        sf.write(path, np.zeros((int(48000 * (seconds + 1)), 2), np.float32), 48000)
    return path


def reframe_timeline(an: VideoAnalysis, audio_file: str, max_duration: float = 15.0, fps: int = 30,
                     exposure: bool = True) -> Timeline:
    segs, t = [], 0.0
    for sh in an.shots:
        if t >= max_duration - 1e-3:
            break
        dur = min(sh.end - sh.start, max_duration - t)
        if dur < 1.0 / fps:
            continue
        segs.append({
            "id": f"shot{sh.index}", "source": an.source, "src_in": sh.start,
            "timeline_in": round(t, 4), "timeline_out": round(t + dur, 4),
            "exposure": sh.exposure if exposure else 0.0,
            "crop": {"mode": "face_track" if sh.face_path else "manual", "bounds": an.content,
                     "path": [list(p) for p in sh.crop_path]},
        })
        t += dur
    return Timeline.model_validate({
        "output": {"width": 1080, "height": 1920, "fps": fps, "duration": round(t, 4)},
        "audio": {"file": audio_file, "start": 0, "end": round(t + 0.5, 4), "fade_out": 0.2},
        "segments": segs,
    })


def debug_sheet(an: VideoAnalysis, clip_path: Path, out: Path, max_shots: int = 12) -> Path:
    """Middle frame of each shot with: content area (yellow), face (green), 9:16 crop (magenta),
    and the shot's scores."""
    from mog.ffmpeg.frames import FrameReader
    from mog.video.ingest import ingest

    clip = ingest(clip_path)
    tiles = []
    for sh in an.shots[:max_shots]:
        mid = (sh.start + sh.end) / 2
        k = min(int(mid * float(clip.fps)), clip.frames - 1)
        with FrameReader(clip.path, clip.fps, k, clip.width, clip.height) as r:
            img = r.frame_at(k).copy()
        h, w = img.shape[:2]
        x0, y0, x1, y1 = an.content
        cv2.rectangle(img, (int(x0 * w), int(y0 * h)), (int(x1 * w) - 1, int(y1 * h) - 1), (0, 220, 255), 3)
        rel = mid - sh.start
        if sh.face_path:
            fp = np.array(sh.face_path)
            fx, fy, fs = (float(np.interp(rel, fp[:, 0], fp[:, i])) for i in (1, 2, 3))
            half = fs * min(w, h) / 2 * 1.2
            cv2.rectangle(img, (int(fx * w - half), int(fy * h - half)), (int(fx * w + half), int(fy * h + half)),
                          (60, 255, 60), 3)
        cp = np.array(sh.crop_path)
        cx, cy = float(np.interp(rel, cp[:, 0], cp[:, 1])), float(np.interp(rel, cp[:, 0], cp[:, 2]))
        m = window_affine(w, h, 1080, 1920, cx, cy, 1.0, bounds=an.content)
        inv = cv2.invertAffineTransform(m)
        corners = np.array([[0, 0], [1079, 0], [1079, 1919], [0, 1919]], np.float32)
        pts = (corners @ inv[:, :2].T + inv[:, 2]).astype(np.int32)
        cv2.polylines(img, [pts], True, (255, 0, 255), 4)
        scale = 360 / h
        tile = cv2.resize(img, (max(2, int(w * scale)), 360), interpolation=cv2.INTER_AREA)
        label = f"#{sh.index} {sh.start:.1f}-{sh.end:.1f}s  stare {sh.stare_score:.2f}  act {sh.action_score:.2f}"
        cv2.rectangle(tile, (0, 0), (tile.shape[1], 22), (0, 0, 0), -1)
        cv2.putText(tile, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    rows, per = [], 4
    for i in range(0, len(tiles), per):
        row = tiles[i:i + per]
        rows.append(np.hstack(row + [np.zeros_like(row[0])] * 0))
    width = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0))) for r in rows]
    cv2.imwrite(str(out), np.vstack(rows))
    return out
