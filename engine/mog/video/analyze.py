"""analyze_video(path) -> VideoAnalysis: shots, faces, motion, scores, crop paths.

One decoding pass samples the clip at up to 10 frames per second, small (240 px), and
records for each sample: faces (YuNet), a tiny grey image (for motion, sharpness and the
content area). Shot boundaries come from PySceneDetect's ContentDetector on every frame.
Then, per shot: pick the main face, smooth its path (Kalman + RTS), score the shot, and
build the 9:16 crop path.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from mog.errors import MogError
from mog.ffmpeg.audio_io import ffmpeg_bin
from mog.video.content import content_bounds
from mog.video.faces import Face, FaceDetector
from mog.video.ingest import Clip, cache_dir, ingest
from mog.video.models import Shot, VideoAnalysis
from mog.video.track import kalman_smooth

ANALYSIS_VERSION = 7
SAMPLE_FPS = 10.0
MAX_SAMPLES = 1800
SAMPLE_DIM = 480        # decode size for face detection
GREY_DIM = 200          # tiny grey copy for motion / sharpness / content
CROP_KEY_DT = 0.2       # crop path keyframe spacing (s)


@dataclass
class Sample:
    t: float
    faces: list[Face]
    grey: np.ndarray
    main: Face | None = None
    flow_mag: float = 0.0       # frame-widths per second
    flow_cx: float = 0.5        # where the motion is (x), for no-face reframing
    sharp: float = 0.0
    bright: float = 0.0
    face_bright: float = float("nan")
    extra: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------- decoding

def _iter_samples(clip: Clip, fps: float):
    s = min(1.0, SAMPLE_DIM / max(clip.width, clip.height))
    w, h = max(2, int(clip.width * s) // 2 * 2), max(2, int(clip.height * s) // 2 * 2)
    cmd = [ffmpeg_bin(), "-v", "error", "-nostdin", "-i", str(clip.path), "-an",
           "-vf", f"fps={fps:.4f},scale={w}:{h}:flags=area:in_color_matrix=bt709:in_range=tv:out_range=full,"
                  "format=bgr24", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**7)
    size = w * h * 3
    i = 0
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield i / fps, np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            i += 1
    finally:
        proc.kill()
        proc.wait()


def detect_shots(clip: Clip) -> list[tuple[float, float]]:
    """Shot boundaries (seconds) with PySceneDetect's ContentDetector."""
    from scenedetect import ContentDetector, detect

    fps = float(clip.fps)
    try:
        scenes = detect(str(clip.path), ContentDetector(threshold=27.0, min_scene_len=max(2, int(0.4 * fps))),
                        show_progress=False)
    except Exception as exc:  # noqa: BLE001 - shot detection failing shouldn't kill the job
        raise MogError(f"Shot detection failed on {clip.source}: {exc}") from exc
    def frame_no(tc) -> int:  # `frame_num` in PySceneDetect >= 0.7, get_frames() before
        return int(tc.frame_num) if hasattr(tc, "frame_num") else int(tc.get_frames())

    out = [(frame_no(s) / fps, frame_no(e) / fps) for s, e in scenes]
    if not out:
        out = [(0.0, clip.duration)]
    out[0] = (0.0, out[0][1])
    out[-1] = (out[-1][0], clip.duration)
    return out


# ----------------------------------------------------------------------------- per sample

def _pick_main(faces: list[Face], prev: Face | None, bounds) -> Face | None:
    x0, y0, x1, y1 = bounds
    best, best_s = None, 0.0
    for f in faces:
        if not (x0 <= f.cx <= x1 and y0 <= f.cy <= y1):
            continue  # a face in the letterbox / UI area (e.g. a profile picture) doesn't count
        s = math.sqrt(f.area) * f.score
        if prev is not None:
            s *= 1.0 / (1.0 + 4.0 * math.hypot(f.cx - prev.cx, f.cy - prev.cy))
        if s > best_s:
            best, best_s = f, s
    return best


def _crop_grey(g: np.ndarray, bounds) -> np.ndarray:
    h, w = g.shape
    x0, y0, x1, y1 = bounds
    return g[int(y0 * h):max(int(y1 * h), int(y0 * h) + 2), int(x0 * w):max(int(x1 * w), int(x0 * w) + 2)]


def _lin(v: float) -> float:
    return max(v, 1e-4) ** 2.2


# ----------------------------------------------------------------------------- main

def analyze_video(path: str | Path, use_cache: bool = True) -> VideoAnalysis:
    clip = ingest(path)
    key = clip.path.stem
    cache = cache_dir().parent / "analysis" / f"{key}-a{ANALYSIS_VERSION}.json"
    if use_cache and cache.exists():
        return VideoAnalysis.model_validate_json(cache.read_text())
    res = _analyze(clip)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(res.model_dump_json())
    return res


def _analyze(clip: Clip) -> VideoAnalysis:
    fps = min(SAMPLE_FPS, MAX_SAMPLES / max(clip.duration, 0.1))
    det = FaceDetector()
    samples: list[Sample] = []
    for t, img in _iter_samples(clip, fps):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        s = min(1.0, GREY_DIM / max(g.shape))
        grey = cv2.resize(g, (max(8, int(g.shape[1] * s)), max(8, int(g.shape[0] * s))), interpolation=cv2.INTER_AREA)
        samples.append(Sample(t=t, faces=det.detect(img), grey=grey))
    if not samples:
        raise MogError(f"Could not read any frames from {clip.source}.")

    idx = np.linspace(0, len(samples) - 1, min(len(samples), 60)).astype(int)
    bounds = content_bounds([samples[i].grey for i in idx])
    shots = detect_shots(clip)

    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
    out_shots: list[Shot] = []
    for si, (a, b) in enumerate(shots):
        ss = [s for s in samples if a <= s.t < b]
        if not ss:  # very short shot between samples: take the nearest sample
            ss = [min(samples, key=lambda s: abs(s.t - (a + b) / 2))]
        prev = None
        prev_grey = None
        for s in ss:
            s.main = _pick_main(s.faces, prev, bounds)
            prev = s.main or prev
            cg = _crop_grey(s.grey, bounds)
            s.bright = float(cg.mean() / 255)
            s.sharp = float(cv2.Laplacian(cg, cv2.CV_32F).var())
            if s.main is not None:
                h, w = s.grey.shape
                fx0, fx1 = int((s.main.cx - s.main.w / 3) * w), int((s.main.cx + s.main.w / 3) * w)
                fy0, fy1 = int((s.main.cy - s.main.h / 3) * h), int((s.main.cy + s.main.h / 3) * h)
                patch = s.grey[max(fy0, 0):max(fy1, fy0 + 1), max(fx0, 0):max(fx1, fx0 + 1)]
                if patch.size:
                    s.face_bright = float(patch.mean() / 255)
            if prev_grey is not None and prev_grey.shape == cg.shape:
                flow = dis.calc(np.ascontiguousarray(prev_grey), np.ascontiguousarray(cg), None)
                mag = np.hypot(flow[..., 0], flow[..., 1])
                # Median: robust to compression noise and small moving bits (a hand, a caption)
                s.flow_mag = float(np.median(mag)) / cg.shape[1] * fps
                wsum = mag.sum(axis=0)
                if wsum.sum() > 1e-6:
                    xs = (np.arange(cg.shape[1]) + 0.5) / cg.shape[1]
                    s.flow_cx = bounds[0] + float((xs * wsum).sum() / wsum.sum()) * (bounds[2] - bounds[0])
            prev_grey = cg
        out_shots.append(_shot(si, a, b, ss, bounds, clip))

    warnings = list(clip.warnings)
    if bounds != (0.0, 0.0, 1.0, 1.0):
        warnings.append(f"{clip.source}: trimmed borders/UI, using area {bounds}.")
    return VideoAnalysis(version=ANALYSIS_VERSION, source=clip.source, duration=clip.duration, fps=float(clip.fps),
                         width=clip.width, height=clip.height, content=bounds, shots=out_shots, warnings=warnings)


def _shot(index: int, a: float, b: float, ss: list[Sample], bounds, clip: Clip) -> Shot:
    t = np.array([s.t for s in ss])
    has = np.array([s.main is not None for s in ss])
    face_frac = float(has.mean())
    face_path = None
    cx_c, cy_c = (bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2

    if has.sum() >= max(2, 0.3 * len(ss)):
        fx = np.array([s.main.cx if s.main else np.nan for s in ss])
        fy = np.array([s.main.cy if s.main else np.nan for s in ss])
        fs = np.array([math.sqrt(s.main.area) if s.main else np.nan for s in ss])
        r = np.where(np.isnan(fs), 1.0, 0.04 * np.nan_to_num(fs) + 0.004)
        sx = kalman_smooth(t, fx, r, q=0.25)
        sy = kalman_smooth(t, fy, r, q=0.25)
        sz = kalman_smooth(t, fs, r, q=0.1)
        face_path = [(round(float(ti - a), 3), round(float(x), 4), round(float(y), 4), round(float(z), 4))
                     for ti, x, y, z in zip(t, sx, sy, sz)]
        # Crop centre: the face, with the window centre a bit below it so the face sits in
        # the upper third of the 9:16 frame (headroom like a real framing)
        path_x, path_y = sx, sy + 0.12 * np.nan_to_num(sz)
        size = float(np.nanmedian(fs))
        center = float(np.nanmean(1 - np.clip(np.abs(fx - cx_c) * 1.2, 0, 0.6)))
        score = float(np.nanmean([s.main.score for s in ss if s.main]))
        size_score = float(np.clip((size - 0.06) / (0.32 - 0.06), 0, 1))
        face_score = face_frac * size_score * center * score
        frontal = float(np.mean([1 - abs(s.main.yaw) for s in ss if s.main]))
    else:
        # No face: follow where the motion is (smoothed hard), else the centre
        mx = np.array([s.flow_cx for s in ss])
        moving = np.array([s.flow_mag for s in ss]) > 0.02
        z = np.where(moving, mx, np.nan)
        path_x = kalman_smooth(t, z, np.full(len(t), 0.08), q=0.05) if moving.any() else np.full(len(t), cx_c)
        path_x = np.where(np.isnan(path_x), cx_c, path_x)
        path_y = np.full(len(t), cy_c)
        face_score, frontal = 0.0, 0.0

    # Keyframes every CROP_KEY_DT
    kt = np.arange(0.0, max(b - a, 1e-3) + 1e-9, CROP_KEY_DT)
    rel = t - a
    crop_path = [(round(float(k), 3), round(float(np.interp(k, rel, path_x)), 4),
                  round(float(np.interp(k, rel, path_y)), 4)) for k in kt]

    motion = float(np.median([s.flow_mag for s in ss[1:]])) if len(ss) > 1 else 0.0
    motion_score = float(np.clip(motion / 0.5, 0, 1))
    bright = float(np.nanmean([s.face_bright for s in ss]) if face_path else np.mean([s.bright for s in ss]))
    if math.isnan(bright):
        bright = float(np.mean([s.bright for s in ss]))
    # Mostly a rescue for under-exposed shots: brighten up to 1.5 stops, darken at most 0.5
    target = 0.50 if face_path else 0.42
    stops = math.log2(_lin(target) / _lin(bright))
    exposure = float(np.clip(stops, -0.5, 1.5)) if abs(stops) > 0.25 else 0.0

    sharp = float(np.median([s.sharp for s in ss]))
    sharp_score = float(np.clip((math.log10(max(sharp, 1)) - 1.3) / (2.6 - 1.3), 0, 1))
    expo_ok = 1.0 if 0.06 < np.mean([s.bright for s in ss]) < 0.94 else 0.4
    bw_px = (bounds[2] - bounds[0]) * clip.width
    bh_px = (bounds[3] - bounds[1]) * clip.height
    enlarge = max(1080 / bw_px, 1920 / bh_px)
    res = float(np.clip(1.5 / enlarge, 0.35, 1.0))
    quality = sharp_score * expo_ok * res
    if b - a < 0.5:
        quality *= 0.6

    stare = face_score * frontal * (1 - motion_score) ** 1.5
    action = motion_score * (0.6 + 0.4 * quality)
    r3 = lambda v: round(float(v), 3)  # noqa: E731
    return Shot(index=index, start=r3(a), end=r3(b), face_score=r3(face_score), frontal=r3(frontal),
                motion_score=r3(motion_score), stare_score=r3(stare), action_score=r3(action),
                quality=r3(quality), brightness=r3(bright), exposure=round(exposure, 2), face_frac=r3(face_frac),
                motion=round(motion, 4), crop_path=crop_path, face_path=face_path)


def dump(an: VideoAnalysis) -> str:
    return json.dumps(an.model_dump(), indent=1)
