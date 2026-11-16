"""Video analysis: content area, tracking, shots, scores (synthetic data with known answers)."""

import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from mog.ffmpeg.frames import FrameWriter
from mog.video.analyze import analyze_video, detect_shots
from mog.video.content import content_bounds
from mog.video.ingest import ingest
from mog.video.reframe import reframe_timeline
from mog.video.track import jitter, kalman_smooth

REAL = Path(__file__).parents[1] / "out" / "real" / "clips"
rng = np.random.default_rng(0)


def footage(n=40, h=200, w=120):
    """Moving textured 'footage': noise blobs drifting over time."""
    base = cv2.GaussianBlur(rng.integers(0, 255, (h * 2, w * 2), dtype=np.uint8), (0, 0), 3)
    base = cv2.normalize(base, None, 0, 255, cv2.NORM_MINMAX)
    return [np.roll(base, (i * 3, i * 2), (0, 1))[:h, :w].copy() for i in range(n)]


def test_letterbox_bars_are_trimmed():
    frames = footage()
    for f in frames:
        f[:20] = 0
        f[-30:] = 0
    x0, y0, x1, y1 = content_bounds(frames)
    assert (x0, x1) == (0.0, 1.0)
    assert 0.1 <= y0 <= 0.12 and 0.83 <= y1 <= 0.85


def test_phone_ui_bars_are_trimmed_but_static_footage_is_kept():
    frames = footage()
    ui = rng.integers(0, 255, (12, 120), dtype=np.uint8)       # textured, frozen status bar
    for f in frames:
        f[:12] = ui
        f[-16:] = ui[:, ::-1].repeat(2, axis=0)[:16]
    x0, y0, x1, y1 = content_bounds(frames)
    assert 0.06 <= y0 <= 0.08 and 0.90 <= y1 <= 0.93
    # a textured but frozen region at the SIDE is real footage (e.g. a still wall): kept
    frames = footage()
    wall = rng.integers(0, 255, (200, 30), dtype=np.uint8)
    for f in frames:
        f[:, -30:] = wall
    assert content_bounds(frames)[2] == 1.0


def test_blur_bars_are_trimmed():
    frames = footage()
    for f in frames:
        f[:20] = cv2.GaussianBlur(f[20:40], (0, 0), 8)
        f[-20:] = cv2.GaussianBlur(f[-40:-20], (0, 0), 8)
    _, y0, _, y1 = content_bounds(frames)
    assert y0 >= 0.09 and y1 <= 0.91


def test_static_clip_is_not_trimmed():
    f = footage(1)[0]
    assert content_bounds([f.copy() for _ in range(10)]) == (0.0, 0.0, 1.0, 1.0)


def test_kalman_smoother_follows_motion_and_rejects_noise():
    t = np.arange(0, 6, 0.1)
    truth = 0.5 + 0.2 * np.sin(t)
    z = truth + rng.normal(0, 0.02, len(t))
    z[15:22] = np.nan          # face lost for 0.7 s
    z[40] = 0.95               # one wrong detection (another face)
    sm = kalman_smooth(t, z, np.full(len(t), 0.02), q=0.25)
    assert np.max(np.abs(sm - truth)) < 0.05
    assert abs(sm[40] - truth[40]) < 0.05
    assert jitter(sm, 0.1) < 0.4 * jitter(np.nan_to_num(z, nan=0.5), 0.1)


def _cut_clip(path: Path) -> Path:
    """3 shots: 2 s moving pattern, 1.5 s different pattern, 2 s static colour."""
    parts = []
    for i, src in enumerate(["testsrc2=s=320x180:r=30:d=2", "mandelbrot=s=320x180:r=30,trim=duration=1.5",
                             "color=c=0x305070:s=320x180:r=30:d=2"]):
        p = path.parent / f"part{i}.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", src, "-t", "2", "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", str(p)], check=True)
        parts.append(p)
    lst = path.parent / "list.txt"
    lst.write_text("".join(f"file '{p}'\n" for p in parts))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


def test_shots_scores_and_reframe_without_faces(tmp_path):
    clip = _cut_clip(tmp_path / "cuts.mp4")
    shots = detect_shots(ingest(clip))
    assert len(shots) == 3
    assert abs(shots[1][0] - 2.0) < 1 / 30 + 1e-6 and abs(shots[2][0] - 3.5) < 1 / 30 + 1e-6
    an = analyze_video(clip, use_cache=False)
    assert [s.face_score for s in an.shots] == [0, 0, 0]
    assert an.shots[0].motion_score > an.shots[2].motion_score   # moving pattern vs flat colour
    assert an.shots[2].action_score < 0.05
    tl = reframe_timeline(an, "a.wav", max_duration=5.5)
    assert [s.src_in for s in tl.segments] == [s.start for s in an.shots]
    assert all(s.crop.bounds == an.content for s in tl.segments)


@pytest.mark.skipif(not (REAL / "clip4.mp4").exists(), reason="real test clips not present")
def test_real_clip_face_and_reframe():
    an = analyze_video(REAL / "clip4.mp4")
    sh = an.shots[0]
    assert sh.face_frac > 0.9 and sh.face_score > 0.3
    xs = [p[1] for p in sh.crop_path]
    assert 0.55 < np.median(xs) < 0.75         # the presenter sits right of centre
    assert jitter(np.array(xs), 0.2) < 1.0


def test_writer_helper_still_works(tmp_path):
    # guards the FrameWriter import used by fixtures elsewhere
    wr = FrameWriter(32, 32, 30, ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(tmp_path / "x.mp4")])
    wr.write(np.zeros((32, 32, 3), np.uint8))
    wr.close()
