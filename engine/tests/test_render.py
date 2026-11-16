"""End-to-end renders of small timelines through the real FFmpeg pipeline."""

import json
import subprocess

import numpy as np
import pytest
import soundfile as sf

from mog.errors import MogError
from mog.ffmpeg.probe import probe_video
from mog.render.renderer import render
from mog.testing.clips import make_barcode_clip, read_barcode
from mog.timeline.models import Timeline
from mog.video.ingest import ingest

W, H, FPS = 360, 640, 30


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    d = tmp_path_factory.mktemp("media")
    # Portrait barcode clips so the 9:16 crop keeps all bars. B's numbers start at 2000.
    make_barcode_clip(d / "a.mp4", 240, fps=30, w=W, h=H)
    make_barcode_clip(d / "b.mp4", 240, fps=30, w=W, h=H, offset=2000)
    make_barcode_clip(d / "c60.mp4", 240, fps=60, w=W, h=H, offset=1000)
    t = np.arange(int(48000 * 6)) / 48000
    sf.write(d / "tone.wav", (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), 48000)
    return d


def tl(segments, duration, **kw):
    d = {"output": {"width": W, "height": H, "fps": FPS, "duration": duration},
         "audio": {"file": "tone.wav", "start": 0.5, "end": 5.5}, "segments": segments}
    d.update(kw)
    return Timeline.model_validate(d)


def decode(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)


def test_cuts_and_speeds_are_frame_exact(media, tmp_path):
    t = tl([
        {"id": "s1", "source": "a.mp4", "src_in": 1.0, "timeline_in": 0.0, "timeline_out": 0.5},
        {"id": "s2", "source": "b.mp4", "src_in": 0.0, "timeline_in": 0.5, "timeline_out": 1.0,
         "speed_curve": [[0, 2.0]]},
        {"id": "s3", "source": "a.mp4", "src_in": 4.0, "timeline_in": 1.0, "timeline_out": 1.5,
         "speed_curve": [[0, 0.5]]},
        {"id": "s4", "source": "c60.mp4", "src_in": 0.5, "timeline_in": 1.5, "timeline_out": 2.0},
    ], 2.0)
    res = render(t, tmp_path / "o.mp4", media, interpolation="off")
    got = [read_barcode(f) for f in decode(res.path)]
    expect = ([30 + i for i in range(15)]                  # a from 1.0 s, 1x
              + [2000 + 2 * i for i in range(15)]          # b at 2x
              + [120 + i // 2 for i in range(15)]          # a from 4.0 s at 0.5x (each frame twice)
              + [1000 + 30 + 2 * i for i in range(15)])    # 60 fps source at 1x into 30 fps output
    assert got == expect


def test_cut_lands_on_nearest_frame(media, tmp_path):
    # A cut at 0.517 s is 15.5 frames at 30 fps -> rounds to frame 16
    t = tl([{"id": "s1", "source": "a.mp4", "src_in": 0, "timeline_in": 0, "timeline_out": 0.517},
            {"id": "s2", "source": "b.mp4", "src_in": 0, "timeline_in": 0.517, "timeline_out": 1.0}], 1.0)
    got = [read_barcode(f) for f in decode(render(t, tmp_path / "o.mp4", media).path)]
    assert got[15] < 2000 and got[16] >= 2000


def test_output_file_is_platform_ready(media, tmp_path):
    t = tl([{"id": "s1", "source": "a.mp4", "src_in": 0, "timeline_in": 0, "timeline_out": 2.0}], 2.0,
           grade={"lut": "bw_crush.cube", "grain": 0.3, "vignette": 0.4, "glow": 0.2, "contrast": 1.2})
    res = render(t, tmp_path / "o.mp4", media)
    info = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json",
                                      str(res.path)], capture_output=True).stdout)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    a = next(s for s in info["streams"] if s["codec_type"] == "audio")
    assert (v["codec_name"], v["profile"], v["pix_fmt"]) == ("h264", "High", "yuv420p")
    assert (v["width"], v["height"], v["r_frame_rate"]) == (W, H, "30/1")
    assert v["color_primaries"] == "bt709" and v["color_transfer"] == "bt709"
    assert (a["codec_name"], a["sample_rate"], a["channels"]) == ("aac", "48000", 2)
    assert abs(float(info["format"]["duration"]) - 2.0) < 0.05
    head = res.path.read_bytes()[:4096]
    assert head.find(b"moov") != -1 and (head.find(b"mdat") == -1 or head.find(b"moov") < head.find(b"mdat"))
    assert probe_video(res.path, count_frames=True).frames == 60


def test_identity_grade_preserves_colour(media, tmp_path):
    # A flat colour clip through the pipeline with no grade must come out the same colour
    src = media / "flat.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c=0x3c78b4:s={W}x{H}:r=30:d=1",
                    "-c:v", "libx264", "-crf", "1", "-pix_fmt", "yuv420p",
                    "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", str(src)], check=True)
    ref = decode_any(src)[5].reshape(-1, 3).mean(0)
    t = tl([{"id": "s1", "source": "flat.mp4", "src_in": 0, "timeline_in": 0, "timeline_out": 0.5}], 0.5)
    out = decode(render(t, tmp_path / "o.mp4", media).path)[5].reshape(-1, 3).mean(0)
    assert np.all(np.abs(out - ref) < 3), (out, ref)
    # and saturation 0 really is black & white
    t2 = tl(t.model_dump()["segments"], 0.5, grade={"saturation": 0.0})
    bw = decode(render(t2, tmp_path / "bw.mp4", media).path)[5].reshape(-1, 3).mean(0)
    assert np.ptp(bw) < 3


def decode_any(path):
    info = probe_video(path)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, info.height, info.width, 3)


def test_preview_is_half_size(media, tmp_path):
    t = tl([{"id": "s1", "source": "a.mp4", "src_in": 0, "timeline_in": 0, "timeline_out": 1.0}], 1.0)
    res = render(t, tmp_path / "p.mp4", media, preview=True)
    assert (res.width, res.height) == (W // 2, H // 2)


def test_same_timeline_renders_identically(media, tmp_path):
    t = tl([{"id": "s1", "source": "a.mp4", "src_in": 0, "timeline_in": 0, "timeline_out": 1.0}], 1.0,
           grade={"grain": 0.5, "lut": "teal_cold.cube"})

    def md5s(p):
        return subprocess.run(["ffmpeg", "-v", "error", "-i", str(p), "-map", "0:v", "-f", "framemd5", "-"],
                              capture_output=True).stdout

    assert md5s(render(t, tmp_path / "1.mp4", media).path) == md5s(render(t, tmp_path / "2.mp4", media).path)


def test_progress_is_reported(media, tmp_path):
    seen = []
    t = tl([{"id": "s1", "source": "a.mp4", "src_in": 0, "timeline_in": 0, "timeline_out": 1.0}], 1.0)
    render(t, tmp_path / "o.mp4", media, progress=lambda s, f: seen.append((s, f)))
    stages = [s for s, _ in seen]
    assert "Rendering" in stages and stages[-1] == "Encoding"
    assert all(0 <= f <= 1 for _, f in seen)


@pytest.mark.parametrize("seg,msg", [
    ({"source": "missing.mp4"}, "Missing clip"),
    ({"src_in": 7.5}, "only 8.00s long"),
])
def test_clear_errors(media, tmp_path, seg, msg):
    s = {"id": "s1", "source": "a.mp4", "src_in": 0, "timeline_in": 0, "timeline_out": 1.0}
    s.update(seg)
    with pytest.raises(MogError, match=msg):
        render(tl([s], 1.0), tmp_path / "o.mp4", media)
    assert not list(tmp_path.glob("*.mp4"))  # no half-written output left behind


def test_ingest_handles_rotation_vfr_and_hdr(tmp_path):
    rot = make_barcode_clip(tmp_path / "rot.mov", 30, fps=30, w=320, h=180, rotation=90)
    c = ingest(rot)
    assert (c.width, c.height) == (180, 320)
    # Variable frame rate: irregular timestamps -> constant-rate output at a standard rate
    vfr = tmp_path / "vfr.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=320x180:r=30:d=3",
                    "-vf", "setpts='PTS+0.01*sin(N)/TB'", "-fps_mode", "vfr", "-c:v", "libx264", str(vfr)], check=True)
    c = ingest(vfr)
    assert float(c.fps) == 30 and abs(c.frames - 90) <= 2
    # HDR (HLG, 10-bit, like iPhone video) is tone-mapped to 8-bit Rec.709
    hdr = tmp_path / "hlg.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=320x180:r=30:d=1",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p10le", "-color_primaries", "bt2020",
                    "-color_trc", "arib-std-b67", "-colorspace", "bt2020nc", str(hdr)], check=True)
    assert probe_video(hdr).is_hdr
    c = ingest(hdr)
    out = probe_video(c.path)
    assert out.pix_fmt == "yuv420p" and out.color_transfer == "bt709"


def test_ingest_rejects_non_video(tmp_path):
    p = tmp_path / "x.mp4"
    p.write_bytes(b"not a video")
    with pytest.raises(MogError, match="doesn't look like a video"):
        ingest(p)


def test_standard_fps_prefers_declared_rate():
    from fractions import Fraction

    from mog.video.ingest import standard_fps

    assert standard_fps(Fraction(1225, 47), Fraction(25)) == 25          # real clip: avg 26.06, declared 25
    assert standard_fps(Fraction(26), Fraction(30)) == 30                # VFR phone clip that dropped frames
    assert standard_fps(Fraction(363900, 6061), Fraction(60)) == 60
    assert standard_fps(Fraction(2997, 100), Fraction(90000, 1)) == Fraction(30000, 1001)  # bogus declared rate
    assert standard_fps(Fraction(240), None) == 60


def test_slow_motion_makes_in_between_frames(media, tmp_path):
    """A bright square moving 8 px per source frame, at 0.25x speed: with interpolation the
    square should advance ~2 px per output frame instead of jumping 8 px every 4th frame."""
    src = media / "square.mp4"
    from mog.ffmpeg.frames import FrameWriter

    wr = FrameWriter(W, H, 30, ["-c:v", "libx264", "-crf", "10", "-pix_fmt", "yuv420p", str(src)])
    for k in range(60):
        f = np.full((H, W, 3), 30, np.uint8)
        x = 40 + 4 * k
        f[300:360, x:x + 60] = 230
        wr.write(f)
    wr.close()

    def positions(interp):
        t = tl([{"id": "s", "source": "square.mp4", "src_in": 0.5, "timeline_in": 0, "timeline_out": 1.0,
                 "speed_curve": [[0, 0.25]]}], 1.0)
        frames = decode(render(t, tmp_path / f"{interp}.mp4", media, interpolation=interp).path)
        xs = []
        for f in frames[2:-2]:
            col = f[330, :, 1].astype(float)
            xs.append(float(np.average(np.arange(W), weights=np.maximum(col - 80, 0))))
        return np.diff(xs)

    steps_off, steps_on = positions("off"), positions("fast")
    assert np.max(steps_off) > 3.0          # without: still, still, still, JUMP
    assert np.max(np.abs(steps_on - np.median(steps_on))) < 1.0  # with: even ~1 px steps
    assert 0.6 < np.median(steps_on) < 1.4


def test_effects_render_without_black_edges(media, tmp_path):
    """Shake + zoom punch + rgb split on a clip with no zoom headroom must never show a
    black (or replicated-edge) border."""
    t = tl([{"id": "s", "source": "a.mp4", "src_in": 0, "timeline_in": 0, "timeline_out": 2.0,
             "effects": [{"type": "shake", "t": 0.2, "intensity": 60, "decay": 1.5},
                         {"type": "zoom_punch", "t": 0.5, "strength": 1.3, "decay": 0.4},
                         {"type": "rgb_split", "t": 1.0, "frames": 6, "offset": 20}]}], 2.0)
    frames = decode(render(t, tmp_path / "o.mp4", media).path)
    for f in frames:
        for edge in (f[0], f[-1], f[:, 0], f[:, -1]):
            assert edge.mean() > 25  # barcode clip background is grey 40; black would be ~0
    # and the picture actually moved during the shake
    assert np.abs(frames[10].astype(int) - frames[0].astype(int)).mean() > 2


def test_flash_lands_on_the_beat_frame(media, tmp_path):
    # Segment starts at 0.5 s; flash at segment time 0.3 => output time 0.8 s => frame 24 at 30 fps
    t = tl([{"id": "a", "source": "a.mp4", "src_in": 0, "timeline_in": 0, "timeline_out": 0.5},
            {"id": "b", "source": "b.mp4", "src_in": 0, "timeline_in": 0.5, "timeline_out": 1.5,
             "effects": [{"type": "flash", "t": 0.3, "frames": 2}]}], 1.5)
    frames = decode(render(t, tmp_path / "o.mp4", media).path)
    bright = [float(f.mean()) for f in frames]
    assert bright[24] > 240                  # full white on the beat frame
    assert 120 < bright[25] < 220            # half strength on the next
    assert bright[23] < 120 and bright[26] < 120


def test_text_overlay_appears_in_its_window(media, tmp_path):
    t = tl([{"id": "s", "source": "a.mp4", "src_in": 0, "timeline_in": 0, "timeline_out": 2.0}], 2.0,
           grade={"lut": "teal_cold.cube"},
           text=[{"content": "HE DOESN'T CHASE", "t_in": 0.5, "t_out": 1.5, "font": "Anton", "y": 0.15,
                  "anim": "pop"}])
    frames = decode(render(t, tmp_path / "o.mp4", media).path)
    band = slice(int(0.1 * H), int(0.2 * H))
    def whiteness(f):
        return float((f[band].min(axis=2) > 200).mean())
    assert whiteness(frames[5]) < 0.01                     # before t_in: no text
    assert whiteness(frames[30]) > 0.05                    # during: white text (not tinted teal)
    assert whiteness(frames[55]) < 0.01                    # after t_out: gone
