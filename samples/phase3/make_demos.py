"""Phase 3 effect demos: one short timeline per effect, rendered on real clips.

Usage (from engine/):  python ../samples/phase3/make_demos.py MEDIA_DIR OUT_DIR
MEDIA_DIR must contain the clips named in CLIPS below plus phonk145.wav
(`python -m mog synth MEDIA_DIR/phonk145.wav --bpm 145`). Swap in your own clips by
editing CLIPS. Timelines are written next to the videos so you can see exactly what
each demo asks for.
"""

import json
import subprocess
import sys
from pathlib import Path

from mog.render.renderer import render
from mog.timeline.models import Timeline

BEAT = 60 / 145
DROP = 13.6114  # start of the synthetic track's drop: beats land at 0, BEAT, 2*BEAT, ...
CLIPS = {"face": ("clip4.mp4", 0.55), "tux": ("clip1.mov", 0.5), "motion": ("clip2.mp4", 0.5),
         "webcam": ("clip3.mp4", 0.5)}
GRADE = {"lut": "teal_cold.cube", "contrast": 1.15, "black_crush": 0.04, "grain": 0.2, "vignette": 0.4}


def seg(sid, clip, src_in, t0, t1, **kw):
    name, x = CLIPS[clip]
    d = {"id": sid, "source": name, "src_in": src_in, "timeline_in": round(t0, 4), "timeline_out": round(t1, 4),
         "crop": {"mode": "manual", "x": x, "y": 0.45}}
    d.update(kw)
    return d


def timeline(segments, duration, grade=GRADE, text=()):
    return {"output": {"width": 1080, "height": 1920, "fps": 30, "duration": round(duration, 4)},
            "audio": {"file": "phonk145.wav", "start": DROP, "end": DROP + duration + 0.5, "fade_out": 0.3},
            "grade": grade, "segments": segments, "text": list(text)}


B = BEAT
DEMOS = {
    # same footage, slowed to 0.3x: left = repeated frames, right = optical-flow in-betweens
    "slowmo": timeline([seg("s", "face", 2.0, 0, 8 * B, speed_curve=[[0, 0.3]])], 8 * B),
    # velocity ramp: 3x, snapping to 0.3x exactly on beat 4, with motion blur in the fast part
    "velocity_ramp": timeline([seg("s", "motion", 1.0, 0, 8 * B,
                                   speed_curve=[[0, 3.0], [4 * B - 0.06, 3.0], [4 * B, 0.3], [8 * B, 0.3]],
                                   ease="cubic", effects=[{"type": "motion_blur", "t_in": 0, "t_out": 4 * B,
                                                           "strength": 0.8}])], 8 * B),
    "zoom_punch": timeline([seg("s", "tux", 0.2, 0, 8 * B, effects=[
        {"type": "zoom_punch", "t": round(k * 2 * B, 4), "strength": 1.25, "decay": 0.3} for k in range(4)])], 8 * B),
    "shake": timeline([seg("s", "face", 0.3, 0, 8 * B, effects=[
        {"type": "shake", "t": 0.0, "intensity": 24, "decay": 0.45},
        {"type": "shake", "t": round(4 * B, 4), "intensity": 32, "decay": 0.5}])], 8 * B),
    "flash": timeline([seg("a", "webcam", 10, 0, 2 * B, effects=[{"type": "flash", "t": 0, "frames": 3}]),
                       seg("b", "tux", 0.5, 2 * B, 4 * B, effects=[{"type": "flash", "t": 0, "frames": 2}]),
                       seg("c", "face", 3.0, 4 * B, 6 * B, effects=[{"type": "flash", "t": 0, "frames": 2}]),
                       seg("d", "motion", 5.0, 6 * B, 8 * B, effects=[{"type": "flash", "t": 0, "frames": 3}])],
                      8 * B),
    "rgb_split": timeline([seg("s", "tux", 0.2, 0, 8 * B, effects=[
        {"type": "rgb_split", "t": round(k * B, 4), "frames": 4, "offset": 14} for k in range(8)])], 8 * B),
    "text": timeline([seg("s", "tux", 0.2, 0, 8 * B, speed_curve=[[0, 0.5]])], 8 * B,
                     grade={**GRADE, "lut": "bw_crush.cube", "saturation": 0},
                     text=[{"content": "he doesn't chase.", "t_in": round(2 * B, 4), "t_out": round(7.5 * B, 4),
                            "font": "Cinzel", "size": 0.05, "y": 0.78, "anim": "pop"}]),
    # same fast footage at 2.5x: left = sharp frames, right = motion blur
    "motion_blur": timeline([seg("s", "motion", 1.0, 0, 8 * B, speed_curve=[[0, 2.5]],
                                 effects=[{"type": "motion_blur", "t_in": 0, "t_out": round(8 * B, 4),
                                           "strength": 1.0}])], 8 * B),
}


def main(media: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, d in DEMOS.items():
        (out / f"{name}.json").write_text(json.dumps(d, indent=1))
        tl = Timeline.model_validate(d)
        if name in ("slowmo", "motion_blur"):
            # Side-by-side comparison: without (left) vs with (right)
            if name == "slowmo":
                a = render(tl, out / "_a.mp4", media, interpolation="off")
            else:
                plain = Timeline.model_validate({**d, "segments": [{**d["segments"][0], "effects": []}]})
                a = render(plain, out / "_a.mp4", media)
            b = render(tl, out / "_b.mp4", media)
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(a.path), "-i", str(b.path), "-filter_complex",
                            "[0:v]scale=540:960[l];[1:v]scale=540:960[r];[l][r]hstack[v]", "-map", "[v]", "-map", "1:a",
                            "-c:v", "libx264", "-crf", "18", "-c:a", "copy", "-movflags", "+faststart",
                            str(out / f"{name}.mp4")], check=True)
            a.path.unlink()
            b.path.unlink()
            print(f"{name}: {b.seconds:.1f}s render (side by side)")
        else:
            r = render(tl, out / f"{name}.mp4", media)
            print(f"{name}: {r.seconds:.1f}s render")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
