"""The final FFmpeg pass: colour grade, glow, vignette, grain, audio, encode.

Inputs: 0 = raw BGR frames from Python, 1 = the music, 2.. = text overlay images.

  1. ONE 3D LUT holding the whole per-pixel grade: look LUT + strength + black crush +
     contrast/brightness/gamma + saturation (baked in grade_lut.py; ~4x faster than
     running those as separate filters)
  2. convert to YUV Rec.709
  3. text overlays (after the grade, so a white caption stays white)
  4. film grain on luma only (last, so the grade doesn't amplify it)

Glow and vignette are applied per frame in Python before frames reach FFmpeg
(pixel_fx.py): measured much cheaper there than as FFmpeg filters.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mog.color.grade_lut import bake
from mog.ffmpeg.filtergraph import F, Graph
from mog.render.text import add_text_overlays
from mog.timeline.models import Audio, Grade, TextOverlay

GRAIN_SEED = 20240611  # fixed so the same timeline renders the same grain


@dataclass
class EncodeSettings:
    preset: str
    crf: int
    maxrate: str
    bufsize: str
    audio_bitrate: str

    @classmethod
    def export(cls) -> "EncodeSettings":
        # ~12-16 Mbps at 1080x1920: capped-quality encode (quality target, bitrate ceiling)
        return cls(preset="medium", crf=17, maxrate="15M", bufsize="15M", audio_bitrate="320k")

    @classmethod
    def preview(cls) -> "EncodeSettings":
        return cls(preset="veryfast", crf=23, maxrate="6M", bufsize="12M", audio_bitrate="160k")


@dataclass
class FinishGraph:
    graph: str
    extra_inputs: list[list[str]]  # FFmpeg args for inputs 2, 3, ... (after video pipe and audio)


def build_graph(grade: Grade, width: int, height: int, duration: float, audio: Audio, fps: int,
                search_dirs: list[Path] = (), texts: list[TextOverlay] = (), text_pngs: list[Path] = ()
                ) -> FinishGraph:
    g = Graph()
    cur = "0:v"

    def step(filters: list[F]) -> None:
        nonlocal cur
        out = g.label("v")
        g.add([cur], filters, [out])
        cur = out

    step([F("format", pix_fmts="gbrp")])

    lut = bake(grade, list(search_dirs))
    if lut is not None:
        step([F("lut3d", file=str(lut), interp="tetrahedral")])

    step([F("scale", out_color_matrix="bt709", out_range="tv"), F("format", pix_fmts="yuv420p")])

    # Text after the grade (so captions keep their colour), before grain (so they sit in the film)
    extra: list[list[str]] = []
    if texts:
        cur, extra = add_text_overlays(g, cur, list(texts), list(text_pngs), 2, width, height, fps)

    last = [F("format", pix_fmts="yuv420p")]
    if grade.grain > 0:
        # Temporal (changes every frame) uniform noise on luma only = monochrome film grain
        last.append(F("noise", c0s=round(grade.grain * 28, 2), c0f="t+u", c0_seed=GRAIN_SEED))
    g.add([cur], last, ["vout"])

    fade_in = min(audio.fade_in, duration / 2)
    fade_out = min(audio.fade_out, duration / 2)
    afilters = [F("aformat", sample_rates=48000, channel_layouts="stereo")]
    if fade_in > 0:
        afilters.append(F("afade", t="in", st=0, d=fade_in))
    if fade_out > 0:
        afilters.append(F("afade", t="out", st=round(duration - fade_out, 6), d=fade_out))
    afilters += [F("apad"), F("atrim", end=round(duration, 6))]
    g.add(["1:a"], afilters, ["aout"])
    return FinishGraph(g.render(), extra)


def encoder_args(fg: FinishGraph, audio_path: Path, audio: Audio, duration: float, fps: int,
                 out_path: Path, settings: EncodeSettings) -> list[str]:
    """Arguments that follow the raw-video pipe input (input 0)."""
    length = min(audio.end - audio.start, duration + 1.0)
    args = ["-ss", f"{audio.start:.6f}", "-t", f"{length:.6f}", "-i", str(audio_path)]
    for extra in fg.extra_inputs:
        args += extra
    return args + [
        "-filter_complex", fg.graph, "-filter_complex_threads", "4",
        "-map", "[vout]", "-map", "[aout]",
        "-c:v", "libx264", "-preset", settings.preset, "-profile:v", "high", "-level:v", "4.2",
        "-crf", str(settings.crf), "-maxrate", settings.maxrate, "-bufsize", settings.bufsize,
        "-g", str(fps * 2), "-pix_fmt", "yuv420p", "-r", str(fps), "-fps_mode", "cfr",
        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
        "-c:a", "aac", "-b:a", settings.audio_bitrate, "-ar", "48000", "-ac", "2",
        "-t", f"{duration:.6f}", "-movflags", "+faststart", str(out_path),
    ]
