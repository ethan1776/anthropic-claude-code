"""Text overlays: one bold line that pops in on a beat.

Each overlay is drawn once with Pillow to a transparent PNG (with a soft dark outline so
it reads on any footage). FFmpeg then animates and composites it AFTER the colour grade,
so a white caption stays white under a teal or purple look.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from mog.errors import MogError
from mog.ffmpeg.filtergraph import F, Graph
from mog.timeline.models import TextOverlay

FONTS_DIR = Path(__file__).resolve().parents[3] / "assets" / "fonts"
FONT_FILES = {"cinzel": "Cinzel.ttf", "anton": "Anton.ttf", "inter": "Inter.ttf"}
POP_S = 0.16  # seconds for the pop to settle


def font_path(name: str) -> Path:
    key = name.lower().replace(" ", "")
    if key in FONT_FILES:
        p = FONTS_DIR / FONT_FILES[key]
        if p.exists():
            return p
    p = Path(name)
    if p.suffix.lower() in (".ttf", ".otf") and p.exists():
        return p
    raise MogError(f"Font '{name}' not found. Built-in fonts: Cinzel, Anton, Inter (or a path to a .ttf).")


def render_text_png(tx: TextOverlay, out_w: int, out_h: int, cache: Path) -> Path:
    px = max(8, int(tx.size * out_h))
    key = hashlib.sha1(f"{tx.content}|{tx.font}|{px}|{tx.color}|{out_w}".encode()).hexdigest()[:16]
    path = cache / f"text-{key}.png"
    if path.exists():
        return path
    font = ImageFont.truetype(str(font_path(tx.font)), px)
    try:
        font.set_variation_by_name("Bold")
    except (OSError, ValueError, AttributeError):
        pass  # not a variable font, or no Bold instance: use as is
    # Shrink to fit 90% of the width
    while True:
        box = font.getbbox(tx.content)
        if box[2] - box[0] <= 0.9 * out_w or font.size <= 8:
            break
        font = font.font_variant(size=int(font.size * 0.92))
    w, h = box[2] - box[0], box[3] - box[1]
    pad = int(font.size * 0.4)
    img = Image.new("RGBA", (w + 2 * pad, h + 2 * pad), (0, 0, 0, 0))
    # Soft shadow/outline for legibility
    shadow = Image.new("L", img.size, 0)
    ImageDraw.Draw(shadow).text((pad - box[0], pad - box[1]), tx.content, font=font, fill=255,
                                stroke_width=max(1, font.size // 14), stroke_fill=255)
    shadow = shadow.filter(ImageFilter.GaussianBlur(max(1, font.size // 10)))
    img.putalpha(shadow.point(lambda v: int(v * 0.7)))
    color = tuple(int(tx.color[i:i + 2], 16) for i in (1, 3, 5))
    ImageDraw.Draw(img).text((pad - box[0], pad - box[1]), tx.content, font=font, fill=(*color, 255))
    cache.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


def add_text_overlays(g: Graph, cur: str, texts: list[TextOverlay], pngs: list[Path], first_input: int,
                      out_w: int, out_h: int, fps: int) -> tuple[str, list[list[str]]]:
    """Add overlay chains to graph g. Returns (new current label, extra FFmpeg input args)."""
    extra: list[list[str]] = []
    for i, (tx, png) in enumerate(zip(texts, pngs)):
        idx = first_input + i
        extra.append(["-loop", "1", "-framerate", str(fps), "-t", f"{tx.t_out + 0.1:.4f}", "-i", str(png)])
        t0, t1 = tx.t_in, tx.t_out
        lab = g.label("tx")
        chain = [F("format", pix_fmts="rgba")]
        if tx.anim == "pop":
            # scale 1.35 -> 1.0, ease-out cubic over POP_S, starting on the beat
            u = f"clip((t-{t0:.4f})/{POP_S},0,1)"
            sc = f"(1+0.35*pow(1-{u},3))"
            chain.append(F("scale", w=f"trunc(iw*{sc}/2)*2", h=f"trunc(ih*{sc}/2)*2", eval="frame"))
            chain.append(F("fade", t="in", st=round(t0, 4), d=0.05, alpha=1))
        elif tx.anim == "fade":
            chain.append(F("fade", t="in", st=round(t0, 4), d=0.2, alpha=1))
        chain.append(F("fade", t="out", st=round(max(t0, t1 - 0.12), 4), d=0.12, alpha=1))
        g.add([f"{idx}:v"], chain, [lab])
        out = g.label("v")
        g.add([cur, lab], [F("overlay", x="(W-w)/2", y=f"{tx.y:.4f}*H-h/2", eval="frame",
                             enable=f"between(t,{t0:.4f},{t1:.4f})", format="yuv420")], [out])
        cur = out
    return cur, extra
