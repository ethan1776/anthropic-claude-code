"""Built-in colour LUTs, generated from maths (no third-party LUT files to license).

A 3D LUT ("look-up table", .cube file) maps every input colour to an output colour. We
compute each look as a function of RGB, sample it on a 33x33x33 grid, and write the .cube
file to the cache the first time it's used. FFmpeg's lut3d filter applies it.

Users can also point a timeline at their own .cube file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from mog.errors import MogError

LUT_VERSION = 1
SIZE = 33
LUMA = np.array([0.2126, 0.7152, 0.0722])  # Rec.709 luminance weights


def _luma(rgb: np.ndarray) -> np.ndarray:
    return rgb @ LUMA


def _s_curve(x: np.ndarray, amount: float) -> np.ndarray:
    """Filmic contrast S-curve around mid-grey. amount 0 = identity."""
    s = x * x * (3 - 2 * x)
    return np.clip((1 - amount) * x + amount * s, 0, 1)


def _saturate(rgb: np.ndarray, s: float) -> np.ndarray:
    y = _luma(rgb)[..., None]
    return np.clip(y + (rgb - y) * s, 0, 1)


def _split_tone(rgb: np.ndarray, shadow: tuple, highlight: tuple, amount: float) -> np.ndarray:
    y = _luma(rgb)[..., None]
    sh = np.array(shadow) * ((1 - y) ** 2)
    hi = np.array(highlight) * (y ** 2)
    return np.clip(rgb + amount * (sh + hi), 0, 1)


def bw_crush(rgb):
    """Dark Triad: black & white, crushed blacks, hard contrast, very slightly cool."""
    y = _luma(rgb)
    y = np.clip((y - 0.05) / 0.9, 0, 1)      # crush the toe, clip the very top
    y = _s_curve(y, 0.85) ** 1.12
    out = np.stack([y, y, y * 1.0 + 0.01 * (1 - y)], -1)
    return np.clip(out, 0, 1)


def teal_cold(rgb):
    """Mog: cold teal, desaturated, high contrast."""
    x = _saturate(rgb, 0.55)
    x = _split_tone(x, shadow=(-0.06, 0.05, 0.09), highlight=(-0.02, 0.02, 0.05), amount=1.0)
    return _s_curve(x, 0.6)


def velocity_punch(rgb):
    """Velocity: colour kept, punchy contrast, orange/teal separation."""
    x = _saturate(rgb, 1.15)
    x = _split_tone(x, shadow=(-0.04, 0.03, 0.06), highlight=(0.06, 0.02, -0.04), amount=1.0)
    return _s_curve(x, 0.5)


def night_drift(rgb):
    """Drift Phonk: saturated night colours, purple shadows, magenta-pink highlights."""
    x = _saturate(rgb, 1.45)
    x = _split_tone(x, shadow=(0.05, -0.04, 0.12), highlight=(0.08, -0.02, 0.06), amount=1.0)
    return _s_curve(x, 0.45) ** 1.05


def clean_film(rgb):
    """Clean Sigma: light, natural contrast, a touch of warmth in highlights."""
    x = _split_tone(rgb, shadow=(-0.01, 0.01, 0.02), highlight=(0.03, 0.015, -0.01), amount=1.0)
    return _s_curve(x, 0.25)


BUILTIN: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "bw_crush.cube": bw_crush,
    "teal_cold.cube": teal_cold,
    "velocity_punch.cube": velocity_punch,
    "night_drift.cube": night_drift,
    "clean_film.cube": clean_film,
}


def lut_table(fn: Callable[[np.ndarray], np.ndarray], size: int = SIZE) -> np.ndarray:
    """(size**3, 3) table in .cube order: red changes fastest, then green, then blue."""
    v = np.linspace(0, 1, size)
    b, g, r = np.meshgrid(v, v, v, indexing="ij")
    rgb = np.stack([r, g, b], -1).reshape(-1, 3)
    return np.clip(fn(rgb), 0, 1)


def write_cube(path: Path, table: np.ndarray, size: int = SIZE, title: str = "") -> Path:
    lines = [f'TITLE "{title}"', f"LUT_3D_SIZE {size}", "DOMAIN_MIN 0.0 0.0 0.0", "DOMAIN_MAX 1.0 1.0 1.0"]
    lines += [f"{r:.6f} {g:.6f} {b:.6f}" for r, g, b in table]
    path.write_text("\n".join(lines) + "\n")
    return path


def lut_cache_dir() -> Path:
    from mog.video.ingest import cache_dir

    d = cache_dir().parent / "luts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def resolve_lut(name: str, search_dirs: list[Path] = ()) -> Path:
    """A built-in LUT name -> generated file; otherwise a .cube path (absolute or in search_dirs)."""
    if name in BUILTIN:
        path = lut_cache_dir() / f"v{LUT_VERSION}-{name}"
        if not path.exists():
            tmp = path.with_suffix(".part")
            write_cube(tmp, lut_table(BUILTIN[name]), title=name)
            tmp.replace(path)
        return path
    for base in [Path("."), *search_dirs]:
        p = (base / name) if not Path(name).is_absolute() else Path(name)
        if p.exists() and p.suffix.lower() == ".cube":
            return p
    raise MogError(f"LUT '{name}' not found. Built-in LUTs: {', '.join(BUILTIN)}; or give a path to a .cube file.")
