"""Bake the whole per-pixel colour grade into ONE 3D LUT.

A grade is several colour operations: the look LUT, its strength, black crush, contrast,
brightness, gamma, saturation. Each one only depends on the pixel's own colour, so the
whole chain is itself a function RGB -> RGB, and a single 3D LUT can hold it. FFmpeg then
does one table lookup per pixel instead of running four filters (measured: ~4x faster).

Vignette, glow and grain depend on *where* a pixel is or on its neighbours, so they can't
go in a LUT (see pixel_fx.py and finish.py).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from mog.color.luts import BUILTIN, LUMA, SIZE, lut_cache_dir, lut_table, resolve_lut, write_cube
from mog.errors import MogError
from mog.timeline.models import Grade

BAKE_VERSION = 1


def read_cube(path: Path) -> tuple[np.ndarray, int]:
    """Parse a .cube file -> ((size**3, 3) table in red-fastest order, size)."""
    size, rows = None, []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        head = line.split()[0].upper()
        if head == "LUT_3D_SIZE":
            size = int(line.split()[1])
        elif head in ("TITLE", "DOMAIN_MIN", "DOMAIN_MAX", "LUT_1D_SIZE", "LUT_3D_INPUT_RANGE"):
            if head == "LUT_1D_SIZE":
                raise MogError(f"{Path(path).name} is a 1D LUT; only 3D .cube LUTs are supported.")
            continue
        else:
            try:
                rows.append([float(v) for v in line.split()[:3]])
            except ValueError:
                raise MogError(f"{Path(path).name} is not a valid .cube file (line: {line[:40]!r}).") from None
    if not size or len(rows) != size**3:
        raise MogError(f"{Path(path).name} is not a valid .cube file (expected {size}^3 entries, got {len(rows)}).")
    return np.clip(np.array(rows), 0, 1), size


def _grid(size: int) -> np.ndarray:
    v = np.linspace(0, 1, size)
    b, g, r = np.meshgrid(v, v, v, indexing="ij")
    return np.stack([r, g, b], -1).reshape(-1, 3)


def black_crush(x: np.ndarray, c: float) -> np.ndarray:
    """Values below c go to black; above, stretched back up to 1, with a soft knee (no harsh edge)."""
    if c <= 0:
        return x
    k = 0.03
    f = lambda v: ((v - c) + np.sqrt((v - c) ** 2 + k * k)) / 2  # noqa: E731 - smooth max(v - c, 0)
    return np.clip((f(x) - f(0.0)) / (f(1.0) - f(0.0)), 0, 1)


def apply_tone(rgb: np.ndarray, g: Grade) -> np.ndarray:
    x = black_crush(rgb, g.black_crush)
    x = (x - 0.5) * g.contrast + 0.5 + g.brightness
    x = np.clip(x, 0, 1) ** (1.0 / g.gamma)
    y = (x @ LUMA)[..., None]
    x = y + (x - y) * g.saturation
    return np.clip(x, 0, 1)


def is_identity(g: Grade) -> bool:
    return (g.lut is None and g.black_crush == 0 and g.contrast == 1 and g.brightness == 0
            and g.gamma == 1 and g.saturation == 1)


def bake(g: Grade, search_dirs: list[Path] = ()) -> Path | None:
    """Return a .cube holding the whole per-pixel grade, or None if the grade does nothing."""
    if is_identity(g):
        return None
    if g.lut is None:
        size, base = SIZE, _grid(SIZE)
    elif g.lut in BUILTIN:
        size, base = SIZE, lut_table(BUILTIN[g.lut])
    else:
        base, size = read_cube(resolve_lut(g.lut, list(search_dirs)))
    ident = _grid(size)
    looked = ident + (base - ident) * g.lut_strength
    table = apply_tone(looked, g)

    key = hashlib.sha1(json.dumps([BAKE_VERSION, g.model_dump(), base[:: max(1, len(base) // 997)].round(5).tolist()])
                       .encode()).hexdigest()[:16]
    path = lut_cache_dir() / f"baked-{key}.cube"
    if not path.exists():
        tmp = path.with_suffix(".part")
        write_cube(tmp, table, size=size, title="mog baked grade")
        tmp.replace(path)
    return path
