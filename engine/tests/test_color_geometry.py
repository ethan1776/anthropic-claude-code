"""Crop geometry (no black edges), LUT files, grade baking, filtergraph escaping."""

import subprocess

import numpy as np
import pytest

from mog.color.grade_lut import apply_tone, bake, black_crush, read_cube
from mog.color.luts import BUILTIN, lut_table, resolve_lut, write_cube
from mog.errors import MogError
from mog.ffmpeg.filtergraph import F, Graph
from mog.render.geometry import crop_center, window_affine
from mog.render.pixel_fx import Glow, Vignette
from mog.timeline.models import Crop, Grade


def visible_source_rect(m, out_w, out_h):
    """Bounding box (source pixels) of where the output's 4 corner pixel centres sample from."""
    inv = np.linalg.inv(np.vstack([m, [0, 0, 1]]))
    pts = np.array([[0, 0, 1], [out_w - 1, 0, 1], [0, out_h - 1, 1], [out_w - 1, out_h - 1, 1]]).T
    xs, ys = (inv @ pts)[:2]
    return xs.min(), ys.min(), xs.max(), ys.max()


@pytest.mark.parametrize("src", [(1920, 1080), (1080, 1920), (960, 720), (720, 720)])
def test_window_never_leaves_the_source(src):
    rng = np.random.default_rng(0)
    sw, sh = src
    for _ in range(300):
        cx, cy = rng.uniform(-0.2, 1.2, 2)
        zoom = rng.uniform(1, 3)
        off = rng.uniform(-200, 200, 2)
        angle = rng.uniform(-4, 4)
        m = window_affine(sw, sh, 1080, 1920, cx, cy, zoom, tuple(off), angle)
        x0, y0, x1, y1 = visible_source_rect(m, 1080, 1920)
        # pixel centres sampled stay within the source's pixel centres (+-0.5 px edge)
        assert x0 >= -0.5 - 1e-6 and y0 >= -0.5 - 1e-6
        assert x1 <= sw - 0.5 + 1e-6 and y1 <= sh - 0.5 + 1e-6


def test_shake_offset_moves_the_picture():
    # Room to move: zoomed 1.5x. Offset +30 px on screen => content moves right by 30 px
    m0 = window_affine(1920, 1080, 1080, 1920, 0.5, 0.5, 1.5)
    m1 = window_affine(1920, 1080, 1080, 1920, 0.5, 0.5, 1.5, (30.0, 0.0))
    p = np.array([960, 540, 1])
    assert (m1 @ p)[0] - (m0 @ p)[0] == pytest.approx(30.0)


def test_zoom_one_fills_output_exactly():
    m = window_affine(1920, 1080, 1080, 1920, 0.5, 0.5, 1.0)
    x0, y0, x1, y1 = visible_source_rect(m, 1080, 1920)
    assert y0 == pytest.approx(-0.5 + 0.5 / m[1, 1], abs=1e-6)        # full height used
    assert (x1 + x0) / 2 == pytest.approx(959.5)                        # centred


def test_same_size_is_identity():
    m = window_affine(1080, 1920, 1080, 1920, 0.5, 0.5, 1.0)
    assert np.allclose(m, [[1, 0, 0], [0, 1, 0]])


def test_crop_path_interpolates():
    c = Crop(mode="face_track", path=[(0.0, 0.2, 0.5), (1.0, 0.8, 0.3)])
    assert crop_center(c, 0.5) == pytest.approx((0.5, 0.4))
    assert crop_center(c, 5.0) == pytest.approx((0.8, 0.3))


def test_cube_round_trip(tmp_path):
    for name, fn in BUILTIN.items():
        t = lut_table(fn, 9)
        p = write_cube(tmp_path / name, t, size=9)
        back, size = read_cube(p)
        assert size == 9 and np.allclose(back, t, atol=1e-6)


def test_bw_lut_is_grey():
    t = lut_table(BUILTIN["bw_crush.cube"], 9)
    assert np.allclose(t[:, 0], t[:, 1], atol=1e-6)


def test_identity_grade_bakes_nothing():
    assert bake(Grade()) is None
    assert bake(Grade(grain=0.5, vignette=0.5, glow=0.2)) is None  # those aren't per-pixel colour ops


def test_baked_grade_matches_tone_maths():
    g = Grade(contrast=1.3, saturation=0.0, black_crush=0.1)
    table, size = read_cube(bake(g))
    v = np.linspace(0, 1, size)
    # grey ramp entries (r = g = b) must equal the tone curve applied to grey
    grey_idx = [i + i * size + i * size * size for i in range(size)]
    expect = apply_tone(np.stack([v, v, v], -1), g)
    assert np.allclose(table[grey_idx], expect, atol=1e-5)
    assert np.allclose(table[:, 0], table[:, 1], atol=1e-5)  # saturation 0 => grey


def test_black_crush_shape():
    x = np.linspace(0, 1, 101)
    y = black_crush(x, 0.1)
    assert y[0] == pytest.approx(0) and y[-1] == pytest.approx(1)
    assert y[5] < 0.03 and np.all(np.diff(y) >= -1e-9)


def test_user_lut_and_missing_lut(tmp_path):
    write_cube(tmp_path / "mine.cube", lut_table(BUILTIN["night_drift.cube"], 5), size=5)
    assert bake(Grade(lut="mine.cube"), [tmp_path]).exists()
    with pytest.raises(MogError, match="not found"):
        resolve_lut("nope.cube")
    bad = tmp_path / "bad.cube"
    bad.write_text("LUT_3D_SIZE 3\n0 0 0\n")
    with pytest.raises(MogError, match="not a valid"):
        read_cube(bad)


def test_filtergraph_escapes_awkward_paths(tmp_path):
    """A LUT path with a colon, comma, quote, brackets and spaces must survive FFmpeg parsing."""
    weird = tmp_path / "it's [my], dir: x"
    weird.mkdir()
    lut = write_cube(weird / "look, v2.cube", lut_table(BUILTIN["teal_cold.cube"], 5), size=5)
    g = Graph()
    g.add(["0:v"], [F("format", pix_fmts="gbrp"), F("lut3d", file=str(lut))], ["out"])
    proc = subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=64x64:d=0.1",
                           "-filter_complex", g.render(), "-map", "[out]", "-f", "null", "-"], capture_output=True)
    assert proc.returncode == 0, proc.stderr.decode()


def test_vignette_darkens_corners_only():
    f = np.full((192, 108, 3), 200, np.uint8)
    out = Vignette(108, 192, 1.0)(f)
    assert out[96, 54, 0] == 200 and out[0, 0, 0] < 60


def test_glow_only_brightens():
    rng = np.random.default_rng(1)
    f = rng.integers(0, 256, (192, 108, 3), dtype=np.uint8)
    out = Glow(108, 192, 0.5)(f)
    assert np.all(out.astype(int) >= f.astype(int))
    dark = np.full((192, 108, 3), 40, np.uint8)
    assert np.array_equal(Glow(108, 192, 1.0)(dark), dark)  # nothing above the threshold


def test_exposure_table():
    from mog.render.pixel_fx import exposure_table

    assert exposure_table(0) is None
    up, down = exposure_table(1.0), exposure_table(-1.0)
    assert up[0] == 0 and up[255] >= 250 and down[0] == 0
    assert np.all(np.diff(up.astype(int)) >= 0) and np.all(np.diff(down.astype(int)) >= 0)
    # +1 stop on mid-grey (~118) is roughly +37% in sRGB values; -1 stop roughly -27%
    assert 150 < up[118] < 175 and 75 < down[118] < 95
    # highlights roll off: nothing below 90% input clips to white
    assert up[230] < 255
