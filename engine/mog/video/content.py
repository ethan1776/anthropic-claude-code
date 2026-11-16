"""Find the part of the frame that is real footage.

Uploads often carry junk around the picture: letterbox bars, a phone's status bar and app
buttons (screen recordings), or a frame around a reposted video. None of that should end up
in the 9:16 crop.

Trick: footage changes over time; borders don't. For sampled frames we measure, per row and
per column, how much the pixels change over time. Edge rows/columns that hardly change AND
are either a flat colour (bars) or perfectly frozen (phone UI) are trimmed, as are top/bottom
"blur bars" with no fine detail (see _trim).
We never trim more than 30% per side.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import median_filter

MAX_TRIM = 0.30


FROZEN_MAX = 0.12  # phone status / navigation bars: at most this far in from the top/bottom


def _trim(activity: np.ndarray, texture: np.ndarray, allow_frozen: bool,
          sharpness: np.ndarray | None = None) -> tuple[int, int]:
    """Indices [a, b) of the 'live' part, given per-row (or column) activity over time and
    spatial texture of the average image.

    A border row is junk only if it hardly changes AND is either a flat colour (letterbox,
    texture < 6) or completely frozen (phone status/navigation bar: activity < 1).
    "Frozen but textured" is only accepted near the top/bottom (allow_frozen, FROZEN_MAX):
    clean studio footage can be perfectly still on one side, and must not be cut off.
    """
    n = len(activity)
    ref = np.percentile(activity, 75)
    if ref < 1.5:  # the whole clip is (almost) static: don't trim anything
        return 0, n
    still = activity < 0.12 * ref
    frozen = (activity < 1.0) if allow_frozen else np.zeros(n, bool)
    blurred = np.zeros(n, bool)
    if sharpness is not None:
        # "Blur bars" (reposts padded with a blurred copy of the video): they move and have
        # colour, but no fine detail at all
        # (median filter: one-row spikes at the very edge, from the filter border, mustn't stop
        # the trim; measured on a real repost: bars ~0.2x the footage's detail)
        sm = median_filter(sharpness, size=5, mode="nearest")
        if n > 8:  # the outermost rows of a downscaled, filtered image are unreliable
            sm[:2], sm[-2:] = sm[2], sm[-3]
        blurred = sm < 0.35 * np.median(sharpness)
    if allow_frozen:
        frozen[int(n * FROZEN_MAX):n - int(n * FROZEN_MAX)] = False
        blurred[int(n * FROZEN_MAX):n - int(n * FROZEN_MAX)] = False
    junk = (still & ((texture < 6.0) | frozen)) | blurred
    max_cut = int(n * MAX_TRIM)
    a = 0
    while a < max_cut and junk[a]:
        a += 1
    b = n
    while n - b < max_cut and junk[b - 1]:
        b -= 1
    return a, b


def content_bounds(frames: list[np.ndarray]) -> tuple[float, float, float, float]:
    """frames: grey (H, W) uint8 samples spread over the clip. Returns (x0, y0, x1, y1) fractions."""
    if len(frames) < 3:
        return (0.0, 0.0, 1.0, 1.0)
    stack = np.stack([f.astype(np.float32) for f in frames])
    std = stack.std(axis=0)                     # how much each pixel changes over time
    rows = np.percentile(std, 90, axis=1)       # a row is live if part of it moves
    cols = np.percentile(std, 90, axis=0)
    mean_img = stack.mean(axis=0)
    lap = np.mean([np.abs(cv2.Laplacian(f, cv2.CV_32F)) for f in frames], axis=0)
    y0, y1 = _trim(rows, mean_img.std(axis=1), allow_frozen=True,    # texture along each row
                   sharpness=lap.mean(axis=1))
    x0, x1 = _trim(cols, mean_img.std(axis=0), allow_frozen=False)  # side columns: bars only
    h, w = std.shape
    # Small margin inside the detected edge (encoders smear a few pixels across a border)
    my, mx = int(0.004 * h) + (1 if y0 > 0 else 0), int(0.004 * w) + (1 if x0 > 0 else 0)
    y0 = min(y0 + my if y0 > 0 else 0, h - 1)
    y1 = max(y1 - my if y1 < h else h, y0 + 1)
    x0 = min(x0 + mx if x0 > 0 else 0, w - 1)
    x1 = max(x1 - mx if x1 < w else w, x0 + 1)
    return (round(x0 / w, 4), round(y0 / h, 4), round(x1 / w, 4), round(y1 / h, 4))
