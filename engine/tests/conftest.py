import numpy as np
import pytest

from mog.audio.analyze import analyze_signal
from mog.testing.synth import make_track

SR = 22050
# One video frame at 60 fps. The quality bar: everything must land within +-1 frame.
FRAME_60 = 1 / 60

# (bpm, offset) pairs covering slow drift-phonk to fast phonk, with awkward start offsets
TRACKS = [(108, 0.6), (130, 1.1), (145, 0.37), (160, 0.05)]

_cache: dict = {}


def analyzed(bpm: float, offset: float, **kw):
    key = (bpm, offset, tuple(sorted(kw.items())))
    if key not in _cache:
        y, truth = make_track(bpm=bpm, offset=offset, sr=SR, **kw)
        _cache[key] = (analyze_signal(y, SR, source=f"synth{bpm}", tracker="librosa"), truth, y)
    return _cache[key]


def match(detected, reference, tol):
    """Return (recall, precision, mean_abs_error_s) for event lists under a tolerance."""
    det = np.asarray(detected, float)
    ref = np.asarray(reference, float)
    if len(det) == 0 or len(ref) == 0:
        return 0.0, 0.0, float("inf")
    err_ref = np.array([np.min(np.abs(det - r)) for r in ref])
    err_det = np.array([np.min(np.abs(ref - d)) for d in det])
    recall = float(np.mean(err_ref < tol))
    precision = float(np.mean(err_det < tol))
    mae = float(err_ref[err_ref < tol].mean()) if np.any(err_ref < tol) else float("inf")
    return recall, precision, mae


@pytest.fixture(params=TRACKS, ids=[f"{b}bpm" for b, _ in TRACKS])
def track(request):
    bpm, offset = request.param
    return analyzed(bpm, offset)
