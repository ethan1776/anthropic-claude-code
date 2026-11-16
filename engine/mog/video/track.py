"""Smooth a noisy face position over time: Kalman filter + Rauch-Tung-Striebel smoother.

Face detections jump around by a few pixels every frame and sometimes drop out. Cropping
straight on them makes the frame jitter. A Kalman filter models the face as moving with a
roughly constant velocity and weighs each detection by how much it trusts it; detections
that are wildly off (a different face for one frame) are ignored.

Because analysis is offline we also run the RTS *smoother* backwards over the result, so
each point uses the past AND the future. A forward-only filter lags behind fast moves; the
smoother doesn't.
"""

from __future__ import annotations

import numpy as np


def kalman_smooth(t: np.ndarray, z: np.ndarray, r: np.ndarray, q: float = 0.02,
                  gate: float = 4.0) -> np.ndarray:
    """1-D constant-velocity smoother.

    t: sample times (s). z: measurements (NaN = missing). r: measurement std per sample.
    q: process noise (how fast the true position can change; units/s^1.5).
    Returns the smoothed position at every t (gaps are filled by the motion model).
    """
    n = len(t)
    if n == 0:
        return np.zeros(0)
    first = np.flatnonzero(~np.isnan(z))
    if first.size == 0:
        return np.full(n, np.nan)
    x = np.array([z[first[0]], 0.0])
    p = np.diag([r[first[0]] ** 2, 1.0])
    xs, ps, xp_s, pp_s, fs = [], [], [], [], []
    prev_t = t[0]
    for i in range(n):
        dt = max(t[i] - prev_t, 1e-3) if i else 0.0
        prev_t = t[i]
        f = np.array([[1.0, dt], [0.0, 1.0]])
        qm = q * q * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
        x = f @ x
        p = f @ p @ f.T + qm
        xp_s.append(x.copy())
        pp_s.append(p.copy())
        fs.append(f)
        if not np.isnan(z[i]):
            s = p[0, 0] + r[i] ** 2
            innov = z[i] - x[0]
            if abs(innov) <= gate * np.sqrt(s):          # outlier gate
                k = p[:, 0] / s
                x = x + k * innov
                p = p - np.outer(k, p[0, :])
        xs.append(x.copy())
        ps.append(p.copy())
    # RTS backward pass
    xsm = [None] * n
    xsm[-1] = xs[-1]
    psm = ps[-1]
    for i in range(n - 2, -1, -1):
        c = ps[i] @ fs[i + 1].T @ np.linalg.inv(pp_s[i + 1])
        xsm[i] = xs[i] + c @ (xsm[i + 1] - xp_s[i + 1])
        psm = ps[i] + c @ (psm - pp_s[i + 1]) @ c.T
    return np.array([v[0] for v in xsm])


def jitter(path: np.ndarray, dt: float) -> float:
    """RMS of the second difference (acceleration) of a path, per second^2. Lower = steadier."""
    if len(path) < 3:
        return 0.0
    acc = np.diff(path, 2) / (dt * dt)
    return float(np.sqrt(np.mean(acc**2)))
