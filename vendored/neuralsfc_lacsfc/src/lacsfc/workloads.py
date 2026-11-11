"""Synthetic box-query workloads on a cell grid."""

from __future__ import annotations

import numpy as np

from .grid import validate_shape


def random_boxes(shape, m: int, rng: np.random.Generator, max_frac: float = 0.5,
                 center=None, spread: float = 0.1):
    """Return (lo, hi) inclusive boxes of shape (m, d).

    Side lengths are uniform in [1, max(1, floor(max_frac * s_k))]. With
    ``center=None`` box positions are uniform; otherwise box centres are drawn
    from a normal around ``center`` (in [0, 1]^d units) with std ``spread``.
    """
    shape = validate_shape(shape)
    sides = np.asarray(shape, dtype=np.int64)
    d = len(shape)
    max_len = np.maximum(1, np.floor(max_frac * sides).astype(np.int64))
    length = rng.integers(1, max_len + 1, size=(m, d))
    if center is None:
        lo = rng.integers(0, sides - length + 1, size=(m, d))
    else:
        c = np.asarray(center, dtype=np.float64) * sides
        mid = rng.normal(c, spread * sides, size=(m, d))
        lo = np.rint(mid - length / 2).astype(np.int64)
        lo = np.clip(lo, 0, sides - length)
    hi = lo + length - 1
    return lo.astype(np.int64), hi.astype(np.int64)


def correlated_boxes(shape, m: int, rng: np.random.Generator, max_frac: float = 0.5,
                     kind: str = "diag", spread: float = 0.05):
    """Boxes whose per-axis intervals are dependent (violates the product model).

    - ``diag``    : centres tied to one scalar along the main diagonal (+ jitter);
                    axis positions are strongly correlated.
    - ``iso``     : side length shared across axes; positions uniform.
    - ``iso_diag``: shared side length and diagonal centres.
    """
    shape = validate_shape(shape)
    sides = np.asarray(shape, dtype=np.int64)
    d = len(shape)
    max_len = np.maximum(1, np.floor(max_frac * sides).astype(np.int64))
    if "iso" in kind:
        length = np.repeat(rng.integers(1, int(max_len.min()) + 1, size=(m, 1)), d, axis=1)
    else:
        length = rng.integers(1, max_len + 1, size=(m, d))
    if "diag" in kind:
        t = rng.uniform(0.0, 1.0, size=(m, 1))
        mid = np.rint(t * sides + rng.normal(0.0, spread * sides, size=(m, d))).astype(np.int64)
        lo = np.clip(mid - length // 2, 0, sides - length)
    else:
        lo = rng.integers(0, sides - length + 1, size=(m, d))
    hi = lo + length - 1
    return lo.astype(np.int64), hi.astype(np.int64)
