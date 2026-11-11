"""Co-coverage weights J_k(u) = J(u, u + e_k) from a box workload.

Queries are inclusive cell boxes [lo, hi]. The pair (u, u + e_k) lies in the box
iff lo <= u <= hi - e_k, so each dimension needs one d-dim difference array.
Returned array has shape (d, N) in C flat order; J[k, u] = 0 when u_k = s_k - 1.
"""

from __future__ import annotations

import itertools

import numpy as np

from .grid import num_cells, validate_shape


def _normalize(shape, lo, hi, weights):
    shape = validate_shape(shape)
    lo = np.atleast_2d(np.asarray(lo, dtype=np.int64))
    hi = np.atleast_2d(np.asarray(hi, dtype=np.int64))
    if lo.shape != hi.shape or lo.shape[1] != len(shape):
        raise ValueError("lo/hi must have shape (m, d)")
    if np.any(lo < 0) or np.any(hi >= np.asarray(shape)) or np.any(lo > hi):
        raise ValueError("boxes must satisfy 0 <= lo <= hi < shape")
    m = lo.shape[0]
    if weights is None:
        weights = np.full(m, 1.0 / m)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (m,):
        raise ValueError("weights must have shape (m,)")
    return shape, lo, hi, weights


def pair_weights(shape, lo, hi, weights=None, dtype=np.float64) -> np.ndarray:
    """J via difference arrays, O(d (m 2^d + d N)) time.

    ``dtype`` controls the diff/output precision; float32 halves the memory of
    large grids at the cost of ~1e-4 absolute noise in J (acceptable for the
    MWST ordering, which is robust to near-ties)."""
    shape, lo, hi, weights = _normalize(shape, lo, hi, weights)
    d = len(shape)
    out = np.zeros((d,) + shape, dtype=dtype)
    corners = list(itertools.product((0, 1), repeat=d))
    for k in range(d):
        hik = hi.copy()
        hik[:, k] -= 1
        valid = np.all(hik >= lo, axis=1)
        if not np.any(valid):
            continue
        l, h, w = lo[valid], hik[valid], weights[valid]
        diff = np.zeros(tuple(s + 1 for s in shape), dtype=dtype)
        for corner in corners:
            idx = tuple(np.where(c == 0, l[:, j], h[:, j] + 1) for j, c in enumerate(corner))
            sign = -1.0 if sum(corner) % 2 else 1.0
            np.add.at(diff, idx, sign * w)
        for ax in range(d):
            np.cumsum(diff, axis=ax, out=diff)
        out[k] = diff[tuple(slice(0, s) for s in shape)]
    return out.reshape(d, num_cells(shape))


def pair_weights_bruteforce(shape, lo, hi, weights=None) -> np.ndarray:
    """Reference implementation for tests."""
    shape, lo, hi, weights = _normalize(shape, lo, hi, weights)
    d = len(shape)
    out = np.zeros((d,) + shape, dtype=np.float64)
    for i in range(lo.shape[0]):
        for k in range(d):
            h = hi[i].copy()
            h[k] -= 1
            if np.any(h < lo[i]):
                continue
            out[(k,) + tuple(slice(a, b + 1) for a, b in zip(lo[i], h))] += weights[i]
    return out.reshape(d, num_cells(shape))
