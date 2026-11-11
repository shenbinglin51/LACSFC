"""Objective, segment counts and the certified upper bound UB0."""

from __future__ import annotations

import numpy as np

from .construct import adjacent_values
from .grid import validate_shape


def path_phi(order: np.ndarray, J: np.ndarray, shape) -> float:
    """Phi of an AC path: sum of J over consecutive cells."""
    order = np.asarray(order, dtype=np.int64)
    return float(adjacent_values(J, order[:-1], order[1:], shape).sum())


def segment_intervals(rank: np.ndarray, lo, hi, shape) -> list[tuple[int, int]]:
    """Return maximal inclusive code intervals induced by the inclusive box [lo, hi]."""
    shape = validate_shape(shape)
    lo = np.asarray(lo, dtype=np.int64)
    hi = np.asarray(hi, dtype=np.int64)
    if lo.shape != (len(shape),) or hi.shape != (len(shape),):
        raise ValueError("lo and hi must have one bound per axis")
    if np.any(lo < 0) or np.any(hi >= np.asarray(shape)) or np.any(lo > hi):
        raise ValueError("invalid inclusive box")
    grid = np.asarray(rank).reshape(shape)
    codes = np.sort(grid[tuple(slice(int(a), int(b) + 1) for a, b in zip(lo, hi))].ravel())
    if codes.size == 0:
        return []
    cuts = np.flatnonzero(np.diff(codes) != 1)
    starts = np.r_[0, cuts + 1]
    ends = np.r_[cuts, codes.size - 1]
    return [(int(codes[a]), int(codes[b])) for a, b in zip(starts, ends)]


def segment_count(rank: np.ndarray, lo, hi, shape) -> int:
    """Number of maximal runs of consecutive ranks inside the box [lo, hi] (any bijection)."""
    return len(segment_intervals(rank, lo, hi, shape))


def avg_segments(rank: np.ndarray, lo, hi, shape, weights=None) -> float:
    lo = np.atleast_2d(lo)
    hi = np.atleast_2d(hi)
    m = lo.shape[0]
    w = np.full(m, 1.0 / m) if weights is None else np.asarray(weights, dtype=np.float64)
    return float(sum(w[i] * segment_count(rank, lo[i], hi[i], shape) for i in range(m)))


def expected_box_size(lo, hi, weights=None) -> float:
    lo = np.atleast_2d(lo)
    hi = np.atleast_2d(hi)
    m = lo.shape[0]
    w = np.full(m, 1.0 / m) if weights is None else np.asarray(weights, dtype=np.float64)
    return float(np.dot(w, np.prod(hi - lo + 1, axis=1)))


def upper_bound_ub0(J: np.ndarray, shape) -> float:
    """UB0 = 1/2 * sum_v (sum of the two largest J on edges incident to v) >= Phi*."""
    shape = validate_shape(shape)
    d = len(shape)
    Jg = np.asarray(J).reshape((d,) + shape)
    inc = []
    for k in range(d):
        fwd = Jg[k]
        bwd = np.zeros_like(fwd)
        dst = [slice(None)] * d
        src = [slice(None)] * d
        dst[k] = slice(1, None)
        src[k] = slice(None, -1)
        bwd[tuple(dst)] = fwd[tuple(src)]
        inc.extend([fwd.ravel(), bwd.ravel()])
    inc = np.stack(inc, axis=0)
    top2 = -np.partition(-inc, 1, axis=0)[:2]
    return float(0.5 * top2.sum())


def optimality_gap(phi: float, ub: float) -> float:
    return 0.0 if ub <= 0 else (ub - phi) / ub
