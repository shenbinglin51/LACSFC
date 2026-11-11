"""Baseline orders. Each function returns ``rank`` (rank[cell] = position)."""

from __future__ import annotations

import numpy as np

from .grid import all_coords, num_cells, validate_shape


def _cube_bits(shape) -> int:
    shape = validate_shape(shape)
    s = shape[0]
    if any(t != s for t in shape) or s & (s - 1):
        raise ValueError("requires a cube with power-of-two side")
    return s.bit_length() - 1


def _interleave(X: list[np.ndarray], bits: int) -> np.ndarray:
    h = np.zeros_like(X[0])
    for level in range(bits - 1, -1, -1):
        for x in X:
            h = (h << 1) | ((x >> level) & 1)
    return h


def snake_rank(shape) -> np.ndarray:
    """Multi-dimensional boustrophedon (AC). Dim k is reflected when counter digit k-1 is odd."""
    shape = validate_shape(shape)
    d = len(shape)
    t = np.stack(np.unravel_index(np.arange(num_cells(shape), dtype=np.int64), shape), axis=1)
    c = t.copy()
    for k in range(1, d):
        odd = (t[:, k - 1] & 1) == 1
        c[odd, k] = shape[k] - 1 - t[odd, k]
    order = np.ravel_multi_index(tuple(c.T), shape).astype(np.int64)
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size, dtype=np.int64)
    return rank


def zorder_rank(shape) -> np.ndarray:
    """Morton order (not AC); dimension 0 is the most significant."""
    bits = _cube_bits(shape)
    coords = all_coords(shape)
    return _interleave([coords[:, k] for k in range(coords.shape[1])], bits)


def hilbert_rank(shape) -> np.ndarray:
    """N-d Hilbert order via Skilling's AxestoTranspose (AC)."""
    bits = _cube_bits(shape)
    coords = all_coords(shape)
    d = coords.shape[1]
    X = [coords[:, k].copy() for k in range(d)]
    M = 1 << (bits - 1)
    Q = M
    while Q > 1:
        P = Q - 1
        for i in range(d):
            hit = (X[i] & Q) != 0
            if i == 0:
                X[0] = np.where(hit, X[0] ^ P, X[0])
                continue
            t = (X[0] ^ X[i]) & P
            X[0], X[i] = np.where(hit, X[0] ^ P, X[0] ^ t), np.where(hit, X[i], X[i] ^ t)
        Q >>= 1
    for i in range(1, d):
        X[i] = X[i] ^ X[i - 1]
    t = np.zeros_like(X[0])
    Q = M
    while Q > 1:
        t = np.where((X[d - 1] & Q) != 0, t ^ (Q - 1), t)
        Q >>= 1
    X = [x ^ t for x in X]
    return _interleave(X, bits)
