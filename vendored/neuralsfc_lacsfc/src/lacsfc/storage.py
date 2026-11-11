"""Point layout by curve rank and per-query I/O proxies (E3).

Points are binned to grid cells and stored sorted by the rank of their cell
(contiguous runs per cell, empty cells occupy no space). For a box query we
report, at cell granularity (no false positives inside cells):

- ``grid_segs``: maximal runs of consecutive ranks inside the box (what Phi models);
- ``seeks``    : maximal runs in the sorted point array (empty cells do not break runs);
- ``pages``    : distinct fixed-size pages touched, for each page size;
- ``points``   : number of points read.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grid import validate_shape


def make_binner(x: np.ndarray, shape, mode: str = "minmax"):
    """Fit per-axis bin edges on points (n, d); return f(values (m, d)) -> cell coords (m, d)."""
    shape = validate_shape(shape)
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(shape):
        raise ValueError("x must be (n, d) with d = len(shape)")
    if mode == "minmax":
        a, b = x.min(axis=0), x.max(axis=0)
        edges = [a[k] + (b[k] - a[k]) * np.arange(1, s) / s for k, s in enumerate(shape)]
    elif mode == "quantile":
        edges = [np.quantile(x[:, k], np.linspace(0.0, 1.0, s + 1)[1:-1]) for k, s in enumerate(shape)]
    else:
        raise ValueError(f"unknown binning mode {mode!r}")

    def binner(v):
        v = np.atleast_2d(np.asarray(v, dtype=np.float64))
        return np.column_stack([np.searchsorted(e, v[:, k], side="right")
                                for k, e in enumerate(edges)]).astype(np.int64)

    return binner


def bin_points(x: np.ndarray, shape, mode: str = "minmax"):
    """Map points (n, d) to grid cells. Returns (flat cell ids, cell coords)."""
    shape = validate_shape(shape)
    coords = make_binner(x, shape, mode)(x)
    cells = np.ravel_multi_index(tuple(coords.T), shape)
    return cells.astype(np.int64), coords


def data_boxes(coords: np.ndarray, shape, m: int, rng: np.random.Generator,
               max_frac: float = 0.25, iso: bool = False):
    """Density-driven boxes: centre = cell of a random data point; side lengths
    log-uniform in [1, floor(max_frac * s_k)] (independent per axis, or one shared
    quantile per query when ``iso``). Returns inclusive (lo, hi)."""
    shape = validate_shape(shape)
    sides = np.asarray(shape, dtype=np.int64)
    d = len(shape)
    max_len = np.maximum(1, np.floor(max_frac * sides)).astype(np.float64)
    u = rng.random((m, 1) if iso else (m, d))
    length = np.clip(np.rint(max_len ** u), 1, max_len).astype(np.int64)
    c = np.asarray(coords, dtype=np.int64)[rng.integers(0, len(coords), size=m)]
    lo = np.clip(c - length // 2, 0, sides - length)
    return lo, lo + length - 1


def continuous_boxes(x: np.ndarray, m: int, rng: np.random.Generator, fmin: float, fmax: float,
                     iso: bool = False):
    """Resolution-independent boxes in data space: centre = a random data point; side on
    axis k = f * (range of axis k), f log-uniform in [fmin, fmax] (shared when ``iso``)."""
    x = np.asarray(x, dtype=np.float64)
    d = x.shape[1]
    span = x.max(axis=0) - x.min(axis=0)
    u = rng.random((m, 1) if iso else (m, d))
    f = np.exp(np.log(fmin) + u * (np.log(fmax) - np.log(fmin)))
    half = 0.5 * f * span
    c = x[rng.integers(0, len(x), size=m)]
    return c - half, c + half


def map_boxes(binner, ql, qh):
    """Smallest cell box covering each continuous box (bins are monotone per axis)."""
    return binner(ql), binner(qh)


def layout(rank: np.ndarray, cell_of_point: np.ndarray) -> np.ndarray:
    """start[r] = offset of the first point of the cell with rank r; start[N] = n."""
    rank = np.asarray(rank, dtype=np.int64)
    cnt = np.bincount(rank[np.asarray(cell_of_point, dtype=np.int64)], minlength=rank.size)
    start = np.zeros(rank.size + 1, dtype=np.int64)
    np.cumsum(cnt, out=start[1:])
    return start


@dataclass
class IOStats:
    grid_segs: np.ndarray
    seeks: np.ndarray
    pages: np.ndarray
    points: np.ndarray
    page_sizes: tuple


def io_stats(rank, cell_of_point, lo, hi, shape, page_sizes=(64,)) -> IOStats:
    shape = validate_shape(shape)
    rank = np.asarray(rank, dtype=np.int64)
    grid = rank.reshape(shape)
    start = layout(rank, cell_of_point)
    lo = np.atleast_2d(lo)
    hi = np.atleast_2d(hi)
    m = lo.shape[0]
    P = len(page_sizes)
    B = np.asarray(page_sizes, dtype=np.int64)[:, None]
    gs = np.zeros(m, dtype=np.int64)
    sk = np.zeros(m, dtype=np.int64)
    pg = np.zeros((P, m), dtype=np.int64)
    npt = np.zeros(m, dtype=np.int64)
    for i in range(m):
        r = np.sort(grid[tuple(slice(int(a), int(b) + 1) for a, b in zip(lo[i], hi[i]))].ravel())
        gs[i] = 1 + np.count_nonzero(np.diff(r) != 1)
        s, e = start[r], start[r + 1]
        keep = e > s
        s, e = s[keep], e[keep]
        if s.size == 0:
            continue
        brk = s[1:] != e[:-1]
        sk[i] = 1 + np.count_nonzero(brk)
        npt[i] = int((e - s).sum())
        rs = s[np.r_[True, brk]]
        re = e[np.r_[brk, True]]
        fp = rs[None, :] // B
        lp = (re[None, :] - 1) // B
        pg[:, i] = (lp - fp + 1).sum(axis=1) - (fp[:, 1:] == lp[:, :-1]).sum(axis=1)
    return IOStats(gs, sk, pg, npt, tuple(int(b) for b in page_sizes))
