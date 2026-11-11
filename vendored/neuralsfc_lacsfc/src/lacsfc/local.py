"""AC-preserving local search on Hamiltonian paths (layer L3).

Moves (each keeps a single Hamiltonian path with unit L1 steps):
  2-opt   : p_i~p_j and p_{i+1}~p_{j+1} adjacent (i+1 < j < N-1): reverse p_{i+1..j}.
            gain = J(p_i,p_j) + J(p_{i+1},p_{j+1}) - J(p_i,p_{i+1}) - J(p_j,p_{j+1})
  end rot : p_{N-1}~p_i (i < N-2): reverse p_{i+1..N-1}.  gain = J(p_i,p_{N-1}) - J(p_i,p_{i+1})
  start rot (fix_start=False only): p_0~p_j (j > 1): reverse p_{0..j-1}.
            gain = J(p_0,p_j) - J(p_{j-1},p_j)
Best-improvement until no move gains more than ``tol``. Phi strictly increases, so it terminates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .construct import adjacent_values
from .grid import num_cells, strides, unravel, validate_shape


@dataclass(frozen=True)
class LocalResult:
    order: np.ndarray
    phi: float
    moves: int


def neighbor_table(shape) -> np.ndarray:
    """(N, 2d) grid neighbours, -1 where absent."""
    shape = validate_shape(shape)
    N = num_cells(shape)
    c = unravel(np.arange(N, dtype=np.int64), shape)
    st = strides(shape)
    cols = []
    for k in range(len(shape)):
        cols.append(np.where(c[:, k] > 0, np.arange(N) - st[k], -1))
        cols.append(np.where(c[:, k] < shape[k] - 1, np.arange(N) + st[k], -1))
    return np.stack(cols, axis=1).astype(np.int64)


def _pairJ(J, x, y, shape):
    return adjacent_values(J, x, y, shape) if x.size else np.zeros(0)


def candidate_moves(p: np.ndarray, J: np.ndarray, shape, coords: np.ndarray, nb: np.ndarray,
                    fix_start: bool = True):
    """All legal reversals p[lo..hi] of the current path and their exact Phi gains."""
    N = p.size
    rank = np.empty(N, dtype=np.int64)
    rank[p] = np.arange(N)
    ew = adjacent_values(J, p[:-1], p[1:], shape)
    i = np.repeat(np.arange(N - 1), nb.shape[1])
    c = nb[p[:-1]].reshape(-1)
    ok = c >= 0
    i, c = i[ok], c[ok]
    j = rank[c]
    ok = j > i + 1
    i, c, j = i[ok], c[ok], j[ok]
    los, his, gains = [], [], []

    end = j == N - 1
    ie = i[end]
    los.append(ie + 1)
    his.append(np.full(ie.size, N - 1, dtype=np.int64))
    gains.append(_pairJ(J, p[ie], c[end], shape) - ew[ie])

    mid = ~end
    im, jm, cm = i[mid], j[mid], c[mid]
    adj = np.abs(coords[p[im + 1]] - coords[p[jm + 1]]).sum(axis=1) == 1
    im, jm, cm = im[adj], jm[adj], cm[adj]
    los.append(im + 1)
    his.append(jm)
    gains.append(_pairJ(J, p[im], cm, shape) + _pairJ(J, p[im + 1], p[jm + 1], shape)
                 - ew[im] - ew[jm])

    if not fix_start:
        cs = nb[p[0]]
        cs = cs[cs >= 0]
        js = rank[cs]
        cs, js = cs[js > 1], js[js > 1]
        los.append(np.zeros(js.size, dtype=np.int64))
        his.append(js - 1)
        gains.append(_pairJ(J, np.full(cs.size, p[0]), cs, shape) - ew[js - 1])

    return np.concatenate(los), np.concatenate(his), np.concatenate(gains)


def _reverse(p: np.ndarray, lo: int, hi: int) -> None:
    p[lo:hi + 1] = p[lo:hi + 1][::-1].copy()


def local_search(order: np.ndarray, J: np.ndarray, shape, fix_start: bool = True,
                 max_moves: int = 1_000_000, tol: float = 1e-12) -> LocalResult:
    shape = validate_shape(shape)
    J = np.asarray(J, dtype=np.float64)
    p = np.asarray(order, dtype=np.int64).copy()
    coords = unravel(np.arange(p.size, dtype=np.int64), shape)
    nb = neighbor_table(shape)
    moves = 0
    while moves < max_moves:
        lo, hi, g = candidate_moves(p, J, shape, coords, nb, fix_start)
        if g.size == 0:
            break
        t = int(np.argmax(g))
        if g[t] <= tol:
            break
        _reverse(p, int(lo[t]), int(hi[t]))
        moves += 1
    phi = float(adjacent_values(J, p[:-1], p[1:], shape).sum())
    return LocalResult(p, phi, moves)


def iterated_local_search(order: np.ndarray, J: np.ndarray, shape, fix_start: bool = True,
                          iters: int = 200, kick: int = 8, seed: int = 0) -> LocalResult:
    """ILS: kick = ``kick`` random legal reversals, then local_search; accept if Phi does not drop."""
    shape = validate_shape(shape)
    J = np.asarray(J, dtype=np.float64)
    rng = np.random.default_rng(seed)
    coords = unravel(np.arange(len(order), dtype=np.int64), shape)
    nb = neighbor_table(shape)
    cur = local_search(order, J, shape, fix_start)
    best = cur
    total = cur.moves
    for _ in range(iters):
        p = cur.order.copy()
        for _ in range(kick):
            lo, hi, _g = candidate_moves(p, J, shape, coords, nb, fix_start)
            if lo.size == 0:
                break
            t = int(rng.integers(lo.size))
            _reverse(p, int(lo[t]), int(hi[t]))
        cand = local_search(p, J, shape, fix_start)
        total += kick + cand.moves
        if cand.phi >= cur.phi - 1e-12:
            cur = cand
            if cur.phi > best.phi + 1e-12:
                best = cur
    return LocalResult(best.order, best.phi, total)
