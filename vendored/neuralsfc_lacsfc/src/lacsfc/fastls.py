"""Scalable windowed ILS (L3 at scale; requires numba, optional extra ``fast``).

The path is cut into position windows [s, e] of length W. Inside a window only 2-opt
reversals with both endpoints p[s], p[e] fixed are used (plus end rotation when
e = N-1), so every window keeps its cell set and the whole order stays a Hamiltonian
AC path starting at p[0]. Windows are disjoint apart from shared fixed endpoints and run
in parallel; alternate passes shift windows by W/2 so cells migrate across boundaries.
Per window: first-improvement local search, ILS with ``kick`` random legal moves,
accept if Phi does not drop, otherwise restore.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from numba import njit, prange

from .construct import adjacent_values
from .grid import num_cells, strides, validate_shape
from .local import neighbor_table

_NEG = -1e300
_TOL = 1e-12


@dataclass(frozen=True)
class WindowedResult:
    order: np.ndarray
    phi: float
    passes: int
    seconds: float
    history: tuple


def neighbor_weights(J: np.ndarray, shape, nb: np.ndarray) -> np.ndarray:
    """JN[x, t] = J of the grid edge (x, nb[x, t]); 0 where the neighbour is absent."""
    shape = validate_shape(shape)
    N = num_cells(shape)
    st = strides(shape)
    x = np.arange(N, dtype=np.int64)
    JN = np.zeros(nb.shape, dtype=np.float64)
    for k in range(len(shape)):
        lo, hi = nb[:, 2 * k] >= 0, nb[:, 2 * k + 1] >= 0
        JN[lo, 2 * k] = J[k, x[lo] - st[k]]
        JN[hi, 2 * k + 1] = J[k, x[hi]]
    return JN


@njit(cache=True)
def _w(nb, JN, x, y):
    for t in range(nb.shape[1]):
        if nb[x, t] == y:
            return JN[x, t]
    return _NEG


@njit(cache=True)
def _reverse(p, pos, a, b):
    while a < b:
        u, v = p[a], p[b]
        p[a], p[b] = v, u
        pos[v], pos[u] = a, b
        a += 1
        b -= 1


@njit(cache=True)
def _rand(state):
    x = state[0]
    x ^= x << np.uint64(13)
    x ^= x >> np.uint64(7)
    x ^= x << np.uint64(17)
    state[0] = x
    return x


@njit(cache=True)
def _move_gain(p, pos, nb, JN, s, e, i, t, allow_end):
    """Gain and reversal end j of move (i, dir t); j = -1 if illegal."""
    x = p[i]
    c = nb[x, t]
    if c < 0:
        return 0.0, -1
    j = pos[c]
    if j <= i + 1 or j > e:
        return 0.0, -1
    w1 = _w(nb, JN, x, p[i + 1])
    if j == e:
        if allow_end:
            return JN[x, t] - w1, j
        return 0.0, -1
    y1 = p[j + 1]
    wy = _w(nb, JN, p[i + 1], y1)
    if wy == _NEG:
        return 0.0, -1
    return JN[x, t] + wy - w1 - _w(nb, JN, c, y1), j


@njit(cache=True)
def _local(p, pos, nb, JN, s, e, allow_end, max_sweeps):
    total = 0.0
    D2 = nb.shape[1]
    for _ in range(max_sweeps):
        improved = False
        for i in range(s, e):
            for t in range(D2):
                g, j = _move_gain(p, pos, nb, JN, s, e, i, t, allow_end)
                if j >= 0 and g > _TOL:
                    _reverse(p, pos, i + 1, j)
                    total += g
                    improved = True
                    break
        if not improved:
            break
    return total


@njit(cache=True)
def _kick(p, pos, nb, JN, s, e, allow_end, kick, state):
    total = 0.0
    done = 0
    D2 = nb.shape[1]
    span = e - s
    for _ in range(50 * kick):
        if done >= kick or span < 2:
            break
        i = s + np.int64(_rand(state) % np.uint64(span))
        t = np.int64(_rand(state) % np.uint64(D2))
        g, j = _move_gain(p, pos, nb, JN, s, e, i, t, allow_end)
        if j >= 0:
            _reverse(p, pos, i + 1, j)
            total += g
            done += 1
    return total


@njit(parallel=True, cache=True)
def _windows(p, pos, nb, JN, starts, ends, iters, kick, seed, max_sweeps):
    N = p.size
    gains = np.zeros(starts.size)
    for w in prange(starts.size):
        s = starts[w]
        e = ends[w]
        if e - s < 3:
            continue
        allow_end = e == N - 1
        state = np.empty(1, dtype=np.uint64)
        state[0] = np.uint64(seed) * np.uint64(1000003) + np.uint64(w) + np.uint64(1)
        acc = _local(p, pos, nb, JN, s, e, allow_end, max_sweeps)
        backup = p[s:e + 1].copy()
        for _ in range(iters):
            delta = _kick(p, pos, nb, JN, s, e, allow_end, kick, state)
            delta += _local(p, pos, nb, JN, s, e, allow_end, max_sweeps)
            if delta >= -_TOL:
                acc += delta
                backup[:] = p[s:e + 1]
            else:
                for k in range(backup.size):
                    p[s + k] = backup[k]
                    pos[backup[k]] = s + k
        gains[w] = acc
    return gains.sum()


def windowed_ils(order: np.ndarray, J: np.ndarray, shape, window: int = 256, passes: int = 4,
                 iters: int = 20, kick: int = 4, seed: int = 0, max_sweeps: int = 50,
                 time_budget: float | None = None) -> WindowedResult:
    shape = validate_shape(shape)
    J = np.asarray(J, dtype=np.float64)
    p = np.asarray(order, dtype=np.int64).copy()
    N = p.size
    pos = np.empty(N, dtype=np.int64)
    pos[p] = np.arange(N)
    nb = neighbor_table(shape)
    JN = neighbor_weights(J, shape, nb)
    phi = float(adjacent_values(J, p[:-1], p[1:], shape).sum())
    hist = [phi]
    t0 = time.perf_counter()
    done = 0
    for r in range(passes):
        if time_budget is not None and time.perf_counter() - t0 >= time_budget:
            break
        off = 0 if r % 2 == 0 else window // 2
        cuts = np.unique(np.concatenate([[0], np.arange(off, N - 1, window), [N - 1]])).astype(np.int64)
        phi += _windows(p, pos, nb, JN, cuts[:-1], cuts[1:], iters, kick, seed * 7919 + r, max_sweeps)
        hist.append(phi)
        done += 1
    sec = time.perf_counter() - t0
    phi_exact = float(adjacent_values(J, p[:-1], p[1:], shape).sum())
    return WindowedResult(p, phi_exact, done, sec, tuple(hist))
