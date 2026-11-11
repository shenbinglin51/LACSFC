"""Split-merge local search (L3; requires numba): AC-preserving compound moves.

2-opt must reject a unit square whose two path edges appear as (a,b) = (p_i, p_{i+1}) and
(d,c) = (p_j, p_{j+1}) with a~c, b~d: swapping them splits the path into
    P = p[0..i] + (a,c) + p[j+1..N-1]   and the cycle   C = p[i+1..j] + (d,b).
In 2D every square of an AC path cut from an AC cycle is of this kind (planarity), so 2-opt
has no moves there.  A split-merge move re-merges C into P with a second square:
    internal : cycle edge (u,v), P edge (x,y), u~x, v~y   -> remove both, add (u,x), (v,y)
    end      : path end e ~ u in C                       -> remove cycle edge (u,v), add (e,u)
The result is again a Hamiltonian AC path with the same start p[0], and
    gain = [J(a,c) + J(b,d) - J(a,b) - J(d,c)] + [second-square gain]   (exact).
Locality: cycle length j - i <= max_loop and the P edge lies within ``reach`` positions of the
loop, so each move rewrites only a local slice of the order.
Windowed mode: moves are confined to position windows [s, e] (end merge only when e = N-1), so
each window keeps its cell set and both end cells; windows run in parallel and alternate passes
shift them by half a window, as in ``fastls.windowed_ils``.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from .construct import adjacent_values
from .fastls import _NEG
from .grid import index_dtype, strides, validate_shape
from .local import LocalResult


def _boundary_mask(shape) -> np.ndarray:
    """bm[u] bit t set iff u has a neighbour in direction t (axis t // 2; t even: -1, odd: +1)."""
    d = len(shape)
    dt = np.uint16 if 2 * d <= 16 else (np.uint32 if 2 * d <= 32 else np.int64)
    bm = np.zeros(shape, dtype=dt)
    for k, s in enumerate(shape):
        c = np.arange(s).reshape((1,) * k + (-1,) + (1,) * (d - k - 1))
        bm |= ((c > 0).astype(dt) << dt(2 * k)) | ((c < s - 1).astype(dt) << dt(2 * k + 1))
    return bm.reshape(-1)


@njit(cache=True, inline="always")
def _nbr(u, t, bm, st):
    """Neighbour of u in direction t; -1 if absent."""
    if (bm[u] >> t) & 1:
        return u + st[t >> 1] if t & 1 else u - st[t >> 1]
    return -1


@njit(cache=True, inline="always")
def _jn(J, u, t, st):
    """J of the edge (u, _nbr(u, t)); the neighbour must exist."""
    k = t >> 1
    return np.float64(J[k, u]) if t & 1 else np.float64(J[k, u - st[k]])


@njit(cache=True)
def _w(J, bm, st, x, y):
    lo = min(x, y)
    delta = abs(y - x)
    for k in range(st.size):
        if delta == st[k]:
            if (bm[lo] >> (2 * k + 1)) & 1:
                return np.float64(J[k, lo])
            return _NEG
    return _NEG


@njit(cache=True)
def _best_merge(p, pos, J, bm, st, i, j, a, c, reach, s, e):
    N = p.size
    D2 = 2 * st.size
    L = j - i
    lo = max(i - reach, s)
    hi = min(j + 1 + reach, e)
    best = _NEG
    kind = 0
    bu = bv = bx = by = -1
    for k in range(L):
        u = p[i + 1 + k]
        v = p[i + 2 + k] if k < L - 1 else p[i + 1]
        wuv = _w(J, bm, st, u, v)
        for t in range(D2):
            x = _nbr(u, t, bm, st)
            y = _nbr(v, t, bm, st)
            if x < 0 or y < 0:
                continue
            px = pos[x]
            py = pos[y]
            if (px > i and px <= j) or (py > i and py <= j):
                continue
            if (x == a and y == c) or (x == c and y == a):
                wxy = _w(J, bm, st, a, c)
            elif abs(px - py) == 1 and lo <= px <= hi and lo <= py <= hi:
                wxy = _w(J, bm, st, x, y)
            else:
                continue
            g = _jn(J, u, t, st) + _jn(J, v, t, st) - wuv - wxy
            if g > best:
                best, kind, bu, bv, bx, by = g, 1, u, v, x, y
    if e == N - 1 and N - 1 <= j + 1 + reach:
        z = p[N - 1]
        for t in range(D2):
            u = _nbr(z, t, bm, st)
            if u < 0:
                continue
            q = pos[u]
            if q <= i or q > j:
                continue
            k = q - (i + 1)
            for sg in (-1, 1):
                v = p[i + 1 + (k + sg) % L]
                g = _jn(J, z, t, st) - _w(J, bm, st, u, v)
                if g > best:
                    best, kind, bu, bv, bx, by = g, 2, u, v, z, -1
    return best, kind, bu, bv, bx, by


@njit(cache=True)
def _emit_cycle(p, pos, i, j, s, end, buf, n):
    """Write cycle p[i+1..j] starting at s, ending at its cycle-neighbour ``end``."""
    L = j - i
    ks = pos[s] - (i + 1)
    kt = pos[end] - (i + 1)
    step = L - 1 if (ks + 1) % L == kt else 1
    k = ks
    for _ in range(L):
        buf[n] = p[i + 1 + k]
        n += 1
        k = (k + step) % L
    return n


@njit(cache=True)
def _apply(p, pos, i, j, kind, u, v, x, y, buf):
    N = p.size
    if kind == 1:
        lo = min(i, pos[x], pos[y])
        hi = max(j + 1, pos[x], pos[y])
    else:
        lo = i
        hi = N - 1
    n = 0
    m = (i - lo + 1) + (hi - j)
    for r in range(m):
        q = lo + r if r <= i - lo else j + 1 + (r - (i - lo + 1))
        z = p[q]
        buf[n] = z
        n += 1
        if kind == 1 and r + 1 < m:
            r2 = r + 1
            q2 = lo + r2 if r2 <= i - lo else j + 1 + (r2 - (i - lo + 1))
            z2 = p[q2]
            if z == x and z2 == y:
                n = _emit_cycle(p, pos, i, j, u, v, buf, n)
            elif z == y and z2 == x:
                n = _emit_cycle(p, pos, i, j, v, u, buf, n)
    if kind == 2:
        n = _emit_cycle(p, pos, i, j, u, v, buf, n)
    for r in range(n):
        p[lo + r] = buf[r]
        pos[buf[r]] = lo + r
    return n == hi - lo + 1


@njit(cache=True)
def _sweep(p, pos, J, bm, st, s, e, max_loop, reach, tol, buf):
    D2 = 2 * st.size
    total = 0.0
    moves = 0
    i = s
    while i < e:
        a = p[i]
        b = p[i + 1]
        wab = _w(J, bm, st, a, b)
        moved = False
        for t in range(D2):
            c = _nbr(a, t, bm, st)
            if c < 0:
                continue
            j = pos[c] - 1
            if j < i + 3 or j - i > max_loop or j + 1 > e:
                continue
            d = p[j]
            wbd = _w(J, bm, st, b, d)
            if wbd == _NEG:
                continue
            g1 = _jn(J, a, t, st) + wbd - wab - _w(J, bm, st, d, c)
            g2, kind, u, v, x, y = _best_merge(p, pos, J, bm, st, i, j, a, c, reach, s, e)
            if kind > 0 and g1 + g2 > tol:
                if not _apply(p, pos, i, j, kind, u, v, x, y, buf):
                    raise RuntimeError("split-merge rebuild length mismatch")
                total += g1 + g2
                moves += 1
                moved = True
                break
        if not moved:
            i += 1
    return total, moves


@njit(parallel=True, cache=True)
def _windows(p, pos, J, bm, st, starts, ends, max_loop, reach, tol):
    moves = np.zeros(starts.size, dtype=np.int64)
    for w in prange(starts.size):
        s = starts[w]
        e = ends[w]
        buf = np.empty(e - s + 1, dtype=p.dtype)
        _g, moves[w] = _sweep(p, pos, J, bm, st, s, e, max_loop, reach, tol, buf)
    return moves.sum()


def split_merge_search(order: np.ndarray, J: np.ndarray, shape, max_loop: int = 64,
                       reach: int = 256, max_sweeps: int = 50, tol: float = 1e-12,
                       window: int | None = None) -> LocalResult:
    """First-improvement split-merge local search; keeps p[0] and the AC property.

    ``window=None``: sequential sweeps over the whole path.  Otherwise parallel position
    windows of length ``window`` (>= 2 * (max_loop + reach) recommended), offsets alternating
    by window // 2, until two consecutive passes make no move or ``max_sweeps`` passes.
    """
    shape = validate_shape(shape)
    J = np.asarray(J)
    if J.dtype not in (np.float32, np.float64):
        J = J.astype(np.float64)
    J = np.ascontiguousarray(J)
    bm = _boundary_mask(shape)
    st = strides(shape).astype(np.int64)
    idx = index_dtype(shape)
    p = np.asarray(order).astype(idx, copy=True)
    N = p.size
    pos = np.empty(N, dtype=idx)
    pos[p] = np.arange(N, dtype=idx)
    moves = 0
    if window is None:
        buf = np.empty(N, dtype=idx)
        for _ in range(max_sweeps):
            _g, mv = _sweep(p, pos, J, bm, st, 0, N - 1, max_loop, reach, tol, buf)
            moves += mv
            if mv == 0:
                break
    else:
        idle = 0
        for r in range(max_sweeps):
            off = 0 if r % 2 == 0 else window // 2
            cuts = np.unique(np.concatenate([[0], np.arange(off, N - 1, window), [N - 1]]))
            mv = int(_windows(p, pos, J, bm, st, cuts[:-1].astype(np.int64),
                              cuts[1:].astype(np.int64), max_loop, reach, tol))
            moves += mv
            idle = idle + 1 if mv == 0 else 0
            if idle >= 2:
                break
    phi = float(adjacent_values(J, p[:-1], p[1:], shape).sum(dtype=np.float64))
    return LocalResult(p, phi, moves)
