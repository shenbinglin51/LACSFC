"""B3: per-region curve selection + MWST-style merging (prototype).

1. Tile the grid into regions of side T (T even, divides every side).
2. Per region choose, by internal Phi, one Hamiltonian cycle from a candidate family:
     comb_k : boustrophedon along axis k over x_k in [1, T-1], columns ordered by a snake
              path of the other axes, closed by the return lane x_k = 0 (runs of length T-1
              along axis k);
     l1     : the Gray-block / Delta-Phi-MWST cycle of the region under its local J.
3. Merge region cycles through ports: a port of adjacent regions R, R+e_k is a cycle edge
   (a1,a2) of R whose endpoints lie in the relative interior of the shared face, such that
   (a1+e_k, a2+e_k) is a cycle edge of R+e_k; exact gain J(a1,b1)+J(a2,b2)-J(a1,a2)-J(b1,b2).
   Ports of different region pairs are edge-disjoint, so the best port per pair plus a
   maximum spanning tree of the region graph is exact for the fixed choice (§5.1, H2).
   If the port graph is disconnected (e.g. T = 2) fall back to greedy Kruskal over all
   parallel face edges with no edge reuse (valid, not exact).
4. Cut at the origin.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .baselines import hilbert_rank
from .construct import (_replace, adjacent_values, apply_merges, cycle_phi, cycle_to_path,
                        delta_phi, initial_cycle, max_spanning_tree, merge_squares)
from .gray import gray_sequence
from .grid import (bits_to_offsets, index_dtype, num_cells, ravel, strides, unravel,
                   validate_shape)


@dataclass(frozen=True)
class RegionCurve:
    order: np.ndarray
    phi: float
    choice: np.ndarray
    names: tuple
    n_merges: int
    exact: bool = False
    ub: float = float("inf")
    phi_in: float = float("nan")
    merge_gain: float = float("nan")
    cut: float = float("nan")

    @property
    def gap(self) -> float:
        if not np.isfinite(self.ub) or self.ub <= 0:
            return float("nan")
        return (self.ub - self.phi) / self.ub


@dataclass(frozen=True)
class RegionCover:
    nbr: np.ndarray
    choice: np.ndarray
    names: tuple
    phis: np.ndarray
    cuts: np.ndarray
    seqs: tuple
    gidx: np.ndarray


def _snake_order(shape) -> np.ndarray:
    """Boustrophedon order of a box of any dimension >= 1 (flat C indices)."""
    shape = tuple(int(s) for s in shape)
    d = len(shape)
    N = num_cells(shape)
    t = np.stack(np.unravel_index(np.arange(N, dtype=np.int64), shape), axis=1)
    c = t.copy()
    prefix = np.zeros(N, dtype=np.int64)
    for k in range(1, d):
        prefix = prefix * shape[k - 1] + t[:, k - 1]
        odd = (prefix & 1) == 1
        c[odd, k] = shape[k] - 1 - t[odd, k]
    return np.ravel_multi_index(tuple(c.T), shape).astype(np.int64)


def comb_cycle(tshape, k: int) -> np.ndarray:
    """Hamiltonian cycle (sequence of local flat indices) of box ``tshape`` with long runs along k."""
    tshape = tuple(int(s) for s in tshape)
    d = len(tshape)
    L = tshape[k]
    others = [a for a in range(d) if a != k]
    oshape = tuple(tshape[a] for a in others)
    S = np.stack(np.unravel_index(_snake_order(oshape), oshape), axis=1)
    M = S.shape[0]
    j = np.arange(1, L, dtype=np.int64)
    X = np.where((np.arange(M) % 2 == 0)[:, None], j[None, :], L - j[None, :])
    seq = np.empty((M * (L - 1) + M, d), dtype=np.int64)
    body = seq[:M * (L - 1)]
    body[:, others] = np.repeat(S, L - 1, axis=0)
    body[:, k] = X.ravel()
    tail = seq[M * (L - 1):]
    tail[:, others] = S[::-1]
    tail[:, k] = 0
    return ravel(seq, tshape)


@lru_cache(maxsize=None)
def _moore_corners(d: int):
    """Entry corners a_i and Hilbert end axes m_i for the 2^d sub-cubes visited in Gray order
    g_i, such that exit b_i = a_i ^ e_{m_i} faces g_{i+1} and a_{i+1} = b_i ^ e_{j_i} (cyclic)."""
    g = gray_sequence(d).tolist()
    n = len(g)
    j = [(g[i] ^ g[(i + 1) % n]).bit_length() - 1 for i in range(n)]

    dead = set()

    def dfs(i, a, a0, acc):
        if (i, a) in dead:
            return None
        for m in range(d):
            b = a ^ (1 << m)
            if ((b >> j[i]) & 1) != ((g[(i + 1) % n] >> j[i]) & 1):
                continue
            nxt = b ^ (1 << j[i])
            if i == n - 1:
                if nxt == a0:
                    return acc + [(a, m)]
                continue
            res = dfs(i + 1, nxt, a0, acc + [(a, m)])
            if res is not None:
                return res
        dead.add((i, a))
        return None

    for a0 in range(n):
        dead.clear()
        res = dfs(0, a0, a0, [])
        if res is not None:
            return tuple(g), tuple(res)
    raise RuntimeError("no Moore corner assignment")


def moore_cycle(tshape) -> np.ndarray:
    """Closed Hilbert-type (Moore) Hamiltonian cycle of a cube with power-of-two side T >= 4:
    2^d sub-cubes in reflected-Gray order, each filled with a reflected/axis-swapped Hilbert path."""
    tshape = tuple(int(s) for s in tshape)
    d = len(tshape)
    T = tshape[0]
    if any(t != T for t in tshape) or T < 4 or T & (T - 1):
        raise ValueError("moore_cycle needs a cube with power-of-two side >= 4")
    s = T // 2
    sub = (s,) * d
    P = unravel(np.argsort(hilbert_rank(sub)), sub)
    g, corners = _moore_corners(d)
    parts = []
    for gi, (a, m) in zip(g, corners):
        Q = P.copy()
        Q[:, [0, m]] = P[:, [m, 0]]
        for k in range(d):
            if (a >> k) & 1:
                Q[:, k] = s - 1 - Q[:, k]
        parts.append(Q + bits_to_offsets(gi, d) * s)
    return ravel(np.concatenate(parts), tshape)


def _l1_cycle(tshape, Jl, sq=None, c0=None) -> np.ndarray:
    sq = merge_squares(tshape) if sq is None else sq
    c0 = initial_cycle(tshape) if c0 is None else c0
    nb = num_cells(tuple(s // 2 for s in tshape))
    if nb > 1:
        tree = max_spanning_tree(nb, sq.block_a, sq.block_b, delta_phi(Jl, sq))
    else:
        tree = np.zeros(0, dtype=bool)
    return cycle_to_path(apply_merges(c0, sq, tree), tshape)


def _region_index(shape, tile):
    """gidx[r, p]: global flat index of local cell p (C order in the tile) of region r."""
    d = len(shape)
    tshape = (tile,) * d
    rshape = tuple(s // tile for s in shape)
    st = strides(shape)
    lbase = unravel(np.arange(tile ** d, dtype=np.int64), tshape) @ st
    rbase = (unravel(np.arange(num_cells(rshape), dtype=np.int64), rshape) * tile) @ st
    return tshape, rshape, (rbase[:, None] + lbase[None, :]).astype(index_dtype(shape))


def _cells(gidx, seq):
    return gidx[:, seq] if seq.ndim == 1 else np.take_along_axis(gidx, seq, axis=1)


def _fill(nbr, cells):
    nbr[cells, 0] = np.roll(cells, 1, axis=1)
    nbr[cells, 1] = np.roll(cells, -1, axis=1)


def _edge_in(nbr, x, y):
    return (nbr[x, 0] == y) | (nbr[x, 1] == y)


def region_cover(shape, J: np.ndarray, tile: int = 8, candidates=("comb", "l1", "moore"),
                 choice=None) -> RegionCover:
    """Cover V by one candidate cycle per region (by internal Phi unless ``choice`` is given)."""
    shape = validate_shape(shape)
    d = len(shape)
    if tile % 2 or any(s % tile for s in shape):
        raise ValueError("tile must be even and divide every side")
    J = np.asarray(J)
    N = num_cells(shape)
    tshape, rshape, gidx = _region_index(shape, tile)
    R, n = gidx.shape

    seqs, names = [], []
    if "comb" in candidates:
        for k in range(d):
            seqs.append(comb_cycle(tshape, k))
            names.append(f"comb{k}")
    if "moore" in candidates and tile >= 4 and tile & (tile - 1) == 0:
        seqs.append(moore_cycle(tshape))
        names.append("moore")
    if "l1" in candidates:
        Jg = J.reshape((d,) + shape)
        rc = unravel(np.arange(R, dtype=np.int64), rshape)
        rows = np.empty((R, n), dtype=np.int64)
        sq, c0 = merge_squares(tshape), initial_cycle(tshape)
        for r in range(R):
            origin = rc[r] * tile
            sl = tuple(slice(o, o + tile) for o in origin)
            Jl = Jg[(slice(None),) + sl].copy()
            for k in range(d):
                idx = [slice(None)] * d
                idx[k] = tile - 1
                Jl[k][tuple(idx)] = 0.0
            rows[r] = _l1_cycle(tshape, Jl.reshape(d, -1), sq, c0)
        seqs.append(rows)
        names.append("l1")
    if not seqs:
        raise ValueError("no candidate cycles")

    C = len(seqs)
    phis = np.empty((R, C))
    cuts = np.empty(C)
    for c, seq in enumerate(seqs):
        cells = _cells(gidx, seq)
        nxt = np.roll(cells, -1, axis=1)
        phis[:, c] = adjacent_values(J, cells.ravel(), nxt.ravel(), shape).reshape(R, n).sum(axis=1)
        p = int(np.flatnonzero(cells[0] == 0)[0])
        cuts[c] = adjacent_values(J, [0, 0], [cells[0, p - 1], cells[0, (p + 1) % n]], shape).min()

    if choice is None:
        choice = phis.argmax(axis=1)
    choice = np.asarray(choice, dtype=np.int64)
    if choice.shape != (R,) or choice.min() < 0 or choice.max() >= C:
        raise ValueError("choice must give one candidate index per region")
    nbr = np.empty((N, 2), dtype=index_dtype(shape))
    for c, seq in enumerate(seqs):
        rows = np.flatnonzero(choice == c)
        if rows.size:
            _fill(nbr, _cells(gidx[rows], seq if seq.ndim == 1 else seq[rows]))
    return RegionCover(nbr, choice, tuple(names), phis, cuts, tuple(seqs), gidx)


def _face_slots(shape, tile, gidx):
    """All potential ports: edges in the relative interior of every shared face."""
    d = len(shape)
    st = strides(shape)
    tshape = (tile,) * d
    tst = strides(tshape)
    rshape = tuple(s // tile for s in shape)
    rst = strides(rshape)
    rc = unravel(np.arange(gidx.shape[0], dtype=np.int64), rshape)
    parts = {key: [] for key in ("u", "v", "a1", "a2", "k", "l")}
    for k in range(d):
        regs = np.flatnonzero(rc[:, k] < rshape[k] - 1)
        if regs.size == 0:
            continue
        for l in range(d):
            if l == k:
                continue
            ranges = [np.arange(1, tile - 1, dtype=np.int64)] * d
            ranges[k] = np.array([tile - 1], dtype=np.int64)
            ranges[l] = np.arange(1, tile - 2, dtype=np.int64)
            p1 = np.stack(np.meshgrid(*ranges, indexing="ij"), axis=-1).reshape(-1, d)
            if p1.shape[0] == 0:
                continue
            q1 = ravel(p1, tshape)
            m = q1.size
            parts["u"].append(np.repeat(regs, m))
            parts["v"].append(np.repeat(regs + rst[k], m))
            parts["a1"].append(gidx[np.ix_(regs, q1)].ravel())
            parts["a2"].append(gidx[np.ix_(regs, q1 + tst[l])].ravel())
            parts["k"].append(np.full(regs.size * m, k, dtype=np.int64))
            parts["l"].append(np.full(regs.size * m, l, dtype=np.int64))
    S = {key: (np.concatenate(v) if v else np.zeros(0, dtype=np.int64)) for key, v in parts.items()}
    S["b1"] = S["a1"] + st[S["k"]]
    S["b2"] = S["a2"] + st[S["k"]]
    return S


def _with_gain(S, J, mask):
    P = {key: v[mask] for key, v in S.items()}
    P["gain"] = (J[P["k"], P["a1"]] + J[P["k"], P["a2"]]
                 - J[P["l"], P["a1"]] - J[P["l"], P["b1"]])
    return P


def _port_table(nbr, J, shape, tile):
    """Ports present in the cover ``nbr`` with their exact gains."""
    shape = validate_shape(shape)
    J = np.asarray(J)
    S = _face_slots(shape, tile, _region_index(shape, tile)[2])
    mask = _edge_in(nbr, S["a1"], S["a2"]) & _edge_in(nbr, S["b1"], S["b2"])
    return _with_gain(S, J, mask)


def _best_ports(P, R):
    """Best port per region pair, then MWST of the region graph; ValueError if disconnected."""
    if P["gain"].size == 0:
        raise ValueError("no ports")
    key = P["u"] * R + P["v"]
    o = np.lexsort((-P["gain"], key))
    ks = key[o]
    best = o[np.r_[True, ks[1:] != ks[:-1]]]
    tree = max_spanning_tree(R, P["u"][best], P["v"][best], P["gain"][best])
    return best[tree]


def _upper_bound(cov, J, shape, tile):
    """H3: sum_R max_c Phi_in + MWST(w*) - min_c cut, w* = best port over all candidate pairs."""
    S = _face_slots(shape, tile, cov.gidx)
    up = np.zeros(S["a1"].size, dtype=bool)
    low = np.zeros_like(up)
    nbr = np.empty_like(cov.nbr)
    for seq in cov.seqs:
        _fill(nbr, _cells(cov.gidx, seq))
        up |= _edge_in(nbr, S["a1"], S["a2"])
        low |= _edge_in(nbr, S["b1"], S["b2"])
    P = _with_gain(S, J, up & low)
    sel = _best_ports(P, cov.gidx.shape[0])
    return float(cov.phis.max(axis=1).sum() + P["gain"][sel].sum() - cov.cuts.min())


def region_curve(shape, J: np.ndarray, tile: int = 8,
                 candidates=("comb", "l1", "moore"), choice=None) -> RegionCurve:
    shape = validate_shape(shape)
    J = np.asarray(J)
    cov = region_cover(shape, J, tile, candidates, choice)
    R = cov.gidx.shape[0]
    nbr = cov.nbr
    phi_in = float(cov.phis[np.arange(R), cov.choice].sum())
    exact, ub, merge_gain = False, float("inf"), float("nan")
    if R == 1:
        exact, merge_gain = True, 0.0
        ub = float(cov.phis.max(axis=1).sum() - cov.cuts.min())
    else:
        P = _port_table(nbr, J, shape, tile)
        try:
            sel = _best_ports(P, R)
        except ValueError:
            sel = None
        if sel is not None:
            a1, a2, b1, b2 = (P[key][sel] for key in ("a1", "a2", "b1", "b2"))
            _replace(nbr, a1, a2, b1)
            _replace(nbr, a2, a1, b2)
            _replace(nbr, b1, b2, a1)
            _replace(nbr, b2, b1, a2)
            exact, merge_gain = True, float(P["gain"][sel].sum())
            ub = _upper_bound(cov, J, shape, tile)
        else:
            _merge_regions(nbr, J, shape, tile, R)
    cyc = cycle_phi(nbr, J, shape)
    order = cycle_to_path(nbr, shape, J)
    phi = float(adjacent_values(J, order[:-1], order[1:], shape).sum())
    return RegionCurve(order, phi, cov.choice, cov.names, R - 1, exact, ub, phi_in,
                       merge_gain, cyc - phi)


def region_curve_auto(shape, J: np.ndarray, tiles=None, min_tile: int = 8,
                      candidates=("comb", "l1", "moore")) -> tuple[RegionCurve, int]:
    """Data-driven tile size: best training Phi over even tiles dividing every side."""
    shape = validate_shape(shape)
    if tiles is None:
        tiles = [t for t in range(2, min(shape) + 1, 2)
                 if t >= min(min_tile, min(shape)) and all(s % t == 0 for s in shape)]
    best = None
    for t in tiles:
        rc = region_curve(shape, J, tile=t, candidates=candidates)
        if best is None or rc.phi > best[0].phi + 1e-12:
            best = (rc, t)
    return best


def _merge_regions(nbr, J, shape, tile, R) -> int:
    d = len(shape)
    N = nbr.shape[0]
    st = strides(shape)
    coords = unravel(np.arange(N, dtype=np.int64), shape)
    rshape = tuple(s // tile for s in shape)
    region = ravel(coords // tile, rshape)
    x = np.repeat(np.arange(N, dtype=np.int64), 2)
    y = nbr.reshape(-1)
    keep = x < y
    a1, a2 = x[keep], y[keep]
    A1, A2, B1, B2, G = [], [], [], [], []
    for k in range(d):
        on = ((coords[a1, k] % tile) == tile - 1) & ((coords[a2, k] % tile) == tile - 1) \
             & (coords[a1, k] < shape[k] - 1)
        e1, e2 = a1[on], a2[on]
        b1, b2 = e1 + st[k], e2 + st[k]
        par = (nbr[b1, 0] == b2) | (nbr[b1, 1] == b2)
        e1, e2, b1, b2 = e1[par], e2[par], b1[par], b2[par]
        g = (adjacent_values(J, e1, b1, shape) + adjacent_values(J, e2, b2, shape)
             - adjacent_values(J, e1, e2, shape) - adjacent_values(J, b1, b2, shape))
        A1.append(e1); A2.append(e2); B1.append(b1); B2.append(b2); G.append(g)
    a1, a2, b1, b2, g = (np.concatenate(v) for v in (A1, A2, B1, B2, G))
    order = np.lexsort((np.arange(g.size), -g))
    parent = list(range(R))

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    used = set()
    taken = 0
    ra, rb = region[a1].tolist(), region[b1].tolist()
    for t in order.tolist():
        u, v = find(ra[t]), find(rb[t])
        if u == v:
            continue
        ea = (int(a1[t]), int(a2[t]))
        eb = (int(b1[t]), int(b2[t]))
        if ea in used or eb in used:
            continue
        used.add(ea)
        used.add(eb)
        parent[u] = v
        _swap(nbr, ea[0], ea[1], eb[0])
        _swap(nbr, ea[1], ea[0], eb[1])
        _swap(nbr, eb[0], eb[1], ea[0])
        _swap(nbr, eb[1], eb[0], ea[1])
        taken += 1
        if taken == R - 1:
            break
    if taken != R - 1:
        raise RuntimeError("region merge graph could not be connected")
    return taken


def _swap(nbr, x, old, new):
    s = 0 if nbr[x, 0] == old else 1
    if nbr[x, s] != old:
        raise AssertionError("merge edge missing")
    nbr[x, s] = new
