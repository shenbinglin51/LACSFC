"""N-d Cover-and-Merge with exact Delta-Phi weights (layer L1).

Pipeline: initial block cycles C0 -> merge squares on the block dual graph ->
Delta-Phi weights -> maximum-weight spanning tree -> merges -> cut at origin.

A cycle is stored as ``nbr`` of shape (N, 2): the two cycle neighbours of each
cell (flat C order). Merge squares have pairwise disjoint cycle edges, so all
selected merges are applied simultaneously.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

try:
    from numba import njit
except ImportError:
    njit = None

from .gray import edge_endpoints, edge_flip_dim, facet_assignment, gray_sequence
from .grid import (bits_to_offsets, block_shape, index_dtype, num_cells, ravel, strides,
                   unravel, validate_shape)


@dataclass(frozen=True)
class MergeSquares:
    """One entry per dual edge (block_a, block_b = block_a + e_dim).

    Local edge (a1, a2) of block A and (b1, b2) of block B satisfy
    b1 = a1 + e_dim and b2 = a2 + e_dim; a1, a2 differ along ``flip``.
    """

    block_a: np.ndarray
    block_b: np.ndarray
    dim: np.ndarray
    flip: np.ndarray
    a1: np.ndarray
    a2: np.ndarray
    b1: np.ndarray
    b2: np.ndarray

    @property
    def size(self) -> int:
        return int(self.dim.size)


@dataclass(frozen=True)
class Curve:
    shape: tuple
    order: np.ndarray
    rank: np.ndarray
    tree: np.ndarray
    squares: MergeSquares
    delta_phi: np.ndarray
    phi: float


_CHUNK = 1 << 22


def _mask_offsets(shape) -> np.ndarray:
    """off[mask] = flat offset of local corner ``mask`` (bit k <-> axis k) inside a 2^d block."""
    d = len(shape)
    return bits_to_offsets(np.arange(1 << d, dtype=np.int64), d) @ strides(shape)


def _block_grid(ranges, shape):
    """Blocks at the C-ordered product of ``ranges`` (block coordinates): flat block index,
    flat cell index of the block origin 2*A, and parity mask; built from sparse grids."""
    st, bst = strides(shape), strides(block_shape(shape))
    g = np.meshgrid(*ranges, indexing="ij", sparse=True)
    full = np.broadcast_shapes(*(x.shape for x in g))
    blk = np.zeros(full, dtype=np.int64)
    base = np.zeros(full, dtype=np.int64)
    par = np.zeros(full, dtype=np.int32)
    for j, x in enumerate(g):
        blk += x * bst[j]
        base += x * (2 * st[j])
        par |= ((x & 1) << j).astype(np.int32)
    return blk.ravel(), base.ravel(), par.ravel()


def initial_cycle(shape) -> np.ndarray:
    """C0: one mirrored reflected-Gray cycle per 2^d block."""
    shape = validate_shape(shape)
    d = len(shape)
    idx = index_dtype(shape)
    _, base, r = _block_grid([np.arange(n, dtype=np.int64) for n in block_shape(shape)], shape)
    local = gray_sequence(d).astype(np.int32)[None, :] ^ r[:, None]
    del r
    cells = _mask_offsets(shape).astype(idx)[local]
    del local
    cells += base.astype(idx)[:, None]
    del base
    nbr = np.empty((num_cells(shape), 2), dtype=cells.dtype)
    nbr[cells, 0] = np.roll(cells, 1, axis=1)
    nbr[cells, 1] = np.roll(cells, -1, axis=1)
    return nbr


def merge_squares(shape) -> MergeSquares:
    shape = validate_shape(shape)
    d = len(shape)
    nb = block_shape(shape)
    sig = facet_assignment(d)
    sig_flip = edge_flip_dim(d, sig)
    st, bst = strides(shape), strides(nb)
    off = _mask_offsets(shape)
    idx = index_dtype(shape)
    bidx = np.int32 if num_cells(nb) <= np.iinfo(np.int32).max else np.int64
    parts = {key: [] for key in ("block_a", "block_b", "dim", "flip", "a1", "a2", "b1", "b2")}
    for k in range(d):
        if nb[k] < 2:
            continue
        ranges = [np.arange(n, dtype=np.int64) for n in nb]
        ranges[k] = np.arange(nb[k] - 1, dtype=np.int64)
        blk, base, rA = _block_grid(ranges, shape)
        even = ((rA >> k) & 1) == 0
        e1, e2 = edge_endpoints(d, sig[k, 1])
        o1, o2 = edge_endpoints(d, sig[k, 0])
        y1 = np.where(even, np.int32(e1), np.int32(o1))
        y2 = np.where(even, np.int32(e2), np.int32(o2))
        rB = rA ^ np.int32(1 << k)
        shift = 2 * int(st[k])
        parts["block_a"].append(blk.astype(bidx))
        parts["block_b"].append((blk + bst[k]).astype(bidx))
        parts["dim"].append(np.full(blk.size, k, dtype=np.int8))
        parts["flip"].append(np.where(even, sig_flip[k, 1], sig_flip[k, 0]).astype(np.int8))
        parts["a1"].append((base + off[y1 ^ rA]).astype(idx))
        parts["a2"].append((base + off[y2 ^ rA]).astype(idx))
        parts["b1"].append((base + (off[y1 ^ rB] + shift)).astype(idx))
        parts["b2"].append((base + (off[y2 ^ rB] + shift)).astype(idx))
        del blk, base, rA, rB, even, y1, y2
    cat = {}
    for key, v in parts.items():
        dt = np.int8 if key in ("dim", "flip") else (bidx if key.startswith("block") else idx)
        cat[key] = np.concatenate(v) if v else np.zeros(0, dtype=dt)
        v.clear()
    return MergeSquares(**cat)


def delta_phi(J: np.ndarray, sq: MergeSquares) -> np.ndarray:
    """Exact Phi gain of each merge: J(a1,b1) + J(a2,b2) - J(a1,a2) - J(b1,b2)."""
    return (J[sq.dim, sq.a1] + J[sq.dim, sq.a2]
            - J[sq.flip, np.minimum(sq.a1, sq.a2)]
            - J[sq.flip, np.minimum(sq.b1, sq.b2)])


def _kruskal_nb(n_nodes, u, v, order):
    parent = np.arange(n_nodes)
    mask = np.zeros(u.size, dtype=np.bool_)
    taken = 0
    for e in order:
        ru = u[e]
        while parent[ru] != ru:
            parent[ru] = parent[parent[ru]]
            ru = parent[ru]
        rv = v[e]
        while parent[rv] != rv:
            parent[rv] = parent[parent[rv]]
            rv = parent[rv]
        if ru != rv:
            parent[ru] = rv
            mask[e] = True
            taken += 1
            if taken == n_nodes - 1:
                break
    return mask, taken


def _walk_nb(nbr, start, nxt):
    N = nbr.shape[0]
    order = np.empty(N, dtype=nbr.dtype)
    order[0] = start
    prev, cur = start, nxt
    for i in range(1, N):
        order[i] = cur
        step = nbr[cur, 0] if nbr[cur, 0] != prev else nbr[cur, 1]
        prev, cur = cur, step
    return order, cur


if njit is not None:
    _kruskal_nb = njit(_kruskal_nb)
    _walk_nb = njit(_walk_nb)


def max_spanning_tree(n_nodes: int, u: np.ndarray, v: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Kruskal; ties broken by edge index (deterministic). Returns a boolean edge mask."""
    order = np.lexsort((np.arange(len(w)), -np.asarray(w)))
    if njit is not None:
        mask, taken = _kruskal_nb(int(n_nodes), np.asarray(u, dtype=np.int64),
                                  np.asarray(v, dtype=np.int64), order.astype(np.int64))
        if taken != n_nodes - 1:
            raise ValueError("dual graph is not connected")
        return mask
    parent = list(range(n_nodes))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    mask = np.zeros(len(w), dtype=bool)
    uu, vv = u.tolist(), v.tolist()
    taken = 0
    for e in order.tolist():
        ru, rv = find(uu[e]), find(vv[e])
        if ru != rv:
            parent[ru] = rv
            mask[e] = True
            taken += 1
            if taken == n_nodes - 1:
                break
    if taken != n_nodes - 1:
        raise ValueError("dual graph is not connected")
    return mask


def _replace(nbr: np.ndarray, x: np.ndarray, old: np.ndarray, new: np.ndarray) -> None:
    slot = (nbr[x, 1] == old).astype(np.int64)
    if not np.all(nbr[x, slot] == old):
        raise AssertionError("merge edge missing from current cycle")
    nbr[x, slot] = new


def apply_merges(nbr: np.ndarray, sq: MergeSquares, tree: np.ndarray,
                 copy: bool = True) -> np.ndarray:
    """Rewire ``nbr`` along the selected merge squares; copies unless ``copy=False``."""
    if copy:
        nbr = nbr.copy()
    a1, a2, b1, b2 = sq.a1[tree], sq.a2[tree], sq.b1[tree], sq.b2[tree]
    _replace(nbr, a1, a2, b1)
    _replace(nbr, a2, a1, b2)
    _replace(nbr, b1, b2, a1)
    _replace(nbr, b2, b1, a2)
    return nbr


def _adjacent_axis(x, y, shape):
    """(axis, lower endpoint) of grid-adjacent pairs; axis = -1 where not adjacent."""
    x = np.asarray(x, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    lo = np.minimum(x, y)
    delta = np.abs(y - x)
    k = np.full(x.shape, -1, dtype=np.int8)
    for a, (s, st) in enumerate(zip(shape, strides(shape))):
        k[(k < 0) & (delta == st) & ((lo // st) % s < s - 1)] = a
    return k, lo


def adjacent_values(J: np.ndarray, x, y, shape) -> np.ndarray:
    """J for grid-adjacent pairs (x, y); raises if some pair is not adjacent."""
    J = np.asarray(J)
    x = np.atleast_1d(np.asarray(x))
    y = np.atleast_1d(np.asarray(y))
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    out = np.empty(x.shape, dtype=J.dtype)
    xf, yf, of = x.reshape(-1), y.reshape(-1), out.reshape(-1)
    for i in range(0, xf.size, _CHUNK):
        k, lo = _adjacent_axis(xf[i:i + _CHUNK], yf[i:i + _CHUNK], shape)
        if np.any(k < 0):
            raise ValueError("pair is not grid-adjacent")
        of[i:i + _CHUNK] = J[k, lo]
    return out


def cycle_phi(nbr: np.ndarray, J: np.ndarray, shape) -> float:
    x = np.repeat(np.arange(nbr.shape[0], dtype=np.int64), 2)
    y = nbr.reshape(-1)
    keep = x < y
    return float(adjacent_values(J, x[keep], y[keep], shape).sum(dtype=np.float64))


def cycle_to_path(nbr: np.ndarray, shape, J: np.ndarray | None = None, start: int = 0) -> np.ndarray:
    """Cut the cycle at ``start`` (dropping the incident edge with smaller J) and walk it."""
    n0, n1 = int(nbr[start, 0]), int(nbr[start, 1])
    if J is not None:
        w0, w1 = adjacent_values(J, [start, start], [n0, n1], shape)
        cut = n0 if (w0 < w1 or (w0 == w1 and n0 < n1)) else n1
    else:
        cut = min(n0, n1)
    nxt = n1 if cut == n0 else n0
    N = nbr.shape[0]
    if njit is not None:
        order, cur = _walk_nb(np.ascontiguousarray(nbr), start, nxt)
        if cur != start:
            raise AssertionError("cover is not a single Hamiltonian cycle")
        return order
    c0, c1 = nbr[:, 0].tolist(), nbr[:, 1].tolist()
    order = [start]
    prev, cur = start, nxt
    for _ in range(N - 1):
        order.append(cur)
        step = c0[cur] if c0[cur] != prev else c1[cur]
        prev, cur = cur, step
    if cur != start:
        raise AssertionError("cover is not a single Hamiltonian cycle")
    return np.asarray(order, dtype=np.int64)


def build_curve(shape, J: np.ndarray, tree: np.ndarray | None = None) -> Curve:
    """Build the AC path. If ``tree`` is None, use the Delta-Phi MWST (Theorem T2)."""
    shape = validate_shape(shape)
    J = np.asarray(J)
    if J.shape != (len(shape), num_cells(shape)):
        raise ValueError("J must have shape (d, N)")
    sq = merge_squares(shape)
    dphi = delta_phi(J, sq)
    n_blocks = num_cells(block_shape(shape))
    if tree is None:
        tree = (max_spanning_tree(n_blocks, sq.block_a, sq.block_b, dphi)
                if n_blocks > 1 else np.zeros(0, dtype=bool))
    tree = np.asarray(tree, dtype=bool)
    nbr = apply_merges(initial_cycle(shape), sq, tree, copy=False)
    order = cycle_to_path(nbr, shape, J)
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size, dtype=rank.dtype)
    phi = float(adjacent_values(J, order[:-1], order[1:], shape).sum(dtype=np.float64))
    return Curve(shape, order, rank, tree, sq, dphi, phi)


def build_curve_permuted(shape, J: np.ndarray, perm) -> np.ndarray:
    """build_curve after relabelling axes (new axis i = old axis perm[i]); returns the order
    in original flat indices. Phi, AC and origin start are invariant under relabelling."""
    shape = validate_shape(shape)
    d = len(shape)
    perm = [int(k) for k in perm]
    pshape = tuple(shape[k] for k in perm)
    Jg = np.asarray(J).reshape((d,) + shape)
    Jp = Jg[perm].transpose([0] + [k + 1 for k in perm]).reshape(d, -1)
    po = build_curve(pshape, Jp).order
    pc = unravel(po, pshape)
    c = np.empty_like(pc)
    c[:, perm] = pc
    return ravel(c, shape)


def build_curve_auto(shape, J: np.ndarray, max_perms: int = 24) -> tuple[np.ndarray, tuple, float]:
    """Data-driven axis-to-Gray-bit assignment: best Phi over all d! relabellings when
    d! <= max_perms, else axes sorted by total J (heaviest axis -> Gray bit 0)."""
    shape = validate_shape(shape)
    d = len(shape)
    J = np.asarray(J)
    if math.factorial(d) <= max_perms:
        cands = list(itertools.permutations(range(d)))
    else:
        cands = [tuple(int(k) for k in np.argsort(-J.sum(axis=1), kind="stable"))]
    best = None
    for perm in cands:
        order = build_curve_permuted(shape, J, perm)
        phi = float(adjacent_values(J, order[:-1], order[1:], shape).sum(dtype=np.float64))
        if best is None or phi > best[2] + 1e-12:
            best = (order, tuple(perm), phi)
    return best


def is_ac_path(order: np.ndarray, shape, start: int | None = 0) -> bool:
    """Bijection onto all cells, unit L1 steps, and (optionally) fixed start."""
    order = np.asarray(order)
    if not np.issubdtype(order.dtype, np.integer):
        order = order.astype(np.int64)
    N = num_cells(shape)
    if order.shape != (N,):
        return False
    if order.min() < 0 or order.max() >= N:
        return False
    seen = np.zeros(N, dtype=bool)
    seen[order] = True
    if not seen.all():
        return False
    del seen
    if start is not None and order[0] != start:
        return False
    for i in range(0, N - 1, _CHUNK):
        a = order[i:i + _CHUNK + 1]
        k, _ = _adjacent_axis(a[:-1], a[1:], shape)
        if np.any(k < 0):
            return False
    return True
