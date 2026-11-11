"""Segment recovery for a box query: boundary enumeration (BRQ, AC curves) vs full scan (any order)."""

from __future__ import annotations

import numpy as np


def scan_segments(grid_rank: np.ndarray, lo, hi):
    """Segments of box [lo, hi] for any order: sort the codes of every cell of Q and split at gaps."""
    r = np.sort(grid_rank[tuple(slice(int(a), int(b) + 1) for a, b in zip(lo, hi))].ravel())
    brk = np.flatnonzero(np.diff(r) != 1)
    return r[np.r_[0, brk + 1]], r[np.r_[brk, r.size - 1]]


def brq_segments(grid_rank: np.ndarray, order_coords: np.ndarray, lo, hi):
    """Segments of box [lo, hi] for an AC order from the boundary cells only (Proposition 1).

    grid_rank: code of every cell, shaped like the grid; order_coords[z] = coordinates of code z.
    Candidates are the cells on the 2d faces of Q plus the curve end if it lies in Q; a candidate
    with code z starts a segment iff z = 0 or cell z-1 is outside Q, and ends one iff z = N-1 or
    cell z+1 is outside Q.
    """
    lo, hi = np.asarray(lo, dtype=np.int64), np.asarray(hi, dtype=np.int64)
    d, N = len(lo), order_coords.shape[0]
    box = [slice(int(a), int(b) + 1) for a, b in zip(lo, hi)]
    parts = []
    for k in range(d):
        for v in {int(lo[k]), int(hi[k])}:
            sl = list(box)
            sl[k] = v
            parts.append(grid_rank[tuple(sl)].ravel())
    end = order_coords[N - 1]
    if np.all((end >= lo) & (end <= hi)):
        parts.append(np.array([N - 1]))
    z = np.unique(np.concatenate(parts))
    zp, zn = np.maximum(z - 1, 0), np.minimum(z + 1, N - 1)
    cp, cn = order_coords[zp], order_coords[zn]
    prev_out = (z == 0) | ~np.all((cp >= lo) & (cp <= hi), axis=1)
    next_out = (z == N - 1) | ~np.all((cn >= lo) & (cn <= hi), axis=1)
    return z[prev_out], z[next_out]


class BRQIndex:
    """Boundary-driven range query (BRQ) as specified in docs/images_paper.pptx, slides 1-2.

    Index: the inflow direction of every cell, stored as 2d bitmaps (inflow along axis k from
    the lower side / from the upper side). Each bitmap is split into the 2^K hyperplanes
    {x_k = c}, laid out row-major over the remaining axes, and supports rank1/select1 (here
    emulated by the sorted positions of the one-bits of each hyperplane: rank = searchsorted,
    select = indexing). Total size 2dN bits.

    Preconditions (PPT): the order is adjacency-continuous and both curve termini lie on the
    grid boundary. The start cell gets a virtual inflow from outside the grid through the
    grid face it lies on; grid-boundary faces of a query carry no entry/exit and are never
    read, so that bit is never used.

    Query: for every face of Q not on the grid boundary, entries are the cells of the inner
    slice whose inflow comes from outside, exits are the inner neighbours of the cells of the
    adjacent outer slice whose inflow comes from inside. Code 0 / N-1 are added if their cell
    is in Q. All codes go into one queue, which is sorted and read pairwise.
    """

    def __init__(self, grid_rank: np.ndarray):
        g = np.asarray(grid_rank)
        self.shape = tuple(int(s) for s in g.shape)
        self.d = d = len(self.shape)
        self.N = N = g.size
        if d < 2:
            raise ValueError("BRQIndex needs d >= 2")
        self.grid = g.astype(np.int64)
        if not np.array_equal(np.sort(self.grid.ravel()), np.arange(N)):
            raise ValueError("grid_rank must be a bijection onto 0..N-1")
        order = np.empty(N, dtype=np.int64)
        order[self.grid.ravel()] = np.arange(N)
        oc = np.stack(np.unravel_index(order, self.shape), 1)
        self.start, self.end = oc[0], oc[N - 1]
        hi_c = np.asarray(self.shape) - 1
        for p, name in ((self.start, "start"), (self.end, "end")):
            if not np.any((p == 0) | (p == hi_c)):
                raise ValueError(f"curve {name} {tuple(p)} is not on the grid boundary")
        delta = oc[:-1] - oc[1:]                       # predecessor - landing cell
        if not np.all(np.abs(delta).sum(axis=1) == 1):
            raise ValueError("order is not adjacency-continuous")
        axis = np.argmax(delta != 0, axis=1)
        upper = delta[np.arange(N - 1), axis] > 0       # predecessor at +e_axis
        inflow = np.empty(N, dtype=np.int64)
        inflow[order[1:]] = 2 * axis + upper
        k0 = int(np.flatnonzero((self.start == 0) | (self.start == hi_c))[0])
        inflow[order[0]] = 2 * k0 + int(self.start[k0] != 0)
        inflow = inflow.reshape(self.shape)
        self.rest = [[j for j in range(d) if j != k] for k in range(d)]
        self.ptr, self.pos = {}, {}
        for k in range(d):
            for s in (0, 1):
                bm = np.moveaxis(inflow == 2 * k + s, k, 0).reshape(self.shape[k], -1)
                sl, loc = np.nonzero(bm)
                self.pos[k, s] = loc.astype(np.int64)
                self.ptr[k, s] = np.searchsorted(sl, np.arange(self.shape[k] + 1))

    def _select_face(self, k, s, c, lo, hi):
        """Local positions of the one-bits of bitmap (k, s) in slice c inside the box's projection."""
        seg = self.pos[k, s][self.ptr[k, s][c]:self.ptr[k, s][c + 1]]
        rest = self.rest[k]
        rshape = [self.shape[j] for j in rest]
        heads = [np.arange(lo[j], hi[j] + 1) for j in rest[:-1]]
        base = np.zeros(1, dtype=np.int64)
        for j, h in enumerate(heads):
            base = (base[:, None] * rshape[j] + h[None]).ravel()
        last = rest[-1]
        a = base * rshape[-1] + lo[last]
        b = base * rshape[-1] + hi[last]
        r0 = np.searchsorted(seg, a, side="left")        # rank1 before the range
        r1 = np.searchsorted(seg, b, side="right")       # rank1 through the range
        if not np.any(r1 > r0):
            return np.zeros((0, self.d), dtype=np.int64)
        idx = np.concatenate([np.arange(u, v) for u, v in zip(r0, r1) if v > u])
        loc = seg[idx]                                   # select1
        out = np.empty((loc.size, self.d), dtype=np.int64)
        out[:, rest] = np.stack(np.unravel_index(loc, rshape), 1)
        out[:, k] = c
        return out

    def segments(self, lo, hi):
        """Maximal contiguous segments of the inclusive box [lo, hi]: (starts, ends)."""
        lo = np.asarray(lo, dtype=np.int64)
        hi = np.asarray(hi, dtype=np.int64)
        queue = []
        if np.all((self.start >= lo) & (self.start <= hi)):
            queue.append(np.array([0]))
        if np.all((self.end >= lo) & (self.end <= hi)):
            queue.append(np.array([self.N - 1]))
        for k in range(self.d):
            if lo[k] > 0:
                ent = self._select_face(k, 0, lo[k], lo, hi)
                ext = self._select_face(k, 1, lo[k] - 1, lo, hi)
                ext[:, k] += 1
                queue += [self.grid[tuple(ent.T)], self.grid[tuple(ext.T)]]
            if hi[k] < self.shape[k] - 1:
                ent = self._select_face(k, 1, hi[k], lo, hi)
                ext = self._select_face(k, 0, hi[k] + 1, lo, hi)
                ext[:, k] -= 1
                queue += [self.grid[tuple(ent.T)], self.grid[tuple(ext.T)]]
        q = np.sort(np.concatenate(queue)) if queue else np.zeros(0, dtype=np.int64)
        return q[0::2], q[1::2]
