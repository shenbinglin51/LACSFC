"""Recursive dyadic curve family F_R and its exact edge-additive DP decoder (Plan B).

F_R on a cube of side 2^K: every aligned dyadic subcube is traversed as one contiguous piece,
entering and leaving at two of its corners. A subcube of side >= 2 visits its 2^d children
along a Hamiltonian path of the hypercube Q_d (children labelled by bit masks, bit k <-> axis k);
child i leaves at corner b_i, and child i+1 is entered at the cell adjacent across their shared
facet, i.e. at corner b_i xor e_k where k is the axis separating the two children. Hilbert curves
(of any dimension) are members, and every member is an AC Hamiltonian path.

``recursive_dp`` maximizes Phi_W = sum of W over consecutive cells exactly within F_R, bottom-up
with state (subcube, entry corner, exit corner) and a chain DP over the children of each node:
O(N * |paths(Q_d)| * 4^d / (2^d - 1)) time. Supported for d in {2, 3}.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .grid import bits_to_offsets, num_cells, ravel, unravel, validate_shape

_NEG = -np.inf


@lru_cache(maxsize=None)
def hypercube_paths(d: int) -> np.ndarray:
    """All directed Hamiltonian paths of Q_d as an (n_paths, 2^d) array of vertex masks."""
    m = 1 << d
    out = []

    def dfs(path, seen):
        if len(path) == m:
            out.append(tuple(path))
            return
        for k in range(d):
            v = path[-1] ^ (1 << k)
            if not seen >> v & 1:
                path.append(v)
                dfs(path, seen | (1 << v))
                path.pop()

    for s in range(m):
        dfs([s], 1 << s)
    return np.array(out, dtype=np.int64)


def _side_bits(shape) -> int:
    shape = validate_shape(shape)
    n, d = shape[0], len(shape)
    if any(s != n for s in shape) or n & (n - 1):
        raise ValueError("requires a cube with power-of-two side")
    if d not in (2, 3):
        raise ValueError("recursive_dp supports d in {2, 3}")
    return n.bit_length() - 1


def recursive_dp(shape, W, start_origin: bool = False):
    """Phi_W-optimal curve in F_R. Returns (order, value); order lists cells in visit order.

    W has the layout of ``pair_weights``: W[k, u] is the weight of the edge (u, u + e_k).
    With ``start_origin`` the curve must start at cell 0.
    """
    shape = validate_shape(shape)
    K, d, n = _side_bits(shape), len(shape), shape[0]
    W = np.asarray(W, dtype=np.float64)
    m = 1 << d
    P = hypercube_paths(d)
    kbit = np.log2(P[:, :-1] ^ P[:, 1:]).astype(np.int64)
    off = bits_to_offsets(np.arange(m), d)

    # level 1: side-2 nodes, children are cells
    g1 = (n >> 1,) * d
    origin = 2 * unravel(np.arange(num_cells(g1)), g1)
    low = ravel(origin[:, None, None, :] + off[P[:, :-1] & P[:, 1:]][None], shape)
    pw = W[kbit[None], low].sum(axis=2)
    M = origin.shape[0]
    V = np.full((M, m, m), _NEG)
    bp1 = np.zeros((M, m, m), dtype=np.int16)
    for p in range(P.shape[0]):
        a, b = P[p, 0], P[p, -1]
        better = pw[:, p] > V[:, a, b]
        V[better, a, b] = pw[better, p]
        bp1[better, a, b] = p

    bps = []
    for lvl in range(2, K + 1):
        s, h = 1 << lvl, 1 << (lvl - 1)
        gl, gc = (n >> lvl,) * d, (n >> (lvl - 1),) * d
        X = unravel(np.arange(num_cells(gl)), gl)
        M = X.shape[0]
        child = ravel(2 * X[:, None, :] + off[None], gc)
        Vc = V[child]
        cell = (s * X)[:, None, None, :] + (h * off)[None, :, None, :] + ((h - 1) * off)[None, None]
        E = np.full((M, m, m, d), _NEG)
        for k in range(d):
            ck = off[:, k][:, None]
            valid = off[None, :, k] == 1 - ck
            lowc = cell - (ck[None, :, :, None] * np.eye(d, dtype=np.int64)[k])
            lowc = np.clip(lowc, 0, n - 1)
            w = W[k, ravel(lowc, shape)]
            E[..., k] = np.where(valid[None], w, _NEG)
        Vn = np.full((M, m, m), _NEG)
        bp_p = np.zeros((M, m, m), dtype=np.int16)
        bp_s = np.zeros((M, m, m, m), dtype=np.int8)
        rows = np.arange(M)
        for p in range(P.shape[0]):
            c = P[p]
            a, b = c[0], c[-1]
            f = Vc[:, c[0], a, :]
            args = []
            for i in range(1, m):
                k = kbit[p, i - 1]
                cand = (f[:, :, None] + E[:, c[i - 1], :, k][:, :, None]
                        + Vc[:, c[i], np.arange(m) ^ (1 << k), :])
                args.append(cand.argmax(axis=1))
                f = cand.max(axis=1)
            val = f[:, b]
            better = val > Vn[:, a, b]
            if not better.any():
                continue
            states = np.empty((M, m), dtype=np.int64)
            states[:, -1] = b
            for i in range(m - 1, 0, -1):
                states[:, i - 1] = args[i - 1][rows, states[:, i]]
            Vn[better, a, b] = val[better]
            bp_p[better, a, b] = p
            bp_s[better, a, b] = states[better]
        bps.append((bp_p, bp_s))
        V = Vn

    root = V[0].copy()
    if start_origin:
        root[1:] = _NEG
    a, b = np.unravel_index(int(np.argmax(root)), root.shape)
    value = float(root[a, b])
    if not np.isfinite(value):
        raise RuntimeError("no feasible curve")

    node, A, B = np.zeros(1, dtype=np.int64), np.array([a]), np.array([b])
    for lvl in range(K, 1, -1):
        bp_p, bp_s = bps[lvl - 2]
        gl, gc = (n >> lvl,) * d, (n >> (lvl - 1),) * d
        p = bp_p[node, A, B].astype(np.int64)
        st = bp_s[node, A, B].astype(np.int64)
        c = P[p]
        ent = np.empty_like(st)
        ent[:, 0] = A
        ent[:, 1:] = st[:, :-1] ^ (1 << kbit[p])
        X = unravel(node, gl)
        node = ravel(2 * X[:, None, :] + off[c], gc).ravel()
        A, B = ent.ravel(), st.ravel()
    p = bp1[node, A, B].astype(np.int64)
    X = unravel(node, g1)
    order = ravel(2 * X[:, None, :] + off[P[p]], shape).ravel()
    return order, value


def recursive_rank(shape, W, start_origin: bool = False) -> np.ndarray:
    order, _ = recursive_dp(shape, W, start_origin)
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size, dtype=np.int64)
    return rank
