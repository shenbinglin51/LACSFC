"""Fast, equivalence-preserving reimplementation of the LACSFC construction.

Goal (option A): produce the SAME code_map as ``weights_to_sfc`` under real
(non-degenerate) weights, but vectorized so large grids (up to ~2^30 cells)
are reachable within the 24GB / time budget measured in P0.

This module is built and verified incrementally against the reference
implementation via ``scripts/verify_fast_construct.py``. Each helper documents
whether it is structural (output-order-invariant -> guaranteed exact) or
ordering-sensitive (must reproduce reference behavior).

Stage map (mirrors profile_construct_stages.py):
    s1  dual edges            -> vectorized (structural)
    s2  dual graph + MST      -> scipy MST (verify tree + edge order)
    s3  circuits              -> vectorized BRGC (structural)
    s4  merge                 -> reference for now (sequential, hardest)
    s5  path extraction       -> flat-id traversal
    s6  code_map              -> vectorized fill
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

LACSFCF_ROOT = os.environ.get(
    "LACSFCF_ROOT",
    str(Path(__file__).resolve().parents[1] / "vendored" / "lacsfcf_repo"))
if LACSFCF_ROOT not in sys.path:
    sys.path.insert(0, LACSFCF_ROOT)


# --------------------------------------------------------------------------- #
# Lightweight grid: same geometry as lacsfcf.data.grid.Grid but WITHOUT the    #
# O(num_blocks) Python loop in _build_dual_edges (which dominated s1 at scale). #
# The fast path never needs grid.dual_edges (scipy_mst_edges uses vectorized    #
# dual_edges_arrays), so this is a drop-in for fast_code_map.                   #
# --------------------------------------------------------------------------- #
class LightGrid:
    __slots__ = ("d", "K", "grid_side", "block_side", "num_cells",
                 "num_blocks", "grid_shape", "block_shape", "num_dual_edges")

    def __init__(self, d: int, K: int):
        self.d = d
        self.K = K
        self.grid_side = 2 ** K
        self.block_side = 2 ** (K - 1)
        self.num_cells = self.grid_side ** d
        self.num_blocks = self.block_side ** d
        self.grid_shape = (self.grid_side,) * d
        self.block_shape = (self.block_side,) * d
        B = self.block_side
        # num dual edges = sum over dims of (#blocks with coord[k] < B-1)
        #               = d * B^(d-1) * (B-1)
        self.num_dual_edges = d * (B ** (d - 1)) * (B - 1)


# --------------------------------------------------------------------------- #
# s1: vectorized dual edges                                                    #
# --------------------------------------------------------------------------- #
def dual_edges_arrays(d: int, block_side: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (b1_flat, b2_flat, dim) arrays for all adjacent block pairs.

    Matches the *order* of Grid._build_dual_edges exactly: outer loop over
    blocks in C-order (np.ndindex), inner loop over dimensions 0..d-1, edge
    added when block[k] < B-1 to neighbor block+e_k.

    Flat indices use C-order ravel over block_shape (= block_side per dim).
    """
    B = block_side
    block_shape = (B,) * d
    num_blocks = B ** d
    # C-order block coordinates for every block id 0..num_blocks-1
    coords = np.array(np.unravel_index(np.arange(num_blocks), block_shape)).T  # (nb, d)
    strides = np.ones(d, dtype=np.int64)
    for k in range(d - 2, -1, -1):
        strides[k] = strides[k + 1] * B

    b1_list, b2_list, dim_list = [], [], []
    for k in range(d):
        # edge exists when coord[k] < B-1; neighbor is +1 in dim k
        mask = coords[:, k] < (B - 1)
        ids = np.where(mask)[0]
        b1_list.append(ids)
        b2_list.append(ids + strides[k])
        dim_list.append(np.full(ids.shape[0], k, dtype=np.int8))

    # Reference order is block-major then dim-minor. To reproduce exactly we
    # must interleave by block first. Build (block_id, dim) sort key.
    b1 = np.concatenate(b1_list)
    b2 = np.concatenate(b2_list)
    dim = np.concatenate(dim_list)
    order = np.lexsort((dim, b1))  # primary b1 (block C-order), secondary dim
    return b1[order], b2[order], dim[order]


# --------------------------------------------------------------------------- #
# s2: scipy maximum spanning tree (drop-in exact in d=2, quality-equiv d>=3)   #
# --------------------------------------------------------------------------- #
def scipy_mst_edges(d: int, block_side: int, weights: np.ndarray):
    """Return tree edges as a list of (b1_tuple, b2_tuple) block-coordinate pairs.

    weights are indexed in the SAME order as Grid.dual_edges (block C-order,
    dim-minor) -- matched by dual_edges_arrays above. A maximum spanning tree is
    computed by negating weights (shift - w) and running scipy's minimum tree.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import minimum_spanning_tree

    b1f, b2f, _dim = dual_edges_arrays(d, block_side)
    nb = block_side ** d
    w = np.asarray(weights, dtype=np.float64)
    if w.shape[0] != b1f.shape[0]:
        raise ValueError(
            f"weights length {w.shape[0]} != num_dual_edges {b1f.shape[0]}")
    shift = float(w.max()) + 1.0 if w.size else 1.0
    transformed = shift - w  # all > 0 so scipy treats them as real edges
    mat = csr_matrix((transformed, (b1f, b2f)), shape=(nb, nb))
    tree = minimum_spanning_tree(mat).tocoo()

    block_shape = (block_side,) * d
    r = np.asarray(tree.row, dtype=np.int64)
    c = np.asarray(tree.col, dtype=np.int64)
    rc = np.unravel_index(r, block_shape)  # tuple of d arrays
    cc = np.unravel_index(c, block_shape)
    edges = []
    m = r.shape[0]
    for i in range(m):
        b1 = tuple(int(rc[k][i]) for k in range(d))
        b2 = tuple(int(cc[k][i]) for k in range(d))
        edges.append((b1, b2))
    return edges


def _grid_strides(d: int, grid_side: int):
    """C-order strides for grid_shape=(grid_side,)*d, matching _Circuits.strides."""
    strides = [1] * d
    for k in range(d - 2, -1, -1):
        strides[k] = strides[k + 1] * grid_side
    return strides


def scipy_mst_edge_arrays(d: int, block_side: int, weights: np.ndarray):
    """Vectorized MST edge data for the fast merge -- NO per-edge Python tuples.

    Returns ``(base_arr, k_arr, lowcoord)`` where, for each tree edge:
      - ``k_arr[i]``   : the single block-dimension in which the two blocks differ
      - ``lowcoord[i]``: block coordinates of the LOW block (b1, b1[k] < b2[k])
      - ``base_arr[i]``: flat CELL id of b1's high-face anchor cell
                         = sum_j 2*b1[j]*cstride[j]  +  cstride[k]
                         (i.e. coord[k]=2*b1[k]+1, coord[j!=k]=2*b1[j])

    These are exactly the quantities apply_mst_fast needs; everything is built
    with numpy so memory stays ~O(m) int arrays instead of ~O(m) Python tuples.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import minimum_spanning_tree

    b1f, b2f, _dim = dual_edges_arrays(d, block_side)
    nb = block_side ** d
    w = np.asarray(weights, dtype=np.float64)
    if w.shape[0] != b1f.shape[0]:
        raise ValueError(
            f"weights length {w.shape[0]} != num_dual_edges {b1f.shape[0]}")
    shift = float(w.max()) + 1.0 if w.size else 1.0
    transformed = shift - w
    mat = csr_matrix((transformed, (b1f, b2f)), shape=(nb, nb))
    tree = minimum_spanning_tree(mat).tocoo()

    block_shape = (block_side,) * d
    r = np.asarray(tree.row, dtype=np.int64)
    c = np.asarray(tree.col, dtype=np.int64)
    rc = np.stack(np.unravel_index(r, block_shape), axis=1)  # (m, d)
    cc = np.stack(np.unravel_index(c, block_shape), axis=1)  # (m, d)

    # Only one dim differs (and by exactly 1); low block = elementwise minimum.
    lowcoord = np.minimum(rc, cc).astype(np.int32)            # (m, d)
    k_arr = np.argmax(np.abs(rc - cc), axis=1).astype(np.int8)  # (m,)

    grid_side = 2 * block_side
    cstride = np.asarray(_grid_strides(d, grid_side), dtype=np.int64)  # (d,)
    base_arr = (2 * lowcoord.astype(np.int64) * cstride[None, :]).sum(axis=1)
    base_arr += cstride[k_arr.astype(np.int64)]
    return base_arr, k_arr, lowcoord


def _edge_arrays_to_tuples(k_arr, lowcoord):
    """Reconstruct (b1_tuple, b2_tuple) edges from array form (fallback only)."""
    m, d = lowcoord.shape
    edges = []
    for i in range(m):
        b1 = tuple(int(x) for x in lowcoord[i])
        k = int(k_arr[i])
        b2 = list(b1); b2[k] += 1; b2 = tuple(b2)
        edges.append((b1, b2))
    return edges


class _EdgeListGraph:
    """Minimal stand-in for an nx.Graph that apply_mst_nd needs in default mode.

    apply_mst_nd with edge_order='default' only calls ``.edges()`` and iterates
    (b1, b2) pairs; it never touches node/edge attributes. This avoids building a
    full networkx graph over 2^{d(k-1)} block nodes (slow + memory heavy at scale).
    """
    __slots__ = ("_edges",)

    def __init__(self, edges):
        self._edges = edges

    def edges(self, data=False):
        if data:
            return [(b1, b2, {}) for (b1, b2) in self._edges]
        return self._edges


# --------------------------------------------------------------------------- #
# s3: vectorized reflected-BRGC circuits build                                 #
# --------------------------------------------------------------------------- #
def build_circuits_fast(grid_shape: Tuple[int, ...]):
    """Vectorized equivalent of cover_merge.build_circuits_nd.

    Produces byte-identical nxt/prv arrays to the reference (verified in the
    harness) by reproducing the exact add_edge fill convention:
        nxt[id(b,0)]   = id(b,1)     prv[id(b,0)]   = id(b,M-1)
        nxt[id(b,i)]   = id(b,i-1)   prv[id(b,i)]   = id(b,i+1)   (1<=i<=M-2)
        nxt[id(b,M-1)] = id(b,M-2)   prv[id(b,M-1)] = id(b,0)
    where id(b,i) is the global flat id of the i-th cell of block b in
    reflected-BRGC order.
    """
    from lacsfcf.core.brgc import gray_code_sequence
    from lacsfcf.core.cover_merge import _Circuits

    d = len(grid_shape)
    block_shape = tuple(s // 2 for s in grid_shape)
    n = 1
    for s in grid_shape:
        n *= s
    M = 1 << d  # cells per block

    circ = _Circuits(n, grid_shape)
    strides = np.asarray(circ.strides, dtype=np.int64)

    nb = 1
    for s in block_shape:
        nb *= s
    bcoords = np.array(np.unravel_index(np.arange(nb), block_shape)).T  # (nb, d) int
    parity = (bcoords & 1).astype(np.int64)
    base2b = (2 * bcoords).astype(np.int64)

    gc = np.array(gray_code_sequence(d), dtype=np.int64)  # (M, d)

    # ids_all[bi, i] = flat id of i-th reflected-BRGC cell of block bi
    ids_all = np.empty((nb, M), dtype=np.int64)
    for i in range(M):
        refl = gc[i][None, :] ^ parity          # (nb, d)
        coord = base2b + refl                   # (nb, d)
        ids_all[:, i] = coord @ strides

    nxt = circ.nxt
    prv = circ.prv
    col0 = ids_all[:, 0]
    colM1 = ids_all[:, M - 1]
    nxt[col0] = ids_all[:, 1]
    prv[col0] = colM1
    for i in range(1, M - 1):
        col = ids_all[:, i]
        nxt[col] = ids_all[:, i - 1]
        prv[col] = ids_all[:, i + 1]
    nxt[colM1] = ids_all[:, M - 2]
    prv[colM1] = col0
    return circ


# --------------------------------------------------------------------------- #
# s5+s6: flat-id cycle traversal that fills code_map directly                  #
# --------------------------------------------------------------------------- #
def cycle_to_code_map(circuits, grid_shape, start_id: int) -> np.ndarray:
    """Traverse the single Hamiltonian cycle and write code_map in one pass.

    Replaces _extract_path_with_cut_optimization (fixed_start) + id2cell loop +
    code_map_from_path. Cuts the edge start->neighbors[0] and walks the rest,
    assigning sequential codes by flat cell id (no per-cell tuple conversion).
    """
    nxt = np.ascontiguousarray(circuits.nxt)
    prv = np.ascontiguousarray(circuits.prv)
    n = circuits.n
    a0 = int(nxt[start_id])
    b0 = int(prv[start_id])
    if a0 == -1 or b0 == -1:
        raise RuntimeError("circuits must be a Hamiltonian cycle (degree-2 graph)")

    code = np.empty(n, dtype=np.int32)

    # --- compiled C fast path (identical traversal; Python fallback below) ---
    import merge_core
    lib = merge_core.load()
    if lib is not None:
        import ctypes
        i32 = ctypes.POINTER(ctypes.c_int32)
        t = lib.cycle_code(
            nxt.ctypes.data_as(i32), prv.ctypes.data_as(i32),
            ctypes.c_long(n), ctypes.c_long(start_id),
            code.ctypes.data_as(i32),
        )
        if t != n:
            raise RuntimeError(
                f"cycle traversal visited {t} of {n} cells -- merge did not "
                f"produce a single Hamiltonian cycle")
        return code.reshape(grid_shape)

    code[start_id] = 0
    prev_id = start_id
    cur = b0  # cut edge start->a0 (neighbors[0]); traverse toward neighbors[1]
    t = 1
    while cur != start_id:
        code[cur] = t
        a = int(nxt[cur])
        b = int(prv[cur])
        nxt_id = a if a != prev_id else b
        prev_id, cur = cur, nxt_id
        t += 1
    if t != n:
        raise RuntimeError(
            f"cycle traversal visited {t} of {n} cells -- merge did not produce "
            f"a single Hamiltonian cycle")
    return code.reshape(grid_shape)


# --------------------------------------------------------------------------- #
# s4: fast matched-portal merge                                                #
# --------------------------------------------------------------------------- #
def _face_offsets_sorted(d: int, strides) -> list:
    """offsets_sorted[k] = sorted flat-id offsets of the 2^(d-1) face cells of a
    block's high face in dim k (relative to the block-base + 1*stride[k] cell),
    summed over the OTHER dimensions' {0,1}*2*stride[j] choices.

    Reproduces sorted(set(_face_cell_ids_np(...))) modulo the constant base, so
    base + offsets_sorted[k] == reference sorted(face_ids) exactly.
    """
    out = []
    for k in range(d):
        others = [j for j in range(d) if j != k]
        m = 1 << (d - 1)
        offs = []
        for bits in range(m):
            o = 0
            for j_idx, j in enumerate(others):
                if (bits >> j_idx) & 1:
                    o += int(strides[j])
            offs.append(o)
        out.append(sorted(offs))
    return out


def apply_mst_fast(circuits, base_arr, k_arr, grid_shape):
    """Matched-portal-only merge that mirrors apply_mst_nd's matched_portal path
    operation-for-operation (same edge order, same first-found portal, same
    add/remove slot semantics) so the resulting nxt/prv is byte-identical.

    Edges are supplied as numpy arrays from ``scipy_mst_edge_arrays``:
      - ``base_arr[i]`` : flat cell id of b1's high-face anchor
      - ``k_arr[i]``    : the split block-dimension

    Returns the merged _Circuits, or None if any edge has NO matched portal
    (signals the caller to fall back to the full reference merge). For grids
    built from a valid MST over reflected-BRGC circuits this never happens
    (matched_portal covers 100% of edges -- verified empirically).
    """
    circuits = circuits.copy()
    d = len(grid_shape)
    n = circuits.n
    strides = [int(s) for s in circuits.strides]
    nxt = circuits.nxt
    prv = circuits.prv

    # --- vectorized DSU init: every cell -> its block-base cell id ----------
    # grid dims are 2^K so each stride is a power of two; clearing the low bit
    # of each coordinate == clearing bit (log2 stride[k]) of the flat id.
    clear_mask = 0
    for s in strides:
        clear_mask |= (s & -s)  # = 1 << bit position of this (power-of-two) stride
    parent = (np.arange(n, dtype=np.int64) & ~clear_mask).astype(np.int64)

    face_off = _face_offsets_sorted(d, strides)

    # --- compiled C fast path (byte-identical; falls back to Python below) ---
    import merge_core
    lib = merge_core.load()
    if lib is not None:
        import ctypes
        fc = 1 << (d - 1)
        face_off_arr = np.ascontiguousarray(
            np.array(face_off, dtype=np.int64))            # (d, fc)
        strides_arr = np.ascontiguousarray(
            np.array(strides, dtype=np.int64))             # (d,)
        ba = np.ascontiguousarray(base_arr.astype(np.int64, copy=False))
        ka = np.ascontiguousarray(k_arr.astype(np.int8, copy=False))
        nxt = np.ascontiguousarray(nxt)
        prv = np.ascontiguousarray(prv)
        circuits.nxt = nxt
        circuits.prv = prv
        i32 = ctypes.POINTER(ctypes.c_int32)
        i64 = ctypes.POINTER(ctypes.c_int64)
        i8 = ctypes.POINTER(ctypes.c_int8)
        rc = lib.merge_run(
            nxt.ctypes.data_as(i32), prv.ctypes.data_as(i32),
            parent.ctypes.data_as(i64), ctypes.c_long(n),
            ba.ctypes.data_as(i64), ka.ctypes.data_as(i8),
            ctypes.c_long(ba.shape[0]),
            strides_arr.ctypes.data_as(i64),
            face_off_arr.ctypes.data_as(i64),
            ctypes.c_long(fc), ctypes.c_long(d),
        )
        if rc != 0:
            return None  # edge (rc-1) had no matched portal -> reference merge
        return circuits

    # --- pure-Python fallback (identical logic) ------------------------------
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    base_list = base_arr.tolist()
    k_list = k_arr.tolist()

    for ei in range(len(base_list)):
        k = k_list[ei]
        base = base_list[ei]
        sdim = strides[k]
        face_ids = [base + o for o in face_off[k]]
        face_set = set(face_ids)

        portal = None
        # Pass 1: v1 restricted to the face; Pass 2: v1 anywhere.
        for require_face in (True, False):
            for u1 in face_ids:
                u2 = u1 + sdim
                if u2 >= n:
                    continue
                if find(u1) == find(u2):
                    continue
                a = int(nxt[u1]); b = int(prv[u1])
                for v1 in (a, b):
                    if v1 == -1:
                        continue
                    if require_face and v1 not in face_set:
                        continue
                    v2 = v1 + sdim
                    if v2 < n and (int(nxt[u2]) == v2 or int(prv[u2]) == v2) \
                            and find(v2) == find(u2):
                        portal = ((u1, v1), (u2, v2))
                        break
                if portal is not None:
                    break
            if portal is not None:
                break

        if portal is None:
            return None  # fall back to reference full merge

        (u1, v1), (u2, v2) = portal
        # remove (u1,v1) and (u2,v2)
        if int(nxt[u1]) == v1: nxt[u1] = -1
        else: prv[u1] = -1
        if int(nxt[v1]) == u1: nxt[v1] = -1
        else: prv[v1] = -1
        if int(nxt[u2]) == v2: nxt[u2] = -1
        else: prv[u2] = -1
        if int(nxt[v2]) == u2: nxt[v2] = -1
        else: prv[v2] = -1
        # add (u1,u2) and (v1,v2)
        if int(nxt[u1]) == -1: nxt[u1] = u2
        else: prv[u1] = u2
        if int(nxt[u2]) == -1: nxt[u2] = u1
        else: prv[u2] = u1
        if int(nxt[v1]) == -1: nxt[v1] = v2
        else: prv[v1] = v2
        if int(nxt[v2]) == -1: nxt[v2] = v1
        else: prv[v2] = v1
        # union components of u1 and u2
        ru, rv = find(u1), find(u2)
        if ru != rv:
            parent[rv] = ru

    return circuits


# --------------------------------------------------------------------------- #
# Full fast pipeline                                                           #
# --------------------------------------------------------------------------- #
def fast_code_map(grid, weights: np.ndarray, merge_policy: str = "full") -> np.ndarray:
    """Vectorized construction returning the code_map directly.

    Equivalence target:
      - d=2: identical code_map to weights_to_sfc (default args).
      - d>=3: quality-equivalent (valid AC path; avg_seg within noise).
    """
    d = grid.d
    base_arr, k_arr, lowcoord = scipy_mst_edge_arrays(d, grid.block_side, weights)
    circuits = build_circuits_fast(grid.grid_shape)

    merged = apply_mst_fast(circuits, base_arr, k_arr, grid.grid_shape)
    if merged is None:
        # Rare: some edge lacks a matched portal -> full reference merge.
        from lacsfcf.core.cover_merge import apply_mst_nd
        edges = _edge_arrays_to_tuples(k_arr, lowcoord)
        merged = apply_mst_nd(
            circuits, _EdgeListGraph(edges), grid.grid_shape,
            edge_order="default", merge_policy=merge_policy,
            portal_selection_policy="first_found",
        )
    start_id = 0  # all-zeros cell -> flat id 0
    return cycle_to_code_map(merged, grid.grid_shape, start_id)
