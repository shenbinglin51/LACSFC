"""General d-dimensional Cover-and-Merge for LACSFCF.

Terminology:
  grid_shape : tuple of d ints, each = 2^K  (e.g. (256, 256) for K=8, d=2)
  block_shape: tuple of d ints, each = 2^(K-1)
  Cell       : d-tuple of ints, global grid coordinates
  Block      : d-tuple of ints, block coordinates in block_shape

Cover phase:
  Each 2×…×2 block gets a BRGC Hamiltonian cycle over its 2^d cells.
  The BRGC local coordinates are reflected per-dimension according to the
  block's coordinate parity (block[k] % 2 == 1 → flip local axis k).
  This ensures that for every pair of adjacent blocks in any dimension k,
  the portal edges on their shared faces are compatible for 2-edge splicing.

Merge phase (2-edge splice):
  For each tree edge (b1, b2) in the MaxST:
    - b1 and b2 differ in exactly one dimension k, with b1[k]+1 == b2[k].
    - Find a matched portal pair:
        (u1,v1) on b1's high face (coord[k] == 2*b1[k]+1), currently in circuits
        (u2,v2) = (u1+e_k, v1+e_k) on b2's low face, currently in circuits
    - Remove (u1,v1) and (u2,v2), add (u1,u2) and (v1,v2).
    - New edges satisfy ℓ₁=1 (geometric pairing φ).
  Portal search is robust: v1 need not remain on the face after prior splices;
  a cross-component check ensures each splice merges (not splits) cycles.
"""
import itertools
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from lacsfcf.core.brgc import gray_code_sequence

Cell  = Tuple[int, ...]
Block = Tuple[int, ...]
Edge  = Tuple[Cell, Cell]


MERGE_STAT_KEYS: Tuple[str, ...] = (
    "matched_portal",
    "general_portal",
    "strict_global",
    "relay",
    "relaxed_global",
    "already_connected_skip",
)

MERGE_POLICIES: Tuple[str, ...] = (
    "matched_only",
    "matched_general",
    "full",
)

PORTAL_SELECTION_POLICIES: Tuple[str, ...] = (
    "first_found",
    "hilbert_gap",
)


def _empty_merge_stats() -> Dict[str, int]:
    return {key: 0 for key in MERGE_STAT_KEYS}


class _RelaySplice:
    """Container for a two-hop relay splice result."""
    __slots__ = ("r1", "r2")
    def __init__(self, r1, r2):
        self.r1 = r1  # ((u1,v1),(um1,vm1))  splice comp_b1 ↔ comp_m
        self.r2 = r2  # ((u2,v2),(um2,vm2))  splice comp_b2 ↔ comp_m


# ---------------------------------------------------------------------------
# Fast numpy-based circuits graph (degree-2 Hamiltonian cycles)
# ---------------------------------------------------------------------------

class _Circuits:
    """Compact adjacency representation for a union of Hamiltonian cycles.

    Each node has exactly degree 2.  Stored as two int32 arrays:
      nxt[i] = one  neighbour of node i
      prv[i] = other neighbour of node i

    Node ids are flat (ravel_multi_index) indices into grid_shape.
    All internal operations use int ids; tuple↔int conversion happens only
    at the public boundary (``cell2id`` / ``id2cell``).
    """
    __slots__ = ("nxt", "prv", "grid_shape", "strides", "n", "coords", "_stride_set")

    def __init__(self, n: int, grid_shape: Tuple[int, ...]):
        self.n          = n
        self.grid_shape = grid_shape
        # Compute C-order strides for ravel_multi_index
        d = len(grid_shape)
        strides = [1] * d
        for k in range(d - 2, -1, -1):
            strides[k] = strides[k + 1] * grid_shape[k + 1]
        self.strides = strides
        self.nxt = np.full(n, -1, dtype=np.int32)
        self.prv = np.full(n, -1, dtype=np.int32)
        # Pre-compute coordinate matrix: coords[i, k] = k-th coordinate of node i.
        # Stored as int16 (max grid size 2^15 per dim is more than enough).
        self.coords = np.array(
            np.unravel_index(np.arange(n), grid_shape), dtype=np.int16
        ).T   # shape (n, d)
        # Set of valid stride values for O(1) adjacency check.
        self._stride_set: Set[int] = set(strides)

    # ---- id ↔ cell conversion (hot path – avoid np overhead) ---------------

    def cell2id(self, cell: Cell) -> int:
        idx = 0
        for k, c in enumerate(cell):
            idx += c * self.strides[k]
        return idx

    def id2cell(self, idx: int) -> Cell:
        cell = []
        rem = idx
        for k in range(len(self.grid_shape) - 1, -1, -1):
            cell.append(rem % self.grid_shape[k])
            rem //= self.grid_shape[k]
        return tuple(reversed(cell))

    # ---- graph mutation -----------------------------------------------------

    def add_edge(self, u: int, v: int) -> None:
        """Add undirected edge u–v (both must have a free slot, i.e. nxt==-1)."""
        if self.nxt[u] == -1:
            self.nxt[u] = v
        else:
            self.prv[u] = v
        if self.nxt[v] == -1:
            self.nxt[v] = u
        else:
            self.prv[v] = u

    def remove_edge(self, u: int, v: int) -> None:
        """Remove undirected edge u–v."""
        if self.nxt[u] == v:
            self.nxt[u] = -1
        else:
            self.prv[u] = -1
        if self.nxt[v] == u:
            self.nxt[v] = -1
        else:
            self.prv[v] = -1

    def has_node(self, uid: int) -> bool:
        return 0 <= uid < self.n

    def has_edge(self, u: int, v: int) -> bool:
        return self.nxt[u] == v or self.prv[u] == v

    def neighbors(self, u: int):
        """Return the (up to) 2 neighbours of u as a list of ints."""
        result = []
        a = int(self.nxt[u]); b = int(self.prv[u])
        if a != -1: result.append(a)
        if b != -1: result.append(b)
        return result

    def copy(self) -> "_Circuits":
        c = _Circuits.__new__(_Circuits)
        c.n          = self.n
        c.grid_shape = self.grid_shape
        c.strides    = self.strides
        c.coords      = self.coords      # immutable – shared reference is safe
        c._stride_set = self._stride_set  # immutable – shared reference is safe
        c.nxt         = self.nxt.copy()
        c.prv         = self.prv.copy()
        return c

    def iter_nodes(self):
        return range(self.n)

    def is_adjacent(self, a: int, b: int) -> bool:
        """Return True iff nodes a and b are grid-adjacent (l1 distance == 1).

        Pure-int implementation: ~10x faster than numpy _l1 for scalar pairs.
        """
        diff = a - b
        if diff < 0:
            diff = -diff
        if diff not in self._stride_set:
            return False
        # Find which dimension this stride corresponds to, then check no wrap.
        coords = self.coords
        strides = self.strides
        for k, s in enumerate(strides):
            if s == diff:
                return abs(int(coords[a, k]) - int(coords[b, k])) == 1
        return False


# ---------------------------------------------------------------------------
# Fast array-based Union-Find (operates on int node ids)
# ---------------------------------------------------------------------------

class _UnionFindArr:
    """Path-compressed, union-by-rank DSU over integer ids [0, n)."""

    __slots__ = ("parent", "rank")

    def __init__(self, n: int) -> None:
        self.parent = np.arange(n, dtype=np.int32)
        self.rank   = np.zeros(n, dtype=np.int32)

    def find(self, x: int) -> int:
        p = self.parent
        while p[x] != x:
            p[x] = p[p[x]]   # path halving
            x = int(p[x])
        return x

    def union(self, x: int, y: int) -> bool:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True

    def same(self, x: int, y: int) -> bool:
        return self.find(x) == self.find(y)

    def find_all(self) -> np.ndarray:
        """Return array p where p[i] = root of node i (with path compression).

        Uses repeated parent[parent] applications until convergence.
        Much faster than calling find() in a Python loop.
        """
        p = self.parent.copy()
        while True:
            pp = p[p]
            if np.array_equal(pp, p):
                break
            p = pp
        self.parent[:] = p   # apply full path compression
        return p


# ---------------------------------------------------------------------------
# Legacy dict-based Union-Find (kept for any callers that pass tuple keys)
# ---------------------------------------------------------------------------

class _UnionFind:
    """Path-compressed, union-by-rank disjoint-set structure (tuple-key version)."""

    __slots__ = ("parent", "rank")

    def __init__(self, nodes) -> None:
        self.parent: Dict = {n: n for n in nodes}
        self.rank:   Dict = {n: 0 for n in nodes}

    def find(self, x):
        while self.parent[x] is not x:
            self.parent[x] = self.parent[self.parent[x]]  # path halving
            x = self.parent[x]
        return x

    def union(self, x, y) -> bool:
        rx, ry = self.find(x), self.find(y)
        if rx is ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True

    def same(self, x, y) -> bool:
        return self.find(x) is self.find(y)


_gc_cache: Dict[int, list] = {}


def _reflected_gc(d: int, block: Block) -> list:
    """Return the BRGC sequence for *block*, reflected per-axis by parity.

    For each dimension k where block[k] is odd, the local coordinate in that
    axis is flipped (0↔1).  This guarantees that adjacent blocks in any
    dimension share a compatible portal edge orientation for 2-edge splicing.
    """
    if d not in _gc_cache:
        _gc_cache[d] = gray_code_sequence(d)
    gc = _gc_cache[d]
    return [
        tuple(1 - s[k] if block[k] % 2 == 1 else s[k] for k in range(d))
        for s in gc
    ]


def _block_cells_ordered(block: Block) -> Tuple[Cell, ...]:
    """Return all 2^d cells of *block* in reflected-BRGC order (global coords)."""
    d  = len(block)
    gc = _reflected_gc(d, block)
    return tuple(tuple(2 * block[k] + s[k] for k in range(d)) for s in gc)


def _face_cells(block: Block, dim: int, side: int) -> Set[Cell]:
    """Set of cells on the given face of *block*.

    side=1 → high face (local coord in *dim* == 1, global == 2*block[dim]+1)
    side=0 → low face  (local coord in *dim* == 0, global == 2*block[dim]  )
    """
    d      = len(block)
    target = 2 * block[dim] + side
    other  = [range(2) for _ in range(d - 1)]
    cells: Set[Cell] = set()
    for coords in itertools.product(*other):
        cell = []
        ci   = 0
        for k in range(d):
            if k == dim:
                cell.append(target)
            else:
                cell.append(2 * block[k] + coords[ci])
                ci += 1
        cells.add(tuple(cell))
    return cells


def _face_cell_ids_np(
    block: Block, dim: int, side: int, strides: List[int], d: int
) -> np.ndarray:
    """Return numpy int32 array of flat node ids on the given face (vectorized).

    Replaces ``{c2id(c) for c in _face_cells(block, dim, side)}``.
    Avoids itertools.product and per-cell tuple/cell2id overhead.
    """
    base: int = (2 * block[dim] + side) * strides[dim]
    other_strides: List[int] = []
    for k in range(d):
        if k != dim:
            base += 2 * block[k] * strides[k]
            other_strides.append(strides[k])
    m = 1 << (d - 1)  # 2^(d-1) cells on the face
    bits = np.arange(m, dtype=np.int32)
    offsets = np.zeros(m, dtype=np.int32)
    for j, s in enumerate(other_strides):
        offsets += ((bits >> j) & 1) * s
    return (base + offsets).astype(np.int32)


def _portal_hilbert_score(circuits: "_Circuits", edge1: Tuple[int, int], edge2: Tuple[int, int]) -> float:
    u1, v1 = edge1
    u2, v2 = edge2
    c_u1 = circuits.id2cell(int(u1))
    c_v1 = circuits.id2cell(int(v1))
    c_u2 = circuits.id2cell(int(u2))
    c_v2 = circuits.id2cell(int(v2))
    d = len(c_u1)
    max_coord = max(
        max(int(x) for x in c_u1),
        max(int(x) for x in c_v1),
        max(int(x) for x in c_u2),
        max(int(x) for x in c_v2),
    )
    p = max(1, int(max_coord).bit_length())
    from hilbertcurve.hilbertcurve import HilbertCurve

    hc = HilbertCurve(p=p, n=d)
    codes = [
        int(hc.distance_from_coordinates(list(c_u1))),
        int(hc.distance_from_coordinates(list(c_v1))),
        int(hc.distance_from_coordinates(list(c_u2))),
        int(hc.distance_from_coordinates(list(c_v2))),
    ]
    return float(abs(codes[0] - codes[2]) + abs(codes[1] - codes[3]))


def _select_portal_candidate(
    circuits: "_Circuits",
    candidates: List[Tuple[Edge, Edge]],
    portal_selection_policy: str,
) -> Optional[Tuple[Edge, Edge]]:
    if not candidates:
        return None
    if portal_selection_policy == "first_found":
        return candidates[0]
    if portal_selection_policy == "hilbert_gap":
        return min(candidates, key=lambda cand: _portal_hilbert_score(circuits, cand[0], cand[1]))
    raise ValueError(f"unsupported portal_selection_policy: {portal_selection_policy}")


def _find_matched_portal(
    circuits: "_Circuits",
    block: Block,
    dim: int,
    d: int,
    uf: "_UnionFindArr",
    portal_selection_policy: str = "first_found",
) -> Optional[Tuple[Edge, Edge]]:
    """Search for a matched portal pair for a merge in *dim*.

    Finds edge (u1,v1) with u1 on block's high face (coord[dim]==2*block[dim]+1)
    such that (u2,v2) = (u1+e_dim, v1+e_dim) also exists in *circuits* and
    u1/u2 belong to *different* connected components (guaranteeing that the
    splice merges two cycles rather than splitting one).

    *uf* is a :class:`_UnionFindArr` maintained incrementally by the caller.

    Returns ((u1,v1), (u2,v2)) or None if not found.  All values are int ids.
    """
    face1_coord = 2 * block[dim] + 1
    strides     = circuits.strides
    stride_dim  = strides[dim]
    n           = circuits.n

    # Vectorized face cell ids (avoids itertools.product + cell2id loop)
    face_ids_arr = _face_cell_ids_np(block, dim, 1, strides, d)
    face_ids     = set(face_ids_arr.tolist())

    nxt_arr = circuits.nxt
    prv_arr = circuits.prv

    def _try(u1id: int, require_v1_in_face: bool) -> Optional[Tuple]:
        u2id = u1id + stride_dim
        if u2id >= n:
            return None
        if uf.same(u1id, u2id):
            return None
        _a = int(nxt_arr[u1id]); _b = int(prv_arr[u1id])
        for v1id in (_x for _x in (_a, _b) if _x != -1):
            if require_v1_in_face and v1id not in face_ids:
                continue
            v2id = v1id + stride_dim
            if v2id < n and circuits.has_edge(u2id, v2id) and uf.same(v2id, u2id):
                return (u1id, v1id), (u2id, v2id)
        return None

    face_ids_sorted = sorted(face_ids)
    candidates: List[Tuple[Edge, Edge]] = []
    for u1id in face_ids_sorted:
        r = _try(u1id, require_v1_in_face=True)
        if r:
            if portal_selection_policy == "first_found":
                return r
            candidates.append(r)

    for u1id in face_ids_sorted:
        r = _try(u1id, require_v1_in_face=False)
        if r:
            if portal_selection_policy == "first_found":
                return r
            candidates.append(r)

    return _select_portal_candidate(circuits, candidates, portal_selection_policy)


def _find_portal_general(
    circuits: "_Circuits",
    block: Block,
    dim: int,
    d: int,
    uf: "_UnionFindArr",
    b2: Optional[Block] = None,
    portal_selection_policy: str = "first_found",
) -> Optional[Tuple[Edge, Edge]]:
    """General-purpose fallback portal search when _find_matched_portal fails.

    Relaxes the v2 = v1+e_dim constraint: instead of requiring both new edges
    to be axis-aligned copies, it only requires:
      - u2 = u1 + e_dim  (l1=1, AC)
      - l1(v1, v2) == 1  (AC for the second new edge)
      - u1/u2 in different components
      - no self-loops or multi-edges

    Search order:
      Phase A: u1 on block's high face (face cells only) – any different comps.
      Phase B: u1 anywhere on coord[dim]==face1_coord – restricted to the
               target pair (comp_b1, comp_b2) to avoid premature merges.

    Args:
        b2: the high-side neighbour block; when provided Phase B only accepts
            splices between comp(b1) and comp(b2).
    """
    face1_coord = 2 * block[dim] + 1
    c2id        = circuits.cell2id
    stride_dim  = circuits.strides[dim]
    strides     = circuits.strides
    coords      = circuits.coords
    n           = circuits.n

    face_ids_arr = _face_cell_ids_np(block, dim, 1, strides, d)
    face_ids     = set(face_ids_arr.tolist())

    # Determine target components for Phase B (if b2 is known)
    if b2 is not None:
        b1_id = c2id(tuple(2 * block[j] for j in range(d)))
        b2_id = c2id(tuple(2 * b2[j]    for j in range(d)))
        comp_b1_target = uf.find(b1_id)
        comp_b2_target = uf.find(b2_id)
    else:
        comp_b1_target = -1
        comp_b2_target = -1

    def _try_general(u1id: int, restrict_comps: bool = False) -> List[Tuple[Edge, Edge]]:
        out: List[Tuple[Edge, Edge]] = []
        u2id = u1id + stride_dim
        if u2id >= n:
            return out
        if uf.same(u1id, u2id):
            return out
        if restrict_comps and comp_b1_target >= 0:
            c1, c2 = uf.find(u1id), uf.find(u2id)
            if not ({c1, c2} == {comp_b1_target, comp_b2_target}):
                return out
        for v1id in circuits.neighbors(u1id):
            if v1id == u2id:
                continue
            for v2id in circuits.neighbors(u2id):
                if v2id == u1id or v2id == v1id:
                    continue
                if circuits.is_adjacent(v1id, v2id):
                    out.append(((u1id, v1id), (u2id, v2id)))
        return out

    candidates: List[Tuple[Edge, Edge]] = []

    # Phase A: face cells only
    for u1id in sorted(face_ids):
        phase_candidates = _try_general(u1id, restrict_comps=False)
        if not phase_candidates:
            continue
        if portal_selection_policy == "first_found":
            return phase_candidates[0]
        candidates.extend(phase_candidates)

    # Phase B: full layer scan – pre-build layer_ids via numpy (fast)
    layer_mask = (coords[:, dim] == face1_coord)
    layer_ids  = np.where(layer_mask)[0]
    face_ids_set = face_ids
    for u1id in layer_ids:
        u1id = int(u1id)
        if u1id in face_ids_set:
            continue
        phase_candidates = _try_general(u1id, restrict_comps=True)
        if not phase_candidates:
            continue
        if portal_selection_policy == "first_found":
            return phase_candidates[0]
        candidates.extend(phase_candidates)

    return _select_portal_candidate(circuits, candidates, portal_selection_policy)


def _find_portal_any_layer(
    circuits: "_Circuits",
    b1: Block,
    b2: Block,
    d: int,
    uf: "_UnionFindArr",
    pass1_only: bool = False,
    comp_to_nodes: Optional[Dict] = None,
) -> Optional[Tuple[Edge, Edge]]:
    """Last-resort portal search across the entire graph.

    Restricted to pairs where one endpoint belongs to the component
    containing b1's cells and the other to the component containing b2's cells.
    """
    c2id   = circuits.cell2id
    coords = circuits.coords
    n      = circuits.n

    b1_id   = c2id(tuple(2 * b1[j] for j in range(d)))
    b2_id   = c2id(tuple(2 * b2[j] for j in range(d)))
    comp_b1 = uf.find(b1_id)
    comp_b2 = uf.find(b2_id)

    # Use pre-built comp_to_nodes if available, else build via find_all (fast).
    if comp_to_nodes is not None:
        nodes_b1 = comp_to_nodes.get(comp_b1, [])
        nodes_b2 = comp_to_nodes.get(comp_b2, [])
    else:
        roots    = uf.find_all()
        nodes_b1 = list(np.where(roots == comp_b1)[0].astype(int))
        nodes_b2 = list(np.where(roots == comp_b2)[0].astype(int))

    strides = circuits.strides
    is_adj  = circuits.is_adjacent

    # Pass 1: strict AC – enumerate stride-1 neighbours of nodes_b1,
    # check if they land in comp_b2.  No numpy per-pair overhead.
    for uid in nodes_b1:
        for dim in range(d):
            stride = strides[dim]
            for upid in (uid + stride, uid - stride):
                if upid < 0 or upid >= n:
                    continue
                if not is_adj(uid, upid):   # guard wrap-around
                    continue
                if uf.find(upid) != comp_b2:
                    continue
                for nuid in circuits.neighbors(uid):
                    if nuid == upid:
                        continue
                    for nupid in circuits.neighbors(upid):
                        if nupid == uid or nupid == nuid:
                            continue
                        if is_adj(nuid, nupid):
                            return (uid, nuid), (upid, nupid)

    if pass1_only:
        return None

    # Pass 2: relaxed – find (u, up) across comp_b1/comp_b2 with minimum l1.
    # To avoid O(n²) broadcast on large components, restrict to nodes whose
    # coordinate in the split dimension is within ±RADIUS of the interface.
    # The interface is estimated as the mean coord[dim] boundary between b1/b2.
    # We use a small RADIUS=3; if that fails, expand to the full set.
    # For each candidate (u, up) pair, require only l1(nuid, nupid)==1.
    RADIUS = 3

    def _pass2_search(cands_b1: List[int], cands_b2: List[int]) -> Optional[Tuple]:
        """Search all (u, up) pairs sorted by l1, return first valid splice."""
        if not cands_b1 or not cands_b2:
            return None
        arr1 = np.array(cands_b1, dtype=np.int32)
        arr2 = np.array(cands_b2, dtype=np.int32)
        dist = np.abs(coords[arr1][:, None, :].astype(np.int32)
                      - coords[arr2][None, :, :].astype(np.int32)).sum(axis=2)
        order = np.argsort(dist.ravel())
        len2  = len(arr2)
        for flat_idx in order:
            uid  = int(arr1[int(flat_idx) // len2])
            upid = int(arr2[int(flat_idx) %  len2])
            for nuid in circuits.neighbors(uid):
                if nuid == upid:
                    continue
                for nupid in circuits.neighbors(upid):
                    if nupid == uid or nupid == nuid:
                        continue
                    if is_adj(nuid, nupid):
                        return (uid, nuid), (upid, nupid)
        return None

    # Estimate interface coordinate: midpoint between b1 high-face and b2 low-face.
    # b1 high-face coord[split_dim] = max coord in nodes_b1 along split_dim.
    # Use the dimension where b1 and b2 are adjacent (largest coord gap).
    # Fallback: use the full node lists if RADIUS search fails.
    b1_coords = coords[np.array(nodes_b1, dtype=np.int32)]  # (|b1|, d)
    b2_coords = coords[np.array(nodes_b2, dtype=np.int32)]  # (|b2|, d)
    # Find the split dimension: largest mean difference between b1 and b2.
    mean_diff = np.abs(b1_coords.mean(axis=0).astype(np.float32)
                       - b2_coords.mean(axis=0).astype(np.float32))
    split_dim = int(np.argmax(mean_diff))
    interface = float(b1_coords[:, split_dim].max() + b2_coords[:, split_dim].min()) / 2.0

    cands_b1 = [uid for uid in nodes_b1
                if abs(float(coords[uid, split_dim]) - interface) <= RADIUS]
    cands_b2 = [uid for uid in nodes_b2
                if abs(float(coords[uid, split_dim]) - interface) <= RADIUS]

    r = _pass2_search(cands_b1, cands_b2)
    if r is not None:
        return r
    # Fallback: full search (safety net – should be very rare)
    return _pass2_search(nodes_b1, nodes_b2)


def _find_relay_splice(
    circuits: "_Circuits",
    b1_id: int,
    b2_id: int,
    d: int,
    uf: "_UnionFindArr",
    comp_to_nodes: Optional[Dict] = None,
):
    """Two-hop relay: merge comp_b1 and comp_b2 via a shared neighbour comp_m."""
    comp_b1 = uf.find(b1_id)
    comp_b2 = uf.find(b2_id)
    n       = circuits.n
    coords  = circuits.coords
    strides = circuits.strides

    # Use pre-built comp_to_nodes if available, else build via find_all (fast).
    if comp_to_nodes is not None:
        comp_nodes = comp_to_nodes
    else:
        roots = uf.find_all()
        unique_roots = np.unique(roots)
        comp_nodes = {
            int(r): list(np.where(roots == r)[0].astype(int))
            for r in unique_roots
        }

    nodes_b1 = comp_nodes.get(comp_b1, [])
    nodes_b2 = comp_nodes.get(comp_b2, [])

    is_adj = circuits.is_adjacent

    def _ac_splice(src_nodes: List[int], tgt_comp: int):
        """Return the first strict-AC splice between src_nodes and tgt_comp."""
        for uid in src_nodes:
            for dim in range(d):
                stride = strides[dim]
                for upid in (uid + stride, uid - stride):
                    if upid < 0 or upid >= n:
                        continue
                    if not is_adj(uid, upid):
                        continue
                    if uf.find(upid) != tgt_comp:
                        continue
                    for nuid in circuits.neighbors(uid):
                        if nuid == upid:
                            continue
                        for nupid in circuits.neighbors(upid):
                            if nupid == uid or nupid == nuid:
                                continue
                            if is_adj(nuid, nupid):
                                return (uid, nuid), (upid, nupid)
        return None

    # Try every other component as relay.
    for comp_m, m_nodes in comp_nodes.items():
        if comp_m == comp_b1 or comp_m == comp_b2:
            continue
        r1 = _ac_splice(nodes_b1, comp_m)
        if r1 is None:
            continue
        r2 = _ac_splice(nodes_b2, comp_m)
        if r2 is None:
            continue
        # Ensure r1 and r2 use disjoint edge sets (no shared edge).
        (u1, v1), (um1, vm1) = r1
        (u2, v2), (um2, vm2) = r2
        edges1 = {(u1, v1), (v1, u1), (um1, vm1), (vm1, um1)}
        if (u2, v2) in edges1 or (v2, u2) in edges1 or \
           (um2, vm2) in edges1 or (vm2, um2) in edges1:
            continue
        return _RelaySplice(r1=r1, r2=r2)

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_circuits_nd(grid_shape: Tuple[int, ...]) -> "_Circuits":
    """Build reflected-BRGC Hamiltonian cycles for every block in the grid.

    Args:
        grid_shape: tuple of d ints, each must be even (= 2^K).

    Returns:
        _Circuits object.  Each connected component is the reflected-BRGC
        Hamiltonian cycle of one 2×…×2 block.
    """
    block_shape = tuple(s // 2 for s in grid_shape)
    n = 1
    for s in grid_shape:
        n *= s

    circ = _Circuits(n, grid_shape)
    c2id = circ.cell2id

    for block in itertools.product(*[range(b) for b in block_shape]):
        cells   = _block_cells_ordered(block)   # uses reflected BRGC
        n_local = len(cells)
        ids     = [c2id(cells[i]) for i in range(n_local)]
        for i in range(n_local):
            circ.add_edge(ids[i], ids[(i + 1) % n_local])

    return circ


def apply_mst_nd(
    circuits: "_Circuits",
    mst: nx.Graph,
    grid_shape: Tuple[int, ...],
    edge_order: str = "default",
    return_stats: bool = False,
    merge_policy: str = "full",
    portal_selection_policy: str = "first_found",
) -> "_Circuits":
    """Apply MaxST to reflected-BRGC circuits via successive 2-edge splices.

    Args:
        circuits  : output of ``build_circuits_nd``; modified in-place copy.
        mst       : Maximum Spanning Tree of the block dual graph.
                    Node labels must be d-tuples (block coordinates).
        grid_shape: same value passed to ``build_circuits_nd``.
        edge_order: order in which MST edges are processed.
                    ``"default"``     – networkx iteration order (default).
                    ``"weight_desc"`` – heaviest edge first.
                    ``"weight_asc"``  – lightest edge first.
        merge_policy: fallback policy for splice search.
                      ``"matched_only"`` keeps only matched portals.
                      ``"matched_general"`` allows matched + general portal.
                      ``"full"`` enables the full fallback stack.
        portal_selection_policy: how to choose among multiple valid portal
                      candidates. ``"first_found"`` keeps the original
                      scan-order behavior. ``"hilbert_gap"`` prefers the
                      candidate with the smallest Hilbert-code discontinuity.

    Returns:
        Modified _Circuits.  If all splices succeed, this is a single
        Hamiltonian cycle over all cells. When ``return_stats`` is True,
        returns ``(circuits, stats)`` where ``stats`` counts which merge
        fallback path was used.

    Raises:
        RuntimeError: if no matched portal is found for some tree edge.
    """
    circuits = circuits.copy()    # protect caller's original
    if merge_policy not in MERGE_POLICIES:
        raise ValueError(f"unsupported merge_policy: {merge_policy}")
    if portal_selection_policy not in PORTAL_SELECTION_POLICIES:
        raise ValueError(f"unsupported portal_selection_policy: {portal_selection_policy}")
    d        = len(grid_shape)
    c2id     = circuits.cell2id
    n        = circuits.n

    # Vectorized block-cell initialization: avoids per-node cycle traversal.
    # For each block, compute all 2^d cell ids directly from block coordinates,
    # union them together, and store in comp_to_nodes – no Python-level
    # append per node, no visited-array bookkeeping.
    block_shape_local = tuple(s // 2 for s in grid_shape)
    block_size = 1 << d  # 2^d cells per block

    # Precompute all 2^d local-offset patterns (strides-weighted, block-independent)
    _strides_np = np.array(circuits.strides, dtype=np.int64)
    _bits_init  = np.arange(block_size, dtype=np.int64)
    _local_offsets = np.zeros(block_size, dtype=np.int64)
    for _k in range(d):
        _local_offsets += ((_bits_init >> _k) & 1) * _strides_np[_k]

    uf = _UnionFindArr(n)
    comp_to_nodes: Dict[int, List[int]] = {}
    merge_stats = _empty_merge_stats()

    for block in itertools.product(*[range(s) for s in block_shape_local]):
        _base: int = 0
        for _k in range(d):
            _base += 2 * block[_k] * int(_strides_np[_k])
        _cell_ids = (_base + _local_offsets).astype(np.int32)
        _s = int(_cell_ids[0])
        for _j in range(1, block_size):
            uf.union(_s, int(_cell_ids[_j]))
        root = uf.find(_s)
        comp_to_nodes[root] = _cell_ids.tolist()

    def _merge_comp(a: int, b: int) -> None:
        """Union a and b in uf; merge their lists in comp_to_nodes."""
        ra = uf.find(a)
        rb = uf.find(b)
        if ra == rb:
            return
        uf.union(a, b)
        new_root = uf.find(a)
        if new_root == ra:
            comp_to_nodes[ra].extend(comp_to_nodes.pop(rb, []))
        else:
            comp_to_nodes[rb].extend(comp_to_nodes.pop(ra, []))

    # Determine edge iteration order.
    if edge_order == "weight_desc":
        edges = sorted(mst.edges(data=True),
                       key=lambda e: e[2].get("weight", 0.0), reverse=True)
        edges = [(b1, b2) for b1, b2, _ in edges]
    elif edge_order == "weight_asc":
        edges = sorted(mst.edges(data=True),
                       key=lambda e: e[2].get("weight", 0.0))
        edges = [(b1, b2) for b1, b2, _ in edges]
    else:
        edges = list(mst.edges())

    for b1, b2 in edges:
        b1 = tuple(b1)
        b2 = tuple(b2)

        diff_dims = [k for k in range(d) if b1[k] != b2[k]]
        if len(diff_dims) != 1:
            raise ValueError(
                f"MST edge {b1}\u2013{b2} must differ in exactly 1 dimension, "
                f"got {len(diff_dims)}"
            )
        k = diff_dims[0]

        # Ensure b1 is the low block (b1[k] < b2[k])
        if b1[k] > b2[k]:
            b1, b2 = b2, b1

        result = _find_matched_portal(
            circuits,
            b1,
            dim=k,
            d=d,
            uf=uf,
            portal_selection_policy=portal_selection_policy,
        )
        merge_kind = "matched_portal" if result is not None else None
        if result is None and merge_policy in {"matched_general", "full"}:
            result = _find_portal_general(
                circuits,
                b1,
                dim=k,
                d=d,
                uf=uf,
                b2=b2,
                portal_selection_policy=portal_selection_policy,
            )
            if result is not None:
                merge_kind = "general_portal"

        b1_id = c2id(tuple(2 * b1[j] for j in range(d)))
        b2_id = c2id(tuple(2 * b2[j] for j in range(d)))

        if result is None and merge_policy == "full":
            if uf.same(b1_id, b2_id):
                # b1 and b2 already connected via a prior splice – skip.
                merge_stats["already_connected_skip"] += 1
                continue
            # Pass 3: strict-AC global search (Pass1 only – no relaxed Pass2).
            result = _find_portal_any_layer(circuits, b1, b2, d, uf,
                                            pass1_only=True,
                                            comp_to_nodes=comp_to_nodes)
            if result is not None:
                merge_kind = "strict_global"

        if result is None and merge_policy == "full":
            # Pass 4: two-hop relay via a shared intermediate component.
            result = _find_relay_splice(circuits, b1_id, b2_id, d, uf,
                                        comp_to_nodes=comp_to_nodes)
            if result is not None:
                merge_kind = "relay"

        if result is None and merge_policy == "full":
            # Pass 5 (last resort): relaxed global search allowing l1(u,up)>1.
            result = _find_portal_any_layer(circuits, b1, b2, d, uf,
                                            pass1_only=False,
                                            comp_to_nodes=comp_to_nodes)
            if result is not None:
                merge_kind = "relaxed_global"

        if result is None:
            raise RuntimeError(
                f"No portal found for blocks {b1} → {b2} in dim {k} "
                f"under merge_policy={merge_policy}."
            )

        # Apply the splice(s).
        if isinstance(result, _RelaySplice):
            merge_stats["relay"] += 1
            # Two-hop relay: first splice comp_b1 ⇔ comp_m, then merged ⇔ comp_b2.
            (u1, v1), (um1, vm1) = result.r1
            circuits.remove_edge(u1, v1)
            circuits.remove_edge(um1, vm1)
            circuits.add_edge(u1, um1)
            circuits.add_edge(v1, vm1)
            _merge_comp(u1, um1)
            # After r1 the component of b1 and comp_m are merged; now splice to b2.
            (u2, v2), (um2, vm2) = result.r2
            circuits.remove_edge(u2, v2)
            circuits.remove_edge(um2, vm2)
            circuits.add_edge(u2, um2)
            circuits.add_edge(v2, vm2)
            _merge_comp(u2, um2)
            continue

        if merge_kind is not None:
            merge_stats[merge_kind] += 1

        (u1, v1), (u2, v2) = result
        circuits.remove_edge(u1, v1)
        circuits.remove_edge(u2, v2)
        circuits.add_edge(u1, u2)   # φ(u1) = u2  →  ℓ₁ = 1
        circuits.add_edge(v1, v2)   # φ(v1) = v2  →  ℓ₁ = 1
        _merge_comp(u1, u2)         # merge components in UF + comp_to_nodes

    if return_stats:
        return circuits, merge_stats
    return circuits


def circuits_to_path(circuits: "_Circuits", start: Cell) -> List[Cell]:
    """Extract a Hamiltonian path by cutting one edge adjacent to *start*.

    The path begins at *start*, cuts the edge to its first neighbor,
    and traverses the resulting open chain.

    Args:
        circuits: a _Circuits object that is a single Hamiltonian cycle
                  (output of ``apply_mst_nd`` when all splices succeeded).
        start:    a cell tuple; the path begins here.

    Returns:
        Ordered list of Cell tuples (the SFC).
    """
    start_id = circuits.cell2id(start)
    id2c     = circuits.id2cell
    nxt_arr  = circuits.nxt
    prv_arr  = circuits.prv
    n        = circuits.n

    nbrs = circuits.neighbors(start_id)
    if len(nbrs) < 2:
        raise RuntimeError("circuits must be a Hamiltonian cycle (degree-2 graph)")

    # Cut edge start→nbrs[0]; traverse start→nbrs[1]→...
    path_ids = [start_id]
    prev_id = start_id
    cur_id  = nbrs[1]   # skip nbrs[0] (the cut edge)
    while cur_id != start_id:
        path_ids.append(cur_id)
        a = int(nxt_arr[cur_id])
        b = int(prv_arr[cur_id])
        nxt_id = a if a != prev_id else b
        prev_id, cur_id = cur_id, nxt_id

    return [id2c(i) for i in path_ids]
