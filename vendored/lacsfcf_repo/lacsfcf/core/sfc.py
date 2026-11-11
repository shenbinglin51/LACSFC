"""End-to-end SFC construction: edge weights → tree backbone → cover-merge → path."""
from __future__ import annotations

import heapq
import itertools
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from lacsfcf.core.cover_merge import apply_mst_nd, build_circuits_nd, circuits_to_path
from lacsfcf.data.grid import Cell, Grid


def _block_flat_index(block: Tuple[int, ...], block_shape: Tuple[int, ...]) -> int:
    return int(np.ravel_multi_index(block, block_shape))


def _compute_hilbert_block_codes(grid: Grid) -> np.ndarray:
    if grid.K <= 1:
        return np.zeros(grid.num_blocks, dtype=np.int64)

    from hilbertcurve.hilbertcurve import HilbertCurve

    hc = HilbertCurve(p=grid.K - 1, n=grid.d)
    codes = np.empty(grid.num_blocks, dtype=np.int64)
    for idx in range(grid.num_blocks):
        block = np.unravel_index(idx, grid.block_shape)
        codes[idx] = int(hc.distance_from_coordinates(list(block)))
    return codes


def _build_dual_graph(
    grid: Grid,
    weights: np.ndarray,
    *,
    tree_hilbert_penalty: float,
) -> nx.Graph:
    dual_g = nx.Graph()
    dual_g.add_nodes_from(np.ndindex(grid.block_shape))

    block_codes: Optional[np.ndarray] = None
    if tree_hilbert_penalty > 0.0:
        block_codes = _compute_hilbert_block_codes(grid)
    max_hilbert_gap = float(max(grid.num_blocks - 1, 1))

    for i, (b1, b2, _) in enumerate(grid.dual_edges):
        raw_weight = float(weights[i])
        hilbert_gap = 0.0
        if block_codes is not None:
            idx1 = _block_flat_index(b1, grid.block_shape)
            idx2 = _block_flat_index(b2, grid.block_shape)
            hilbert_gap = abs(int(block_codes[idx1]) - int(block_codes[idx2])) / max_hilbert_gap
        score = raw_weight - tree_hilbert_penalty * hilbert_gap
        dual_g.add_edge(
            b1,
            b2,
            weight=score,
            raw_weight=raw_weight,
            hilbert_gap=hilbert_gap,
        )
    return dual_g


def _build_locality_aware_tree(
    dual_g: nx.Graph,
    *,
    start_block: Tuple[int, ...],
    tree_branch_penalty: float,
) -> nx.Graph:
    tree = nx.Graph()
    tree.add_nodes_from(dual_g.nodes())
    if dual_g.number_of_nodes() <= 1:
        return tree

    if start_block not in dual_g:
        start_block = next(iter(dual_g.nodes()))

    in_tree = {start_block}
    degrees: Dict[Tuple[int, ...], int] = {start_block: 0}
    heap: List[Tuple[float, int, Tuple[int, ...], Tuple[int, ...]]] = []
    push_id = 0

    def current_score(u: Tuple[int, ...], v: Tuple[int, ...]) -> float:
        data = dual_g[u][v]
        degree_penalty = tree_branch_penalty * float(degrees.get(u, 0) + degrees.get(v, 0))
        return float(data["weight"]) - degree_penalty

    def push_frontier(node: Tuple[int, ...]) -> int:
        next_push_id = push_id
        for nbr in dual_g.neighbors(node):
            if nbr in in_tree:
                continue
            score = current_score(node, nbr)
            heapq.heappush(heap, (-score, next_push_id, node, nbr))
            next_push_id += 1
        return next_push_id

    push_id = push_frontier(start_block)

    while len(in_tree) < dual_g.number_of_nodes():
        if not heap:
            raise RuntimeError("Locality-aware tree builder could not find a spanning frontier")

        neg_score, _, u, v = heapq.heappop(heap)
        u_in = u in in_tree
        v_in = v in in_tree
        if u_in and v_in:
            continue
        if (not u_in) and (not v_in):
            continue

        parent, child = (u, v) if u_in else (v, u)
        score_now = current_score(parent, child)
        score_then = -neg_score
        if score_now + 1e-12 < score_then:
            heapq.heappush(heap, (-score_now, push_id, parent, child))
            push_id += 1
            continue

        data = dual_g[parent][child]
        tree.add_edge(
            parent,
            child,
            weight=score_now,
            raw_weight=float(data.get("raw_weight", data.get("weight", 0.0))),
            hilbert_gap=float(data.get("hilbert_gap", 0.0)),
        )
        in_tree.add(child)
        degrees[parent] = degrees.get(parent, 0) + 1
        degrees[child] = degrees.get(child, 0) + 1
        push_id = push_frontier(child)

    return tree


def _path_from_cut(circuits, start_id: int, cut_neighbor_id: int) -> List[int]:
    nbrs = [int(nbr) for nbr in circuits.neighbors(start_id)]
    if len(nbrs) < 2:
        raise RuntimeError("circuits must be a Hamiltonian cycle (degree-2 graph)")
    if cut_neighbor_id not in nbrs:
        raise ValueError("cut_neighbor_id must be adjacent to start")

    nxt_arr = circuits.nxt
    prv_arr = circuits.prv
    other_neighbor_id = nbrs[0] if int(nbrs[1]) == int(cut_neighbor_id) else nbrs[1]

    path_ids = [start_id]
    prev_id = start_id
    cur_id = int(other_neighbor_id)
    while cur_id != start_id:
        path_ids.append(cur_id)
        a = int(nxt_arr[cur_id])
        b = int(prv_arr[cur_id])
        nxt_id = a if a != prev_id else b
        prev_id, cur_id = cur_id, nxt_id
    return path_ids


def _evaluate_cut_l1(path_ids: Sequence[int], circuits) -> float:
    coords = [circuits.id2cell(int(pid)) for pid in path_ids]
    if len(coords) <= 1:
        return 0.0
    total = 0.0
    for a, b in zip(coords, coords[1:]):
        total += float(sum(abs(int(a[k]) - int(b[k])) for k in range(len(a))))
    return total / float(len(coords) - 1)


def _evaluate_cut_hilbert(path_ids: Sequence[int], circuits) -> float:
    if not path_ids:
        return 0.0
    cells = np.asarray([circuits.id2cell(int(pid)) for pid in path_ids], dtype=np.int64)
    d = int(cells.shape[1]) if cells.ndim == 2 else 0
    if d <= 0:
        return 0.0

    max_coord = int(np.max(cells)) if cells.size else 0
    p = max(1, max_coord.bit_length())
    from hilbertcurve.hilbertcurve import HilbertCurve

    hc = HilbertCurve(p=p, n=d)
    codes = [int(hc.distance_from_coordinates(cell.tolist())) for cell in cells]
    if len(codes) <= 1:
        return 0.0
    total_gap = 0.0
    for a, b in zip(codes, codes[1:]):
        total_gap += abs(int(a) - int(b))
    return total_gap / float(len(codes) - 1)


def _extract_path_with_cut_optimization(
    circuits,
    start: Cell,
    *,
    cut_mode: str,
    cut_eval: str,
    cut_samples: int,
) -> Tuple[List[Cell], Dict[str, object]]:
    start_id = circuits.cell2id(start)
    neighbors = [int(nbr) for nbr in circuits.neighbors(start_id)]
    if len(neighbors) < 2:
        raise RuntimeError("circuits must be a Hamiltonian cycle (degree-2 graph)")

    if cut_mode == "fixed_start":
        path_ids = _path_from_cut(circuits, start_id, neighbors[0])
        stats = {
            "cut_mode": cut_mode,
            "cut_eval": cut_eval,
            "cut_candidates": 1,
            "cut_choice": int(neighbors[0]),
            "cut_score": 0.0,
        }
        return [circuits.id2cell(int(i)) for i in path_ids], stats

    unique_ids = np.unique(np.concatenate((circuits.nxt, circuits.prv))).astype(np.int64)
    candidate_ids = [int(i) for i in unique_ids if int(i) != int(start_id)]
    candidate_ids.extend(neighbors)
    candidate_ids = list(dict.fromkeys(candidate_ids))

    if cut_samples > 0 and len(candidate_ids) > cut_samples:
        step = len(candidate_ids) / float(cut_samples)
        sampled = [candidate_ids[int(i * step)] for i in range(cut_samples)]
        candidate_ids = list(dict.fromkeys(sampled + neighbors))

    best_path_ids: Optional[List[int]] = None
    best_cut_id: Optional[int] = None
    best_score: Optional[float] = None

    for cut_id in candidate_ids:
        try:
            path_ids = _path_from_cut(circuits, start_id, int(cut_id))
        except ValueError:
            continue
        if cut_eval == "hilbert_gap":
            score = _evaluate_cut_hilbert(path_ids, circuits)
        else:
            score = _evaluate_cut_l1(path_ids, circuits)
        if best_score is None or float(score) < float(best_score):
            best_score = float(score)
            best_cut_id = int(cut_id)
            best_path_ids = path_ids

    if best_path_ids is None or best_cut_id is None or best_score is None:
        raise RuntimeError("Cut optimisation could not find a valid path cut")

    stats = {
        "cut_mode": cut_mode,
        "cut_eval": cut_eval,
        "cut_candidates": int(len(candidate_ids)),
        "cut_choice": int(best_cut_id),
        "cut_score": float(best_score),
    }
    return [circuits.id2cell(int(i)) for i in best_path_ids], stats


def weights_to_sfc(
    grid: Grid,
    weights: np.ndarray,
    start: Tuple[int, ...] = None,
    return_stats: bool = False,
    merge_policy: str = "full",
    tree_branch_penalty: float = 0.0,
    tree_hilbert_penalty: float = 0.0,
    cut_mode: str = "fixed_start",
    cut_eval: str = "l1",
    cut_samples: int = 64,
    portal_selection_policy: str = "first_found",
) -> List[Cell]:
    """Full pipeline: dual-graph edge weights → AC Hamiltonian path.

    Steps:
      1. Build block dual graph with the given weights.
      2. Compute a merge backbone tree.
      3. Build BRGC circuits (Cover phase).
      4. Apply the tree via 2-edge splices (Merge phase).
      5. Cut one edge to obtain a Hamiltonian path.

    Args:
        grid: Grid object (defines d, K, block structure).
        weights: Float array of shape ``(num_dual_edges,)`` indexed by
            ``grid.dual_edges`` order.
        start: Starting cell for the path (default: all-zeros cell).
        merge_policy: Merge fallback policy passed to ``apply_mst_nd``.
        tree_branch_penalty: Dynamic degree penalty used by the locality-aware
            Prim prototype. ``0.0`` recovers the original behavior.
        tree_hilbert_penalty: Static penalty for Hilbert-distant neighboring
            blocks when scoring tree edges. ``0.0`` disables it.
        cut_mode: Path extraction mode. ``fixed_start`` keeps the original
            behavior. ``sampled_start_cuts`` tests multiple cuts adjacent to
            the start cell and keeps the best one under ``cut_eval``.
        cut_eval: Surrogate used for cut selection.
        cut_samples: Maximum number of cut candidates to evaluate.
        portal_selection_policy: how to choose among multiple valid merge
            portals. ``first_found`` keeps the original scan-order behavior,
            while ``hilbert_gap`` prefers lower Hilbert discontinuity.

    Returns:
        Ordered list of cells – the SFC path. When ``return_stats`` is True,
        returns ``(path, stats)`` where ``stats`` summarizes tree/merge
        construction behavior.
    """
    if start is None:
        start = tuple(0 for _ in range(grid.d))

    dual_g = _build_dual_graph(
        grid,
        weights,
        tree_hilbert_penalty=float(tree_hilbert_penalty),
    )

    start_block = tuple(0 for _ in range(grid.d))
    if float(tree_branch_penalty) > 0.0:
        mst = _build_locality_aware_tree(
            dual_g,
            start_block=start_block,
            tree_branch_penalty=float(tree_branch_penalty),
        )
        tree_builder = "locality_aware_prim"
    else:
        mst = nx.maximum_spanning_tree(dual_g, algorithm="prim")
        tree_builder = "hilbert_weighted_maxst" if float(tree_hilbert_penalty) > 0.0 else "maxst"

    construction_stats: Dict[str, object] = {
        "tree_builder": tree_builder,
        "tree_edge_count": int(mst.number_of_edges()),
        "tree_weight_sum": float(
            sum(float(data.get("raw_weight", data.get("weight", 0.0))) for _, _, data in mst.edges(data=True))
        ),
        "tree_score_sum": float(sum(float(data.get("weight", 0.0)) for _, _, data in mst.edges(data=True))),
        "tree_hilbert_gap_sum": float(sum(float(data.get("hilbert_gap", 0.0)) for _, _, data in mst.edges(data=True))),
        "tree_max_degree": int(max((deg for _, deg in mst.degree()), default=0)),
        "tree_branch_penalty": float(tree_branch_penalty),
        "tree_hilbert_penalty": float(tree_hilbert_penalty),
        "cut_mode": str(cut_mode),
        "cut_eval": str(cut_eval),
        "cut_samples": int(cut_samples),
        "portal_selection_policy": str(portal_selection_policy),
        "edge_order": "default",
        "merge_policy": merge_policy,
        "retried_for_nac": False,
    }

    d = grid.d

    # 3. Build BRGC circuits once – apply_mst_nd copies internally so the
    #    original is safe to reuse for retries with different edge orderings.
    circuits = build_circuits_nd(grid.grid_shape)

    def _run(edge_order: str):
        """Apply the tree to a fresh copy of circuits with given edge ordering."""
        merged, merge_stats = apply_mst_nd(
            circuits,
            mst,
            grid.grid_shape,
            edge_order=edge_order,
            return_stats=True,
            merge_policy=merge_policy,
            portal_selection_policy=str(portal_selection_policy),
        )
        path, cut_stats = _extract_path_with_cut_optimization(
            merged,
            start,
            cut_mode=str(cut_mode),
            cut_eval=str(cut_eval),
            cut_samples=int(cut_samples),
        )
        stats = dict(construction_stats)
        stats["edge_order"] = edge_order
        stats.update(cut_stats)
        stats.update({
            key: int(value)
            for key, value in merge_stats.items()
        })
        stats["merge_total"] = int(sum(int(value) for value in merge_stats.values()))
        return path, stats

    def _has_nac(path: List[Cell]) -> bool:
        return any(
            sum(abs(path[i][k] - path[i + 1][k]) for k in range(d)) != 1
            for i in range(len(path) - 1)
        )

    # 4-5. Merge (default order) → path
    path, stats = _run("default")

    # If any non-AC step survived (rare last-resort splice with l1(u,up)>1),
    # retry with weight-descending then weight-ascending order. In practice
    # the two alternative orderings are complementary and cover all observed
    # edge cases.
    if _has_nac(path):
        path2, stats2 = _run("weight_desc")
        if not _has_nac(path2):
            stats2["retried_for_nac"] = True
            if return_stats:
                return path2, stats2
            return path2
        path3, stats3 = _run("weight_asc")
        if not _has_nac(path3):
            stats3["retried_for_nac"] = True
            if return_stats:
                return path3, stats3
            return path3
        stats["retried_for_nac"] = True

    if return_stats:
        return path, stats
    return path
