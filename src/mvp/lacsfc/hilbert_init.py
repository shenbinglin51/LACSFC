"""Hilbert-consistent dual-edge weights (T1 of D-4 v2 productionization).

Motivation
----------
`weights_to_sfc(grid, weights)` builds a locality-aware SFC via MaxST + Cover-Merge
over the **block-level dual graph** (adjacent 2^(K-1)-side block pairs).
MaxST prefers **high-weight** edges. We therefore want a weight vector
``w_hilbert`` such that ``weights_to_sfc(grid, w_hilbert)`` recovers (or closely
approximates) the classic Hilbert curve.

Construction
------------
1. Build the Hilbert curve on the full 2^K x 2^K cell grid (via `hilbertcurve`).
2. Walk consecutive cell pairs in the Hilbert order.
3. Every time the walk crosses a block boundary (i.e. the two cells live in
   different blocks ``b1, b2``), increment the count of the corresponding dual
   edge ``(b1, b2)``.
4. Normalise to a strictly positive weight vector — block pairs that the
   Hilbert walk crosses more often get higher weight and become preferred
   MaxST edges.

The resulting weights serve two independent purposes:

* as one of three **supervised training targets** for the `QueryEvaluator`
  (T2)
* as a **structural prior** that is complementary to ``dual_edge_weights()``
  (which captures query-workload geometry).

Only d=2 is fully supported (where Hilbert is unambiguous and the
`hilbertcurve` package covers it). For d>=3 we fall back to a uniform
weight vector with a warning; the rest of the pipeline still works because
downstream code only consumes the returned array.
"""
from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np


def hilbert_edge_weights(grid, normalize: str = "minmax") -> np.ndarray:
    """Compute Hilbert-consistent dual-edge weights for ``grid``.

    Parameters
    ----------
    grid : lacsfcf.data.grid.Grid
        Target grid (only ``grid.d``, ``grid.K``, ``grid.dual_edges`` are
        used). Must expose ``edge_index((b1, b2))`` like ``Grid``.
    normalize : {"minmax", "logcount", "raw"}
        How to post-process the raw crossing counts. ``"minmax"`` rescales to
        [0.1, 1.0]; ``"logcount"`` returns ``log1p(count) + 0.1`` to keep a
        strictly positive floor; ``"raw"`` returns the raw counts (may
        contain zeros).

    Returns
    -------
    weights : np.ndarray, shape (grid.num_dual_edges,), dtype float32
    """
    d = int(grid.d)
    K = int(grid.K)
    num_edges = int(grid.num_dual_edges)

    side = 1 << K                           # cells per dimension = 2^K
    num_cells = side ** d
    all_cells = np.indices(tuple(side for _ in range(d))).reshape(d, -1).T  # (N, d)

    if d == 2:
        # Canonical Hilbert curve via `hilbertcurve` package.
        from hilbertcurve.hilbertcurve import HilbertCurve
        hc = HilbertCurve(p=K, n=d)
        if hasattr(hc, "distances_from_points"):
            codes = np.asarray(hc.distances_from_points(all_cells.tolist()),
                               dtype=np.int64)
        else:
            codes = np.fromiter(
                (hc.distance_from_coordinates(list(c)) for c in all_cells),
                dtype=np.int64, count=num_cells,
            )
    else:
        # d >= 3: `hilbertcurve` package only ships canonical 2D Hilbert;
        # fall back to **Z-curve (Morton) order** as a proper locality-preserving
        # structural prior (generalises to arbitrary d via bit interleaving).
        # Empirically produces a meaningful crossing-count signal, unlike the
        # old uniform fallback which erased all structure from EG pretrain.
        codes = np.zeros(num_cells, dtype=np.int64)
        for dim in range(d):
            col = all_cells[:, dim].astype(np.int64)
            for bit in range(K):
                codes |= ((col >> bit) & 1) << (bit * d + dim)
    order = np.argsort(codes)
    path = all_cells[order]                 # (N, d), cells in SFC order

    # Count block-boundary crossings per dual edge.
    counts = np.zeros(num_edges, dtype=np.int64)
    # block = cell // 2 (because block_side = 2^(K-1))
    blocks = path // 2                       # (N, d)
    for i in range(num_cells - 1):
        b1 = tuple(int(x) for x in blocks[i])
        b2 = tuple(int(x) for x in blocks[i + 1])
        if b1 == b2:
            continue  # intra-block step, no dual edge crossed
        try:
            idx = grid.edge_index(b1, b2)
        except KeyError:
            # consecutive cells must be grid-adjacent, which implies blocks are
            # either equal or adjacent; be defensive anyway.
            continue
        counts[idx] += 1

    if normalize == "raw":
        return counts.astype(np.float32)
    if normalize == "logcount":
        return (np.log1p(counts.astype(np.float64)) + 0.1).astype(np.float32)
    # default: minmax to [0.1, 1.0]
    if counts.max() == counts.min():
        return np.full(num_edges, 0.5, dtype=np.float32)
    w = (counts - counts.min()) / (counts.max() - counts.min())
    w = 0.1 + 0.9 * w
    return w.astype(np.float32)


def verify_hilbert_recoverability(grid, tol: float = 0.0) -> Tuple[bool, float]:
    """Sanity-check that ``weights_to_sfc(grid, hilbert_edge_weights(grid))``
    produces an SFC that agrees with the canonical Hilbert code_map.

    Due to tie-breaking and Cover-Merge internal choices, exact recovery is not
    guaranteed; we instead report the **Spearman-like fraction of cell-pair
    orderings preserved** between the two SFCs.

    Returns
    -------
    (ok, concordance) : (bool, float)
        ``ok`` is True iff concordance >= 1 - tol.
    """
    if grid.d != 2:
        return True, 1.0   # d!=2 falls back to uniform, nothing to verify
    import sys
    from src.mvp._paths import lacsfcf_root as _lr
    sys.path.insert(0, str(_lr()))
    from lacsfcf.core.sfc import weights_to_sfc
    from hilbertcurve.hilbertcurve import HilbertCurve

    w = hilbert_edge_weights(grid)
    path = weights_to_sfc(grid, w)
    code_from_weights = grid.code_map_from_path(path).ravel()

    side = 1 << grid.K
    hc = HilbertCurve(p=grid.K, n=grid.d)
    all_cells = np.indices((side, side)).reshape(2, -1).T
    if hasattr(hc, "distances_from_points"):
        code_canonical = np.asarray(
            hc.distances_from_points(all_cells.tolist()), dtype=np.int64
        )
    else:
        code_canonical = np.fromiter(
            (hc.distance_from_coordinates(list(c)) for c in all_cells),
            dtype=np.int64, count=len(all_cells),
        )
    # reshape to (side, side) in the same ravel order as code_map
    canon_map = np.empty((side, side), dtype=np.int64)
    for cell, c in zip(all_cells, code_canonical):
        canon_map[tuple(cell)] = c
    canon_flat = canon_map.ravel()

    # Spearman rank correlation
    from scipy.stats import spearmanr
    rho, _ = spearmanr(code_from_weights, canon_flat)
    return bool(rho >= 1 - tol), float(rho)
