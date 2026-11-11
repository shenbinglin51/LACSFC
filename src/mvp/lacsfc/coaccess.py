from __future__ import annotations

from typing import Sequence

import numpy as np


def empirical_block_coaccess(queries: Sequence[object], grid: object, normalize: bool = True) -> np.ndarray:
    """Estimate adjacent block co-access weights from quantized query boxes.

    The returned vector follows ``grid.dual_edges`` order and is therefore
    compatible with ``weights_to_sfc(grid, weights)``.
    """
    block_shape = tuple(int(x) for x in grid.block_shape)
    weights = np.zeros(int(grid.num_dual_edges), dtype=np.float64)
    if not queries:
        return weights.astype(np.float32)
    edge_index = getattr(grid, "edge_index", None)

    for q in queries:
        lo = np.asarray(getattr(q, "lo"), dtype=np.int64)
        hi = np.asarray(getattr(q, "hi"), dtype=np.int64)
        qlo = np.minimum(lo, hi)
        qhi = np.maximum(lo, hi)
        blo = np.asarray(grid.cell_to_block(tuple(int(x) for x in qlo))[0], dtype=np.int64)
        bhi = np.asarray(grid.cell_to_block(tuple(int(x) for x in qhi))[0], dtype=np.int64)
        blo = np.clip(blo, 0, np.asarray(block_shape, dtype=np.int64) - 1)
        bhi = np.clip(bhi, 0, np.asarray(block_shape, dtype=np.int64) - 1)

        ranges = [range(int(blo[dim]), int(bhi[dim]) + 1) for dim in range(len(block_shape))]
        for b1 in np.ndindex(*(len(r) for r in ranges)):
            b1_coord = tuple(int(ranges[dim][b1[dim]]) for dim in range(len(block_shape)))
            for axis in range(len(block_shape)):
                b2_list = list(b1_coord)
                b2_list[axis] += 1
                if b2_list[axis] > int(bhi[axis]):
                    continue
                b2_coord = tuple(b2_list)
                idx = None
                if callable(edge_index):
                    try:
                        idx = int(edge_index(b1_coord, b2_coord))
                    except Exception:
                        idx = None
                elif edge_index is not None:
                    idx = edge_index.get((b1_coord, b2_coord))
                    if idx is None:
                        idx = edge_index.get((b2_coord, b1_coord))
                if idx is None:
                    idx = None
                    for j, (e1, e2, _axis) in enumerate(grid.dual_edges):
                        if (tuple(e1) == b1_coord and tuple(e2) == b2_coord) or (tuple(e1) == b2_coord and tuple(e2) == b1_coord):
                            idx = j
                            break
                if idx is not None:
                    weights[int(idx)] += 1.0

    weights /= float(len(queries))
    if normalize and weights.size:
        mn = float(np.min(weights))
        mx = float(np.max(weights))
        if mx > mn:
            weights = (weights - mn) / (mx - mn)
    return weights.astype(np.float32)
