"""
Query evaluation utilities for BitAllocLZ.
Computes page_access / segment / candidate costs given a BitAllocZCurve.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from .curve import BitAllocZCurve


def _enumerate_cells(
    low_cell: np.ndarray, high_cell: np.ndarray, max_cells: int
) -> np.ndarray:
    low = np.asarray(low_cell, dtype=np.int64)
    high = np.asarray(high_cell, dtype=np.int64)
    d = int(low.shape[0])
    sizes = high - low + 1
    total = int(np.prod(sizes))
    if total > int(max_cells):
        raise ValueError(f"enumeration too large: {total} cells (max_cells={max_cells})")
    grids = [np.arange(int(low[i]), int(high[i]) + 1, dtype=np.int64) for i in range(d)]
    mesh = np.meshgrid(*grids, indexing="ij")
    return np.stack([m.reshape(-1) for m in mesh], axis=1)


def _merge_sorted_codes(codes: np.ndarray) -> List[Tuple[int, int]]:
    """Merge sorted code array into (start, end) interval list."""
    if codes.size == 0:
        return []
    start = int(codes[0])
    end = int(codes[0])
    out: List[Tuple[int, int]] = []
    for v in codes[1:]:
        iv = int(v)
        if iv <= end + 1:
            end = iv
        else:
            out.append((start, end))
            start = iv
            end = iv
    out.append((start, end))
    return out


def intervals_from_query(
    *,
    curve: BitAllocZCurve,
    low_cell: np.ndarray,
    high_cell: np.ndarray,
    max_cells: int,
) -> List[Tuple[int, int]]:
    """Return Z-code intervals covering all cells in [low_cell, high_cell]."""
    cells = _enumerate_cells(low_cell, high_cell, max_cells)
    codes = curve.encode_cells(cells)
    codes.sort()
    return _merge_sorted_codes(codes)


def _count_pages(
    intervals: List[Tuple[int, int]],
    page_start_codes: np.ndarray,
) -> int:
    """Count distinct pages touched by the given code intervals.

    page_start_codes[i] is the first code in page i (sorted ascending).
    Page i spans [page_start_codes[i], page_start_codes[i+1]-1].
    """
    if len(page_start_codes) == 0 or not intervals:
        return 0
    pages: set = set()
    for s, e in intervals:
        # first page whose start <= s  =>  searchsorted(side='right') - 1
        first = int(np.searchsorted(page_start_codes, s, side="right")) - 1
        if first < 0:
            first = 0
        # last page whose start <= e
        last = int(np.searchsorted(page_start_codes, e, side="right")) - 1
        if last < 0:
            continue
        last = min(last, len(page_start_codes) - 1)
        pages.update(range(first, last + 1))
    return len(pages)


def build_page_start_codes(
    points: np.ndarray,
    curve: BitAllocZCurve,
    quantizer,
    b: int,
) -> np.ndarray:
    """Quantize points, encode with curve, sort, return page boundary codes."""
    cells = quantizer.quantize_points(points)
    codes = curve.encode_cells(cells)
    codes_sorted = np.sort(codes)
    return codes_sorted[::int(b)].astype(np.int64)


def eval_workload_cost(
    *,
    curve: BitAllocZCurve,
    points: np.ndarray,
    queries: Sequence,
    b: int,
    max_cells: int,
    w_seg: float,
    w_pa: float,
    w_cand: float,
    quantizer,
) -> float:
    """Compute weighted average workload cost for a given curve.

    Cost = w_seg * avg_segments + w_pa * avg_page_accesses + w_cand * avg_candidates

    queries: sequence of objects with low_cell / high_cell attributes (QuantizedQuery)
             OR raw (QueryWindow, QuantizedQuery) tuple pairs.
    """
    page_start_codes = build_page_start_codes(points, curve, quantizer, b)

    total_seg = 0
    total_pa = 0
    total_cand = 0
    n_q = 0

    for qobj in queries:
        if hasattr(qobj, "low_cell"):
            low_cell = np.asarray(qobj.low_cell, dtype=np.int64)
            high_cell = np.asarray(qobj.high_cell, dtype=np.int64)
        else:
            raise TypeError(f"Unsupported query type: {type(qobj)}")

        try:
            intervals = intervals_from_query(
                curve=curve,
                low_cell=low_cell,
                high_cell=high_cell,
                max_cells=max_cells,
            )
        except ValueError:
            continue

        if not intervals:
            continue

        n_seg = len(intervals)
        n_pa = _count_pages(intervals, page_start_codes)

        total_seg += n_seg
        total_pa += n_pa
        n_q += 1

    if n_q == 0:
        return 0.0

    cost = (
        float(w_seg) * float(total_seg) / n_q
        + float(w_pa) * float(total_pa) / n_q
        + float(w_cand) * float(total_cand) / n_q
    )
    return float(cost)
