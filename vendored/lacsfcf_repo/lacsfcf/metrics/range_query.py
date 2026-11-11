"""Range-query evaluation metrics for LACSFCF.

All metrics operate on a *code_map* (ndarray mapping cell coords → SFC code)
and a list of Query objects.
"""
import math
from typing import Dict, List, Optional

import numpy as np

from lacsfcf.data.query_workload import Query


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _hit_codes(code_map: np.ndarray, q: Query) -> np.ndarray:
    """Return sorted 1-D array of SFC codes hit by query q."""
    slices = tuple(slice(q.lo[k], q.hi[k] + 1) for k in range(len(q.lo)))
    codes  = code_map[slices].ravel().copy()
    codes.sort()
    return codes


def _count_segments(sorted_codes: np.ndarray) -> int:
    """Count contiguous segments in a sorted integer array."""
    if sorted_codes.size == 0:
        return 0
    return int(1 + np.sum(np.diff(sorted_codes) > 1))


# ---------------------------------------------------------------------------
# Public metric functions
# ---------------------------------------------------------------------------

def mean_seg(code_map: np.ndarray, queries: List[Query]) -> float:
    """Mean number of contiguous segments (#intervals) over queries.

    Lower is better.
    """
    if not queries:
        return 0.0
    vals = [_count_segments(_hit_codes(code_map, q)) for q in queries]
    return float(np.mean(vals))


def mean_span(code_map: np.ndarray, queries: List[Query]) -> float:
    """Mean span = max(code) - min(code) + 1 over queries.

    Lower is better.
    """
    if not queries:
        return 0.0
    vals = []
    for q in queries:
        codes = _hit_codes(code_map, q)
        if codes.size == 0:
            vals.append(0)
        else:
            vals.append(int(codes[-1]) - int(codes[0]) + 1)
    return float(np.mean(vals))


def mean_span_per_cell(code_map: np.ndarray, queries: List[Query]) -> float:
    """Mean (span / #hit_cells): code-interval sparsity.

    A value ≈ 1 means nearly contiguous; large values indicate holes.
    Lower is better.
    """
    if not queries:
        return 0.0
    vals = []
    for q in queries:
        codes = _hit_codes(code_map, q)
        if codes.size <= 1:
            vals.append(1.0)
            continue
        span    = int(codes[-1]) - int(codes[0]) + 1
        n_cells = codes.size
        vals.append(span / n_cells)
    return float(np.mean(vals))


def mean_page_amp(
    code_map:  np.ndarray,
    queries:   List[Query],
    page_size: int = 64,
) -> float:
    """Mean page amplification ratio over queries.

    page_amp(q) = pages_touched(q) / ceil(hits(q) / page_size)

    page_amp = 1.0  → ideal (no wasted pages)
    page_amp > 1.0  → wasted I/O

    Lower is better.
    """
    if not queries:
        return 1.0

    vals = []
    for q in queries:
        codes = _hit_codes(code_map, q)
        hits  = codes.size
        if hits == 0:
            continue
        pages_min     = math.ceil(hits / page_size)
        pages_touched = len(set(int(c) // page_size for c in codes))
        vals.append(pages_touched / pages_min)

    return float(np.mean(vals)) if vals else 1.0


def evaluate_all(
    code_map:  np.ndarray,
    queries:   List[Query],
    page_size: int = 64,
) -> Dict[str, float]:
    """Compute all metrics and return as a dict."""
    return {
        "mean_seg":          mean_seg(code_map, queries),
        "mean_span":         mean_span(code_map, queries),
        "mean_span_per_cell": mean_span_per_cell(code_map, queries),
        "mean_page_amp":     mean_page_amp(code_map, queries, page_size),
    }
