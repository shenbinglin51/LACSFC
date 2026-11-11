from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from .paging import build_pages
from .paging_sortdim import build_pages_with_sortdim, get_page_start_codes
from .paging_adaptive import build_pages_adaptive
from .quantizer import Quantizer
from .types import LayoutArtifacts, Record, SFCPlugin, QueryWindow


def build_artifacts(
    points: np.ndarray,
    sfc: SFCPlugin,
    quantizer: Quantizer,
    b: int,
    rids: Optional[Sequence[int]] = None,
    use_sortdim: bool = False,
    training_queries: Optional[Sequence[QueryWindow]] = None,
    use_adaptive_paging: bool = False,
    adaptive_alpha: float = 0.8,
    adaptive_b_min: int = 32,
    adaptive_b_max: int = 512,
) -> LayoutArtifacts:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != quantizer.d:
        raise ValueError(f"points must have shape (n,{quantizer.d})")

    n = points.shape[0]
    if rids is None:
        rids = range(n)
    if len(rids) != n:
        raise ValueError("rids length must match number of points")

    cells = quantizer.quantize_points(points)

    codes: np.ndarray
    point_encoder = getattr(sfc, "encode_points", None)
    if callable(point_encoder):
        codes = np.asarray(point_encoder(points, cells), dtype=np.int64)
    else:
        encoder = getattr(sfc, "_encoder", None)
        if encoder is not None and hasattr(encoder, "encode"):
            codes = np.asarray(encoder.encode(cells), dtype=np.int64)
        else:
            codes = np.empty(n, dtype=np.int64)
            for i in range(n):
                codes[i] = int(sfc.encode_cell(cells[i]))

    records: List[Record] = []
    for i, rid in enumerate(rids):
        records.append(Record(rid=int(rid), x=points[i], cell=cells[i], code=int(codes[i])))

    order = np.argsort(codes, kind="mergesort")
    records_sorted = [records[int(i)] for i in order]

    # Use Sort Dimension if requested and training queries provided
    if use_adaptive_paging:
        pages, page_start_codes = build_pages_adaptive(
            records_sorted, b=b,
            b_min=adaptive_b_min, b_max=adaptive_b_max,
            alpha=adaptive_alpha,
        )
    elif use_sortdim and training_queries:
        # Get masks from SFC
        masks = getattr(sfc, 'masks', None)
        if masks is not None:
            pages = build_pages_with_sortdim(
                records_sorted, 
                b=b, 
                training_queries=training_queries, 
                quantizer=quantizer,
                masks=masks,
            )
            page_start_codes = get_page_start_codes(pages)
        else:
            # Fallback to standard paging if no masks
            pages, page_start_codes = build_pages(records_sorted, b=b)
    else:
        pages, page_start_codes = build_pages(records_sorted, b=b)

    return LayoutArtifacts(
        d=quantizer.d,
        k=quantizer.k,
        m=quantizer.m,
        b=b,
        sfc_name=getattr(sfc, "name", sfc.__class__.__name__),
        pages=pages,
        page_start_codes=page_start_codes,
    )
