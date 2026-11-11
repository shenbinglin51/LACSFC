"""
Sort Dimension Paging for LMSFC

This module implements the Sort Dimension optimization from the LMSFC paper.
Each page selects an optimal dimension to sort its records, enabling binary search
along that dimension during query processing.

Key improvement: Uses Query Partition to select Sort Dimension based on partitioned queries.
"""

from __future__ import annotations
from typing import List, Sequence, Tuple
import numpy as np
from .types import Page, QueryWindow, Record, QuantizedQuery
from .quantizer import Quantizer
from .query_partition import partition_query


def build_pages_with_sortdim(
    records_sorted: Sequence[Record],
    b: int,
    training_queries: Sequence[QueryWindow],
    quantizer: Quantizer,
    masks: List[int],
) -> Tuple[Page, ...]:
    """
    Build pages with Sort Dimension selection.
    
    Key improvement: Uses Query Partition to select Sort Dimension based on
    partitioned queries, which is how the original LMSFC paper does it.
    
    Args:
        records_sorted: Records sorted by Z-order code
        b: Page size
        training_queries: Training queries for Sort Dimension selection
        quantizer: Quantizer for converting coordinates
        masks: LMSFC bit masks
    
    Returns:
        Tuple of pages with Sort Dimension selected
    """
    n = len(records_sorted)
    pages: List[Page] = []
    
    # Partition all training queries
    partitioned_queries = []
    for q in training_queries:
        try:
            partitioned = partition_query(q, quantizer, masks, threshold=500)
            partitioned_queries.extend(partitioned)
        except Exception:
            # If partition fails, use original query
            qq = quantizer.quantize_window(q)
            partitioned_queries.append(qq)
    
    # Build pages
    for start in range(0, n, b):
        chunk = list(records_sorted[start : min(start + b, n)])
        if not chunk:
            continue
        
        page_id = len(pages)
        
        # Select Sort Dimension for this page
        sort_dim = _select_sort_dim_for_page(
            chunk, partitioned_queries, quantizer
        )
        
        # Sort by selected dimension if valid
        if sort_dim >= 0:
            chunk_sorted = sorted(
                chunk, 
                key=lambda r: r.cell[sort_dim] if r.cell is not None else 0
            )
        else:
            chunk_sorted = chunk
        
        pages.append(Page(page_id=page_id, records=tuple(chunk_sorted), sort_dim=sort_dim))
    
    return tuple(pages)


def _select_sort_dim_for_page(
    records: List[Record],
    partitioned_queries: Sequence[QuantizedQuery],
    quantizer: Quantizer,
) -> int:
    """
    Select the optimal Sort Dimension for a page.
    
    Strategy: Choose the dimension with the smallest overlap ratio across
    partitioned queries. This dimension provides the best selectivity for
    binary search.
    
    Args:
        records: Records in the page
        partitioned_queries: Partitioned training queries
        quantizer: Quantizer
    
    Returns:
        Selected dimension index, or -1 if no good dimension found
    """
    if not records or not partitioned_queries:
        return -1
    
    # Get dimensionality
    d = quantizer.d
    
    # Calculate page MBR
    cells = [r.cell for r in records if r.cell is not None]
    if not cells:
        return -1
    
    cells_array = np.array(cells, dtype=np.int64)
    page_min = np.min(cells_array, axis=0)
    page_max = np.max(cells_array, axis=0)
    
    # Count overlaps for each dimension
    overlap_counts = np.zeros(d, dtype=np.float64)
    total_queries = 0
    
    for qq in partitioned_queries:
        q_min = np.asarray(qq.low_cell, dtype=np.int64)
        q_max = np.asarray(qq.high_cell, dtype=np.int64)
        
        # Check if query overlaps with page
        overlaps = True
        for i in range(d):
            if q_max[i] < page_min[i] or q_min[i] > page_max[i]:
                overlaps = False
                break
        
        if not overlaps:
            continue
        
        total_queries += 1
        
        # Calculate overlap ratio for each dimension
        for i in range(d):
            overlap_min = max(q_min[i], page_min[i])
            overlap_max = min(q_max[i], page_max[i])
            
            if overlap_max >= overlap_min:
                overlap_len = overlap_max - overlap_min + 1
                page_len = page_max[i] - page_min[i] + 1
                
                if page_len > 0:
                    overlap_ratio = float(overlap_len) / float(page_len)
                    overlap_counts[i] += overlap_ratio
    
    if total_queries == 0:
        return -1
    
    # Average overlap ratio per dimension
    avg_overlaps = overlap_counts / total_queries
    
    # Select dimension with minimum average overlap (best selectivity)
    best_dim = int(np.argmin(avg_overlaps))
    min_overlap = float(avg_overlaps[best_dim])
    avg_overlap = float(np.mean(avg_overlaps))
    
    # If minimum is close to average, no clear winner
    if min_overlap > 0.8 * avg_overlap:
        return -1
    
    return best_dim


def get_page_start_codes(pages: Sequence[Page]) -> np.ndarray:
    """
    Extract page start codes from pages.
    
    Args:
        pages: Sequence of pages
    
    Returns:
        Array of start codes
    """
    codes = []
    for p in pages:
        if p.records:
            codes.append(p.records[0].code if p.records[0].code is not None else 0)
        else:
            codes.append(0)
    return np.array(codes, dtype=np.int64)
