"""
Query Partition for LMSFC

This module implements the query partitioning algorithm from the LMSFC paper.
Query partitioning splits large query windows into smaller sub-queries to reduce
unnecessary page accesses and improve query performance.
"""

from __future__ import annotations
from typing import List, Tuple
import numpy as np
from .types import QueryWindow, QuantizedQuery
from .quantizer import Quantizer


def partition_query(
    query: QueryWindow,
    quantizer: Quantizer,
    masks: List[int],
    threshold: int = 500,
) -> List[QuantizedQuery]:
    """
    Partition a query window into multiple sub-queries.
    
    This is a key optimization in LMSFC that reduces the number of pages accessed
    by splitting large queries into smaller, more precise sub-queries.
    
    Args:
        query: Original query window
        quantizer: Quantizer for converting to cell coordinates
        masks: LMSFC bit masks for each dimension
        threshold: Partition threshold (estimated page range size)
    
    Returns:
        List of partitioned sub-queries
    """
    qq = quantizer.quantize_window(query)
    low_cell = np.asarray(qq.low_cell, dtype=np.uint64)
    high_cell = np.asarray(qq.high_cell, dtype=np.uint64)
    
    # Calculate Z-order codes
    low_code = encode_zorder(low_cell, masks)
    high_code = encode_zorder(high_cell, masks)
    
    # Recursively partition
    result = []
    _partition_recursive(low_cell, high_cell, low_code, high_code, masks, threshold, result)
    
    return [QuantizedQuery(low_cell=tuple(low), high_cell=tuple(high)) 
            for low, high in result]


def _partition_recursive(
    low_cell: np.ndarray,
    high_cell: np.ndarray,
    low_code: int,
    high_code: int,
    masks: List[int],
    threshold: int,
    result: List[Tuple[np.ndarray, np.ndarray]],
):
    """
    Recursively partition a query based on Z-order code differences.
    
    The algorithm finds the most significant bit (MSB) where the low and high
    Z-order codes differ, determines which dimension that bit belongs to,
    and splits the query along that dimension.
    """
    # Estimate page range (simplified: use code range as proxy)
    code_range = high_code - low_code
    if code_range < threshold:
        result.append((low_cell.copy(), high_cell.copy()))
        return
    
    # Find the MSB where codes differ
    xor = low_code ^ high_code
    msb = _find_msb(xor)
    
    if msb == 0:
        result.append((low_cell.copy(), high_cell.copy()))
        return
    
    d = len(masks)
    for i in range(d):
        if msb & masks[i]:
            # Split along dimension i
            # Calculate split position
            mask_bits = masks[i] & (msb - 1)
            count = bin(mask_bits).count('1')
            
            # Split point
            split_pos = (int(high_cell[i]) >> count) << count
            
            # Ensure split is within bounds
            if split_pos <= int(low_cell[i]) or split_pos > int(high_cell[i]):
                result.append((low_cell.copy(), high_cell.copy()))
                return
            
            # Create two sub-queries
            low1, high1 = low_cell.copy(), high_cell.copy()
            low2, high2 = low_cell.copy(), high_cell.copy()
            
            high1[i] = split_pos - 1
            low2[i] = split_pos
            
            # Recursively partition
            code1_low = encode_zorder(low1, masks)
            code1_high = encode_zorder(high1, masks)
            code2_low = encode_zorder(low2, masks)
            code2_high = encode_zorder(high2, masks)
            
            _partition_recursive(low1, high1, code1_low, code1_high, masks, threshold, result)
            _partition_recursive(low2, high2, code2_low, code2_high, masks, threshold, result)
            return
    
    # If no split dimension found, return as is
    result.append((low_cell.copy(), high_cell.copy()))


def _find_msb(x: int) -> int:
    """
    Find the most significant bit (MSB) of x.
    
    Uses bit manipulation to efficiently find the highest set bit.
    """
    if x == 0:
        return 0
    x |= x >> 1
    x |= x >> 2
    x |= x >> 4
    x |= x >> 8
    x |= x >> 16
    x |= x >> 32
    return (x + 1) >> 1


def encode_zorder(cell: np.ndarray, masks: List[int]) -> int:
    """
    Encode a cell coordinate as a Z-order code using LMSFC masks.
    
    Args:
        cell: Cell coordinates (d-dimensional)
        masks: LMSFC bit masks for each dimension
    
    Returns:
        Z-order code
    """
    code = 0
    for i, mask in enumerate(masks):
        code |= _pdep(mask, int(cell[i]))
    return code


def _pdep(mask: int, value: int) -> int:
    """
    PDEP operation (Parallel Deposit).
    
    Deposits bits from value into positions specified by mask.
    This is equivalent to the BMI2 PDEP instruction.
    
    Example:
        mask  = 0b00101010
        value = 0b00001111
        result = 0b00001010 (bits of value deposited at positions of 1s in mask)
    """
    result = 0
    bit_pos = 0
    for i in range(64):
        if mask & (1 << i):
            if value & (1 << bit_pos):
                result |= (1 << i)
            bit_pos += 1
    return result
