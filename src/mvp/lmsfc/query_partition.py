from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from ..types import Interval
from .zaddress import cal_z_address


@dataclass
class _QPState:
    lb: np.ndarray
    ub: np.ndarray
    lbz: int
    ubz: int
    start: int
    end: int


def _popcount(x: int) -> int:
    return int(x).bit_count()


def _highest_one_bit(x: int) -> int:
    x = int(x)
    if x <= 0:
        return 0
    return 1 << (x.bit_length() - 1)


def _search_range_lb(page_start_codes: np.ndarray, key: int) -> int:
    s = page_start_codes
    if s.size == 0:
        return 0
    key_u = np.uint64(int(key) & ((1 << 64) - 1))
    if key_u > s[-1]:
        return int(s.size - 1)
    i = int(np.searchsorted(s, key_u, side="left") - 1)
    if i < 0:
        i = 0
    return i


def _search_range_ub(page_start_codes: np.ndarray, key: int) -> int:
    s = page_start_codes
    if s.size == 0:
        return 0
    key_u = np.uint64(int(key) & ((1 << 64) - 1))
    if key_u > s[-1]:
        return int(s.size)
    return int(np.searchsorted(s, key_u, side="right"))


def _merge_intervals(intervals: Sequence[Interval]) -> List[Interval]:
    if not intervals:
        return []
    arr = sorted(intervals, key=lambda it: (int(it.start), int(it.end)))
    out: List[Interval] = [Interval(int(arr[0].start), int(arr[0].end))]
    for it in arr[1:]:
        s = int(it.start)
        e = int(it.end)
        last = out[-1]
        if s <= last.end + 1:
            out[-1] = Interval(last.start, max(last.end, e))
        else:
            out.append(Interval(s, e))
    return out


def partition_intervals(
    low_cell: np.ndarray,
    high_cell: np.ndarray,
    masks: List[int],
    page_start_codes: np.ndarray,
    threshold: int,
) -> List[Interval]:
    low_cell = np.asarray(low_cell, dtype=np.uint32)
    high_cell = np.asarray(high_cell, dtype=np.uint32)
    if low_cell.ndim != 1 or high_cell.ndim != 1:
        raise ValueError("low_cell/high_cell must be 1D")
    if low_cell.shape != high_cell.shape:
        raise ValueError("low_cell/high_cell shape mismatch")
    if low_cell.shape[0] != len(masks):
        raise ValueError("dimension mismatch with masks")

    page_start_codes = np.asarray(page_start_codes, dtype=np.uint64)

    d = int(low_cell.shape[0])
    threshold = int(threshold)

    lb = np.minimum(low_cell, high_cell)
    ub = np.maximum(low_cell, high_cell)

    lbz = cal_z_address(lb, masks)
    ubz = cal_z_address(ub, masks)
    start = _search_range_lb(page_start_codes, lbz)
    end = _search_range_ub(page_start_codes, ubz)

    root = _QPState(lb=lb, ub=ub, lbz=lbz, ubz=ubz, start=start, end=end)

    leaves: List[_QPState] = []

    def rec(q: _QPState) -> None:
        if (int(q.end) - int(q.start)) < threshold:
            leaves.append(q)
            return

        msb = _highest_one_bit(int(q.lbz) ^ int(q.ubz))
        if msb == 0:
            leaves.append(q)
            return

        for i in range(d):
            if msb & int(masks[i]):
                tmp = int(masks[i]) & (msb - 1)
                count = _popcount(tmp)

                pos = (int(q.ub[i]) >> count) << count
                if pos <= int(q.lb[i]) or pos > int(q.ub[i]) or pos == 0:
                    leaves.append(q)
                    return

                q1 = _QPState(lb=q.lb.copy(), ub=q.ub.copy(), lbz=q.lbz, ubz=q.ubz, start=q.start, end=q.end)
                q2 = _QPState(lb=q.lb.copy(), ub=q.ub.copy(), lbz=q.lbz, ubz=q.ubz, start=q.start, end=q.end)

                q1.ub[i] = np.uint32(pos - 1)
                q2.lb[i] = np.uint32(pos)

                q1.ubz = cal_z_address(q1.ub, masks)
                q2.lbz = cal_z_address(q2.lb, masks)

                q2.start = _search_range_lb(page_start_codes, q2.lbz)
                q1.end = _search_range_ub(page_start_codes, q1.ubz)

                rec(q1)
                rec(q2)
                return

        leaves.append(q)

    rec(root)

    intervals = [Interval(int(min(q.lbz, q.ubz)), int(max(q.lbz, q.ubz))) for q in leaves]
    return _merge_intervals(intervals)
