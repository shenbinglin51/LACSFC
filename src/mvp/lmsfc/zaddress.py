from __future__ import annotations

from typing import List

import numpy as np


def _pdep_u64(value: int, mask: int) -> int:
    value = int(value)
    mask = int(mask)
    res = 0
    bit = 1
    while mask:
        lsb = mask & -mask
        if value & bit:
            res |= lsb
        mask ^= lsb
        bit <<= 1
    return res


def cal_z_address(cell: np.ndarray, masks: List[int]) -> int:
    cell = np.asarray(cell, dtype=np.uint64)
    if cell.ndim != 1:
        raise ValueError("cell must be 1D")
    if cell.shape[0] != len(masks):
        raise ValueError("cell dimension mismatch")

    z = 0
    for i, m in enumerate(masks):
        z |= _pdep_u64(int(cell[i]), int(m))
    return int(z & ((1 << 64) - 1))
