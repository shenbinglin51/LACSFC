from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ..types import Interval, QuantizedQuery, SFCPlugin
from .masks import standard_morton_masks, validate_masks
from .params import PARTITION_PAGE_RANGE_THRESHOLD
from .query_partition import partition_intervals
from .zaddress import cal_z_address


@dataclass
class LMSFCPlugin(SFCPlugin):
    name: str
    grid_size: Tuple[int, ...]
    masks: List[int]
    threshold: int = PARTITION_PAGE_RANGE_THRESHOLD
    _page_start_codes: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        validate_masks(self.masks, len(self.masks))

    @classmethod
    def make_standard(cls, d: int, k: int, name: str = "LMSFC") -> "LMSFCPlugin":
        m = 1 << int(k)
        masks = standard_morton_masks(int(d))
        return cls(name=name, grid_size=tuple([m] * int(d)), masks=masks)

    def set_page_start_codes(self, page_start_codes: np.ndarray) -> None:
        self._page_start_codes = np.asarray(page_start_codes, dtype=np.uint64)
        # 自适应 threshold：若固定常量大于总页数，递归永远在第一层停止（seg 恒=1）
        # 调整为总页数的 1/4，确保递归能真正分割，最小为 1
        total_pages = int(self._page_start_codes.size)
        adaptive = max(1, total_pages // 4)
        if adaptive < self.threshold:
            self.threshold = adaptive

    def encode_cell(self, cell: np.ndarray) -> int:
        cell = np.asarray(cell, dtype=np.uint64)
        if cell.ndim != 1 or cell.shape[0] != len(self.masks):
            raise ValueError("cell dimension mismatch")
        return int(cal_z_address(cell, self.masks))

    def intervals(self, q: QuantizedQuery) -> Optional[List[Interval]]:
        if self._page_start_codes is None:
            return None
        low = np.asarray(q.low_cell, dtype=np.uint32)
        high = np.asarray(q.high_cell, dtype=np.uint32)
        out = partition_intervals(
            low_cell=low,
            high_cell=high,
            masks=self.masks,
            page_start_codes=self._page_start_codes,
            threshold=int(self.threshold),
        )
        return out
