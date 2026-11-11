from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .types import QuantizedQuery, QueryWindow


@dataclass(frozen=True)
class Quantizer:
    d: int
    k: int = 8

    @property
    def m(self) -> int:
        return 1 << self.k

    def quantize_points(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != self.d:
            raise ValueError(f"points must have shape (n,{self.d})")
        m = self.m
        cells = np.floor(points * m).astype(np.int64)
        cells = np.clip(cells, 0, m - 1)
        return cells

    def quantize_window(self, q: QueryWindow) -> QuantizedQuery:
        low = np.asarray(q.low, dtype=np.float64)
        high = np.asarray(q.high, dtype=np.float64)
        if low.shape != (self.d,) or high.shape != (self.d,):
            raise ValueError(f"query bounds must have shape ({self.d},)")
        m = self.m
        low_cell = np.floor(low * m).astype(np.int64)
        high_cell = np.floor(high * m).astype(np.int64)
        low_cell = np.clip(low_cell, 0, m - 1)
        high_cell = np.clip(high_cell, 0, m - 1)
        low_cell = np.minimum(low_cell, high_cell)
        high_cell = np.maximum(low_cell, high_cell)
        return QuantizedQuery(low_cell=low_cell, high_cell=high_cell)
