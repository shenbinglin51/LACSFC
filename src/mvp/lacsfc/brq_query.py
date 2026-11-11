from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np


_CPP_DIR = Path(__file__).resolve().parent / "cpp"
if str(_CPP_DIR) not in sys.path:
    sys.path.insert(0, str(_CPP_DIR))


class LACSFCBRQQuery:
    def __init__(
        self,
        path: Sequence[Sequence[int]],
        grid_size: int,
        d: int,
        hybrid: bool = False,
        hybrid_threshold: int = 128,
    ):
        self.grid_size = int(grid_size)
        self.d = int(d)
        self.hybrid = bool(hybrid)
        self.hybrid_threshold = int(hybrid_threshold)
        self.path = np.asarray(path, dtype=np.int64)

        if self.path.ndim != 2 or self.path.shape[1] != self.d:
            raise ValueError("path must have shape (num_cells, d)")
        if self.d < 2 or self.d > 6:
            raise ValueError("AC-BRQ v2 Python binding currently supports 2 <= d <= 6")

        try:
            import lacsfc_brq
        except ImportError as exc:
            raise ImportError(
                "AC-BRQ v2 backend requested but the lacsfc_brq extension is not built. "
                "Build it from src/mvp/lacsfc/cpp with: "
                "python setup_brq.py build_ext --inplace"
            ) from exc

        cls = getattr(lacsfc_brq, f"BRQIndex{self.d}")
        self._index = cls(np.ascontiguousarray(self.path, dtype=np.int64), self.grid_size)

    def get_intervals(self, low_cell: Tuple[int, ...], high_cell: Tuple[int, ...]) -> List[Tuple[int, int]]:
        low = [int(x) for x in low_cell]
        high = [int(x) for x in high_cell]
        if self.hybrid:
            segments = self._index.query_segments_hybrid(low, high, self.hybrid_threshold)
        else:
            segments = self._index.query_segments(low, high)
        return [(int(s), int(e)) for s, e in segments]

    def index_stats(self) -> dict:
        return {
            "bitvector_count": int(self._index.index_bitvector_count()),
            "raw_bits": int(self._index.index_raw_bits()),
        }
