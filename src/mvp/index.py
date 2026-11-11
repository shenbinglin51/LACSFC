from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from .types import Interval


@dataclass(frozen=True)
class BoundaryIndex:
    page_start_codes: np.ndarray  # sorted ascending

    def lookup_interval(self, interval: Interval) -> Tuple[int, int]:
        """Return inclusive page_id range that may overlap [start,end]."""
        s = self.page_start_codes
        if s.size == 0:
            return (0, -1)

        lo = int(interval.start)
        hi = int(interval.end)

        # page whose start_code <= lo
        # IMPORTANT: use side='left' to handle duplicates in page_start_codes.
        # If many records share the same code, that code can span multiple pages;
        # we must include pages before the first page whose start_code == lo.
        i_start = int(np.searchsorted(s, lo, side="left") - 1)
        if i_start < 0:
            i_start = 0

        # last page whose start_code <= hi
        i_end = int(np.searchsorted(s, hi, side="right") - 1)
        return (i_start, i_end)

    def lookup_intervals(self, intervals: Sequence[Interval]) -> List[int]:
        pages: List[int] = []
        for itv in intervals:
            a, b = self.lookup_interval(itv)
            if a <= b:
                pages.extend(range(a, b + 1))
        # preserve order but de-duplicate
        seen = set()
        out: List[int] = []
        for p in pages:
            if p not in seen:
                seen.add(p)
                out.append(p)
        return out
