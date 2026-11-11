from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from .types import Page, Record


def build_pages(records_sorted: Sequence[Record], b: int) -> Tuple[Tuple[Page, ...], np.ndarray]:
    if b <= 0:
        raise ValueError("b must be > 0")

    pages: List[Page] = []
    page_start_codes: List[int] = []

    n = len(records_sorted)
    for start in range(0, n, b):
        chunk = tuple(records_sorted[start : min(start + b, n)])
        if not chunk:
            continue
        page_id = len(pages)
        pages.append(Page(page_id=page_id, records=chunk))
        if chunk[0].code is None:
            raise ValueError("record.code must be populated before paging")
        page_start_codes.append(int(chunk[0].code))

    return tuple(pages), np.asarray(page_start_codes, dtype=np.int64)
