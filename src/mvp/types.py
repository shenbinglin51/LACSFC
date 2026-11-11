from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol, Tuple

import numpy as np


ArrayF = np.ndarray
ArrayI = np.ndarray


@dataclass(frozen=True)
class QueryWindow:
    low: ArrayF
    high: ArrayF


@dataclass(frozen=True)
class QuantizedQuery:
    low_cell: ArrayI
    high_cell: ArrayI


@dataclass(frozen=True, order=True)
class Interval:
    start: int
    end: int


@dataclass
class Record:
    rid: int
    x: ArrayF
    cell: Optional[ArrayI] = None
    code: Optional[int] = None


@dataclass(frozen=True)
class Page:
    page_id: int
    records: Tuple[Record, ...]
    sort_dim: int = -1  # -1 means no sort dimension selected


@dataclass(frozen=True)
class LayoutArtifacts:
    d: int
    k: int
    m: int
    b: int
    sfc_name: str
    pages: Tuple[Page, ...]
    page_start_codes: np.ndarray


class SFCPlugin(Protocol):
    name: str

    def encode_cell(self, cell: ArrayI) -> int: ...

