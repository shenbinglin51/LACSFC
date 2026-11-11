"""Full-grid rank-table SFC plugin, and GBO / QUILTS builders taken unchanged from the NeuralSFC lacsfc repo.

The rank is a permutation of the (2^k)^d grid cells in C order (cell (c_0, ..., c_{d-1}) has flat index
np.ravel_multi_index(c, (2^k,)*d)), the convention of lacsfc's rank arrays. Both builders see only the points
and the *training* queries; test queries are never passed.

  GBO     e7_recursive.gbo_rank(shape, cells, A_low, A_high, alpha=3.0, seed): Yang-He-Chen (TKDE 2016); not AC.
  QUILTS  ch7_export.build_quilts: design_multi with the A-box width patterns, admission rule (a).

LACSFC_ROOT (env var; defaults to the vendored NeuralSFC-lacsfc copy under <repo>/vendored/) and
QUILTS_ROOT (read by ch7_export) locate the sources.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .quantizer import Quantizer
from .types import Interval, QuantizedQuery
from ._paths import neuralsfc_root

LACSFC_ROOT = Path(os.environ.get("LACSFC_ROOT", str(neuralsfc_root())))


class _RankEncoder:
    def __init__(self, rank: np.ndarray, shape: Tuple[int, ...]):
        self.rank = rank
        self.shape = shape

    def encode(self, cells: np.ndarray) -> np.ndarray:
        cells = np.asarray(cells, dtype=np.int64).reshape(-1, len(self.shape))
        return self.rank[np.ravel_multi_index(tuple(cells.T), self.shape)]


@dataclass
class RankTableSFC:
    name: str
    k: int
    d: int
    rank: np.ndarray
    info: dict

    def __post_init__(self) -> None:
        self.grid_size = (1 << int(self.k),) * int(self.d)
        n = int(np.prod(self.grid_size))
        rank = np.asarray(self.rank, dtype=np.int64)
        if rank.shape != (n,) or not np.array_equal(np.sort(rank), np.arange(n)):
            raise ValueError(f"{self.name}: rank is not a permutation of the {n} grid cells")
        self.rank = rank
        self._encoder = _RankEncoder(rank, self.grid_size)

    def encode_cell(self, cell: np.ndarray) -> int:
        cell = np.asarray(cell, dtype=np.int64)
        if cell.shape != (int(self.d),):
            raise ValueError(f"cell must have shape ({self.d},)")
        return int(self._encoder.encode(cell)[0])

    def intervals(self, q: QuantizedQuery) -> Optional[List[Interval]]:
        return None


def _lacsfc_paths() -> None:
    for p in (LACSFC_ROOT / "src", LACSFC_ROOT / "experiments"):
        if not p.is_dir():
            raise FileNotFoundError(f"{p} not found; set LACSFC_ROOT")
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def train_boxes(train_qq: Sequence[QuantizedQuery], d: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(train_qq) == 0:
        raise ValueError("rank-table builders need at least one training query")
    lo = np.stack([np.asarray(q.low_cell, dtype=np.int64) for q in train_qq]).reshape(-1, d)
    hi = np.stack([np.asarray(q.high_cell, dtype=np.int64) for q in train_qq]).reshape(-1, d)
    return lo, hi


def build_gbo_rank(k: int, d: int, cells: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                   seed: int = 0, alpha: float = 3.0) -> Tuple[np.ndarray, dict]:
    _lacsfc_paths()
    from e7_recursive import gbo_rank
    shape = (1 << int(k),) * int(d)
    flat = np.ravel_multi_index(tuple(np.asarray(cells, dtype=np.int64).T), shape)
    rank = np.asarray(gbo_rank(shape, flat, lo, hi, alpha=float(alpha), seed=int(seed)), dtype=np.int64)
    return rank, {"source": "lacsfc experiments/e7_recursive.gbo_rank", "alpha": float(alpha), "seed": int(seed),
                  "ac": False, "occupied_cells": int(np.unique(flat).size)}


def build_quilts_rank(k: int, d: int, cells: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> Tuple[np.ndarray, dict]:
    _lacsfc_paths()
    import ch7_export
    v = {"cells": np.asarray(cells, dtype=np.int64), "K": int(k), "A_low": lo, "A_high": hi}
    rank, info = ch7_export.build_quilts(v, dict(ch7_export.DEFAULTS["quilts"]))
    return np.asarray(rank, dtype=np.int64), {"source": "lacsfc experiments/ch7_export.build_quilts", **info}


def make_rank_table_sfc(name: str, k: int, d: int, points: np.ndarray, train_qq: Sequence[QuantizedQuery],
                        quantizer: Optional[Quantizer] = None, seed: int = 0) -> RankTableSFC:
    """Build GBO or QUILTS from the points and the quantized training queries only."""
    q = quantizer if quantizer is not None else Quantizer(d=int(d), k=int(k))
    if int(q.d) != int(d) or int(q.k) != int(k):
        raise ValueError("quantizer d/k differ from the requested grid")
    cells = q.quantize_points(np.asarray(points, dtype=np.float64))
    lo, hi = train_boxes(train_qq, int(d))
    if name == "GBO":
        rank, info = build_gbo_rank(k, d, cells, lo, hi, seed=seed)
    elif name == "QUILTS":
        rank, info = build_quilts_rank(k, d, cells, lo, hi)
    else:
        raise ValueError(f"unknown rank-table method {name!r}")
    info["n_train_queries"] = int(lo.shape[0])
    return RankTableSFC(name=name, k=int(k), d=int(d), rank=rank, info=info)
