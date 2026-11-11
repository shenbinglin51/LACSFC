"""
BitAllocZPlugin: SFCPlugin adapter for BitAllocLZ in sfcBenchmark.

Implements the SFCPlugin protocol:
  - encode_cell(cell) -> int
  - intervals(q: QuantizedQuery) -> List[Interval]

Usage:
    plugin = BitAllocZPlugin(d=2, k=8, b=256, optimizer='bo', bo_iters=80)
    plugin.build(points, quantized_queries)
    code = plugin.encode_cell(np.array([3, 7]))
    ivs  = plugin.intervals(quantized_query)
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .curve import BitAllocZCurve
from .query_eval import intervals_from_query, build_page_start_codes
from ..types import Interval, QuantizedQuery


class BitAllocZPlugin:
    """Workload-driven Z-order curve with learned bit allocation.

    Parameters
    ----------
    d : int
        Number of spatial dimensions.
    k : int
        Bits per dimension in the grid (grid_size = 2^k per axis).
    b : int
        Page size (records per page).
    total_bits : int | None
        Total bit budget for the Z-code. Defaults to d*k (standard Z-curve).
    optimizer : str
        'bo' (Bayesian Optimization, faster) or 'gd' (Gradient Descent).
    bo_init : int
        BO random exploration points before acquisition (default 5).
    bo_iters : int
        BO acquisition iterations (default 80).
    gd_iters : int
        GD iterations (default 100).
    gd_lr : float
        GD learning rate (default 0.1).
    w_seg : float
        Weight for segment count in cost (default 1.0).
    w_pa : float
        Weight for page accesses in cost (default 1.0).
    w_cand : float
        Weight for candidate count in cost (default 0.0).
    max_cells : int
        Maximum cells to enumerate per query (default 2_000_000).
    seed : int
        Random seed (default 0).
    verbose : bool
        Print optimization progress (default False).
    name : str
        Display name for this plugin.
    """

    def __init__(
        self,
        *,
        d: int = 2,
        k: int = 8,
        b: int = 256,
        total_bits: Optional[int] = None,
        optimizer: str = "bo",
        bo_init: int = 5,
        bo_iters: int = 80,
        gd_iters: int = 100,
        gd_lr: float = 0.1,
        w_seg: float = 1.0,
        w_pa: float = 1.0,
        w_cand: float = 0.0,
        max_cells: int = 2_000_000,
        seed: int = 0,
        verbose: bool = False,
        name: str = "BitAllocZ",
    ) -> None:
        self.name = str(name)
        self._d = int(d)
        self._k = int(k)
        self._b = int(b)
        self._total_bits = int(total_bits) if total_bits is not None else int(d) * int(k)
        self._optimizer = str(optimizer)
        self._bo_init = int(bo_init)
        self._bo_iters = int(bo_iters)
        self._gd_iters = int(gd_iters)
        self._gd_lr = float(gd_lr)
        self._w_seg = float(w_seg)
        self._w_pa = float(w_pa)
        self._w_cand = float(w_cand)
        self._max_cells = int(max_cells)
        self._seed = int(seed)
        self._verbose = bool(verbose)

        self._curve: Optional[BitAllocZCurve] = None
        self._quantizer = None
        self._learn_result = None

    def build(
        self,
        points: np.ndarray,
        quantized_queries: Sequence[QuantizedQuery],
        quantizer=None,
    ) -> None:
        """Learn bit allocation from training data and build the curve.

        Parameters
        ----------
        points : np.ndarray, shape (n, d)
            Training points in [0, 1)^d.
        quantized_queries : sequence of QuantizedQuery
            Pre-quantized training queries (low_cell / high_cell).
        quantizer : Quantizer instance
            sfcBenchmark Quantizer used to re-quantize points for page layout.
        """
        from .optimizer import learn_alloc_bo, learn_alloc_gd

        if quantizer is None:
            from ..quantizer import Quantizer
            quantizer = Quantizer(k=self._k, d=self._d)
        self._quantizer = quantizer

        if self._total_bits == self._d * self._k and self._verbose:
            print(
                f"[BitAllocZ] NOTE: total_bits={self._total_bits} == d*k={self._d*self._k}. "
                f"Only uniform alloc [{self._k}]*{self._d} is feasible at full budget. "
                f"Set total_bits < d*k (e.g. {self._d*self._k - self._k}) for non-uniform allocation."
            )
        if self._verbose:
            print(
                f"[BitAllocZ] building {self.name}: d={self._d} k={self._k} "
                f"total_bits={self._total_bits} optimizer={self._optimizer} "
                f"n_pts={len(points)} n_qry={len(quantized_queries)}"
            )

        if self._optimizer == "bo":
            result = learn_alloc_bo(
                points=points,
                train_queries=quantized_queries,
                d=self._d,
                k=self._k,
                b=self._b,
                total_bits=self._total_bits,
                max_cells=self._max_cells,
                bo_init=self._bo_init,
                bo_iters=self._bo_iters,
                w_seg=self._w_seg,
                w_pa=self._w_pa,
                w_cand=self._w_cand,
                seed=self._seed,
                quantizer=quantizer,
                verbose=self._verbose,
            )
        elif self._optimizer == "gd":
            result = learn_alloc_gd(
                points=points,
                train_queries=quantized_queries,
                d=self._d,
                k=self._k,
                b=self._b,
                total_bits=self._total_bits,
                max_cells=self._max_cells,
                n_iterations=self._gd_iters,
                learning_rate=self._gd_lr,
                w_seg=self._w_seg,
                w_pa=self._w_pa,
                w_cand=self._w_cand,
                seed=self._seed,
                quantizer=quantizer,
                verbose=self._verbose,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self._optimizer!r}. Use 'bo' or 'gd'.")

        self._learn_result = result
        self._curve = BitAllocZCurve(
            d=self._d,
            k=self._k,
            alloc=result.alloc,
            total_bits=self._total_bits,
        )

        if self._verbose:
            print(
                f"[BitAllocZ] learned alloc={list(result.alloc)} "
                f"cost={result.cost:.4f} n_evals={result.n_evals}"
            )

    # ── SFCPlugin protocol ─────────────────────────────────────────────────────

    def encode_cell(self, cell: np.ndarray) -> int:
        """Encode a d-dimensional grid cell to Z-code integer."""
        if self._curve is None:
            raise RuntimeError("BitAllocZPlugin.build() must be called before encode_cell()")
        return self._curve.encode_cell(np.asarray(cell, dtype=np.int64))

    def intervals(self, q: QuantizedQuery) -> Optional[List[Interval]]:
        """Return Z-code intervals covering the query window.

        Returns None if build() hasn't been called yet.
        Skips queries that enumerate too many cells (returns empty list).
        """
        if self._curve is None:
            return None

        low_cell = np.asarray(q.low_cell, dtype=np.int64)
        high_cell = np.asarray(q.high_cell, dtype=np.int64)

        try:
            raw = intervals_from_query(
                curve=self._curve,
                low_cell=low_cell,
                high_cell=high_cell,
                max_cells=self._max_cells,
            )
        except ValueError:
            return []

        return [Interval(int(s), int(e)) for s, e in raw]

    # ── Accessors ──────────────────────────────────────────────────────────────

    @property
    def curve(self) -> Optional[BitAllocZCurve]:
        return self._curve

    @property
    def alloc(self):
        return self._curve.alloc if self._curve is not None else None

    def summary(self) -> str:
        if self._curve is None:
            return f"{self.name} [not built]"
        return (
            f"{self.name} alloc={list(self._curve.alloc)} "
            f"total_bits={self._total_bits} cost={self._learn_result.cost:.4f}"
        )
