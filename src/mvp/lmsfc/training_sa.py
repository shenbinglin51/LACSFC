from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from ..layout import build_artifacts
from ..quantizer import Quantizer
from ..query_engine import QueryEngine
from ..types import QueryWindow
from .masks import standard_morton_masks, validate_masks
from .plugin import LMSFCPlugin


@dataclass(frozen=True)
class SATrainResult:
    masks: List[int]
    cost: float


def _mask_bit_positions(mask: int) -> List[int]:
    out: List[int] = []
    m = int(mask)
    idx = 0
    while m:
        if m & 1:
            out.append(idx)
        idx += 1
        m >>= 1
    return out


def _positions_to_mask(pos: Sequence[int]) -> int:
    m = 0
    for p in pos:
        m |= 1 << int(p)
    return int(m)


def _swap_one_bit_between_dims(rng: np.random.Generator, masks: Sequence[int]) -> List[int]:
    d = len(masks)
    if d < 2:
        return list(masks)

    positions = [_mask_bit_positions(m) for m in masks]
    sizes = [len(p) for p in positions]
    if any(s == 0 for s in sizes):
        return list(masks)

    a = int(rng.integers(0, d))
    b = int(rng.integers(0, d - 1))
    if b >= a:
        b += 1

    ia = int(rng.integers(0, len(positions[a])))
    ib = int(rng.integers(0, len(positions[b])))

    positions[a][ia], positions[b][ib] = positions[b][ib], positions[a][ia]

    out = [_positions_to_mask(p) for p in positions]
    validate_masks(out, d)
    return out


def _random_masks_with_widths(
    rng: np.random.Generator, widths: Sequence[int], total_width: int, d: int
) -> List[int]:
    if total_width > 64:
        raise ValueError("total_width must be <= 64")

    perm = rng.permutation(total_width)
    out: List[int] = []
    idx = 0
    for i in range(d):
        w = int(widths[i])
        bits = perm[idx : idx + w]
        out.append(_positions_to_mask(bits.tolist()))
        idx += w

    validate_masks(out, d)
    return out


def train_sa(
    *,
    points: np.ndarray,
    queries: Sequence[QueryWindow],
    quantizer: Quantizer,
    b: int,
    threshold: Optional[int] = None,
    objective: str = "metrics",
    semantic: str = "cell",
    timing_repeats: int = 3,
    timing_warmup: bool = True,
    iters: int = 200,
    per: int = 10,
    restarts: int = 5,
    w_page_accesses: float = 1.0,
    w_segments: float = 1.0,
    w_span_per_cell: float = 1.0,
    locality_max_cells: int = 20_000,
    locality_qn: int = 0,
    initial_masks: Optional[Sequence[int]] = None,
    seed: int = 0,
) -> SATrainResult:
    if per <= 0:
        raise ValueError("per must be > 0")
    if iters <= 0:
        raise ValueError("iters must be > 0")
    if restarts <= 0:
        raise ValueError("restarts must be > 0")
    objective = str(objective)
    if objective not in {"metrics", "query_s", "page_accesses", "segments", "exact_locality"}:
        raise ValueError("objective must be one of: metrics, query_s, page_accesses, segments, exact_locality")
    if timing_repeats <= 0:
        raise ValueError("timing_repeats must be > 0")

    semantic = str(semantic)
    if semantic != "cell":
        raise ValueError("semantic must be 'cell'")

    rng = np.random.default_rng(int(seed))

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != quantizer.d:
        raise ValueError(f"points must have shape (n,{quantizer.d})")

    d = int(quantizer.d)

    if initial_masks is None:
        base_masks = standard_morton_masks(d)
    else:
        base_masks = [int(m) for m in initial_masks]
    validate_masks(base_masks, d)

    widths = [int(int(m).bit_count()) for m in base_masks]
    total_width = int(sum(widths))
    if total_width <= 0:
        raise ValueError("invalid mask widths")
    if total_width > 64:
        raise ValueError("mask widths exceed 64 bits")

    n = points.shape[0]
    n_train = max(1, n // int(per))
    perm_points = rng.permutation(n)
    training_points = points[perm_points[:n_train]]

    b_train = max(1, int(b) // int(per))

    if objective == "exact_locality":
        qn = len(queries)
        if int(locality_qn) > 0:
            qn_use = int(min(int(locality_qn), int(qn)))
        else:
            qn_use = int(min(int(qn), max(1, int(qn) // int(per))))
        rng_queries = np.random.default_rng(int(seed) + 1337)
        if qn_use >= qn:
            eval_queries: List[QueryWindow] = list(queries)
        else:
            idx = rng_queries.permutation(qn)[:qn_use]
            eval_queries = [queries[int(i)] for i in idx]

        def _enumerate_cells_nd(low_cell: np.ndarray, high_cell: np.ndarray) -> np.ndarray:
            low = np.asarray(low_cell, dtype=np.int64)
            high = np.asarray(high_cell, dtype=np.int64)
            ranges = [np.arange(int(low[i]), int(high[i]) + 1, dtype=np.uint64) for i in range(low.shape[0])]
            grids = np.meshgrid(*ranges, indexing="ij")
            return np.stack([g.reshape(-1) for g in grids], axis=1)

        def eval_cost(masks: Sequence[int]) -> float:
            sfc = LMSFCPlugin(name="LMSFC", grid_size=tuple([quantizer.m] * d), masks=list(masks))
            if threshold is not None:
                sfc.threshold = int(threshold)

            seg_sum = 0.0
            spc_sum = 0.0
            n_eval = 0

            for q in eval_queries:
                qq = quantizer.quantize_window(q)
                low = np.asarray(qq.low_cell, dtype=np.int64)
                high = np.asarray(qq.high_cell, dtype=np.int64)
                lb = np.minimum(low, high)
                ub = np.maximum(low, high)
                sizes = ub - lb + 1
                n_cells = int(np.prod(sizes))
                if n_cells <= 0:
                    continue
                if int(locality_max_cells) > 0 and n_cells > int(locality_max_cells):
                    continue

                cells = _enumerate_cells_nd(lb, ub)
                codes = [int(sfc.encode_cell(cells[i])) for i in range(int(cells.shape[0]))]
                if not codes:
                    continue
                codes.sort()
                span = int(int(codes[-1]) - int(codes[0]) + 1)
                seg = 1
                prev = int(codes[0])
                for c in codes[1:]:
                    ci = int(c)
                    if ci != prev + 1:
                        seg += 1
                    prev = ci

                seg_sum += float(seg)
                spc_sum += float(span) / float(len(codes))
                n_eval += 1

            if n_eval <= 0:
                return float("inf")
            mean_seg = float(seg_sum) / float(n_eval)
            mean_spc = float(spc_sum) / float(n_eval)
            return float(w_segments * mean_seg + w_span_per_cell * mean_spc)

    else:
        def eval_cost(masks: Sequence[int]) -> float:
            sfc = LMSFCPlugin(name="LMSFC", grid_size=tuple([quantizer.m] * d), masks=list(masks))
            if threshold is not None:
                sfc.threshold = int(threshold)
            # Use Sort Dimension during training evaluation
            artifacts = build_artifacts(
                points=training_points, 
                sfc=sfc, 
                quantizer=quantizer, 
                b=b_train,
                use_sortdim=True,
                training_queries=queries,
            )
            sfc.set_page_start_codes(artifacts.page_start_codes)
            engine = QueryEngine(artifacts=artifacts, sfc=sfc, quantizer=quantizer, semantic=semantic)

            if objective == "query_s":
                if timing_warmup and queries:
                    engine.query(queries[0])

                best = float("inf")
                for _ in range(int(timing_repeats)):
                    t0 = time.perf_counter()
                    for q in queries:
                        engine.query(q)
                    t1 = time.perf_counter()
                    best = min(best, float(t1 - t0))
                
                return float(best)

            page_accesses = 0
            segments = 0
            for q in queries:
                r = engine.query(q)
                page_accesses += int(r.metrics.physical_pages)
                segments += int(r.metrics.segments)

            qn = max(1, len(queries))
            avg_pa = page_accesses / qn
            avg_seg = segments / qn

            if objective == "page_accesses":
                return float(avg_pa)
            if objective == "segments":
                return float(avg_seg)
            return float(w_page_accesses * avg_pa + w_segments * avg_seg)

    iters_per_restart = max(1, int(iters) // int(restarts))

    global_best_cost = float("inf")
    global_best_masks = list(base_masks)

    for restart in range(int(restarts)):
        if restart == 0:
            current_masks = list(base_masks)
        else:
            current_masks = _random_masks_with_widths(rng=rng, widths=widths, total_width=total_width, d=d)

        best_masks = list(current_masks)
        best_cost = eval_cost(best_masks)

        prev_masks = list(current_masks)
        T = max(1e-12, 0.9 * best_cost)
        decay = 0.95

        for _ in range(int(iters_per_restart)):
            cand_masks = _swap_one_bit_between_dims(rng=rng, masks=prev_masks)
            cand_cost = eval_cost(cand_masks)

            accept = False
            if cand_cost < best_cost:
                accept = True
                best_cost = cand_cost
                best_masks = list(cand_masks)
            else:
                a = (best_cost - cand_cost) / max(1e-12, T)
                if float(rng.random()) < float(np.exp(a)):
                    accept = True

            if accept:
                prev_masks = list(cand_masks)
                prev_cost = cand_cost
                T = decay * T

        if best_cost < global_best_cost:
            global_best_cost = best_cost
            global_best_masks = list(best_masks)

    return SATrainResult(masks=global_best_masks, cost=float(global_best_cost))
