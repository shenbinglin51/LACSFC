from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from ..index import BoundaryIndex
from ..layout import build_artifacts
from ..quantizer import Quantizer
from ..query_engine import QueryEngine
from ..types import Interval, QueryWindow
from .masks import standard_morton_masks, validate_masks
from .plugin import LMSFCPlugin


@dataclass(frozen=True)
class BOTrainResult:
    masks: List[int]
    cost: float
    n_evaluated: int


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


def _feature_from_masks(masks: Sequence[int], total_width: int, d: int) -> np.ndarray:
    owner = np.full((int(total_width),), fill_value=-1, dtype=np.int64)
    for dim in range(d):
        bit_positions = _mask_bit_positions(int(masks[dim]))
        for b in bit_positions:
            if 0 <= int(b) < int(total_width):
                owner[int(b)] = int(dim)
    if np.any(owner < 0):
        raise ValueError("invalid masks: incomplete bit ownership")
    return owner


def train_bo(
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
    per: int = 10,
    w_page_accesses: float = 1.0,
    w_segments: float = 1.0,
    w_span_per_cell: float = 1.0,
    locality_max_cells: int = 20_000,
    locality_qn: int = 0,
    initial_masks: Optional[Sequence[int]] = None,
    init_size: int = 120,
    rounds: int = 8,
    evals_per_round: int = 40,
    neighbor_candidates: int = 2000,
    random_candidates: int = 2000,
    topk: int = 3,
    rf_init_points: int = 8,
    rf_n_iter: int = 16,
    use_bayes_opt: bool = True,
    seed: int = 0,
) -> BOTrainResult:
    if per <= 0:
        raise ValueError("per must be > 0")
    if init_size <= 0:
        raise ValueError("init_size must be > 0")
    if rounds < 0:
        raise ValueError("rounds must be >= 0")
    if evals_per_round <= 0:
        raise ValueError("evals_per_round must be > 0")
    if neighbor_candidates < 0 or random_candidates < 0:
        raise ValueError("neighbor_candidates/random_candidates must be >= 0")
    if topk <= 0:
        raise ValueError("topk must be > 0")

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
        rng_queries = np.random.default_rng(int(seed) + 7331)
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
            sfc = LMSFCPlugin(name="LMSFC-BO", grid_size=tuple([quantizer.m] * d), masks=list(masks))
            if threshold is not None:
                sfc.threshold = int(threshold)
            artifacts = build_artifacts(points=training_points, sfc=sfc, quantizer=quantizer, b=b_train)
            setter = getattr(sfc, "set_page_start_codes", None)
            if callable(setter):
                setter(artifacts.page_start_codes)

            if objective == "query_s":
                engine = QueryEngine(artifacts=artifacts, sfc=sfc, quantizer=quantizer, semantic=semantic)
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

            # Fast path for page_accesses / segments / metrics:
            # Skip Python record scanning; use sfc.intervals() + BoundaryIndex directly.
            index = BoundaryIndex(page_start_codes=artifacts.page_start_codes)
            intervals_fn = getattr(sfc, "intervals", None)

            page_accesses = 0
            segments = 0
            for q in queries:
                qq = quantizer.quantize_window(q)
                if callable(intervals_fn):
                    raw = intervals_fn(qq)
                else:
                    raw = None
                if raw is not None:
                    ivals: List[Interval] = [Interval(int(it.start), int(it.end)) for it in raw]
                else:
                    # Fallback: enumerate cells and encode
                    low_cell = np.asarray(qq.low_cell, dtype=np.int64)
                    high_cell = np.asarray(qq.high_cell, dtype=np.int64)
                    sizes = (high_cell - low_cell + 1).astype(np.int64)
                    total_cells = int(np.prod(sizes))
                    if total_cells > 200_000:
                        continue
                    grids = [np.arange(int(low_cell[i]), int(high_cell[i]) + 1, dtype=np.int64) for i in range(d)]
                    mesh = np.meshgrid(*grids, indexing="ij")
                    cells = np.stack([m.reshape(-1) for m in mesh], axis=1)
                    codes_arr = np.empty(cells.shape[0], dtype=np.int64)
                    for ci in range(cells.shape[0]):
                        codes_arr[ci] = int(sfc.encode_cell(cells[ci]))
                    codes_arr.sort(kind="mergesort")
                    ivals = []
                    if codes_arr.size > 0:
                        s, e = int(codes_arr[0]), int(codes_arr[0])
                        for cv in codes_arr[1:]:
                            iv = int(cv)
                            if iv <= e + 1:
                                e = iv
                            else:
                                ivals.append(Interval(s, e))
                                s, e = iv, iv
                        ivals.append(Interval(s, e))
                pages = index.lookup_intervals(ivals)
                page_accesses += int(len(pages))
                segments += int(len(ivals))

            qn2 = max(1, len(queries))
            avg_pa = page_accesses / qn2
            avg_seg = segments / qn2

            if objective == "page_accesses":
                return float(avg_pa)
            if objective == "segments":
                return float(avg_seg)
            return float(w_page_accesses * avg_pa + w_segments * avg_seg)

    pool_x: List[np.ndarray] = []
    pool_y: List[float] = []
    seen = set()

    best_masks = list(base_masks)
    best_cost = float("inf")

    def _push_eval(masks: Sequence[int]) -> float:
        nonlocal best_masks, best_cost
        mm = tuple(int(x) for x in masks)
        if mm in seen:
            return float("inf")
        seen.add(mm)
        c = float(eval_cost(mm))
        if not np.isfinite(c):
            return c
        x = _feature_from_masks(mm, total_width=total_width, d=d)
        pool_x.append(np.asarray(x, dtype=np.float64))
        pool_y.append(float(c))
        if c < best_cost:
            best_cost = c
            best_masks = list(mm)
        return c

    _push_eval(base_masks)
    while len(pool_y) < int(init_size):
        _push_eval(_random_masks_with_widths(rng=rng, widths=widths, total_width=total_width, d=d))

    def _fit_rf(
        x: np.ndarray,
        y: np.ndarray,
        rng_local: np.random.Generator,
    ) -> Optional[Callable[[np.ndarray], np.ndarray]]:
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import cross_val_score
        except Exception:
            return None

        params = {
            "n_estimators": 200,
            "min_samples_split": 2,
            "max_features": 0.5,
            "max_depth": 12,
            "random_state": int(rng_local.integers(0, 2**31 - 1)),
        }

        if bool(use_bayes_opt) and x.shape[0] >= 12:
            try:
                from bayes_opt import BayesianOptimization

                def _rf_score(n_estimators: float, min_samples_split: float, max_features: float, max_depth: float) -> float:
                    model = RandomForestRegressor(
                        n_estimators=max(10, int(n_estimators)),
                        min_samples_split=max(2, int(min_samples_split)),
                        max_features=min(max(0.1, float(max_features)), 0.999),
                        max_depth=max(2, int(max_depth)),
                        random_state=0,
                    )
                    folds = int(min(5, max(2, x.shape[0] // 4)))
                    val = cross_val_score(model, x, y, scoring="neg_mean_absolute_error", cv=folds).mean()
                    return float(val)

                opt = BayesianOptimization(
                    f=_rf_score,
                    pbounds={
                        "n_estimators": (20, 400),
                        "min_samples_split": (2, 24),
                        "max_features": (0.1, 0.999),
                        "max_depth": (4, 20),
                    },
                    random_state=int(rng_local.integers(0, 2**31 - 1)),
                    verbose=0,
                )
                opt.maximize(init_points=max(2, int(rf_init_points)), n_iter=max(0, int(rf_n_iter)))
                p = opt.max["params"]
                params.update(
                    {
                        "n_estimators": max(10, int(p["n_estimators"])),
                        "min_samples_split": max(2, int(p["min_samples_split"])),
                        "max_features": min(max(0.1, float(p["max_features"])), 0.999),
                        "max_depth": max(2, int(p["max_depth"])),
                    }
                )
            except Exception:
                pass

        model = RandomForestRegressor(**params)
        model.fit(x, y)

        def _predict(xx: np.ndarray) -> np.ndarray:
            return np.asarray(model.predict(xx), dtype=np.float64)

        return _predict

    for _ in range(int(rounds)):
        x_arr = np.asarray(pool_x, dtype=np.float64)
        y_arr = np.asarray(pool_y, dtype=np.float64)
        predictor = _fit_rf(x=x_arr, y=y_arr, rng_local=rng)

        cand_masks: List[List[int]] = []
        base_for_neighbors = list(best_masks)

        for _n in range(int(neighbor_candidates)):
            m = list(base_for_neighbors)
            n_swaps = int(rng.integers(1, 4))
            for _s in range(n_swaps):
                m = _swap_one_bit_between_dims(rng=rng, masks=m)
            cand_masks.append(m)

        for _n in range(int(random_candidates)):
            cand_masks.append(_random_masks_with_widths(rng=rng, widths=widths, total_width=total_width, d=d))

        dedup_cands: List[List[int]] = []
        local_seen = set()
        for m in cand_masks:
            mm = tuple(int(x) for x in m)
            if mm in seen or mm in local_seen:
                continue
            local_seen.add(mm)
            dedup_cands.append(list(mm))

        if not dedup_cands:
            continue

        chosen: List[List[int]]
        if predictor is not None:
            fx = np.asarray([_feature_from_masks(m, total_width=total_width, d=d) for m in dedup_cands], dtype=np.float64)
            pred = predictor(fx)
            order = np.argsort(pred)
            take = min(int(evals_per_round), len(dedup_cands))
            chosen = [dedup_cands[int(i)] for i in order[:take].tolist()]
        else:
            take = min(int(evals_per_round), len(dedup_cands))
            idx = rng.permutation(len(dedup_cands))[:take]
            chosen = [dedup_cands[int(i)] for i in idx]

        for m in chosen:
            _push_eval(m)

    top_idx = np.argsort(np.asarray(pool_y, dtype=np.float64))[: max(1, int(topk))]
    for i in top_idx.tolist():
        m_feat = np.asarray(pool_x[int(i)], dtype=np.int64)
        recon = [0] * d
        for bit, owner in enumerate(m_feat.tolist()):
            recon[int(owner)] |= 1 << int(bit)
        validate_masks(recon, d)
        c = float(eval_cost(recon))
        if c < best_cost:
            best_cost = c
            best_masks = list(recon)

    return BOTrainResult(masks=list(best_masks), cost=float(best_cost), n_evaluated=int(len(pool_y)))
