from __future__ import annotations

import os
import sys

if __name__ == "__main__" and (__package__ is None or __package__ == ""):
    _ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    sys.path[0] = _ROOT
    __package__ = "src.mvp"

import argparse
import csv
import dataclasses
import hashlib
import json
import time
from dataclasses import dataclass
from collections import Counter
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .layout import build_artifacts
from .lmsfc.masks_io import load_masks_json, save_masks_json
from .lmsfc.plugin import LMSFCPlugin
from .lmsfc.training_bo import train_bo
from .lmsfc.training_sa import train_sa
from .lmsfc.training_reinforce import train_reinforce
from .quantizer import Quantizer
from .query_engine import QueryEngine
from .sfc_plugins import (
    make_column_major,
    make_hilbert,
    make_hilbert_nd,
    make_lacsfc_h,
    make_lacsfc_from_weights,
    make_lacsfc_neural,
    make_lacsfc_neural_v2,
    make_row_major,
    make_z_curve,
    make_piecewise_sfc,
    make_piecewise_mcts,
    make_bitallocz_bo,
    make_bitallocz_gd,
    make_bitallocz_uniform,
)
from .adacurve.paper_repro import make_adacurve_paper_repro
from .rank_table import make_rank_table_sfc
from .types import QueryWindow
from .workloads import (
    generate_points_anti_correlated,
    generate_points_correlated,
    generate_points_diagonal,
    generate_points_gmm,
    generate_points_normal,
    generate_points_skewed,
    generate_points_uniform,
    generate_points_zipf,
    generate_queries_aspect_ratio,
    generate_queries_hotspot_correlated,
    generate_queries_loguniform_area,
    load_points_real,
)
from .lacsfc.workload_shift import WorkloadShiftConfig, evaluate_shift_from_summary


def _points_file_raw_bounds(path: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if not path:
        return None
    p = os.path.abspath(str(path))
    if not os.path.exists(p) or os.path.splitext(p)[1].lower() in {".csv", ".npy"}:
        return None
    if os.path.getsize(p) < 4:
        return None
    count_arr = np.fromfile(p, dtype="<u4", count=1)
    if count_arr.size != 1:
        return None
    count = int(count_arr[0])
    expected_size = 4 + count * 2 * 4
    if count <= 0 or expected_size != os.path.getsize(p):
        return None
    mm = np.memmap(p, dtype="<u4", mode="r", offset=4, shape=(count, 2))
    return np.asarray(mm.min(axis=0), dtype=np.float64), np.asarray(mm.max(axis=0), dtype=np.float64)


def _load_query_windows_file(
    path: str,
    d: int,
    k: int,
    max_qn: int = 0,
    raw_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple[QueryWindow, ...]:
    p = os.path.abspath(str(path))
    if not os.path.exists(p):
        raise FileNotFoundError(path)

    suffix = os.path.splitext(p)[1].lower()
    if suffix == ".npy":
        arr = np.load(p, allow_pickle=False)
    elif suffix == ".json":
        with open(p, "r", encoding="utf-8") as f:
            raw = json.load(f)
        rows = []
        for item in raw:
            if isinstance(item, dict):
                if "low" in item and "high" in item:
                    rows.append(list(item["low"]) + list(item["high"]))
                elif "low" in item and "size" in item:
                    rows.append(list(item["low"]) + list(item["size"]))
                else:
                    raise ValueError("JSON query records must contain low/high or low/size")
            else:
                rows.append(list(item))
        arr = np.asarray(rows, dtype=np.float64)
    elif suffix == ".csv":
        rows = []
        with open(p, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                try:
                    vals = [float(x) for x in row]
                except ValueError:
                    continue
                rows.append(vals)
                if max_qn > 0 and len(rows) >= int(max_qn):
                    break
        arr = np.asarray(rows, dtype=np.float64)
    else:
        raise ValueError("--queries-file must be .csv, .json, or .npy")

    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2 * int(d):
        raise ValueError(f"query windows must have shape (n,{2 * int(d)})")
    if max_qn > 0 and len(arr) > int(max_qn):
        arr = arr[: int(max_qn)]

    lows = arr[:, : int(d)]
    tail = arr[:, int(d) :]
    highs = tail if np.all(tail >= lows) else lows + tail

    if raw_bounds is not None and lows.size and (float(np.nanmin(lows)) < 0.0 or float(np.nanmax(highs)) > 1.0):
        raw_lo, raw_hi = raw_bounds
        raw_lo = np.asarray(raw_lo, dtype=np.float64)[: int(d)]
        raw_hi = np.asarray(raw_hi, dtype=np.float64)[: int(d)]
        raw_range = raw_hi - raw_lo
        raw_range[raw_range == 0] = 1.0
        lows = (lows - raw_lo) / raw_range
        highs = (highs - raw_lo) / raw_range
    elif lows.size and (float(np.nanmin(lows)) < 0.0 or float(np.nanmax(highs)) > 1.0):
        scale = float(2 ** int(k))
        lows = lows / scale
        highs = highs / scale

    lows = np.clip(lows, 0.0, 1.0)
    highs = np.clip(highs, 0.0, 1.0)
    eps = np.finfo(np.float64).eps
    queries = []
    for low, high in zip(lows, highs):
        q_low = np.minimum(low, high).astype(np.float64)
        q_high = np.maximum(low, high).astype(np.float64)
        q_high = np.maximum(q_high, q_low + eps)
        q_high = np.clip(q_high, 0.0, 1.0)
        queries.append(QueryWindow(low=q_low, high=q_high))
    return tuple(queries)


def _lacsfc_artifact_file(
    args: argparse.Namespace,
    seed: int,
    curve_name: str = "LACSFC-Neural",
    meta_enable_effective: Optional[bool] = None,
) -> str:
    artifact_dir = str(getattr(args, "lacsfc_artifact_dir", "") or "")
    if not artifact_dir:
        return ""
    payload = {
        "points": str(args.points),
        "points_file": os.path.abspath(str(args.points_file)) if str(args.points_file) else "",
        "n": int(args.n),
        "qn": int(args.qn),
        "train_qn": int(args.train_qn),
        "d": int(args.d),
        "k": int(args.k),
        "b": int(args.b),
        "seed": int(seed),
        "query_dist": str(args.query_dist),
        "train_query_dist": str(args.train_query_dist) if str(args.train_query_dist) else str(args.query_dist),
        "queries_file": os.path.abspath(str(args.queries_file)) if str(getattr(args, "queries_file", "")) else "",
        "train_queries_file": os.path.abspath(str(args.train_queries_file)) if str(getattr(args, "train_queries_file", "")) else "",
        "area_min": float(args.area_min),
        "area_max": float(args.area_max),
        "hotspot_ratio": float(args.hotspot_ratio),
        "aspect_ratio": float(args.aspect_ratio),
        "lacsfc_k": int(args.lacsfc_k),
        "lacsfc_max_evals": int(args.lacsfc_max_evals),
        "lacsfc_mode": str(args.lacsfc_mode),
        "lacsfc_func_space_m": int(args.lacsfc_func_space_m),
        "lacsfc_sigma0": float(args.lacsfc_sigma0),
        "lacsfc_popsize": int(args.lacsfc_popsize) if args.lacsfc_popsize is not None else None,
        "lacsfc_cover_weight": float(args.lacsfc_cover_weight),
        "lacsfc_runtime_surrogate": str(args.lacsfc_runtime_surrogate),
        "lacsfc_runtime_weight": float(args.lacsfc_runtime_weight),
        "lacsfc_runtime_normalize": str(args.lacsfc_runtime_normalize),
        "lacsfc_runtime_interval_weight": float(args.lacsfc_runtime_interval_weight),
        "lacsfc_objective": str(args.lacsfc_objective),
        "lacsfc_learn_obj": bool(args.lacsfc_learn_obj),
        "lacsfc_init_obj": str(args.lacsfc_init_obj),
        "lacsfc_init_strategy": str(args.lacsfc_init_strategy),
        "lacsfc_curve_name": str(curve_name),
        "lacsfc_meta_enable": bool(getattr(args, "lacsfc_meta_enable", False) if meta_enable_effective is None else meta_enable_effective),
        "lacsfc_meta_model_dir": os.path.abspath(str(getattr(args, "lacsfc_meta_model_dir", ""))) if str(getattr(args, "lacsfc_meta_model_dir", "")) else "",
        "lacsfc_meta_model_name": str(getattr(args, "lacsfc_meta_model_name", "")),
        "lacsfc_meta_inner_steps": int(getattr(args, "lacsfc_meta_inner_steps", 0)),
        "lacsfc_meta_fallback_cmaes": bool(getattr(args, "lacsfc_meta_fallback_cmaes", True)),
        "lacsfc_es_epochs": int(getattr(args, "lacsfc_es_epochs", 30)),
        "lacsfc_es_n_perturb": int(getattr(args, "lacsfc_es_n_perturb", 16)),
        "lacsfc_es_sigma": float(getattr(args, "lacsfc_es_sigma", 0.1)),
        "lacsfc_es_dataset_size": int(getattr(args, "lacsfc_es_dataset_size", 64)),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    os.makedirs(artifact_dir, exist_ok=True)
    return os.path.join(artifact_dir, f"lacsfc_neural_s{int(seed)}_{digest}.npz")


@dataclass(frozen=True)
class CurveResult:
    name: str
    chosen_threshold: Optional[int]
    build_s: float
    query_s: float
    avg_page_accesses: float
    avg_segments: float
    avg_exact_cell_segments: Optional[float]
    avg_span: Optional[float]
    avg_page_amplification: float
    intervals_s: float
    lookup_s: float
    filter_s: float
    total_s: float
    avg_interval_count: float
    avg_covered_code_len: float
    avg_records_scanned: float
    avg_pass_code: float
    avg_pass_semantic: float
    avg_hit_segments: float
    avg_page_count: float
    avg_touched_page_record_count: float
    avg_candidate_page_count: float
    avg_candidate_page_record_count: float
    avg_empty_candidate_page_count: float
    avg_candidate_page_local_span_sum: float
    avg_candidate_page_local_span_mean: float
    avg_candidate_page_local_span_max: float
    avg_page_interval_visits: float
    avg_page_interval_overlaps: float
    avg_candidate_slice_count: float
    avg_candidate_slice_len_mean: float
    avg_empty_slice_count: float
    avg_nonempty_candidate_slice_count: float
    avg_candidate_slice_len_nonempty_mean: float
    avg_candidate_slice_len_max: float
    avg_vectorized_candidate_slice_count: float
    avg_scalar_candidate_slice_count: float
    avg_vectorized_candidate_record_count: float
    avg_scalar_candidate_record_count: float
    avg_batched_candidate_slice_count: float
    avg_batched_candidate_record_count: float
    avg_candidate_record_count: float
    avg_semantic_check_count: float
    avg_accepted_record_count: float
    filter_bisect_s: float
    filter_candidate_scan_s: float
    filter_candidate_extract_s: float
    filter_candidate_array_s: float
    filter_semantic_check_s: float
    filter_materialize_s: float


@dataclass(frozen=True)
class CurveStats:
    name: str
    chosen_threshold_mode: Optional[int]
    build_s_mean: float
    build_s_std: float
    query_s_mean: float
    query_s_std: float
    avg_page_accesses_mean: float
    avg_page_accesses_std: float
    avg_segments_mean: float
    avg_segments_std: float
    avg_exact_cell_segments_mean: Optional[float]
    avg_exact_cell_segments_std: Optional[float]
    avg_span_mean: Optional[float]
    avg_span_std: Optional[float]
    avg_page_amplification_mean: float
    avg_page_amplification_std: float
    intervals_s_mean: float
    intervals_s_std: float
    lookup_s_mean: float
    lookup_s_std: float
    filter_s_mean: float
    filter_s_std: float
    total_s_mean: float
    total_s_std: float
    avg_interval_count_mean: float
    avg_interval_count_std: float
    avg_covered_code_len_mean: float
    avg_covered_code_len_std: float
    avg_records_scanned_mean: float
    avg_records_scanned_std: float
    avg_pass_code_mean: float
    avg_pass_code_std: float
    avg_pass_semantic_mean: float
    avg_pass_semantic_std: float
    avg_hit_segments_mean: float
    avg_hit_segments_std: float
    avg_page_count_mean: float
    avg_page_count_std: float
    avg_touched_page_record_count_mean: float
    avg_touched_page_record_count_std: float
    avg_candidate_page_count_mean: float
    avg_candidate_page_count_std: float
    avg_candidate_page_record_count_mean: float
    avg_candidate_page_record_count_std: float
    avg_empty_candidate_page_count_mean: float
    avg_empty_candidate_page_count_std: float
    avg_candidate_page_local_span_sum_mean: float
    avg_candidate_page_local_span_sum_std: float
    avg_candidate_page_local_span_mean_mean: float
    avg_candidate_page_local_span_mean_std: float
    avg_candidate_page_local_span_max_mean: float
    avg_candidate_page_local_span_max_std: float
    avg_page_interval_visits_mean: float
    avg_page_interval_visits_std: float
    avg_page_interval_overlaps_mean: float
    avg_page_interval_overlaps_std: float
    avg_candidate_slice_count_mean: float
    avg_candidate_slice_count_std: float
    avg_candidate_slice_len_mean_mean: float
    avg_candidate_slice_len_mean_std: float
    avg_empty_slice_count_mean: float
    avg_empty_slice_count_std: float
    avg_nonempty_candidate_slice_count_mean: float
    avg_nonempty_candidate_slice_count_std: float
    avg_candidate_slice_len_nonempty_mean_mean: float
    avg_candidate_slice_len_nonempty_mean_std: float
    avg_candidate_slice_len_max_mean: float
    avg_candidate_slice_len_max_std: float
    avg_vectorized_candidate_slice_count_mean: float
    avg_vectorized_candidate_slice_count_std: float
    avg_scalar_candidate_slice_count_mean: float
    avg_scalar_candidate_slice_count_std: float
    avg_vectorized_candidate_record_count_mean: float
    avg_vectorized_candidate_record_count_std: float
    avg_scalar_candidate_record_count_mean: float
    avg_scalar_candidate_record_count_std: float
    avg_batched_candidate_slice_count_mean: float
    avg_batched_candidate_slice_count_std: float
    avg_batched_candidate_record_count_mean: float
    avg_batched_candidate_record_count_std: float
    avg_candidate_record_count_mean: float
    avg_candidate_record_count_std: float
    avg_semantic_check_count_mean: float
    avg_semantic_check_count_std: float
    avg_accepted_record_count_mean: float
    avg_accepted_record_count_std: float
    filter_bisect_s_mean: float
    filter_bisect_s_std: float
    filter_candidate_scan_s_mean: float
    filter_candidate_scan_s_std: float
    filter_candidate_extract_s_mean: float
    filter_candidate_extract_s_std: float
    filter_candidate_array_s_mean: float
    filter_candidate_array_s_std: float
    filter_semantic_check_s_mean: float
    filter_semantic_check_s_std: float
    filter_materialize_s_mean: float
    filter_materialize_s_std: float


def _run_one(
    *,
    name: str,
    chosen_threshold: Optional[int],
    make_sfc: Callable[[], object],
    points: np.ndarray,
    queries: Sequence[QueryWindow],
    quantizer: Quantizer,
    b: int,
    profile: bool,
    semantic: str,
    locality_qn: int,
    locality_max_cells: int,
    use_sortdim: bool = False,
    training_queries: Optional[Sequence[QueryWindow]] = None,
) -> CurveResult:
    sfc = make_sfc()

    def _exact_locality(q: QueryWindow) -> Optional[Tuple[int, int]]:
        try:
            qq = quantizer.quantize_window(q)
            low = np.asarray(qq.low_cell, dtype=np.int64)
            high = np.asarray(qq.high_cell, dtype=np.int64)
            if low.ndim != 1 or low.shape != high.shape or low.shape[0] < 1:
                return None
            lo = np.minimum(low, high)
            hi = np.maximum(low, high)
            extents = hi - lo + 1
            if np.any(extents <= 0):
                return None
            n_cells = int(np.prod(extents.astype(np.int64)))
            if n_cells <= 0:
                return None
            if int(locality_max_cells) > 0 and n_cells > int(locality_max_cells):
                return None
            grids = [np.arange(int(lo[i]), int(hi[i]) + 1, dtype=np.int64)
                     for i in range(lo.shape[0])]
            mesh = np.meshgrid(*grids, indexing="ij")
            cells = np.stack([m.reshape(-1) for m in mesh], axis=1)
            encode_cells = getattr(sfc, "encode_cells", None)
            if callable(encode_cells):
                codes = np.asarray(encode_cells(cells.astype(np.uint64)), dtype=np.int64)
            else:
                codes = np.asarray(
                    [int(sfc.encode_cell(np.asarray(c, dtype=np.uint64))) for c in cells],
                    dtype=np.int64,
                )
            if codes.size == 0:
                return None
            codes = np.unique(codes)
            span = int(codes[-1] - codes[0] + 1)
            seg = int(1 + np.count_nonzero(np.diff(codes) != 1))
            return int(seg), int(span)
        except Exception:
            return None

    t0 = time.perf_counter()
    artifacts = build_artifacts(
        points=points, 
        sfc=sfc, 
        quantizer=quantizer, 
        b=int(b),
        use_sortdim=use_sortdim,
        training_queries=training_queries,
    )
    setter = getattr(sfc, "set_page_start_codes", None)
    if callable(setter):
        setter(artifacts.page_start_codes)
    t1 = time.perf_counter()

    engine = QueryEngine(artifacts=artifacts, sfc=sfc, quantizer=quantizer, semantic=str(semantic))

    page_accesses = 0
    segments = 0
    util = 0.0

    intervals_s = 0.0
    lookup_s = 0.0
    filter_s = 0.0
    total_s = 0.0

    interval_count = 0
    covered_code_len = 0

    records_scanned = 0
    pass_code = 0
    pass_semantic = 0
    hit_segments = 0
    page_count = 0
    touched_page_record_count = 0
    candidate_page_count = 0
    candidate_page_record_count = 0
    empty_candidate_page_count = 0
    candidate_page_local_span_sum = 0
    candidate_page_local_span_mean_sum = 0.0
    candidate_page_local_span_max = 0
    page_interval_visits = 0
    page_interval_overlaps = 0
    candidate_slice_count = 0
    candidate_slice_len_mean_sum = 0.0
    empty_slice_count = 0
    nonempty_candidate_slice_count = 0
    candidate_slice_len_nonempty_mean_sum = 0.0
    candidate_slice_len_max = 0
    vectorized_candidate_slice_count = 0
    scalar_candidate_slice_count = 0
    vectorized_candidate_record_count = 0
    scalar_candidate_record_count = 0
    batched_candidate_slice_count = 0
    batched_candidate_record_count = 0
    candidate_record_count = 0
    semantic_check_count = 0
    accepted_record_count = 0
    filter_bisect_s = 0.0
    filter_candidate_scan_s = 0.0
    filter_candidate_extract_s = 0.0
    filter_candidate_array_s = 0.0
    filter_semantic_check_s = 0.0
    filter_materialize_s = 0.0

    exact_cell_segments = 0
    span = 0
    locality_n = 0

    if int(locality_qn) > 0:
        for i, q in enumerate(queries):
            if i >= int(locality_qn):
                break
            loc = _exact_locality(q)
            if loc is None:
                continue
            seg_i, span_i = loc
            exact_cell_segments += int(seg_i)
            span += int(span_i)
            locality_n += 1

    t2 = time.perf_counter()
    for q in queries:
        if profile:
            pr = engine.query_profiled(q)
            r = pr.result
            page_accesses += int(r.metrics.physical_pages)
            segments += int(r.metrics.segments)
            util += float(r.metrics.mean_page_amp)
            intervals_s += float(pr.profile.intervals_s)
            lookup_s += float(pr.profile.lookup_s)
            filter_s += float(pr.profile.filter_s)
            total_s += float(pr.profile.total_s)
            interval_count += int(getattr(pr.profile, "interval_count", 0))
            covered_code_len += int(getattr(pr.profile, "covered_code_len", 0))
            records_scanned += int(pr.profile.records_scanned)
            pass_code += int(pr.profile.pass_code)
            pass_semantic += int(pr.profile.pass_semantic)
            hit_segments += int(pr.profile.hit_segments)
            page_count += int(getattr(pr.profile, "page_count", 0))
            touched_page_record_count += int(getattr(pr.profile, "touched_page_record_count", 0))
            candidate_page_count += int(getattr(pr.profile, "candidate_page_count", 0))
            candidate_page_record_count += int(getattr(pr.profile, "candidate_page_record_count", 0))
            empty_candidate_page_count += int(getattr(pr.profile, "empty_candidate_page_count", 0))
            candidate_page_local_span_sum += int(getattr(pr.profile, "candidate_page_local_span_sum", 0))
            candidate_page_local_span_mean_sum += float(getattr(pr.profile, "candidate_page_local_span_mean", 0.0))
            candidate_page_local_span_max += int(getattr(pr.profile, "candidate_page_local_span_max", 0))
            page_interval_visits += int(getattr(pr.profile, "page_interval_visits", 0))
            page_interval_overlaps += int(getattr(pr.profile, "page_interval_overlaps", 0))
            candidate_slice_count += int(getattr(pr.profile, "candidate_slice_count", 0))
            candidate_slice_len_mean_sum += float(getattr(pr.profile, "candidate_slice_len_mean", 0.0))
            empty_slice_count += int(getattr(pr.profile, "empty_slice_count", 0))
            nonempty_candidate_slice_count += int(getattr(pr.profile, "nonempty_candidate_slice_count", 0))
            candidate_slice_len_nonempty_mean_sum += float(getattr(pr.profile, "candidate_slice_len_nonempty_mean", 0.0))
            candidate_slice_len_max += int(getattr(pr.profile, "candidate_slice_len_max", 0))
            vectorized_candidate_slice_count += int(getattr(pr.profile, "vectorized_candidate_slice_count", 0))
            scalar_candidate_slice_count += int(getattr(pr.profile, "scalar_candidate_slice_count", 0))
            vectorized_candidate_record_count += int(getattr(pr.profile, "vectorized_candidate_record_count", 0))
            scalar_candidate_record_count += int(getattr(pr.profile, "scalar_candidate_record_count", 0))
            batched_candidate_slice_count += int(getattr(pr.profile, "batched_candidate_slice_count", 0))
            batched_candidate_record_count += int(getattr(pr.profile, "batched_candidate_record_count", 0))
            candidate_record_count += int(getattr(pr.profile, "candidate_record_count", 0))
            semantic_check_count += int(getattr(pr.profile, "semantic_check_count", 0))
            accepted_record_count += int(getattr(pr.profile, "accepted_record_count", 0))
            filter_bisect_s += float(getattr(pr.profile, "filter_bisect_s", 0.0))
            filter_candidate_scan_s += float(getattr(pr.profile, "filter_candidate_scan_s", 0.0))
            filter_candidate_extract_s += float(getattr(pr.profile, "filter_candidate_extract_s", 0.0))
            filter_candidate_array_s += float(getattr(pr.profile, "filter_candidate_array_s", 0.0))
            filter_semantic_check_s += float(getattr(pr.profile, "filter_semantic_check_s", 0.0))
            filter_materialize_s += float(getattr(pr.profile, "filter_materialize_s", 0.0))
        else:
            r = engine.query(q)
            page_accesses += int(r.metrics.physical_pages)
            segments += int(r.metrics.segments)
            util += float(r.metrics.mean_page_amp)
    t3 = time.perf_counter()

    qn = max(1, len(queries))
    loc_qn = max(1, int(locality_n))
    return CurveResult(
        name=str(name),
        chosen_threshold=(int(chosen_threshold) if chosen_threshold is not None else None),
        build_s=float(t1 - t0),
        query_s=float(t3 - t2),
        avg_page_accesses=float(page_accesses / qn),
        avg_segments=float(segments / qn),
        avg_exact_cell_segments=(float(exact_cell_segments / loc_qn) if int(locality_qn) > 0 and int(locality_n) > 0 else None),
        avg_span=(float(span / loc_qn) if int(locality_qn) > 0 and int(locality_n) > 0 else None),
        avg_page_amplification=float(util / qn),
        intervals_s=float(intervals_s),
        lookup_s=float(lookup_s),
        filter_s=float(filter_s),
        total_s=float(total_s),
        avg_interval_count=float(interval_count / qn),
        avg_covered_code_len=float(covered_code_len / qn),
        avg_records_scanned=float(records_scanned / qn),
        avg_pass_code=float(pass_code / qn),
        avg_pass_semantic=float(pass_semantic / qn),
        avg_hit_segments=float(hit_segments / qn),
        avg_page_count=float(page_count / qn),
        avg_touched_page_record_count=float(touched_page_record_count / qn),
        avg_candidate_page_count=float(candidate_page_count / qn),
        avg_candidate_page_record_count=float(candidate_page_record_count / qn),
        avg_empty_candidate_page_count=float(empty_candidate_page_count / qn),
        avg_candidate_page_local_span_sum=float(candidate_page_local_span_sum / qn),
        avg_candidate_page_local_span_mean=float(candidate_page_local_span_mean_sum / qn),
        avg_candidate_page_local_span_max=float(candidate_page_local_span_max / qn),
        avg_page_interval_visits=float(page_interval_visits / qn),
        avg_page_interval_overlaps=float(page_interval_overlaps / qn),
        avg_candidate_slice_count=float(candidate_slice_count / qn),
        avg_candidate_slice_len_mean=float(candidate_slice_len_mean_sum / qn),
        avg_empty_slice_count=float(empty_slice_count / qn),
        avg_nonempty_candidate_slice_count=float(nonempty_candidate_slice_count / qn),
        avg_candidate_slice_len_nonempty_mean=float(candidate_slice_len_nonempty_mean_sum / qn),
        avg_candidate_slice_len_max=float(candidate_slice_len_max / qn),
        avg_vectorized_candidate_slice_count=float(vectorized_candidate_slice_count / qn),
        avg_scalar_candidate_slice_count=float(scalar_candidate_slice_count / qn),
        avg_vectorized_candidate_record_count=float(vectorized_candidate_record_count / qn),
        avg_scalar_candidate_record_count=float(scalar_candidate_record_count / qn),
        avg_batched_candidate_slice_count=float(batched_candidate_slice_count / qn),
        avg_batched_candidate_record_count=float(batched_candidate_record_count / qn),
        avg_candidate_record_count=float(candidate_record_count / qn),
        avg_semantic_check_count=float(semantic_check_count / qn),
        avg_accepted_record_count=float(accepted_record_count / qn),
        filter_bisect_s=float(filter_bisect_s),
        filter_candidate_scan_s=float(filter_candidate_scan_s),
        filter_candidate_extract_s=float(filter_candidate_extract_s),
        filter_candidate_array_s=float(filter_candidate_array_s),
        filter_semantic_check_s=float(filter_semantic_check_s),
        filter_materialize_s=float(filter_materialize_s),
    )


def _parse_threshold_sweep(s: str) -> List[int]:
    out: List[int] = []
    for part in str(s).split(","):
        p = part.strip()
        if not p:
            continue
        v = int(p)
        if v <= 0:
            continue
        out.append(v)
    # de-dup while preserving order
    seen = set()
    uniq: List[int] = []
    for v in out:
        if v in seen:
            continue
        uniq.append(v)
        seen.add(v)
    return uniq


def _tune_lmsfc_threshold(
    *,
    make_base_sfc: Callable[[], object],
    thresholds: Sequence[int],
    points: np.ndarray,
    train_queries: Sequence[QueryWindow],
    quantizer: Quantizer,
    b: int,
    profile: bool,
    semantic: str,
) -> int:
    if not thresholds:
        s0 = make_base_sfc()
        return int(getattr(s0, "threshold", 0) or 0)

    best_th = int(thresholds[0])
    best_qs = float("inf")
    for th in thresholds:
        th = int(th)

        def _make() -> object:
            s = make_base_sfc()
            if hasattr(s, "threshold"):
                setattr(s, "threshold", int(th))
            return s

        r = _run_one(
            name="_tune",
            chosen_threshold=int(th),
            make_sfc=_make,
            points=points,
            queries=train_queries,
            quantizer=quantizer,
            b=int(b),
            profile=bool(profile),
            semantic=str(semantic),
            locality_qn=0,
            locality_max_cells=0,
        )
        qs = float(r.query_s)
        if (qs < best_qs) or (qs == best_qs and int(th) < int(best_th)):
            best_qs = qs
            best_th = int(th)
    return int(best_th)


def _aggregate(name: str, runs: Sequence[CurveResult]) -> CurveStats:
    ths = [int(r.chosen_threshold) for r in runs if r.chosen_threshold is not None]
    th_mode: Optional[int]
    if ths:
        c = Counter(ths)
        th_mode = int(c.most_common(1)[0][0])
    else:
        th_mode = None

    arr_build = np.asarray([r.build_s for r in runs], dtype=np.float64)
    arr_query = np.asarray([r.query_s for r in runs], dtype=np.float64)
    arr_pa = np.asarray([r.avg_page_accesses for r in runs], dtype=np.float64)
    arr_seg = np.asarray([r.avg_segments for r in runs], dtype=np.float64)
    vals_exact_seg = [r.avg_exact_cell_segments for r in runs if r.avg_exact_cell_segments is not None]
    vals_span = [r.avg_span for r in runs if r.avg_span is not None]
    arr_util = np.asarray([r.avg_page_amplification for r in runs], dtype=np.float64)
    arr_intervals = np.asarray([r.intervals_s for r in runs], dtype=np.float64)
    arr_lookup = np.asarray([r.lookup_s for r in runs], dtype=np.float64)
    arr_filter = np.asarray([r.filter_s for r in runs], dtype=np.float64)
    arr_total = np.asarray([r.total_s for r in runs], dtype=np.float64)
    arr_intv_cnt = np.asarray([r.avg_interval_count for r in runs], dtype=np.float64)
    arr_cover_len = np.asarray([r.avg_covered_code_len for r in runs], dtype=np.float64)
    arr_scanned = np.asarray([r.avg_records_scanned for r in runs], dtype=np.float64)
    arr_pass_code = np.asarray([r.avg_pass_code for r in runs], dtype=np.float64)
    arr_pass_sem = np.asarray([r.avg_pass_semantic for r in runs], dtype=np.float64)
    arr_hit_seg = np.asarray([r.avg_hit_segments for r in runs], dtype=np.float64)
    arr_page_count = np.asarray([r.avg_page_count for r in runs], dtype=np.float64)
    arr_touched_page_record_count = np.asarray([r.avg_touched_page_record_count for r in runs], dtype=np.float64)
    arr_candidate_page_count = np.asarray([r.avg_candidate_page_count for r in runs], dtype=np.float64)
    arr_candidate_page_record_count = np.asarray([r.avg_candidate_page_record_count for r in runs], dtype=np.float64)
    arr_empty_candidate_page_count = np.asarray([r.avg_empty_candidate_page_count for r in runs], dtype=np.float64)
    arr_candidate_page_local_span_sum = np.asarray([r.avg_candidate_page_local_span_sum for r in runs], dtype=np.float64)
    arr_candidate_page_local_span_mean = np.asarray([r.avg_candidate_page_local_span_mean for r in runs], dtype=np.float64)
    arr_candidate_page_local_span_max = np.asarray([r.avg_candidate_page_local_span_max for r in runs], dtype=np.float64)
    arr_page_interval_visits = np.asarray([r.avg_page_interval_visits for r in runs], dtype=np.float64)
    arr_page_interval_overlaps = np.asarray([r.avg_page_interval_overlaps for r in runs], dtype=np.float64)
    arr_candidate_slice_count = np.asarray([r.avg_candidate_slice_count for r in runs], dtype=np.float64)
    arr_candidate_slice_len_mean = np.asarray([r.avg_candidate_slice_len_mean for r in runs], dtype=np.float64)
    arr_empty_slice_count = np.asarray([r.avg_empty_slice_count for r in runs], dtype=np.float64)
    arr_nonempty_candidate_slice_count = np.asarray([r.avg_nonempty_candidate_slice_count for r in runs], dtype=np.float64)
    arr_candidate_slice_len_nonempty_mean = np.asarray([r.avg_candidate_slice_len_nonempty_mean for r in runs], dtype=np.float64)
    arr_candidate_slice_len_max = np.asarray([r.avg_candidate_slice_len_max for r in runs], dtype=np.float64)
    arr_vectorized_candidate_slice_count = np.asarray([r.avg_vectorized_candidate_slice_count for r in runs], dtype=np.float64)
    arr_scalar_candidate_slice_count = np.asarray([r.avg_scalar_candidate_slice_count for r in runs], dtype=np.float64)
    arr_vectorized_candidate_record_count = np.asarray([r.avg_vectorized_candidate_record_count for r in runs], dtype=np.float64)
    arr_scalar_candidate_record_count = np.asarray([r.avg_scalar_candidate_record_count for r in runs], dtype=np.float64)
    arr_batched_candidate_slice_count = np.asarray([r.avg_batched_candidate_slice_count for r in runs], dtype=np.float64)
    arr_batched_candidate_record_count = np.asarray([r.avg_batched_candidate_record_count for r in runs], dtype=np.float64)
    arr_candidate_record_count = np.asarray([r.avg_candidate_record_count for r in runs], dtype=np.float64)
    arr_semantic_check_count = np.asarray([r.avg_semantic_check_count for r in runs], dtype=np.float64)
    arr_accepted_record_count = np.asarray([r.avg_accepted_record_count for r in runs], dtype=np.float64)
    arr_filter_bisect = np.asarray([r.filter_bisect_s for r in runs], dtype=np.float64)
    arr_filter_candidate_scan = np.asarray([r.filter_candidate_scan_s for r in runs], dtype=np.float64)
    arr_filter_candidate_extract = np.asarray([r.filter_candidate_extract_s for r in runs], dtype=np.float64)
    arr_filter_candidate_array = np.asarray([r.filter_candidate_array_s for r in runs], dtype=np.float64)
    arr_filter_semantic_check = np.asarray([r.filter_semantic_check_s for r in runs], dtype=np.float64)
    arr_filter_materialize = np.asarray([r.filter_materialize_s for r in runs], dtype=np.float64)

    def mean_std(a: np.ndarray) -> Tuple[float, float]:
        if a.size <= 1:
            return float(a.mean() if a.size else 0.0), 0.0
        return float(a.mean()), float(a.std(ddof=1))

    def mean_std_list(a: List[float]) -> Tuple[Optional[float], Optional[float]]:
        if not a:
            return None, None
        arr = np.asarray(a, dtype=np.float64)
        if arr.size <= 1:
            return float(arr.mean()), 0.0
        return float(arr.mean()), float(arr.std(ddof=1))

    build_mean, build_std = mean_std(arr_build)
    query_mean, query_std = mean_std(arr_query)
    pa_mean, pa_std = mean_std(arr_pa)
    seg_mean, seg_std = mean_std(arr_seg)
    exact_seg_mean, exact_seg_std = mean_std_list([float(v) for v in vals_exact_seg])
    span_mean, span_std = mean_std_list([float(v) for v in vals_span])
    util_mean, util_std = mean_std(arr_util)
    intervals_mean, intervals_std = mean_std(arr_intervals)
    lookup_mean, lookup_std = mean_std(arr_lookup)
    filter_mean, filter_std = mean_std(arr_filter)
    total_mean, total_std = mean_std(arr_total)
    intv_cnt_mean, intv_cnt_std = mean_std(arr_intv_cnt)
    cover_len_mean, cover_len_std = mean_std(arr_cover_len)
    scanned_mean, scanned_std = mean_std(arr_scanned)
    pass_code_mean, pass_code_std = mean_std(arr_pass_code)
    pass_sem_mean, pass_sem_std = mean_std(arr_pass_sem)
    hit_seg_mean, hit_seg_std = mean_std(arr_hit_seg)
    page_count_mean, page_count_std = mean_std(arr_page_count)
    touched_page_record_count_mean, touched_page_record_count_std = mean_std(arr_touched_page_record_count)
    candidate_page_count_mean, candidate_page_count_std = mean_std(arr_candidate_page_count)
    candidate_page_record_count_mean, candidate_page_record_count_std = mean_std(arr_candidate_page_record_count)
    empty_candidate_page_count_mean, empty_candidate_page_count_std = mean_std(arr_empty_candidate_page_count)
    candidate_page_local_span_sum_mean, candidate_page_local_span_sum_std = mean_std(arr_candidate_page_local_span_sum)
    candidate_page_local_span_mean_mean, candidate_page_local_span_mean_std = mean_std(arr_candidate_page_local_span_mean)
    candidate_page_local_span_max_mean, candidate_page_local_span_max_std = mean_std(arr_candidate_page_local_span_max)
    page_interval_visits_mean, page_interval_visits_std = mean_std(arr_page_interval_visits)
    page_interval_overlaps_mean, page_interval_overlaps_std = mean_std(arr_page_interval_overlaps)
    candidate_slice_count_mean, candidate_slice_count_std = mean_std(arr_candidate_slice_count)
    candidate_slice_len_mean_mean, candidate_slice_len_mean_std = mean_std(arr_candidate_slice_len_mean)
    empty_slice_count_mean, empty_slice_count_std = mean_std(arr_empty_slice_count)
    nonempty_candidate_slice_count_mean, nonempty_candidate_slice_count_std = mean_std(arr_nonempty_candidate_slice_count)
    candidate_slice_len_nonempty_mean_mean, candidate_slice_len_nonempty_mean_std = mean_std(arr_candidate_slice_len_nonempty_mean)
    candidate_slice_len_max_mean, candidate_slice_len_max_std = mean_std(arr_candidate_slice_len_max)
    vectorized_candidate_slice_count_mean, vectorized_candidate_slice_count_std = mean_std(arr_vectorized_candidate_slice_count)
    scalar_candidate_slice_count_mean, scalar_candidate_slice_count_std = mean_std(arr_scalar_candidate_slice_count)
    vectorized_candidate_record_count_mean, vectorized_candidate_record_count_std = mean_std(arr_vectorized_candidate_record_count)
    scalar_candidate_record_count_mean, scalar_candidate_record_count_std = mean_std(arr_scalar_candidate_record_count)
    batched_candidate_slice_count_mean, batched_candidate_slice_count_std = mean_std(arr_batched_candidate_slice_count)
    batched_candidate_record_count_mean, batched_candidate_record_count_std = mean_std(arr_batched_candidate_record_count)
    candidate_record_count_mean, candidate_record_count_std = mean_std(arr_candidate_record_count)
    semantic_check_count_mean, semantic_check_count_std = mean_std(arr_semantic_check_count)
    accepted_record_count_mean, accepted_record_count_std = mean_std(arr_accepted_record_count)
    filter_bisect_mean, filter_bisect_std = mean_std(arr_filter_bisect)
    filter_candidate_scan_mean, filter_candidate_scan_std = mean_std(arr_filter_candidate_scan)
    filter_candidate_extract_mean, filter_candidate_extract_std = mean_std(arr_filter_candidate_extract)
    filter_candidate_array_mean, filter_candidate_array_std = mean_std(arr_filter_candidate_array)
    filter_semantic_check_mean, filter_semantic_check_std = mean_std(arr_filter_semantic_check)
    filter_materialize_mean, filter_materialize_std = mean_std(arr_filter_materialize)

    return CurveStats(
        name=str(name),
        chosen_threshold_mode=th_mode,
        build_s_mean=build_mean,
        build_s_std=build_std,
        query_s_mean=query_mean,
        query_s_std=query_std,
        avg_page_accesses_mean=pa_mean,
        avg_page_accesses_std=pa_std,
        avg_segments_mean=seg_mean,
        avg_segments_std=seg_std,
        avg_exact_cell_segments_mean=exact_seg_mean,
        avg_exact_cell_segments_std=exact_seg_std,
        avg_span_mean=span_mean,
        avg_span_std=span_std,
        avg_page_amplification_mean=util_mean,
        avg_page_amplification_std=util_std,
        intervals_s_mean=intervals_mean,
        intervals_s_std=intervals_std,
        lookup_s_mean=lookup_mean,
        lookup_s_std=lookup_std,
        filter_s_mean=filter_mean,
        filter_s_std=filter_std,
        total_s_mean=total_mean,
        total_s_std=total_std,
        avg_interval_count_mean=intv_cnt_mean,
        avg_interval_count_std=intv_cnt_std,
        avg_covered_code_len_mean=cover_len_mean,
        avg_covered_code_len_std=cover_len_std,
        avg_records_scanned_mean=scanned_mean,
        avg_records_scanned_std=scanned_std,
        avg_pass_code_mean=pass_code_mean,
        avg_pass_code_std=pass_code_std,
        avg_pass_semantic_mean=pass_sem_mean,
        avg_pass_semantic_std=pass_sem_std,
        avg_hit_segments_mean=hit_seg_mean,
        avg_hit_segments_std=hit_seg_std,
        avg_page_count_mean=page_count_mean,
        avg_page_count_std=page_count_std,
        avg_touched_page_record_count_mean=touched_page_record_count_mean,
        avg_touched_page_record_count_std=touched_page_record_count_std,
        avg_candidate_page_count_mean=candidate_page_count_mean,
        avg_candidate_page_count_std=candidate_page_count_std,
        avg_candidate_page_record_count_mean=candidate_page_record_count_mean,
        avg_candidate_page_record_count_std=candidate_page_record_count_std,
        avg_empty_candidate_page_count_mean=empty_candidate_page_count_mean,
        avg_empty_candidate_page_count_std=empty_candidate_page_count_std,
        avg_candidate_page_local_span_sum_mean=candidate_page_local_span_sum_mean,
        avg_candidate_page_local_span_sum_std=candidate_page_local_span_sum_std,
        avg_candidate_page_local_span_mean_mean=candidate_page_local_span_mean_mean,
        avg_candidate_page_local_span_mean_std=candidate_page_local_span_mean_std,
        avg_candidate_page_local_span_max_mean=candidate_page_local_span_max_mean,
        avg_candidate_page_local_span_max_std=candidate_page_local_span_max_std,
        avg_page_interval_visits_mean=page_interval_visits_mean,
        avg_page_interval_visits_std=page_interval_visits_std,
        avg_page_interval_overlaps_mean=page_interval_overlaps_mean,
        avg_page_interval_overlaps_std=page_interval_overlaps_std,
        avg_candidate_slice_count_mean=candidate_slice_count_mean,
        avg_candidate_slice_count_std=candidate_slice_count_std,
        avg_candidate_slice_len_mean_mean=candidate_slice_len_mean_mean,
        avg_candidate_slice_len_mean_std=candidate_slice_len_mean_std,
        avg_empty_slice_count_mean=empty_slice_count_mean,
        avg_empty_slice_count_std=empty_slice_count_std,
        avg_nonempty_candidate_slice_count_mean=nonempty_candidate_slice_count_mean,
        avg_nonempty_candidate_slice_count_std=nonempty_candidate_slice_count_std,
        avg_candidate_slice_len_nonempty_mean_mean=candidate_slice_len_nonempty_mean_mean,
        avg_candidate_slice_len_nonempty_mean_std=candidate_slice_len_nonempty_mean_std,
        avg_candidate_slice_len_max_mean=candidate_slice_len_max_mean,
        avg_candidate_slice_len_max_std=candidate_slice_len_max_std,
        avg_vectorized_candidate_slice_count_mean=vectorized_candidate_slice_count_mean,
        avg_vectorized_candidate_slice_count_std=vectorized_candidate_slice_count_std,
        avg_scalar_candidate_slice_count_mean=scalar_candidate_slice_count_mean,
        avg_scalar_candidate_slice_count_std=scalar_candidate_slice_count_std,
        avg_vectorized_candidate_record_count_mean=vectorized_candidate_record_count_mean,
        avg_vectorized_candidate_record_count_std=vectorized_candidate_record_count_std,
        avg_scalar_candidate_record_count_mean=scalar_candidate_record_count_mean,
        avg_scalar_candidate_record_count_std=scalar_candidate_record_count_std,
        avg_batched_candidate_slice_count_mean=batched_candidate_slice_count_mean,
        avg_batched_candidate_slice_count_std=batched_candidate_slice_count_std,
        avg_batched_candidate_record_count_mean=batched_candidate_record_count_mean,
        avg_batched_candidate_record_count_std=batched_candidate_record_count_std,
        avg_candidate_record_count_mean=candidate_record_count_mean,
        avg_candidate_record_count_std=candidate_record_count_std,
        avg_semantic_check_count_mean=semantic_check_count_mean,
        avg_semantic_check_count_std=semantic_check_count_std,
        avg_accepted_record_count_mean=accepted_record_count_mean,
        avg_accepted_record_count_std=accepted_record_count_std,
        filter_bisect_s_mean=filter_bisect_mean,
        filter_bisect_s_std=filter_bisect_std,
        filter_candidate_scan_s_mean=filter_candidate_scan_mean,
        filter_candidate_scan_s_std=filter_candidate_scan_std,
        filter_candidate_extract_s_mean=filter_candidate_extract_mean,
        filter_candidate_extract_s_std=filter_candidate_extract_std,
        filter_candidate_array_s_mean=filter_candidate_array_mean,
        filter_candidate_array_s_std=filter_candidate_array_std,
        filter_semantic_check_s_mean=filter_semantic_check_mean,
        filter_semantic_check_s_std=filter_semantic_check_std,
        filter_materialize_s_mean=filter_materialize_mean,
        filter_materialize_s_std=filter_materialize_std,
    )


def _print_table(results: Sequence[CurveStats], *, profile: bool, locality: bool, cover: bool) -> None:
    headers = [
        ("curve", 18),
        ("cfg", 10),
        ("build_s", 18),
        ("query_s", 18),
        ("avg_pages", 18),
        ("avg_seg", 18),
        ("span", 18),
        ("avg_page_amp", 20),
    ]
    if not locality:
        headers = [h for h in headers if h[0] not in {"span"}]
    if profile:
        headers += [
            ("t_int", 18),
            ("t_lookup", 18),
            ("t_filter", 18),
            ("t_total", 18),
        ]
        if cover:
            headers += [
                ("intv_cnt", 18),
                ("cover_len", 18),
            ]
        headers += [
            ("scan", 18),
            ("pass_code", 18),
            ("pass_cell", 18),
            ("hit_seg", 18),
            ("page_cnt", 18),
            ("pg_intv", 18),
            ("slice_cnt", 18),
            ("slice_len", 18),
            ("empty_slice", 18),
        ]

    def fmt_row(vals: List[str]) -> str:
        parts = []
        for (h, w), v in zip(headers, vals):
            parts.append(str(v).ljust(w))
        return " ".join(parts).rstrip()

    print(fmt_row([h for h, _ in headers]))
    print(fmt_row(["-" * len(h) for h, _ in headers]))
    for r in results:
        cfg = "-"
        if r.chosen_threshold_mode is not None:
            cfg = f"th={int(r.chosen_threshold_mode)}"

        avg_seg_s = f"{r.avg_segments_mean:.4f}±{r.avg_segments_std:.4f}"
        if locality and r.avg_exact_cell_segments_mean is not None and r.avg_exact_cell_segments_std is not None:
            avg_seg_s = f"{r.avg_exact_cell_segments_mean:.2f}±{r.avg_exact_cell_segments_std:.2f}"

        span_s = "-"
        if locality and r.avg_span_mean is not None and r.avg_span_std is not None:
            span_s = f"{r.avg_span_mean:.2f}±{r.avg_span_std:.2f}"

        row = [
            r.name,
            cfg,
            f"{r.build_s_mean:.4f}±{r.build_s_std:.4f}",
            f"{r.query_s_mean:.4f}±{r.query_s_std:.4f}",
            f"{r.avg_page_accesses_mean:.4f}±{r.avg_page_accesses_std:.4f}",
            avg_seg_s,
            span_s,
            f"{r.avg_page_amplification_mean:.6f}±{r.avg_page_amplification_std:.6f}",
        ]
        if not locality:
            row = [v for (h, _), v in zip([
                ("curve", 18),
                ("cfg", 10),
                ("build_s", 18),
                ("query_s", 18),
                ("avg_pages", 18),
                ("avg_seg", 18),
                ("span", 18),
                ("avg_page_amp", 20),
            ], row) if h not in {"span"}]
        if profile:
            row += [
                f"{r.intervals_s_mean:.4f}±{r.intervals_s_std:.4f}",
                f"{r.lookup_s_mean:.4f}±{r.lookup_s_std:.4f}",
                f"{r.filter_s_mean:.4f}±{r.filter_s_std:.4f}",
                f"{r.total_s_mean:.4f}±{r.total_s_std:.4f}",
            ]
            if cover:
                row += [
                    f"{r.avg_interval_count_mean:.2f}±{r.avg_interval_count_std:.2f}",
                    f"{r.avg_covered_code_len_mean:.2f}±{r.avg_covered_code_len_std:.2f}",
                ]
            row += [
                f"{r.avg_records_scanned_mean:.2f}±{r.avg_records_scanned_std:.2f}",
                f"{r.avg_pass_code_mean:.2f}±{r.avg_pass_code_std:.2f}",
                f"{r.avg_pass_semantic_mean:.2f}±{r.avg_pass_semantic_std:.2f}",
                f"{r.avg_hit_segments_mean:.2f}±{r.avg_hit_segments_std:.2f}",
                f"{r.avg_page_count_mean:.2f}±{r.avg_page_count_std:.2f}",
                f"{r.avg_page_interval_visits_mean:.2f}±{r.avg_page_interval_visits_std:.2f}",
                f"{r.avg_candidate_slice_count_mean:.2f}±{r.avg_candidate_slice_count_std:.2f}",
                f"{r.avg_candidate_slice_len_mean_mean:.2f}±{r.avg_candidate_slice_len_mean_std:.2f}",
                f"{r.avg_empty_slice_count_mean:.2f}±{r.avg_empty_slice_count_std:.2f}",
            ]
        print(
            fmt_row(
                row
            )
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--qn", type=int, default=1_000)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--b", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--train-qn", type=int, default=0)
    ap.add_argument("--train-iters", type=int, default=200)
    ap.add_argument("--train-per", type=int, default=10)
    ap.add_argument("--train-restarts", type=int, default=5)
    ap.add_argument("--train-reinforce-lr", type=float, default=0.05)
    ap.add_argument("--train-reinforce-tau-start", type=float, default=1.0)
    ap.add_argument("--train-reinforce-tau-end", type=float, default=0.2)
    ap.add_argument("--train-reinforce-baseline-momentum", type=float, default=0.9)
    ap.add_argument(
        "--train-objective",
        type=str,
        default="page_accesses",
        choices=["query_s", "page_accesses", "segments", "metrics", "exact_locality"],
    )
    ap.add_argument("--train-w-page-accesses", type=float, default=1.0)
    ap.add_argument("--train-w-segments", type=float, default=1.0)
    ap.add_argument("--train-w-span-per-cell", type=float, default=1.0)
    ap.add_argument("--train-locality-max-cells", type=int, default=20000)
    ap.add_argument("--train-locality-qn", type=int, default=0)
    ap.add_argument("--train-bo-init-size", type=int, default=120)
    ap.add_argument("--train-bo-rounds", type=int, default=8)
    ap.add_argument("--train-bo-evals-per-round", type=int, default=40)
    ap.add_argument("--train-bo-neighbor-candidates", type=int, default=2000)
    ap.add_argument("--train-bo-random-candidates", type=int, default=2000)
    ap.add_argument("--train-bo-topk", type=int, default=3)
    ap.add_argument("--train-bo-rf-init-points", type=int, default=8)
    ap.add_argument("--train-bo-rf-n-iter", type=int, default=16)
    ap.add_argument("--train-bo-no-bayes", action="store_true")
    ap.add_argument("--train-save-masks-file", type=str, default="")
    ap.add_argument("--timing-repeats", type=int, default=3)
    ap.add_argument("--timing-no-warmup", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--profile-cover", action="store_true")
    ap.add_argument("--semantic", type=str, default="cell", choices=["cell"])
    ap.add_argument("--area-min", type=float, default=1e-4)
    ap.add_argument("--area-max", type=float, default=0.2)
    ap.add_argument("--d", type=int, default=2,
                    help="Data dimensionality (default 2). d>2 requires LACSFC/Z/Hilbert curves only.")
    ap.add_argument(
        "--query-dist",
        type=str,
        default="uniform",
        choices=["uniform", "hotspot", "aspect_ratio", "hotspot_mixture", "aspect_hotspot"],
        help="Test query distribution (default: uniform)",
    )
    ap.add_argument(
        "--train-query-dist",
        type=str,
        default="",
        choices=["", "uniform", "hotspot", "aspect_ratio", "hotspot_mixture", "aspect_hotspot"],
        help="Train query distribution (default: same as --query-dist). Use for workload shift experiments.",
    )
    ap.add_argument("--queries-file", type=str, default="")
    ap.add_argument("--train-queries-file", type=str, default="")
    ap.add_argument("--hotspot-ratio", type=float, default=0.5)
    ap.add_argument("--aspect-ratio", type=float, default=8.0)
    ap.add_argument("--points", type=str, default="uniform")
    ap.add_argument("--points-file", type=str, default="")
    ap.add_argument(
        "--require-n",
        action="store_true",
        help=(
            "For --points=real: fail if the source file has fewer than --n "
            "points instead of silently using all available points. Use this "
            "for scale experiments to prevent silent under-fill."
        ),
    )
    ap.add_argument("--curves", type=str, default="row,column,z,hilbert")
    ap.add_argument("--lmsfc-masks-file", type=str, default="")
    ap.add_argument("--lmsfc-threshold", type=int, default=4)
    ap.add_argument("--tune-lmsfc-threshold", action="store_true")
    ap.add_argument(
        "--lmsfc-threshold-sweep",
        type=str,
        default="64,32,16,8,4,2,1",
        help="Comma-separated thresholds to sweep on train queries when --lmsfc-threshold is not set",
    )
    ap.add_argument("--locality-qn", type=int, default=0)
    ap.add_argument("--locality-max-cells", type=int, default=200_000)
    ap.add_argument("--use-sortdim", action="store_true", help="Enable Sort Dimension optimization")
    ap.add_argument("--lacsfc-k", type=int, default=5, help="LACSFC grid bits per dim (default 5)")
    ap.add_argument("--lacsfc-selected-weights", type=str, default="",
                    help="Path to selected_weights.npy for lacsfc_backendaware curve")
    ap.add_argument("--lacsfc-max-evals", type=int, default=1500, help="CMA-ES max evaluations for LACSFC-Opt")
    ap.add_argument("--lacsfc-mode", default="block_space",
                    choices=["block_space", "hierarchical", "func_space", "edge_space", "generator_es"],
                    help="CMA-ES optimisation mode for LACSFC-Opt (default: block_space); "
                         "generator_es = 路径二(3-目标 ES Generator)")
    ap.add_argument("--lacsfc-func-space-m", type=int, default=200,
                    help="Number of RBF centres for func_space mode (default: 200)")
    ap.add_argument("--lacsfc-sigma0", type=float, default=0.5,
                    help="CMA-ES initial step size for LACSFC-Opt (default: 0.5)")
    ap.add_argument("--lacsfc-popsize", type=int, default=None,
                    help="CMA-ES population size for LACSFC-Opt (default: auto)")
    ap.add_argument("--lacsfc-es-epochs", type=int, default=30,
                    help="Generator-ES training epochs (default: 30)")
    ap.add_argument("--lacsfc-es-n-perturb", type=int, default=16,
                    help="Generator-ES antithetic perturbation pairs per scene (default: 16)")
    ap.add_argument("--lacsfc-es-sigma", type=float, default=0.1,
                    help="Generator-ES perturbation std (default: 0.1)")
    ap.add_argument("--lacsfc-es-dataset-size", type=int, default=64,
                    help="Generator-ES scene dataset size (default: 64)")
    ap.add_argument("--lacsfc-es-learn-obj", action="store_true", default=True,
                    help="Learn objective weights (seg/span/pa) jointly via ES (default: True)")
    ap.add_argument("--lacsfc-es-no-learn-obj", dest="lacsfc_es_learn_obj", action="store_false",
                    help="Fix objective weights (do not learn them)")
    ap.add_argument("--lacsfc-es-init-obj", type=str, default="0.60,0.15,0.25",
                    help="Initial (or fixed) objective weights as 'w_seg,w_span,w_pa' (default: 0.60,0.15,0.25)")
    ap.add_argument("--lacsfc-v2-ckpt", type=str, default="",
                    help="Path to pre-trained FG checkpoint for legacy LACSFC-Neural-v2 "
                         "(produced by examples/train_lacsfc_v2_offline.py). "
                         "Required if 'lacsfc_neural_v2' is in --curves.")
    ap.add_argument("--lacsfc-cover-weight", type=float, default=0.0,
                    help="A-1: cover_len penalty weight in CMA-ES objective (0 = page_amp-only, 0.3-1.0 recommended)")
    ap.add_argument("--lacsfc-runtime-surrogate", default="none",
                    choices=["none", "empty_slice", "page_interval", "candidate_slice", "candidate_record"],
                    help="Runtime-aware CMA-ES surrogate penalty: none, empty_slice, page_interval, candidate_slice, or candidate_record")
    ap.add_argument("--lacsfc-runtime-weight", type=float, default=0.0,
                    help="Weight for the runtime-aware surrogate penalty in the CMA-ES objective")
    ap.add_argument("--lacsfc-runtime-normalize", default="raw",
                    choices=["raw", "relative"],
                    help="Runtime-aware penalty normalization: raw preserves legacy loss; relative uses warm-start-normalized pa and surrogate terms")
    ap.add_argument("--lacsfc-runtime-interval-weight", type=float, default=0.0,
                    help="Weight for interval-count cost in the runtime-aware CMA-ES objective")
    ap.add_argument("--lacsfc-objective", default="page_amp",
                    choices=["page_amp", "weighted_mvp"],
                    help="CMA-ES objective for LACSFC-Opt: exact page_amp only, or weighted MVP (seg+span+pa)")
    ap.add_argument("--lacsfc-learn-obj", action="store_true", default=True,
                    help="Adapt weighted_mvp objective weights online during CMA-ES (default: True)")
    ap.add_argument("--lacsfc-no-learn-obj", dest="lacsfc_learn_obj", action="store_false",
                    help="Use fixed weighted_mvp objective weights during CMA-ES")
    ap.add_argument("--lacsfc-init-obj", type=str, default="0.60,0.15,0.25",
                    help="Initial (or fixed) CMA-ES objective weights as 'w_seg,w_span,w_pa' (default: 0.60,0.15,0.25)")
    ap.add_argument("--lacsfc-init-strategy", default="heuristic",
                    choices=["heuristic", "random", "uniform"],
                    help="CMA-ES warm-start strategy for LACSFC-Opt (P1 ablation). "
                         "heuristic = dual_edge_weights-derived (default); "
                         "random = N(mean(log bd), 2); uniform = bv=0.")
    ap.add_argument("--lacsfc-artifact-dir", type=str, default="",
                    help="Optional directory for reusing LACSFC-Opt path/code_map artifacts across compatible runs.")
    ap.add_argument("--lacsfc-meta-enable", action="store_true",
                    help="Enable optional Tier3 meta warm-start path for LACSFC-Opt")
    ap.add_argument("--lacsfc-meta-model-dir", type=str, default="",
                    help="Directory containing <model_name>_generator.pt and optionally <model_name>_evaluator.pt")
    ap.add_argument("--lacsfc-meta-model-name", type=str, default="lacsfc_pretrained",
                    help="Base model name for Tier3 meta warm-start checkpoints")
    ap.add_argument("--lacsfc-meta-inner-steps", type=int, default=0,
                    help="Optional Tier3 inner-loop adaptation steps on the target workload")
    ap.add_argument("--lacsfc-meta-inner-lr", type=float, default=1e-4,
                    help="Learning rate for optional Tier3 inner-loop adaptation")
    ap.add_argument("--lacsfc-meta-use-evaluator", action="store_true", default=True,
                    help="Load and use evaluator during Tier3 inner-loop adaptation (default: True)")
    ap.add_argument("--lacsfc-meta-no-evaluator", dest="lacsfc_meta_use_evaluator", action="store_false",
                    help="Do not require/load evaluator; generator-only warm-start")
    ap.add_argument("--lacsfc-meta-fallback-cmaes", action="store_true", default=True,
                    help="Refine meta warm-start with the normal CMA-ES path (default: True)")
    ap.add_argument("--lacsfc-meta-direct", dest="lacsfc_meta_fallback_cmaes", action="store_false",
                    help="Directly deploy meta-generated weights without CMA-ES refinement")
    ap.add_argument("--lacsfc-meta-hidden", type=int, default=128,
                    help="Tier3 generator hidden dimension")
    ap.add_argument("--lacsfc-meta-layers", type=int, default=3,
                    help="Tier3 generator layer count")
    ap.add_argument("--lacsfc-meta-evaluator-hidden", type=int, default=128,
                    help="Tier3 evaluator hidden dimension")
    ap.add_argument("--lacsfc-meta-evaluator-layers", type=int, default=3,
                    help="Tier3 evaluator layer count")
    ap.add_argument("--lacsfc-shift-monitor-enable", action="store_true",
                    help="Enable experimental workload-shift monitor metadata path")
    ap.add_argument("--lacsfc-shift-window-size", type=int, default=50,
                    help="Rolling window size for experimental workload-shift monitor")
    ap.add_argument("--lacsfc-shift-pa-threshold", type=float, default=1.20,
                    help="Trigger ratio for rolling page amplification drift")
    ap.add_argument("--lacsfc-shift-score-threshold", type=float, default=1.20,
                    help="Trigger ratio for rolling three-factor score drift")
    ap.add_argument("--lacsfc-reopt-enable", action="store_true",
                    help="Enable experimental warm-start re-optimization metadata path when shift monitor triggers")
    ap.add_argument("--lacsfc-reopt-sigma-scale", type=float, default=0.25,
                    help="Reduced-sigma scale for experimental re-optimization")
    ap.add_argument("--lacsfc-reopt-max-evals", type=int, default=50,
                    help="Reduced max evaluations for experimental re-optimization")
    ap.add_argument("--lacsfc-state-dir", type=str, default="",
                    help="Optional state directory for experimental LACSFC shift/reopt metadata")
    ap.add_argument("--adacurve-block-size", type=int, default=256,
                    help="AdaCurve block_size_points (default 256)")
    ap.add_argument("--adacurve-compress-dim", type=int, default=8,
                    help="AdaCurve compress_dim (default 8)")
    ap.add_argument("--adacurve-epochs", type=int, default=50,
                    help="AdaCurve training epochs (default 50)")
    ap.add_argument("--adacurve-lr", type=float, default=1e-3,
                    help="AdaCurve paper-repro learning rate (default 1e-3)")
    ap.add_argument("--adacurve-query-batch-size", type=int, default=16,
                    help="AdaCurve paper-repro training queries per epoch (default 16)")
    ap.add_argument("--adacurve-train-sample-size", type=int, default=4096,
                    help="AdaCurve paper-repro tuple samples per epoch (default 4096)")
    ap.add_argument("--adacurve-fp-weight", type=float, default=1.0,
                    help="AdaCurve paper-repro false-positive block proxy weight (default 1.0)")
    ap.add_argument("--adacurve-tp-weight", type=float, default=1.0,
                    help="AdaCurve paper-repro true-positive clustering proxy weight (default 1.0)")
    ap.add_argument("--adacurve-recon-weight", type=float, default=0.05,
                    help="AdaCurve paper-repro autoencoder reconstruction weight (default 0.05)")
    ap.add_argument("--adacurve-smooth-weight", type=float, default=0.01,
                    help="AdaCurve paper-repro adjacent spatial smoothness weight (default 0.01)")
    ap.add_argument("--adacurve-margin", type=float, default=1.0,
                    help="AdaCurve paper-repro false-positive spread margin (default 1.0)")
    ap.add_argument("--bitallocz-bo-iters", type=int, default=80, help="BitAllocZ BO acquisition iterations (default 80)")
    ap.add_argument("--bitallocz-bo-init", type=int, default=5, help="BitAllocZ BO random init points (default 5)")
    ap.add_argument("--bitallocz-gd-iters", type=int, default=100, help="BitAllocZ GD iterations (default 100)")
    ap.add_argument("--bitallocz-gd-lr", type=float, default=0.1, help="BitAllocZ GD learning rate (default 0.1)")
    ap.add_argument("--bitallocz-total-bits", type=int, default=None, help="BitAllocZ total bit budget (default d*k)")
    ap.add_argument("--bitallocz-w-seg", type=float, default=1.0, help="BitAllocZ segments cost weight (default 1.0)")
    ap.add_argument("--bitallocz-w-pa", type=float, default=1.0, help="BitAllocZ page_accesses cost weight (default 1.0)")
    ap.add_argument("--bitallocz-w-cand", type=float, default=0.0, help="BitAllocZ candidates cost weight (default 0.0)")
    ap.add_argument("--bitallocz-max-cells", type=int, default=2_000_000, help="BitAllocZ max cells per query enum (default 2000000)")
    ap.add_argument("--output", type=str, default="",
                    help="Append one JSONL record per run to this file (default: no file output)")
    ap.add_argument("--exp-tag", type=str, default="",
                    help="Experiment tag stored in JSONL output (e.g. E1, E2)")
    args = ap.parse_args()

    repeats = int(args.repeat)
    if repeats <= 0:
        raise ValueError("--repeat must be > 0")

    curve_tokens = [t.strip() for t in str(args.curves).split(",") if t.strip()]

    _d = int(args.d)

    factories: Dict[str, Tuple[str, Callable[[], object]]] = {
        "row": ("RowMajor", lambda: make_row_major(k=int(args.k))),
        "column": ("ColumnMajor", lambda: make_column_major(k=int(args.k))),
        "z": ("ZCurve", lambda: make_z_curve(k=int(args.k), d=_d)),
        "hilbert": ("Hilbert", lambda: make_hilbert(k=int(args.k))),
        "hilbert_nd": ("Hilbert", lambda: make_hilbert_nd(k=int(args.k), d=_d)),
        "piecewise": ("PiecewiseSFC", lambda: make_piecewise_sfc(k=int(args.k))),
        "piecewise_mcts": ("PiecewiseSFC-MCTS", None),
        "lmsfc_std": (
            "LMSFC-Std",
            lambda: (lambda s: (setattr(s, "threshold", int(args.lmsfc_threshold)) if args.lmsfc_threshold is not None else None) or s)(
                LMSFCPlugin.make_standard(d=_d, k=int(args.k))
            ),
        ),
        "lmsfc_exact": (
            "LMSFC-Exact",
            lambda: (lambda s: (setattr(s, "intervals", (lambda _q: None))) or s)(
                LMSFCPlugin.make_standard(d=_d, k=int(args.k))
            ),
        ),
        "lmsfc_sa": (
            "LMSFC-SA",
            lambda: (lambda s: (setattr(s, "threshold", int(args.lmsfc_threshold)) if args.lmsfc_threshold is not None else None) or s)(
                LMSFCPlugin.make_standard(d=_d, k=int(args.k))
            ),
        ),
        "lmsfc_reinforce": (
            "LMSFC-REINFORCE",
            lambda: (lambda s: (setattr(s, "intervals", (lambda _q: None))) or s)(
                LMSFCPlugin.make_standard(d=_d, k=int(args.k))
            ),
        ),
        "lmsfc_bo": (
            "LMSFC-BO",
            lambda: (lambda s: (setattr(s, "threshold", int(args.lmsfc_threshold)) if args.lmsfc_threshold is not None else None) or s)(
                LMSFCPlugin.make_standard(d=_d, k=int(args.k))
            ),
        ),
        "adacurve": (
            "AdaCurve",
            None,
        ),
        "lacsfc_h": (
            "LACSFC-H",
            None,
        ),
        "lacsfc_neural": (
            "LACSFC-Neural",
            None,
        ),
        "lacsfc_opt": (
            "LACSFC-Opt",
            None,
        ),
        "lacsfc_fgeg": (
            "LACSFC-FG/EG",
            None,
        ),
        "lacsfc_meta": (
            "LACSFC-Meta",
            None,
        ),
        "lacsfc_backendaware": (
            "LACSFC-BackendAware",
            None,
        ),
        "lacsfc_neural_v2": (
            "LACSFC-Neural-v2",
            None,
        ),
        "bitallocz_bo": (
            "BitAllocZ-BO",
            None,
        ),
        "bitallocz_gd": (
            "BitAllocZ-GD",
            None,
        ),
        "bitallocz_uniform": (
            "BitAllocZ-Uniform",
            None,
        ),
        "gbo": ("GBO", None),
        "quilts": ("QUILTS", None),
    }

    learned_tokens = {"lmsfc", "lmsfc_exact"}
    if any(tok in learned_tokens for tok in curve_tokens):
        if not str(args.lmsfc_masks_file):
            raise ValueError("--lmsfc-masks-file is required when including any of: lmsfc, lmsfc_exact")
        mf = load_masks_json(str(args.lmsfc_masks_file))
        if _d != 2:
            raise ValueError("learned-mask LMSFC paths are currently only supported for --d=2")
        if int(mf.d) != 2:
            raise ValueError(f"masks file d={mf.d} != 2")
        if int(mf.k) != int(args.k):
            raise ValueError(f"masks file k={mf.k} != runner k={args.k}")

        def _make_learned() -> LMSFCPlugin:
            quantizer = Quantizer(d=2, k=int(args.k))
            sfc = LMSFCPlugin(name="LMSFC", grid_size=tuple([quantizer.m] * 2), masks=list(mf.masks))
            if mf.threshold is not None:
                sfc.threshold = int(mf.threshold)
            if args.lmsfc_threshold is not None:
                sfc.threshold = int(args.lmsfc_threshold)
            return sfc

        def _make_learned_exact() -> LMSFCPlugin:
            sfc = _make_learned()
            setattr(sfc, "intervals", (lambda _q: None))
            return sfc

        factories["lmsfc"] = ("LMSFC", _make_learned)
        factories["lmsfc_exact"] = ("LMSFC-Exact", _make_learned_exact)

    selected: List[Tuple[str, Callable[[], object]]] = []
    if _d > 2:
        allowed_nd_tokens = {
            "z", "hilbert", "hilbert_nd",
            "piecewise", "piecewise_mcts",
            "lmsfc_std", "lmsfc_sa", "lmsfc_bo", "lmsfc_reinforce",
            "bitallocz_uniform", "bitallocz_gd", "bitallocz_bo",
            "adacurve", "gbo", "quilts",
            "lacsfc_h", "lacsfc_neural", "lacsfc_opt", "lacsfc_fgeg",
            "lacsfc_meta", "lacsfc_backendaware", "lacsfc_neural_v2",
        }
        invalid_tokens = [tok for tok in curve_tokens if tok not in allowed_nd_tokens]
        if invalid_tokens:
            raise ValueError(
                "for --d > 2, only these curves are supported: "
                + ", ".join(sorted(allowed_nd_tokens))
                + "; "
                f"got unsupported tokens: {', '.join(invalid_tokens)}"
            )
        curve_tokens = ["hilbert_nd" if tok == "hilbert" else tok for tok in curve_tokens]
    for tok in curve_tokens:
        if tok not in factories:
            raise ValueError(f"unknown curve token: {tok}")
        selected.append(factories[tok])

    qn_train = int(args.train_qn)
    if qn_train < 0:
        raise ValueError("--train-qn must be >= 0")

    threshold_sweep = _parse_threshold_sweep(str(args.lmsfc_threshold_sweep))
    locality_qn = int(args.locality_qn)
    locality_max_cells = int(args.locality_max_cells)
    tune_lmsfc_threshold = bool(args.tune_lmsfc_threshold)

    all_runs: Dict[str, List[CurveResult]] = {name: [] for name, _ in selected}
    lacsfc_optional_status: List[dict] = []

    for rep in range(repeats):
        seed = int(args.seed) + int(rep)
        rng = np.random.default_rng(seed)

        if str(args.points) == "real":
            if not str(args.points_file):
                raise ValueError("--points-file is required when --points=real")
            points = load_points_real(str(args.points_file), k=int(args.k), max_n=int(args.n),
                                       require_n=bool(getattr(args, "require_n", False)))
            points_raw_bounds = _points_file_raw_bounds(str(args.points_file))
        elif str(args.points) == "uniform":
            points = generate_points_uniform(rng=rng, n=int(args.n), d=_d)
            points_raw_bounds = None
        elif str(args.points) == "gmm":
            points = generate_points_gmm(rng=rng, n=int(args.n), d=_d)
            points_raw_bounds = None
        elif str(args.points) == "normal":
            points = generate_points_normal(rng=rng, n=int(args.n), d=_d)
            points_raw_bounds = None
        elif str(args.points) == "skewed":
            points = generate_points_skewed(rng=rng, n=int(args.n), d=_d)
            points_raw_bounds = None
        elif str(args.points) == "zipf":
            points = generate_points_zipf(rng=rng, n=int(args.n), d=_d)
            points_raw_bounds = None
        elif str(args.points) == "diagonal":
            points = generate_points_diagonal(rng=rng, n=int(args.n), d=_d)
            points_raw_bounds = None
        elif str(args.points) == "correlated":
            points = generate_points_correlated(rng=rng, n=int(args.n), d=_d)
            points_raw_bounds = None
        elif str(args.points) == "anti_correlated":
            points = generate_points_anti_correlated(rng=rng, n=int(args.n), d=_d)
            points_raw_bounds = None
        else:
            raise ValueError(
                "unsupported --points (use uniform, gmm, normal, skewed, zipf, "
                "diagonal, correlated, anti_correlated, or real)"
            )

        def _make_queries(dist: str) -> tuple:
            if dist == "hotspot":
                return generate_queries_hotspot_correlated(
                    rng=rng, points=points, qn=int(args.qn), d=_d,
                    area_min=float(args.area_min), area_max=float(args.area_max),
                )
            elif dist == "hotspot_mixture":
                hotspot_ratio = float(np.clip(float(args.hotspot_ratio), 0.0, 1.0))
                n_hotspot = int(round(int(args.qn) * hotspot_ratio))
                n_uniform = int(args.qn) - n_hotspot
                hotspot_queries = generate_queries_hotspot_correlated(
                    rng=rng, points=points, qn=n_hotspot, d=_d,
                    area_min=float(args.area_min), area_max=float(args.area_max),
                ) if n_hotspot > 0 else tuple()
                uniform_queries = generate_queries_loguniform_area(
                    rng=rng, qn=n_uniform, d=_d,
                    area_min=float(args.area_min), area_max=float(args.area_max),
                ) if n_uniform > 0 else tuple()
                return tuple(hotspot_queries) + tuple(uniform_queries)
            elif dist == "aspect_ratio":
                return generate_queries_aspect_ratio(
                    rng=rng, qn=int(args.qn), d=_d,
                    area_min=float(args.area_min), area_max=float(args.area_max),
                    aspect=float(args.aspect_ratio),
                )
            elif dist == "aspect_hotspot":
                qn = int(args.qn)
                area_min = float(args.area_min)
                area_max = float(args.area_max)
                aspect = float(args.aspect_ratio)
                if area_min <= 0.0 or area_max <= 0.0 or area_min > area_max:
                    raise ValueError("area_min/area_max invalid")
                log_min = float(np.log(area_min))
                log_max = float(np.log(area_max))
                shrink = aspect ** (1.0 / max(_d - 1, 1))
                queries = []
                n_points = points.shape[0]
                for _ in range(qn):
                    idx = int(rng.integers(0, n_points))
                    center = points[idx]
                    area = float(np.exp(rng.uniform(log_min, log_max)))
                    side = area ** (1.0 / _d)
                    sides = np.full(_d, side / shrink, dtype=np.float64)
                    stretch_dim = int(rng.integers(0, _d))
                    sides[stretch_dim] = side * aspect
                    low = np.clip(center - sides / 2.0, 0.0, 1.0)
                    high = np.clip(center + sides / 2.0, 0.0, 1.0)
                    queries.append(QueryWindow(low=low, high=high))
                return tuple(queries)
            else:
                return generate_queries_loguniform_area(
                    rng=rng, qn=int(args.qn), d=_d,
                    area_min=float(args.area_min), area_max=float(args.area_max),
                )

        test_dist = str(args.query_dist)
        train_dist = str(args.train_query_dist) if str(args.train_query_dist) else test_dist
        queries_file = str(args.queries_file)
        train_queries_file = str(args.train_queries_file)

        if queries_file:
            test_queries = _load_query_windows_file(
                queries_file, d=_d, k=int(args.k), max_qn=int(args.qn), raw_bounds=points_raw_bounds
            )
            if train_queries_file:
                train_queries = _load_query_windows_file(
                    train_queries_file, d=_d, k=int(args.k), max_qn=int(args.train_qn), raw_bounds=points_raw_bounds
                )
            elif qn_train > 0 and qn_train < len(test_queries):
                train_queries = test_queries[:qn_train]
                test_queries = test_queries[qn_train:]
            else:
                train_queries = test_queries
            if not test_queries:
                raise ValueError("--queries-file produced no test queries")
            if not train_queries:
                raise ValueError("--queries-file/--train-queries-file produced no train queries")
        elif train_dist == test_dist:
            queries = _make_queries(test_dist)
            if qn_train <= 0 or qn_train > len(queries):
                train_queries = queries
                test_queries = queries
            else:
                train_queries = queries[:qn_train]
                test_queries = queries[qn_train:]
                if not test_queries:
                    test_queries = queries
        else:
            train_queries = _make_queries(train_dist)
            test_queries = _make_queries(test_dist)

        qn_tr_run = int(len(train_queries))
        qn_te_run = int(len(test_queries))

        quantizer = Quantizer(d=_d, k=int(args.k))

        sa_result = None
        sa_saved = False

        bo_result = None
        bo_saved = False

        reinforce_result = None
        reinforce_saved = False

        train_objective = str(args.train_objective)
        train_w_pa = float(args.train_w_page_accesses)
        train_w_seg = float(args.train_w_segments)
        train_w_spc = float(args.train_w_span_per_cell)
        train_loc_max_cells = int(args.train_locality_max_cells)
        train_loc_qn = int(args.train_locality_qn)

        for name, make_sfc in selected:
            if name in {"LMSFC-SA"}:
                if sa_result is None:
                    sa_result = train_sa(
                        points=points,
                        queries=train_queries,
                        quantizer=quantizer,
                        b=int(args.b),
                        threshold=(int(args.lmsfc_threshold) if args.lmsfc_threshold is not None else None),
                        objective=train_objective,
                        semantic=str(args.semantic),
                        timing_repeats=int(args.timing_repeats),
                        timing_warmup=(not bool(args.timing_no_warmup)),
                        iters=int(args.train_iters),
                        per=int(args.train_per),
                        restarts=int(args.train_restarts),
                        w_page_accesses=train_w_pa,
                        w_segments=train_w_seg,
                        w_span_per_cell=train_w_spc,
                        locality_max_cells=train_loc_max_cells,
                        locality_qn=train_loc_qn,
                        seed=seed,
                    )

                if str(args.train_save_masks_file) and (not sa_saved):
                    save_masks_json(
                        str(args.train_save_masks_file),
                        d=int(quantizer.d),
                        k=int(args.k),
                        masks=list(sa_result.masks),
                        threshold=(int(args.lmsfc_threshold) if args.lmsfc_threshold is not None else None),
                        seed=int(seed),
                        objective=train_objective,
                        trainer="sa",
                        train_config={
                            "n": int(args.n),
                            "qn_total": int(args.qn),
                            "qn_train": int(qn_tr_run),
                            "qn_test": int(qn_te_run),
                            "k": int(args.k),
                            "b": int(args.b),
                            "points": str(args.points),
                            "points_file": str(args.points_file),
                            "area_min": float(args.area_min),
                            "area_max": float(args.area_max),
                            "iters": int(args.train_iters),
                            "per": int(args.train_per),
                            "restarts": int(args.train_restarts),
                            "threshold": (int(args.lmsfc_threshold) if args.lmsfc_threshold is not None else None),
                            "semantic": str(args.semantic),
                            "timing_repeats": int(args.timing_repeats),
                            "timing_warmup": (not bool(args.timing_no_warmup)),
                            "objective": str(train_objective),
                            "w_page_accesses": float(train_w_pa),
                            "w_segments": float(train_w_seg),
                            "w_span_per_cell": float(train_w_spc),
                            "locality_max_cells": int(train_loc_max_cells),
                            "locality_qn": int(train_loc_qn),
                        },
                        cost=float(sa_result.cost),
                        extra={"curve": "LMSFC-SA"},
                    )
                    sa_saved = True

                def _make_learned_sa() -> LMSFCPlugin:
                    sfc = LMSFCPlugin(
                        name="LMSFC-SA",
                        grid_size=tuple([quantizer.m] * _d),
                        masks=list(sa_result.masks),
                    )
                    if args.lmsfc_threshold is not None:
                        sfc.threshold = int(args.lmsfc_threshold)
                    return sfc

                make = _make_learned_sa

                chosen_th: Optional[int]
                if tune_lmsfc_threshold:
                    chosen_th = _tune_lmsfc_threshold(
                        make_base_sfc=make,
                        thresholds=threshold_sweep,
                        points=points,
                        train_queries=train_queries,
                        quantizer=quantizer,
                        b=int(args.b),
                        profile=bool(args.profile),
                        semantic=str(args.semantic),
                    )
                else:
                    chosen_th = int(args.lmsfc_threshold)

                def _make_sa_final() -> LMSFCPlugin:
                    sfc = make()
                    if hasattr(sfc, "threshold") and chosen_th is not None:
                        setattr(sfc, "threshold", int(chosen_th))
                    return sfc

                r = _run_one(
                    name=name,
                    chosen_threshold=chosen_th,
                    make_sfc=_make_sa_final,
                    points=points,
                    queries=test_queries,
                    quantizer=quantizer,
                    b=int(args.b),
                    profile=bool(args.profile),
                    semantic=str(args.semantic),
                    locality_qn=int(locality_qn),
                    locality_max_cells=int(locality_max_cells),
                    use_sortdim=bool(args.use_sortdim),
                    training_queries=train_queries if args.use_sortdim else None,
                )
            elif name in {"LMSFC-BO"}:
                if bo_result is None:
                    bo_result = train_bo(
                        points=points,
                        queries=train_queries,
                        quantizer=quantizer,
                        b=int(args.b),
                        threshold=(int(args.lmsfc_threshold) if args.lmsfc_threshold is not None else None),
                        objective=train_objective,
                        semantic=str(args.semantic),
                        timing_repeats=int(args.timing_repeats),
                        timing_warmup=(not bool(args.timing_no_warmup)),
                        per=int(args.train_per),
                        w_page_accesses=train_w_pa,
                        w_segments=train_w_seg,
                        w_span_per_cell=train_w_spc,
                        locality_max_cells=train_loc_max_cells,
                        locality_qn=train_loc_qn,
                        init_size=int(args.train_bo_init_size),
                        rounds=int(args.train_bo_rounds),
                        evals_per_round=int(args.train_bo_evals_per_round),
                        neighbor_candidates=int(args.train_bo_neighbor_candidates),
                        random_candidates=int(args.train_bo_random_candidates),
                        topk=int(args.train_bo_topk),
                        rf_init_points=int(args.train_bo_rf_init_points),
                        rf_n_iter=int(args.train_bo_rf_n_iter),
                        use_bayes_opt=(not bool(args.train_bo_no_bayes)),
                        seed=seed,
                    )

                if str(args.train_save_masks_file) and (not bo_saved):
                    save_masks_json(
                        str(args.train_save_masks_file),
                        d=int(quantizer.d),
                        k=int(args.k),
                        masks=list(bo_result.masks),
                        threshold=(int(args.lmsfc_threshold) if args.lmsfc_threshold is not None else None),
                        seed=int(seed),
                        objective=train_objective,
                        trainer="bo",
                        train_config={
                            "n": int(args.n),
                            "qn_total": int(args.qn),
                            "qn_train": int(qn_tr_run),
                            "qn_test": int(qn_te_run),
                            "k": int(args.k),
                            "b": int(args.b),
                            "points": str(args.points),
                            "points_file": str(args.points_file),
                            "area_min": float(args.area_min),
                            "area_max": float(args.area_max),
                            "per": int(args.train_per),
                            "threshold": (int(args.lmsfc_threshold) if args.lmsfc_threshold is not None else None),
                            "semantic": str(args.semantic),
                            "timing_repeats": int(args.timing_repeats),
                            "timing_warmup": (not bool(args.timing_no_warmup)),
                            "objective": str(train_objective),
                            "w_page_accesses": float(train_w_pa),
                            "w_segments": float(train_w_seg),
                            "w_span_per_cell": float(train_w_spc),
                            "locality_max_cells": int(train_loc_max_cells),
                            "locality_qn": int(train_loc_qn),
                            "bo_init_size": int(args.train_bo_init_size),
                            "bo_rounds": int(args.train_bo_rounds),
                            "bo_evals_per_round": int(args.train_bo_evals_per_round),
                            "bo_neighbor_candidates": int(args.train_bo_neighbor_candidates),
                            "bo_random_candidates": int(args.train_bo_random_candidates),
                            "bo_topk": int(args.train_bo_topk),
                            "bo_rf_init_points": int(args.train_bo_rf_init_points),
                            "bo_rf_n_iter": int(args.train_bo_rf_n_iter),
                            "bo_use_bayes_opt": (not bool(args.train_bo_no_bayes)),
                            "bo_n_evaluated": int(bo_result.n_evaluated),
                        },
                        cost=float(bo_result.cost),
                        extra={"curve": "LMSFC-BO"},
                    )
                    bo_saved = True

                def _make_learned_bo() -> LMSFCPlugin:
                    sfc = LMSFCPlugin(
                        name="LMSFC-BO",
                        grid_size=tuple([quantizer.m] * _d),
                        masks=list(bo_result.masks),
                    )
                    if args.lmsfc_threshold is not None:
                        sfc.threshold = int(args.lmsfc_threshold)
                    return sfc

                make = _make_learned_bo

                chosen_th: Optional[int]
                if tune_lmsfc_threshold:
                    chosen_th = _tune_lmsfc_threshold(
                        make_base_sfc=make,
                        thresholds=threshold_sweep,
                        points=points,
                        train_queries=train_queries,
                        quantizer=quantizer,
                        b=int(args.b),
                        profile=bool(args.profile),
                        semantic=str(args.semantic),
                    )
                else:
                    chosen_th = int(args.lmsfc_threshold)

                def _make_bo_final() -> LMSFCPlugin:
                    sfc = make()
                    if hasattr(sfc, "threshold") and chosen_th is not None:
                        setattr(sfc, "threshold", int(chosen_th))
                    return sfc

                r = _run_one(
                    name=name,
                    chosen_threshold=chosen_th,
                    make_sfc=_make_bo_final,
                    points=points,
                    queries=test_queries,
                    quantizer=quantizer,
                    b=int(args.b),
                    profile=bool(args.profile),
                    semantic=str(args.semantic),
                    locality_qn=int(locality_qn),
                    locality_max_cells=int(locality_max_cells),
                )
            elif name in {"LMSFC-REINFORCE"}:
                if train_objective == "exact_locality":
                    raise ValueError("--train-objective=exact_locality is only supported for LMSFC-SA")
                if reinforce_result is None:
                    reinforce_result = train_reinforce(
                        points=points,
                        queries=train_queries,
                        quantizer=quantizer,
                        b=int(args.b),
                        objective=train_objective,
                        semantic=str(args.semantic),
                        timing_repeats=int(args.timing_repeats),
                        timing_warmup=(not bool(args.timing_no_warmup)),
                        iters=int(args.train_iters),
                        per=int(args.train_per),
                        w_page_accesses=float(train_w_pa),
                        w_segments=float(train_w_seg),
                        lr=float(args.train_reinforce_lr),
                        tau_start=float(args.train_reinforce_tau_start),
                        tau_end=float(args.train_reinforce_tau_end),
                        baseline_momentum=float(args.train_reinforce_baseline_momentum),
                        force_fallback_intervals=True,
                        seed=seed,
                    )

                if str(args.train_save_masks_file) and (not reinforce_saved):
                    save_masks_json(
                        str(args.train_save_masks_file),
                        d=int(quantizer.d),
                        k=int(args.k),
                        masks=list(reinforce_result.masks),
                        threshold=None,
                        seed=int(seed),
                        objective=train_objective,
                        trainer="reinforce",
                        train_config={
                            "n": int(args.n),
                            "qn_total": int(args.qn),
                            "qn_train": int(qn_tr_run),
                            "qn_test": int(qn_te_run),
                            "k": int(args.k),
                            "b": int(args.b),
                            "points": str(args.points),
                            "points_file": str(args.points_file),
                            "area_min": float(args.area_min),
                            "area_max": float(args.area_max),
                            "iters": int(args.train_iters),
                            "per": int(args.train_per),
                            "semantic": str(args.semantic),
                            "timing_repeats": int(args.timing_repeats),
                            "timing_warmup": (not bool(args.timing_no_warmup)),
                            "objective": str(train_objective),
                            "w_page_accesses": float(train_w_pa),
                            "w_segments": float(train_w_seg),
                            "lr": float(args.train_reinforce_lr),
                            "tau_start": float(args.train_reinforce_tau_start),
                            "tau_end": float(args.train_reinforce_tau_end),
                            "baseline_momentum": float(args.train_reinforce_baseline_momentum),
                            "force_fallback_intervals": True,
                        },
                        cost=float(reinforce_result.cost),
                        extra={"curve": "LMSFC-REINFORCE"},
                    )
                    reinforce_saved = True

                def _make_reinforce_final() -> LMSFCPlugin:
                    sfc = LMSFCPlugin(
                        name="LMSFC-REINFORCE",
                        grid_size=tuple([quantizer.m] * _d),
                        masks=list(reinforce_result.masks),
                    )
                    setattr(sfc, "intervals", (lambda _q: None))
                    return sfc

                r = _run_one(
                    name=name,
                    chosen_threshold=None,
                    make_sfc=_make_reinforce_final,
                    points=points,
                    queries=test_queries,
                    quantizer=quantizer,
                    b=int(args.b),
                    profile=bool(args.profile),
                    semantic=str(args.semantic),
                    locality_qn=int(locality_qn),
                    locality_max_cells=int(locality_max_cells),
                )
            elif name in {"AdaCurve"}:
                def _make_adacurve_built() -> object:
                    sfc = make_adacurve_paper_repro(
                        k=int(args.k),
                        block_size_points=int(args.adacurve_block_size),
                        compress_dim=int(args.adacurve_compress_dim),
                        epochs=int(args.adacurve_epochs),
                        learning_rate=float(args.adacurve_lr),
                        query_batch_size=int(args.adacurve_query_batch_size),
                        train_sample_size=int(args.adacurve_train_sample_size),
                        fp_weight=float(args.adacurve_fp_weight),
                        tp_weight=float(args.adacurve_tp_weight),
                        recon_weight=float(args.adacurve_recon_weight),
                        smooth_weight=float(args.adacurve_smooth_weight),
                        margin=float(args.adacurve_margin),
                        seed=int(seed),
                        name="AdaCurve",
                    )
                    sfc.build(points, training_queries=list(train_queries))
                    return sfc

                r = _run_one(
                    name=name,
                    chosen_threshold=None,
                    make_sfc=_make_adacurve_built,
                    points=points,
                    queries=test_queries,
                    quantizer=quantizer,
                    b=int(args.b),
                    profile=bool(args.profile),
                    semantic=str(args.semantic),
                    locality_qn=int(locality_qn),
                    locality_max_cells=int(locality_max_cells),
                )
            elif name in {"PiecewiseSFC"}:
                def _make_piecewise_built() -> object:
                    sfc = make_piecewise_sfc(k=int(args.k), d=_d)
                    sfc.build(points)
                    return sfc

                r = _run_one(
                    name=name,
                    chosen_threshold=None,
                    make_sfc=_make_piecewise_built,
                    points=points,
                    queries=test_queries,
                    quantizer=quantizer,
                    b=int(args.b),
                    profile=bool(args.profile),
                    semantic=str(args.semantic),
                    locality_qn=int(locality_qn),
                    locality_max_cells=int(locality_max_cells),
                )
            elif name in {"PiecewiseSFC-MCTS"}:
                def _make_piecewise_mcts_built() -> object:
                    sfc = make_piecewise_mcts(k=int(args.k), d=_d)
                    sfc.build(points, queries=list(train_queries))
                    return sfc

                r = _run_one(
                    name=name,
                    chosen_threshold=None,
                    make_sfc=_make_piecewise_mcts_built,
                    points=points,
                    queries=test_queries,
                    quantizer=quantizer,
                    b=int(args.b),
                    profile=bool(args.profile),
                    semantic=str(args.semantic),
                    locality_qn=int(locality_qn),
                    locality_max_cells=int(locality_max_cells),
                )
            elif name in {"LACSFC-H"}:
                lacsfc_k = int(args.lacsfc_k)
                lacsfc_quantizer = Quantizer(d=_d, k=lacsfc_k)

                def _make_lacsfc_h_built() -> object:
                    return make_lacsfc_h(
                        k=lacsfc_k,
                        d=_d,
                        points=points,
                        seed=seed,
                    )

                r = _run_one(
                    name=name,
                    chosen_threshold=None,
                    make_sfc=_make_lacsfc_h_built,
                    points=points,
                    queries=test_queries,
                    quantizer=lacsfc_quantizer,
                    b=int(args.b),
                    profile=bool(args.profile),
                    semantic=str(args.semantic),
                    locality_qn=int(locality_qn),
                    locality_max_cells=int(locality_max_cells),
                )
            elif name in {"LACSFC-BackendAware"}:
                lacsfc_k = int(args.lacsfc_k)
                lacsfc_quantizer = Quantizer(d=_d, k=lacsfc_k)
                weights_path = str(args.lacsfc_selected_weights)
                if not weights_path:
                    raise ValueError("--lacsfc-selected-weights is required when 'lacsfc_backendaware' is in --curves")
                if not os.path.exists(weights_path):
                    raise FileNotFoundError(weights_path)

                def _make_lacsfc_backendaware_built() -> object:
                    weights = np.load(weights_path, allow_pickle=False)
                    return make_lacsfc_from_weights(
                        k=lacsfc_k,
                        d=_d,
                        weights=weights,
                        points=points,
                        name="LACSFC-BackendAware",
                        seed=seed,
                    )

                r = _run_one(
                    name=name,
                    chosen_threshold=None,
                    make_sfc=_make_lacsfc_backendaware_built,
                    points=points,
                    queries=test_queries,
                    quantizer=lacsfc_quantizer,
                    b=int(args.b),
                    profile=bool(args.profile),
                    semantic=str(args.semantic),
                    locality_qn=int(locality_qn),
                    locality_max_cells=int(locality_max_cells),
                )
            elif name in {"LACSFC-Neural", "LACSFC-Opt", "LACSFC-Meta"}:
                lacsfc_k = int(args.lacsfc_k)
                lacsfc_max_evals = int(args.lacsfc_max_evals)
                lacsfc_mode = str(args.lacsfc_mode)
                lacsfc_objective = str(args.lacsfc_objective)
                lacsfc_learn_obj = bool(args.lacsfc_learn_obj)
                lacsfc_init_obj = tuple(float(x) for x in args.lacsfc_init_obj.split(","))
                lacsfc_init_strategy = str(args.lacsfc_init_strategy)
                lacsfc_func_space_m = int(args.lacsfc_func_space_m)
                lacsfc_sigma0 = float(args.lacsfc_sigma0)
                lacsfc_popsize = int(args.lacsfc_popsize) if args.lacsfc_popsize is not None else None
                lacsfc_runtime_surrogate = str(args.lacsfc_runtime_surrogate)
                lacsfc_runtime_weight = float(args.lacsfc_runtime_weight)
                lacsfc_runtime_normalize = str(args.lacsfc_runtime_normalize)
                lacsfc_runtime_interval_weight = float(args.lacsfc_runtime_interval_weight)
                lacsfc_quantizer = Quantizer(d=_d, k=lacsfc_k)

                lacsfc_es_epochs       = int(args.lacsfc_es_epochs)
                lacsfc_es_n_perturb    = int(args.lacsfc_es_n_perturb)
                lacsfc_es_sigma        = float(args.lacsfc_es_sigma)
                lacsfc_es_dataset_size = int(args.lacsfc_es_dataset_size)
                lacsfc_es_learn_obj    = bool(args.lacsfc_es_learn_obj)
                lacsfc_es_init_obj     = tuple(float(x) for x in args.lacsfc_es_init_obj.split(","))
                lacsfc_status_idx = len(lacsfc_optional_status)
                lacsfc_meta_enable_effective = bool(args.lacsfc_meta_enable) or name == "LACSFC-Meta"
                if name == "LACSFC-Meta" and not str(args.lacsfc_meta_model_dir):
                    raise ValueError("--lacsfc-meta-model-dir is required when 'lacsfc_meta' is in --curves")
                lacsfc_artifact_file = _lacsfc_artifact_file(
                    args,
                    seed,
                    curve_name=name,
                    meta_enable_effective=lacsfc_meta_enable_effective,
                )

                def _make_lacsfc_neural_built() -> object:
                    if lacsfc_artifact_file and os.path.exists(lacsfc_artifact_file):
                        from .lacsfc.plugin import LACSFCPlugin

                        data = np.load(lacsfc_artifact_file)
                        sfc = LACSFCPlugin(k=lacsfc_k, d=_d, seed=seed)
                        sfc.name = name
                        sfc._grid = None
                        sfc._path = np.asarray(data["path"], dtype=np.int64)
                        sfc._code_map = np.asarray(data["code_map"], dtype=np.int64)
                        sfc._is_built = True
                        sfc._init_query_processor()
                        lacsfc_optional_status.append({
                            "seed": int(seed),
                            "artifact_loaded": True,
                            "meta": getattr(sfc, "_meta_status", {}),
                            "shift": getattr(sfc, "_shift_status", {}),
                        })
                        print(f"Loaded {name} artifact: {lacsfc_artifact_file}")
                        return sfc

                    sfc = make_lacsfc_neural(
                        d=_d,
                        k=lacsfc_k,
                        points=points,
                        query_windows=list(train_queries),
                        quantizer=lacsfc_quantizer,
                        max_evals=lacsfc_max_evals,
                        seed=seed,
                        cmaes_mode=lacsfc_mode,
                        cmaes_objective=lacsfc_objective,
                        cmaes_learn_obj_weights=lacsfc_learn_obj,
                        cmaes_init_obj_weights=lacsfc_init_obj,
                        cmaes_init_strategy=lacsfc_init_strategy,
                        func_space_M=lacsfc_func_space_m,
                        sigma0=lacsfc_sigma0,
                        popsize=lacsfc_popsize,
                        b=int(args.b),
                        es_epochs=lacsfc_es_epochs,
                        es_n_perturbations=lacsfc_es_n_perturb,
                        es_sigma=lacsfc_es_sigma,
                        es_dataset_size=lacsfc_es_dataset_size,
                        es_learn_obj_weights=lacsfc_es_learn_obj,
                        es_init_obj_weights=lacsfc_es_init_obj,
                        cover_weight=float(args.lacsfc_cover_weight),
                        runtime_surrogate=lacsfc_runtime_surrogate,
                        runtime_weight=lacsfc_runtime_weight,
                        runtime_normalize=lacsfc_runtime_normalize,
                        runtime_interval_weight=lacsfc_runtime_interval_weight,
                        name=name,
                        meta_enable=lacsfc_meta_enable_effective,
                        meta_model_dir=str(args.lacsfc_meta_model_dir),
                        meta_model_name=str(args.lacsfc_meta_model_name),
                        meta_inner_steps=int(args.lacsfc_meta_inner_steps),
                        meta_inner_lr=float(args.lacsfc_meta_inner_lr),
                        meta_use_evaluator=bool(args.lacsfc_meta_use_evaluator),
                        meta_fallback_cmaes=bool(args.lacsfc_meta_fallback_cmaes),
                        meta_hidden=int(args.lacsfc_meta_hidden),
                        meta_layers=int(args.lacsfc_meta_layers),
                        meta_evaluator_hidden=int(args.lacsfc_meta_evaluator_hidden),
                        meta_evaluator_layers=int(args.lacsfc_meta_evaluator_layers),
                    )
                    lacsfc_optional_status.append({
                        "seed": int(seed),
                        "artifact_loaded": False,
                        "meta": getattr(sfc, "_meta_status", {}),
                        "shift": getattr(sfc, "_shift_status", {}),
                    })
                    if lacsfc_artifact_file and getattr(sfc, "_path", None) is not None and getattr(sfc, "_code_map", None) is not None:
                        np.savez_compressed(
                            lacsfc_artifact_file,
                            path=np.asarray(sfc._path, dtype=np.int64),
                            code_map=np.asarray(sfc._code_map, dtype=np.int64),
                        )
                        print(f"Saved {name} artifact: {lacsfc_artifact_file}")
                    return sfc

                r = _run_one(
                    name=name,
                    chosen_threshold=None,
                    make_sfc=_make_lacsfc_neural_built,
                    points=points,
                    queries=test_queries,
                    quantizer=lacsfc_quantizer,
                    b=int(args.b),
                    profile=bool(args.profile),
                    semantic=str(args.semantic),
                    locality_qn=int(locality_qn),
                    locality_max_cells=int(locality_max_cells),
                )
                if len(lacsfc_optional_status) > lacsfc_status_idx:
                    shift_cfg = WorkloadShiftConfig(
                        monitor_enabled=bool(args.lacsfc_shift_monitor_enable),
                        window_size=int(args.lacsfc_shift_window_size),
                        pa_threshold=float(args.lacsfc_shift_pa_threshold),
                        score_threshold=float(args.lacsfc_shift_score_threshold),
                        reopt_enabled=bool(args.lacsfc_reopt_enable),
                        reopt_sigma_scale=float(args.lacsfc_reopt_sigma_scale),
                        reopt_max_evals=int(args.lacsfc_reopt_max_evals),
                        state_dir=str(args.lacsfc_state_dir),
                    )
                    three_factor_score = (
                        float(r.avg_page_amplification)
                        + float(r.avg_segments)
                        + float(r.avg_candidate_page_local_span_mean)
                    )
                    lacsfc_optional_status[lacsfc_status_idx]["shift"] = evaluate_shift_from_summary(
                        config=shift_cfg,
                        train_query_dist=("file" if (str(args.queries_file) or str(args.train_queries_file)) else str(train_dist)),
                        test_query_dist=("file" if str(args.queries_file) else str(test_dist)),
                        current_avg_amp=float(r.avg_page_amplification),
                        baseline_avg_amp=None,
                        current_three_factor_score=three_factor_score,
                        baseline_three_factor_score=None,
                        sigma0=float(lacsfc_sigma0),
                    )
            elif name in {"LACSFC-Neural-v2", "LACSFC-FG/EG"}:
                lacsfc_k = int(args.lacsfc_k)
                lacsfc_quantizer = Quantizer(d=_d, k=lacsfc_k)
                v2_ckpt = str(args.lacsfc_v2_ckpt)
                if not v2_ckpt:
                    raise ValueError(
                        "--lacsfc-v2-ckpt is required when 'lacsfc_neural_v2' or 'lacsfc_fgeg' is in --curves"
                    )

                def _make_lacsfc_neural_v2_built() -> object:
                    return make_lacsfc_neural_v2(
                        k=lacsfc_k,
                        d=_d,
                        points=points,
                        v2_ckpt_path=v2_ckpt,
                        seed=seed,
                        name=name,
                    )

                r = _run_one(
                    name=name,
                    chosen_threshold=None,
                    make_sfc=_make_lacsfc_neural_v2_built,
                    points=points,
                    queries=test_queries,
                    quantizer=lacsfc_quantizer,
                    b=int(args.b),
                    profile=bool(args.profile),
                    semantic=str(args.semantic),
                    locality_qn=int(locality_qn),
                    locality_max_cells=int(locality_max_cells),
                )
            elif name in {"GBO", "QUILTS"}:
                _rt_train_qq = [quantizer.quantize_window(q) for q in train_queries]

                def _make_rank_table_built(_name: str = name) -> object:
                    return make_rank_table_sfc(
                        _name, k=int(args.k), d=_d, points=points,
                        train_qq=_rt_train_qq, quantizer=quantizer, seed=seed,
                    )

                r = _run_one(
                    name=name,
                    chosen_threshold=None,
                    make_sfc=_make_rank_table_built,
                    points=points,
                    queries=test_queries,
                    quantizer=quantizer,
                    b=int(args.b),
                    profile=bool(args.profile),
                    semantic=str(args.semantic),
                    locality_qn=int(locality_qn),
                    locality_max_cells=int(locality_max_cells),
                )
            elif name in {"BitAllocZ-Uniform"}:
                _baz_k = int(args.k)
                _baz_quantizer = Quantizer(d=_d, k=_baz_k)
                _baz_total_bits = getattr(args, "bitallocz_total_bits", None)
                if _baz_total_bits is not None and int(_baz_total_bits) > _d * _baz_k:
                    raise ValueError(
                        f"--bitallocz-total-bits={_baz_total_bits} exceeds "
                        f"d*k={_d*_baz_k} (physically infeasible for d={_d}, k={_baz_k})"
                    )

                def _make_baz_uniform_built() -> object:
                    return make_bitallocz_uniform(
                        d=_d, k=_baz_k, b=int(args.b),
                        total_bits=_baz_total_bits,
                        name="BitAllocZ-Uniform",
                    )

                r = _run_one(
                    name=name,
                    chosen_threshold=None,
                    make_sfc=_make_baz_uniform_built,
                    points=points,
                    queries=test_queries,
                    quantizer=_baz_quantizer,
                    b=int(args.b),
                    profile=bool(args.profile),
                    semantic=str(args.semantic),
                    locality_qn=int(locality_qn),
                    locality_max_cells=int(locality_max_cells),
                )
            elif name in {"BitAllocZ-BO", "BitAllocZ-GD"}:
                _baz_k = int(args.k)
                _baz_quantizer = Quantizer(d=_d, k=_baz_k)
                _baz_total_bits = getattr(args, "bitallocz_total_bits", None)
                if _baz_total_bits is not None and int(_baz_total_bits) > _d * _baz_k:
                    raise ValueError(
                        f"--bitallocz-total-bits={_baz_total_bits} exceeds "
                        f"d*k={_d*_baz_k} (physically infeasible for d={_d}, k={_baz_k})"
                    )
                _baz_max_cells = int(getattr(args, "bitallocz_max_cells", 2_000_000))
                _baz_w_seg = float(getattr(args, "bitallocz_w_seg", 1.0))
                _baz_w_pa = float(getattr(args, "bitallocz_w_pa", 1.0))
                _baz_w_cand = float(getattr(args, "bitallocz_w_cand", 0.0))

                _baz_train_qq = [
                    _baz_quantizer.quantize_window(q) for q in train_queries
                ]

                if name == "BitAllocZ-BO":
                    _baz_bo_iters = int(getattr(args, "bitallocz_bo_iters", 80))
                    _baz_bo_init = int(getattr(args, "bitallocz_bo_init", 5))

                    def _make_baz_bo_built() -> object:
                        return make_bitallocz_bo(
                            d=_d, k=_baz_k, b=int(args.b),
                            total_bits=_baz_total_bits,
                            bo_init=_baz_bo_init,
                            bo_iters=_baz_bo_iters,
                            w_seg=_baz_w_seg, w_pa=_baz_w_pa, w_cand=_baz_w_cand,
                            max_cells=_baz_max_cells,
                            seed=seed,
                            verbose=False,
                            name="BitAllocZ-BO",
                            points=points,
                            quantized_queries=_baz_train_qq,
                            quantizer=_baz_quantizer,
                        )

                    r = _run_one(
                        name=name,
                        chosen_threshold=None,
                        make_sfc=_make_baz_bo_built,
                        points=points,
                        queries=test_queries,
                        quantizer=_baz_quantizer,
                        b=int(args.b),
                        profile=bool(args.profile),
                        semantic=str(args.semantic),
                        locality_qn=int(locality_qn),
                        locality_max_cells=int(locality_max_cells),
                    )
                else:
                    _baz_gd_iters = int(getattr(args, "bitallocz_gd_iters", 100))
                    _baz_gd_lr = float(getattr(args, "bitallocz_gd_lr", 0.1))

                    def _make_baz_gd_built() -> object:
                        return make_bitallocz_gd(
                            d=_d, k=_baz_k, b=int(args.b),
                            total_bits=_baz_total_bits,
                            gd_iters=_baz_gd_iters,
                            gd_lr=_baz_gd_lr,
                            w_seg=_baz_w_seg, w_pa=_baz_w_pa, w_cand=_baz_w_cand,
                            max_cells=_baz_max_cells,
                            seed=seed,
                            verbose=False,
                            name="BitAllocZ-GD",
                            points=points,
                            quantized_queries=_baz_train_qq,
                            quantizer=_baz_quantizer,
                        )

                    r = _run_one(
                        name=name,
                        chosen_threshold=None,
                        make_sfc=_make_baz_gd_built,
                        points=points,
                        queries=test_queries,
                        quantizer=_baz_quantizer,
                        b=int(args.b),
                        profile=bool(args.profile),
                        semantic=str(args.semantic),
                        locality_qn=int(locality_qn),
                        locality_max_cells=int(locality_max_cells),
                    )
            else:
                chosen_th = None
                make_final = make_sfc
                if name in {"LMSFC", "LMSFC-Learned"}:
                    if tune_lmsfc_threshold:
                        chosen_th = _tune_lmsfc_threshold(
                            make_base_sfc=make_sfc,
                            thresholds=threshold_sweep,
                            points=points,
                            train_queries=train_queries,
                            quantizer=quantizer,
                            b=int(args.b),
                            profile=bool(args.profile),
                            semantic=str(args.semantic),
                        )
                    else:
                        chosen_th = int(args.lmsfc_threshold)

                    def _make_final() -> object:
                        sfc = make_sfc()
                        if hasattr(sfc, "threshold") and chosen_th is not None:
                            setattr(sfc, "threshold", int(chosen_th))
                        return sfc

                    make_final = _make_final

                r = _run_one(
                    name=name,
                    chosen_threshold=chosen_th,
                    make_sfc=make_final,
                    points=points,
                    queries=test_queries,
                    quantizer=quantizer,
                    b=int(args.b),
                    profile=bool(args.profile),
                    semantic=str(args.semantic),
                    locality_qn=int(locality_qn),
                    locality_max_cells=int(locality_max_cells),
                    use_sortdim=bool(args.use_sortdim),
                    training_queries=train_queries if args.use_sortdim else None,
                )
            all_runs[name].append(r)

    stats: List[CurveStats] = []
    for name, _ in selected:
        stats.append(_aggregate(name=name, runs=all_runs[name]))

    stats.sort(key=lambda x: (x.query_s_mean, x.avg_page_accesses_mean, x.avg_segments_mean))

    qn_total = int(args.qn)
    if qn_train <= 0 or qn_train > qn_total:
        qn_tr = qn_total
        qn_te = qn_total
    else:
        qn_tr = int(qn_train)
        qn_te = qn_total - qn_tr
    print(
        f"n={int(args.n)} qn_total={qn_total} qn_train={qn_tr} qn_test={qn_te} "
        f"k={args.k} b={args.b} seed={args.seed} repeat={repeats}"
    )
    _print_table(
        stats,
        profile=bool(args.profile),
        locality=(int(locality_qn) > 0),
        cover=(bool(args.profile) and bool(args.profile_cover)),
    )

    if str(args.output):
        result_payload = {}
        for s in stats:
            item = {
                "build_s": round(s.build_s_mean, 4),
                "query_s": round(s.query_s_mean, 4),
                "avg_pa": round(s.avg_page_accesses_mean, 4),
                "avg_seg": round(s.avg_segments_mean, 4),
                "avg_amp": round(s.avg_page_amplification_mean, 4),
            }
            if s.avg_exact_cell_segments_mean is not None:
                item["avg_exact_cell_segments"] = round(s.avg_exact_cell_segments_mean, 4)
                item["avg_exact_cell_segments_std"] = round(s.avg_exact_cell_segments_std, 4)
            if s.avg_span_mean is not None:
                item["avg_exact_span"] = round(s.avg_span_mean, 4)
            if bool(args.profile):
                item.update({
                    "intervals_s": round(s.intervals_s_mean, 4),
                    "lookup_s": round(s.lookup_s_mean, 4),
                    "filter_s": round(s.filter_s_mean, 4),
                    "total_s": round(s.total_s_mean, 4),
                    "avg_interval_count": round(s.avg_interval_count_mean, 4),
                    "avg_covered_code_len": round(s.avg_covered_code_len_mean, 4),
                    "avg_records_scanned": round(s.avg_records_scanned_mean, 4),
                    "avg_pass_code": round(s.avg_pass_code_mean, 4),
                    "avg_pass_semantic": round(s.avg_pass_semantic_mean, 4),
                    "avg_hit_segments": round(s.avg_hit_segments_mean, 4),
                    "avg_page_count": round(s.avg_page_count_mean, 4),
                    "avg_touched_page_record_count": round(s.avg_touched_page_record_count_mean, 4),
                    "avg_candidate_page_count": round(s.avg_candidate_page_count_mean, 4),
                    "avg_candidate_page_record_count": round(s.avg_candidate_page_record_count_mean, 4),
                    "avg_empty_candidate_page_count": round(s.avg_empty_candidate_page_count_mean, 4),
                    "avg_candidate_page_local_span_sum": round(s.avg_candidate_page_local_span_sum_mean, 4),
                    "avg_candidate_page_local_span_mean": round(s.avg_candidate_page_local_span_mean_mean, 4),
                    "avg_candidate_page_local_span_max": round(s.avg_candidate_page_local_span_max_mean, 4),
                    "candidate_page_local_span_per_candidate_record": round(
                        s.avg_candidate_page_local_span_sum_mean / s.avg_candidate_record_count_mean,
                        6,
                    ) if s.avg_candidate_record_count_mean > 0 else 0.0,
                    "candidate_pages_per_touched_page": round(
                        s.avg_candidate_page_count_mean / s.avg_page_count_mean,
                        6,
                    ) if s.avg_page_count_mean > 0 else 0.0,
                    "records_scanned_per_touched_page_record": round(
                        s.avg_records_scanned_mean / s.avg_touched_page_record_count_mean,
                        6,
                    ) if s.avg_touched_page_record_count_mean > 0 else 0.0,
                    "records_scanned_per_candidate_page_record": round(
                        s.avg_records_scanned_mean / s.avg_candidate_page_record_count_mean,
                        6,
                    ) if s.avg_candidate_page_record_count_mean > 0 else 0.0,
                    "avg_page_interval_visits": round(s.avg_page_interval_visits_mean, 4),
                    "avg_page_interval_overlaps": round(s.avg_page_interval_overlaps_mean, 4),
                    "avg_candidate_slice_count": round(s.avg_candidate_slice_count_mean, 4),
                    "avg_candidate_slice_len_mean": round(s.avg_candidate_slice_len_mean_mean, 4),
                    "avg_empty_slice_count": round(s.avg_empty_slice_count_mean, 4),
                    "avg_nonempty_candidate_slice_count": round(s.avg_nonempty_candidate_slice_count_mean, 4),
                    "avg_candidate_slice_len_nonempty_mean": round(s.avg_candidate_slice_len_nonempty_mean_mean, 4),
                    "avg_candidate_slice_len_max": round(s.avg_candidate_slice_len_max_mean, 4),
                    "avg_vectorized_candidate_slice_count": round(s.avg_vectorized_candidate_slice_count_mean, 4),
                    "avg_scalar_candidate_slice_count": round(s.avg_scalar_candidate_slice_count_mean, 4),
                    "avg_vectorized_candidate_record_count": round(s.avg_vectorized_candidate_record_count_mean, 4),
                    "avg_scalar_candidate_record_count": round(s.avg_scalar_candidate_record_count_mean, 4),
                    "avg_batched_candidate_slice_count": round(s.avg_batched_candidate_slice_count_mean, 4),
                    "avg_batched_candidate_record_count": round(s.avg_batched_candidate_record_count_mean, 4),
                    "avg_candidate_record_count": round(s.avg_candidate_record_count_mean, 4),
                    "avg_semantic_check_count": round(s.avg_semantic_check_count_mean, 4),
                    "avg_accepted_record_count": round(s.avg_accepted_record_count_mean, 4),
                    "filter_bisect_s": round(s.filter_bisect_s_mean, 6),
                    "filter_candidate_scan_s": round(s.filter_candidate_scan_s_mean, 6),
                    "filter_candidate_extract_s": round(s.filter_candidate_extract_s_mean, 6),
                    "filter_candidate_array_s": round(s.filter_candidate_array_s_mean, 6),
                    "filter_semantic_check_s": round(s.filter_semantic_check_s_mean, 6),
                    "filter_materialize_s": round(s.filter_materialize_s_mean, 6),
                })
            result_payload[s.name] = item
        record = {
            "exp": str(args.exp_tag),
            "n": int(args.n),
            "d": int(args.d),
            "k": int(args.k),
            "b": int(args.b),
            "seed": int(args.seed),
            "repeat": int(args.repeat),
            "curves": [name for name, _ in selected],
            "data_dist": str(args.points),
            "query_dist": "file" if str(args.queries_file) else str(args.query_dist),
            "train_query_dist": "file" if (str(args.queries_file) or str(args.train_queries_file)) else (str(args.train_query_dist) if str(args.train_query_dist) else str(args.query_dist)),
            "queries_file": os.path.abspath(str(args.queries_file)) if str(args.queries_file) else "",
            "train_queries_file": os.path.abspath(str(args.train_queries_file)) if str(args.train_queries_file) else "",
            "profile": bool(args.profile),
            "profile_cover": bool(args.profile_cover),
            "lacsfc_config": {
                "lacsfc_k": int(args.lacsfc_k),
                "lacsfc_max_evals": int(args.lacsfc_max_evals),
                "lacsfc_sigma0": float(args.lacsfc_sigma0),
                "lacsfc_popsize": int(args.lacsfc_popsize) if args.lacsfc_popsize is not None else None,
                "lacsfc_mode": str(args.lacsfc_mode),
                "lacsfc_coarsen_factor": int(getattr(args, "lacsfc_coarsen_factor", 2)),
                "lacsfc_func_space_m": int(args.lacsfc_func_space_m),
                "lacsfc_objective": str(args.lacsfc_objective),
                "lacsfc_learn_obj": bool(args.lacsfc_learn_obj),
                "lacsfc_init_obj": str(args.lacsfc_init_obj),
                "lacsfc_init_strategy": str(args.lacsfc_init_strategy),
                "lacsfc_cover_weight": float(args.lacsfc_cover_weight),
                "lacsfc_runtime_surrogate": str(args.lacsfc_runtime_surrogate),
                "lacsfc_runtime_weight": float(args.lacsfc_runtime_weight),
                "lacsfc_runtime_normalize": str(args.lacsfc_runtime_normalize),
                "lacsfc_runtime_interval_weight": float(args.lacsfc_runtime_interval_weight),
                "lacsfc_meta_enable": bool(args.lacsfc_meta_enable),
                "lacsfc_meta_model_dir": os.path.abspath(str(args.lacsfc_meta_model_dir)) if str(args.lacsfc_meta_model_dir) else "",
                "lacsfc_meta_model_name": str(args.lacsfc_meta_model_name),
                "lacsfc_meta_inner_steps": int(args.lacsfc_meta_inner_steps),
                "lacsfc_meta_inner_lr": float(args.lacsfc_meta_inner_lr),
                "lacsfc_meta_use_evaluator": bool(args.lacsfc_meta_use_evaluator),
                "lacsfc_meta_fallback_cmaes": bool(args.lacsfc_meta_fallback_cmaes),
            },
            "results": result_payload,
        }
        if lacsfc_optional_status:
            record["lacsfc_optional_status"] = lacsfc_optional_status
        with open(str(args.output), "a") as _f:
            _f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
