#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.mvp.workloads import load_points_real


DEFAULT_SOURCES: Dict[str, str] = {
    "osm_time_3d": "data/OSM/osm25mbinary",
    "osm_time_pop_4d": "data/OSM/osm25mbinary",
    "geonames_time_3d": "data/tiger_tweets/geonames_2d.csv",
    "geonames_time_cat_4d": "data/tiger_tweets/geonames_2d.csv",
    "tiger_time_3d": "data/tiger_tweets/tiger_us_2d.csv",
    "tiger_time_class_4d": "data/tiger_tweets/tiger_us_2d.csv",
}


def _clip01(x: np.ndarray) -> np.ndarray:
    eps = np.finfo(np.float64).eps
    return np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0 - eps)


def _density_score(points2: np.ndarray, grid: int = 32) -> np.ndarray:
    pts = _clip01(points2[:, :2])
    ij = np.floor(pts * int(grid)).astype(np.int64)
    ij = np.clip(ij, 0, int(grid) - 1)
    flat = ij[:, 0] * int(grid) + ij[:, 1]
    counts = np.bincount(flat, minlength=int(grid) * int(grid)).astype(np.float64)
    vals = counts[flat]
    lo = float(np.percentile(vals, 5))
    hi = float(np.percentile(vals, 99))
    if hi <= lo:
        return np.zeros((len(points2),), dtype=np.float64)
    return np.clip((vals - lo) / (hi - lo), 0.0, 1.0)


def _augment_time(points2: np.ndarray, density: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = points2[:, 0]
    y = points2[:, 1]
    dense_center = np.mod(0.18 + 0.55 * density + 0.17 * x + 0.07 * y, 1.0)
    background = rng.random(len(points2), dtype=np.float64)
    use_dense = rng.random(len(points2), dtype=np.float64) < (0.35 + 0.55 * density)
    noise = rng.normal(0.0, 0.055, size=len(points2)).astype(np.float64)
    t = np.where(use_dense, dense_center + noise, background)
    return _clip01(np.mod(t, 1.0))


def _augment_popularity(density: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    raw = np.exp(rng.normal(0.0 + 1.8 * density, 0.75, size=len(density))).astype(np.float64)
    lo = float(np.percentile(raw, 1))
    hi = float(np.percentile(raw, 99))
    if hi <= lo:
        return np.zeros_like(density)
    return _clip01((raw - lo) / (hi - lo))


def _augment_category(points2: np.ndarray, density: np.ndarray, bins: int, rng: np.random.Generator) -> np.ndarray:
    logits = 0.45 * points2[:, 0] + 0.35 * points2[:, 1] + 0.55 * density
    noisy = logits + rng.normal(0.0, 0.12, size=len(points2))
    cat = np.floor(np.clip(noisy, 0.0, 1.0 - np.finfo(np.float64).eps) * int(bins)).astype(np.int64)
    jitter = rng.uniform(0.05, 0.95, size=len(points2))
    return _clip01((cat.astype(np.float64) + jitter) / float(bins))


def make_points(workload: str, source: str, n: int, seed: int, k: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    base = load_points_real(source, k=int(k), max_n=max(int(n) * 4, int(n)))
    base = np.asarray(base, dtype=np.float64)
    if base.ndim != 2 or base.shape[1] < 2:
        raise ValueError("source points must have at least two dimensions")
    if len(base) > int(n):
        idx = rng.choice(len(base), size=int(n), replace=False)
        base = base[idx]
    points2 = _clip01(base[:, :2])
    density = _density_score(points2)
    time = _augment_time(points2, density, rng)

    if workload.endswith("_3d"):
        return _clip01(np.column_stack([points2, time]))
    if workload == "osm_time_pop_4d":
        extra = _augment_popularity(density, rng)
    elif workload == "geonames_time_cat_4d":
        extra = _augment_category(points2, density, bins=16, rng=rng)
    elif workload == "tiger_time_class_4d":
        extra = _augment_category(points2, density, bins=8, rng=rng)
    else:
        raise ValueError(f"unsupported workload: {workload}")
    return _clip01(np.column_stack([points2, time, extra]))


def make_queries(points: np.ndarray, qn: int, seed: int, spatial_area_min: float, spatial_area_max: float) -> np.ndarray:
    rng = np.random.default_rng(int(seed) + 10_000)
    pts = np.asarray(points, dtype=np.float64)
    d = int(pts.shape[1])
    density = _density_score(pts[:, :2])
    weights = 0.15 + density
    weights = weights / np.sum(weights)
    rows = []
    log_sp_min = float(np.log(float(spatial_area_min)))
    log_sp_max = float(np.log(float(spatial_area_max)))
    log_t_min = float(np.log(0.015))
    log_t_max = float(np.log(0.25))
    for _ in range(int(qn)):
        mode = rng.choice(3, p=np.array([0.55, 0.30, 0.15], dtype=np.float64))
        if mode == 0:
            center = pts[int(rng.choice(len(pts), p=weights))].copy()
            spatial_area = float(np.exp(rng.uniform(log_sp_min, log_sp_max)))
        elif mode == 1:
            center = pts[int(rng.integers(0, len(pts)))].copy()
            spatial_area = float(np.exp(rng.uniform(np.log(float(spatial_area_min) * 4.0), np.log(min(float(spatial_area_max) * 8.0, 0.35)))))
        else:
            center = rng.random(d, dtype=np.float64)
            spatial_area = float(np.exp(rng.uniform(log_sp_min, log_sp_max)))
        spatial_side = float(np.sqrt(np.clip(spatial_area, 1e-12, 1.0)))
        half = np.full(d, spatial_side / 2.0, dtype=np.float64)
        if d >= 3:
            half[2] = float(np.exp(rng.uniform(log_t_min, log_t_max))) / 2.0
        if d >= 4:
            half[3] = float(rng.choice([0.035, 0.08, 0.16, 0.32], p=[0.35, 0.35, 0.2, 0.1])) / 2.0
        low = np.clip(center - half, 0.0, 1.0)
        high = np.clip(center + half, 0.0, 1.0)
        high = np.maximum(high, low + np.finfo(np.float64).eps)
        high = np.clip(high, 0.0, 1.0)
        rows.append(np.concatenate([low, high]))
    return np.asarray(rows, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", required=True, choices=sorted(DEFAULT_SOURCES))
    parser.add_argument("--source-points-file", default="")
    parser.add_argument("--out-dir", default="data/realistic_highdim_v1")
    parser.add_argument(
        "--out-prefix",
        default="",
        help=(
            "Optional output file stem (without _points.npy / _queries.csv). "
            "When empty, defaults to '<workload>_s<seed>'. Use a scale-unique "
            "prefix (e.g. '<workload>_s<seed>_n<N>') to avoid different scales "
            "silently reusing the same cached file."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--qn", type=int, default=50)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--spatial-area-min", type=float, default=1e-5)
    parser.add_argument("--spatial-area-max", type=float, default=5e-3)
    args = parser.parse_args()

    source = args.source_points_file or DEFAULT_SOURCES[str(args.workload)]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = str(args.out_prefix).strip() or f"{args.workload}_s{int(args.seed)}"
    prefix = out_dir / stem

    points = make_points(str(args.workload), source, int(args.n), int(args.seed), int(args.k))
    queries = make_queries(points, int(args.qn), int(args.seed), float(args.spatial_area_min), float(args.spatial_area_max))

    points_file = prefix.with_name(prefix.name + "_points.npy")
    queries_file = prefix.with_name(prefix.name + "_queries.csv")
    np.save(points_file, points.astype(np.float64), allow_pickle=False)
    with queries_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(queries.tolist())
    print(f"workload={args.workload} seed={args.seed} d={points.shape[1]} n={len(points)} qn={len(queries)}")
    print(f"points_file={points_file}")
    print(f"queries_file={queries_file}")


if __name__ == "__main__":
    main()
