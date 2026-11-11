"""Real-data-aware meta-learning dataset for LACSFC-Neural-v2.

Mirrors the API of :class:`MetaLearningDataset` (in ``meta_learning.py``) but
builds each training task from **real dataset points** instead of the purely
synthetic hotspot mixture.

Motivation (T3 / §6 limitation #1 in D4_v2_PaperExperiments.md):
    The stock ``MetaLearningDataset`` samples hotspot centres uniformly in
    ``[0,1]^d``. On benchmarks derived from real spatial data (Tiger-2D,
    OSM-2D), the point density is heavily clustered (roads, cities, …),
    and the offline-trained v2 checkpoint trained on uniform hotspots fails
    to match the per-build LACSFC-Neural on these datasets.

    This dataset fixes that distribution shift by drawing each task's
    hotspot centres from **KMeans centroids of the real points**, matching
    the pattern used in ``LACSFCPlugin._sample_workload_from_data``. The
    evaluation workload (``generate_queries_loguniform_area``) is *not*
    changed — it remains point-independent — so no training-eval query
    overlap is created.

Train/test seed discipline:
    The benchmark scripts (``run_mainTable_*.sh``) use eval seeds
    ``0..N_SEEDS-1`` (10 for OSM-2D, 10 for the T1 upgrade). To keep
    training tasks strictly disjoint, this dataset uses seeds starting at
    :pyattr:`TRAIN_SEED_OFFSET` = 10_000.
"""
from __future__ import annotations

import os
import sys
from typing import List, Optional

import numpy as np

from .._paths import lacsfcf_root as _lr
sys.path.insert(0, str(_lr()))

from lacsfcf.data.grid import Grid
from lacsfcf.data.scene import SceneEncoder
from lacsfcf.data.query_workload import (
    QueryWorkload,
    _sample_one_query,
)


TRAIN_SEED_OFFSET = 10_000


# Same six distribution profiles as MetaLearningDataset, kept in sync so the
# only controlled difference between the two datasets is the hotspot source.
_DISTRIBUTIONS = [
    {'num_hotspots': 1, 'hotspot_std': 0.1,  'win_min_frac': 0.01,  'win_max_frac': 0.15},
    {'num_hotspots': 3, 'hotspot_std': 0.15, 'win_min_frac': 0.01,  'win_max_frac': 0.20},
    {'num_hotspots': 5, 'hotspot_std': 0.10, 'win_min_frac': 0.005, 'win_max_frac': 0.10},
    {'num_hotspots': 2, 'hotspot_std': 0.20, 'win_min_frac': 0.10,  'win_max_frac': 0.40},
    {'num_hotspots': 2, 'hotspot_std': 0.15, 'win_min_frac': 0.001, 'win_max_frac': 0.05},
    {'num_hotspots': 1, 'hotspot_std': 0.20, 'win_min_frac': 0.02,  'win_max_frac': 0.30},
]


def _kmeans_centers(
    points: np.ndarray, n_clusters: int, seed: int,
) -> np.ndarray:
    """KMeans centroids in [0,1]^d (points are assumed already normalised)."""
    from sklearn.cluster import KMeans
    n_clusters = int(max(1, min(n_clusters, len(points))))
    km = KMeans(n_clusters=n_clusters, random_state=int(seed), n_init=5)
    km.fit(points)
    return np.clip(km.cluster_centers_.astype(np.float64), 0.0, 1.0)


def _sample_real_workload(
    grid: Grid,
    points: np.ndarray,
    num_queries: int,
    num_hotspots: int,
    hotspot_std: float,
    win_min_frac: float,
    win_max_frac: float,
    seed: int,
) -> QueryWorkload:
    """Sample a QueryWorkload whose hotspots come from real-point KMeans.

    Matches the structure of :func:`lacsfcf.data.query_workload.sample_workload`
    but replaces uniform random hotspots with KMeans centres of ``points``
    (plus per-query Gaussian jitter of scale ``hotspot_std``).
    """
    rng = np.random.default_rng(int(seed))
    d = int(points.shape[1])

    centers = _kmeans_centers(points, num_hotspots, seed=seed)

    queries = []
    for _ in range(int(num_queries)):
        c0 = centers[int(rng.integers(0, len(centers)))]
        jitter = rng.normal(0.0, float(hotspot_std), size=d)
        center = tuple(float(np.clip(c0[k] + jitter[k], 0.0, 1.0)) for k in range(d))
        queries.append(_sample_one_query(rng, grid, center, float(win_min_frac), float(win_max_frac)))

    return QueryWorkload(queries=queries, grid=grid)


class RealWorkloadDataset:
    """Real-points-aware counterpart of :class:`MetaLearningDataset`.

    Each task is built from one of the six distribution profiles in
    :data:`_DISTRIBUTIONS`, but its hotspot centres are drawn from
    KMeans centroids of ``points`` instead of uniform randoms.
    """

    def __init__(
        self,
        grid: Grid,
        encoder: SceneEncoder,
        points: np.ndarray,
        n_tasks: int = 50,
        queries_per_task: int = 100,
        seed: int = 42,
    ):
        if points is None or points.ndim != 2:
            raise ValueError("points must have shape (n, d)")
        if points.shape[1] != grid.d:
            raise ValueError(
                f"points dim {points.shape[1]} != grid.d {grid.d}"
            )

        self.grid = grid
        self.encoder = encoder
        self.points = np.clip(np.asarray(points, dtype=np.float64), 0.0, 1.0)
        self.n_tasks = int(n_tasks)
        self.queries_per_task = int(queries_per_task)
        self.tasks: List[dict] = []

        self._build_tasks(int(seed))

    def _build_tasks(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        # Enforce train/eval seed-space disjointness: every per-task seed
        # used to sample queries starts at TRAIN_SEED_OFFSET. Eval seeds
        # 0..999 stay reserved for benchmark scripts.
        for task_id in range(self.n_tasks):
            cfg = _DISTRIBUTIONS[task_id % len(_DISTRIBUTIONS)]
            task_seed = TRAIN_SEED_OFFSET + int(rng.integers(0, 10**9))

            wl = _sample_real_workload(
                self.grid,
                self.points,
                num_queries=self.queries_per_task,
                num_hotspots=cfg['num_hotspots'],
                hotspot_std=cfg['hotspot_std'],
                win_min_frac=cfg['win_min_frac'],
                win_max_frac=cfg['win_max_frac'],
                seed=task_seed,
            )
            queries = wl.queries
            feats = self.encoder.encode(wl)
            base_weights = wl.dual_edge_weights()

            self.tasks.append({
                'task_id': task_id,
                'feats': feats,
                'queries': queries,
                'base_weights': base_weights,
                'dist_type': task_id % len(_DISTRIBUTIONS),
                'task_seed': task_seed,
            })

    def __len__(self) -> int:
        return len(self.tasks)

    def __getitem__(self, idx: int) -> dict:
        return self.tasks[idx]

    def sample_tasks(self, n: int, rng: Optional[np.random.Generator] = None) -> List[dict]:
        if rng is None:
            rng = np.random.default_rng()
        indices = rng.choice(len(self.tasks), size=int(n), replace=False)
        return [self.tasks[i] for i in indices]


def load_points_for_training(
    points_file: str,
    d: int,
    max_n: int = 50_000,
) -> np.ndarray:
    """Wrapper around ``src.mvp.workloads.load_points_real`` for use here.

    Uses the same loader that :mod:`src.mvp.compare_runner` uses at benchmark
    time, so the point cloud seen during training is identical in
    preprocessing (range clipping, percentile normalisation for integer
    grids, etc.) to the point cloud seen at eval time.
    """
    from src.mvp.workloads import load_points_real
    pts = load_points_real(points_file, k=1, max_n=int(max_n))
    if pts.shape[1] != int(d):
        raise ValueError(
            f"points file {points_file} has dim {pts.shape[1]}, expected d={d}"
        )
    return pts


__all__ = [
    "RealWorkloadDataset",
    "TRAIN_SEED_OFFSET",
    "load_points_for_training",
]
