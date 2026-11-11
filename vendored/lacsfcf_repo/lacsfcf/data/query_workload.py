"""Synthetic query workload generation for d-dimensional grids."""
import itertools
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from lacsfcf.data.grid import Grid

Cell  = Tuple[int, ...]
Block = Tuple[int, ...]


@dataclass
class Query:
    """An axis-aligned window query.

    lo[k] and hi[k] are inclusive cell coordinate bounds in dimension k.
    """
    lo: Tuple[int, ...]   # lower corner (inclusive)
    hi: Tuple[int, ...]   # upper corner (inclusive)

    def hits(self, cell: Cell) -> bool:
        return all(self.lo[k] <= cell[k] <= self.hi[k] for k in range(len(cell)))

    def hit_cells_count(self, grid_shape: Tuple[int, ...]) -> int:
        return int(np.prod([self.hi[k] - self.lo[k] + 1 for k in range(len(grid_shape))]))
    
    def get_sizes(self) -> Tuple[int, ...]:
        """Return query box size in each dimension."""
        return tuple(self.hi[k] - self.lo[k] + 1 for k in range(len(self.lo)))
    
    def get_aspect_ratio(self) -> float:
        """Return aspect ratio (max_size / min_size) for 2D queries.
        
        For d>2, returns max_size / min_size across all dimensions.
        """
        sizes = self.get_sizes()
        max_size = max(sizes)
        min_size = min(sizes)
        return float(max_size) / float(min_size) if min_size > 0 else 1.0
    
    def get_center(self) -> Tuple[float, ...]:
        """Return query box center coordinates (normalized to [0,1])."""
        return tuple((self.lo[k] + self.hi[k]) / 2.0 for k in range(len(self.lo)))
    
    def get_volume(self) -> int:
        """Return query box volume (number of cells)."""
        return int(np.prod(self.get_sizes()))


class QueryWorkload:
    """Container for a list of queries and their statistics."""

    def __init__(self, queries: List[Query], grid: Grid):
        self.queries = queries
        self.grid    = grid
        self._heatmap:  Optional[np.ndarray] = None
        self._block_density: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    @property
    def heatmap(self) -> np.ndarray:
        """Cell-level query hit density, shape = grid.grid_shape, float32."""
        if self._heatmap is None:
            self._heatmap = self._compute_heatmap()
        return self._heatmap

    def _compute_heatmap(self) -> np.ndarray:
        heat = np.zeros(self.grid.grid_shape, dtype=np.float32)
        for q in self.queries:
            slices = tuple(slice(q.lo[k], q.hi[k] + 1) for k in range(self.grid.d))
            heat[slices] += 1.0
        if len(self.queries) > 0:
            heat /= len(self.queries)
        return heat

    @property
    def block_density(self) -> np.ndarray:
        """Block-level query hit density, shape = grid.block_shape, float32."""
        if self._block_density is None:
            h = self.heatmap
            # Sum over 2^d sub-cells in each block
            blk = h
            for k in range(self.grid.d):
                # reshape and sum pairs along dim k
                s = list(blk.shape)
                s[k] = blk.shape[k] // 2
                s.insert(k + 1, 2)
                blk = blk.reshape(s).sum(axis=k + 1)
            self._block_density = blk.astype(np.float32)
        return self._block_density

    def dual_edge_weights(self, noise_std: float = 0.0,
                          rng: Optional[np.random.Generator] = None,
                          marginal_alpha: float = 0.0) -> np.ndarray:
        """Edge weights for each dual edge based on query co-access.

        Weight = marginal_alpha * (bd[b1] + bd[b2]) + joint(b1, b2)
        where joint(b1, b2) = fraction of queries hitting BOTH blocks.

        When marginal_alpha=0 (default), weights equal the empirical block
        co-access probability J(b1,b2), aligned with the cost-model theorem
        (min E[nr] <=> max sum J).  Setting marginal_alpha>0 blends in
        marginal block density as a fallback signal (legacy behaviour used
        marginal_alpha=1).

        Returns float32 array of shape (num_dual_edges,).
        """
        bd = self.block_density
        d  = self.grid.d
        B  = self.grid.block_side
        E  = self.grid.num_dual_edges

        # Degenerate regime: coarse K with small block_side can produce zero
        # dual edges (num_dual_edges == 0). Return an empty float32 array so
        # that callers can branch, rather than crashing on weights.min().
        if E == 0:
            return np.zeros(0, dtype=np.float32)

        # --- Fully vectorized weight computation (block-space box-increment) ---
        # Numerically identical to the reference O(Q*E) double loop, but never
        # materializes grid.dual_edges and runs in O(Q*d + box_volumes) instead
        # of O(Q*E), so it scales to d=2 K=14 (134M edges) in seconds.
        nb = self.grid.num_blocks
        block_shape = self.grid.block_shape

        # C-order strides over block_shape.
        strides = np.ones(d, dtype=np.int64)
        for k in range(d - 2, -1, -1):
            strides[k] = strides[k + 1] * B

        # valid[blockid, dim] = True iff an edge exists from this block in
        # direction dim (block coord in dim < B-1). Boolean-mask flattening of
        # a (nb, d) array with this mask yields elements in block-major,
        # dim-minor order -- exactly Grid._build_dual_edges()'s edge order.
        block_ids = np.arange(nb, dtype=np.int64)
        valid = np.empty((nb, d), dtype=bool)
        for k in range(d):
            valid[:, k] = ((block_ids // strides[k]) % B) < (B - 1)

        # joint(b1, b1+e_dim) = (#queries hitting BOTH blocks) / Q. A query hits
        # both iff b1 lies in the per-query block box with the dim-axis upper
        # bound reduced by 1 (so b2 = b1+e_dim is also inside). Accumulate via
        # per-direction block-space box increments.
        joint_full = np.zeros((nb, d), dtype=np.float64)
        if self.queries:
            counts = [np.zeros(block_shape, dtype=np.float64) for _ in range(d)]
            for q in self.queries:
                blo = [q.lo[k] // 2 for k in range(d)]
                bhi = [min(q.hi[k] // 2, B - 1) for k in range(d)]
                for dim in range(d):
                    hi_dim = bhi[dim] - 1
                    if hi_dim < blo[dim]:
                        continue  # box empty in this direction
                    slc = tuple(
                        slice(blo[k], (hi_dim if k == dim else bhi[k]) + 1)
                        for k in range(d)
                    )
                    counts[dim][slc] += 1.0
            inv_q = 1.0 / len(self.queries)
            for dim in range(d):
                joint_full[:, dim] = counts[dim].ravel() * inv_q

        # marginal term: marginal_alpha * (bd[b1] + bd[b2]) per edge.
        if marginal_alpha != 0.0:
            bd_flat = bd.ravel().astype(np.float64)  # C-order over blocks
            weights_full = marginal_alpha * bd_flat[:, None] + joint_full
            for dim in range(d):
                neigh = np.clip(block_ids + strides[dim], 0, nb - 1)
                weights_full[:, dim] += marginal_alpha * bd_flat[neigh]
        else:
            weights_full = joint_full

        # Gather to reference edge order (block-major, dim-minor) via the mask.
        weights = weights_full[valid].astype(np.float32)

        # Min-max normalise to [0, 1] for consistent scale across d/K configs.
        w_min, w_max = weights.min(), weights.max()
        if w_max > w_min:
            weights = (weights - w_min) / (w_max - w_min)
        else:
            weights[:] = 0.5

        if noise_std > 0 and rng is not None:
            weights += rng.normal(0.0, noise_std, size=weights.shape).astype(np.float32)
        return weights.astype(np.float32)

    def sample_query_driven_weights(self, strategy: str = "hotspot",
                                    rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Sample edge weights based on query workload distribution.
        
        Args:
            strategy: Sampling strategy
                - "hotspot": Perturb weights based on hotspot regions
                - "gradient": Sample based on density gradient
                - "correlation": Sample based on spatial correlation
                - "mixed": Random mix of above strategies
        
        Returns:
            Edge weights array of shape (num_dual_edges,).
        """
        if rng is None:
            rng = np.random.default_rng()
        
        bd = self.block_density
        base_weights = self.dual_edge_weights()
        
        if strategy == "hotspot":
            # 在热点区域增加扰动，冷区域减少扰动
            weights = np.zeros(self.grid.num_dual_edges, dtype=np.float32)
            for i, (b1, b2, _) in enumerate(self.grid.dual_edges):
                density = (bd[b1] + bd[b2]) / 2.0
                # 热点区域 (高密度) → 大扰动，探索更多可能性
                noise_scale = 0.1 + 0.5 * density
                noise = rng.normal(0, noise_scale)
                weights[i] = base_weights[i] + noise
        
        elif strategy == "gradient":
            # 基于密度梯度采样：梯度大的边界更重要
            weights = np.zeros(self.grid.num_dual_edges, dtype=np.float32)
            for i, (b1, b2, _) in enumerate(self.grid.dual_edges):
                gradient = abs(bd[b1] - bd[b2])
                # 梯度大 → 权重重要性高 → 采样接近启发式
                # 梯度小 → 权重不重要 → 更大随机性
                noise_scale = max(0.05, 0.5 * (1.0 - gradient))  # 确保至少0.05
                noise = rng.normal(0, noise_scale)
                weights[i] = base_weights[i] + noise
        
        elif strategy == "correlation":
            # 基于空间相关性：相邻块密度相似 → 权重应该相似
            weights = base_weights.copy()
            # 添加全局随机扰动，但保持局部相关性
            global_noise = rng.normal(0, 0.2, size=weights.shape)
            weights += global_noise
        
        elif strategy == "mixed":
            # 随机选择一种策略
            sub_strategy = rng.choice(["hotspot", "gradient", "correlation"])
            weights = self.sample_query_driven_weights(sub_strategy, rng)
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        return weights.astype(np.float32)


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def _sample_one_query(
    rng: np.random.Generator,
    grid: Grid,
    center: Tuple[float, ...],
    win_min_frac: float,
    win_max_frac: float,
) -> Query:
    d  = grid.d
    N  = grid.grid_side
    lo_list, hi_list = [], []
    for k in range(d):
        frac = float(rng.uniform(win_min_frac, win_max_frac))
        half = max(1, int(round(frac * N / 2)))
        c    = int(round(center[k] * (N - 1)))
        lo   = int(np.clip(c - half, 0, N - 2 * half))
        hi   = min(lo + 2 * half - 1, N - 1)
        lo_list.append(lo)
        hi_list.append(hi)
    return Query(lo=tuple(lo_list), hi=tuple(hi_list))


def sample_workload(
    grid:         Grid,
    num_queries:  int,
    num_hotspots: int  = 3,
    hotspot_std:  float = 0.08,
    win_min_frac: float = 0.05,
    win_max_frac: float = 0.25,
    seed:         Optional[int] = None,
) -> QueryWorkload:
    """Generate a synthetic hotspot-based query workload.

    Hotspots are sampled uniformly in [0.1, 0.9]^d and then slightly
    perturbed with Gaussian noise.  Each query is centered at a random
    hotspot with a random window size.

    Args:
        grid:          Grid object.
        num_queries:   Total number of queries.
        num_hotspots:  Number of hotspot centers.
        hotspot_std:   Std dev of hotspot position jitter.
        win_min_frac:  Min window side as fraction of grid_side.
        win_max_frac:  Max window side as fraction of grid_side.
        seed:          RNG seed.

    Returns:
        QueryWorkload instance.
    """
    rng = np.random.default_rng(seed)
    d   = grid.d

    base_centers = rng.uniform(0.1, 0.9, size=(num_hotspots, d))
    hotspots = []
    for c in base_centers:
        jitter = rng.normal(0.0, hotspot_std, size=d)
        hotspots.append(tuple(float(np.clip(c[k] + jitter[k], 0.0, 1.0)) for k in range(d)))

    queries = []
    for _ in range(num_queries):
        center = hotspots[int(rng.integers(0, num_hotspots))]
        queries.append(_sample_one_query(rng, grid, center, win_min_frac, win_max_frac))

    return QueryWorkload(queries=queries, grid=grid)
