"""LACSFC Neural Network Training Module.

Implements the complete training pipeline for LACSFC with joint Evaluator-Generator training.
Based on LACSFCF train_v3.py with three-weight strategy.
"""
import sys
import os
import time
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.optim as optim

from .._paths import lacsfcf_root as _lr
sys.path.insert(0, str(_lr()))

from lacsfcf.core.sfc import weights_to_sfc
from lacsfcf.data.grid import Grid
from lacsfcf.data.scene import SceneEncoder
from lacsfcf.data.query_workload import sample_workload, QueryWorkload
from lacsfcf.model.evaluator import QueryEvaluator
from lacsfcf.model.generator import EdgeWeightGenerator
from collections import deque

# Import LACSFC's query algorithms
from .boundary_query import LACSFCBoundaryQuery

# Backward-compat alias.  The legacy precomputed-cpp processor was removed
# (see docs/LACSFC/QueryAlgo_Audit.md §2.2); training and quality eval
# always go through ``LACSFCBoundaryQuery`` now.  Kept as ``False`` so
# downstream callers that still reference ``PRECOMPUTED_AVAILABLE`` keep
# working without code changes.
PRECOMPUTED_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Fast construction hook for *training* quality evaluation.                    #
#                                                                             #
# The inner quality call (compute_quality_mvp) historically builds the SFC via #
# weights_to_sfc (~31s at K=8), which forced all v2 checkpoints to be trained  #
# at K=4. fast_code_map is byte-identical for d=2 and quality-equivalent for   #
# d>=3 (locked by tests/test_fast_construct_regression.py), so swapping it in  #
# preserves training semantics while unlocking large-K training. Guarded by    #
# LACSFC_FAST_TRAIN (default off) to keep existing behavior unchanged.         #
# --------------------------------------------------------------------------- #
from pathlib import Path as _Path

_FAST_SCRIPTS_DIR = _Path(__file__).resolve().parents[3] / "scripts"


def _fast_train_enabled() -> bool:
    return (os.environ.get("LACSFC_FAST_TRAIN", "") or "").strip().lower() in {
        "1", "true", "yes", "on"
    }


def _get_fast_code_map():
    """Return the ``fast_code_map`` callable, or None if unavailable."""
    try:
        if str(_FAST_SCRIPTS_DIR) not in sys.path:
            sys.path.insert(0, str(_FAST_SCRIPTS_DIR))
        from fast_construct import fast_code_map
        return fast_code_map
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[LACSFC] fast_construct unavailable, using weights_to_sfc: {exc}")
        return None


def _build_code_map(grid: Grid, weights: np.ndarray) -> np.ndarray:
    """Build a code_map from edge weights, using the fast path when enabled."""
    if _fast_train_enabled():
        fast = _get_fast_code_map()
        if fast is not None:
            return np.asarray(fast(grid, np.asarray(weights, dtype=np.float64)),
                              dtype=np.int64)
    path = weights_to_sfc(grid, weights)
    return grid.code_map_from_path(path)

class SceneDataset:
    """On-the-fly synthetic scene dataset for training."""
    
    def __init__(self, grid: Grid, enc: SceneEncoder, queries_list: List, size: int = 128):
        self.grid = grid
        self.enc = enc
        self.size = size
        self._scenes: List = []
        self._build(queries_list)
    
    def _build(self, queries_list: List):
        """Build dataset from provided queries or generate synthetic ones.

        If queries_list contains a flat list of Query objects (single workload),
        each scene gets a random subsample of 50-100 queries to ensure diversity.
        If queries_list is a list-of-lists, each scene uses its own query list.
        """
        rng = np.random.default_rng(42)

        # Detect if queries_list is a flat list of Query objects (single workload repeated)
        is_flat = (queries_list and not isinstance(queries_list[0], list)
                   and hasattr(queries_list[0], 'lo'))

        for i in range(self.size):
            if is_flat:
                # Random subsample from the single shared query pool
                n_q = int(rng.integers(30, min(100, len(queries_list)) + 1))
                idxs = rng.choice(len(queries_list), size=n_q, replace=False)
                queries = [queries_list[j] for j in idxs]
            elif queries_list and i < len(queries_list):
                src = queries_list[i]
                # src may itself be a list of Query objects
                if src and hasattr(src, '__len__') and not hasattr(src, 'lo'):
                    queries = src
                else:
                    queries = src
            else:
                wl = sample_workload(
                    self.grid,
                    num_queries=100,
                    num_hotspots=3,
                    hotspot_std=0.15,
                    win_min_frac=0.01,
                    win_max_frac=0.2,
                    seed=int(rng.integers(0, 10**9)),
                )
                queries = wl.queries

            wl = QueryWorkload(queries, self.grid)
            feats = self.enc.encode(wl)
            base_weights = wl.dual_edge_weights()

            self._scenes.append((feats, queries, base_weights))
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self._scenes[i] for i in range(*idx.indices(len(self)))]
        return self._scenes[idx]


def compute_quality_from_codemap(
    code_map: np.ndarray,
    queries,
    grid_size: int,
    d: int,
    b: int = 256,
) -> Tuple[float, float, float]:
    """Fast quality computation directly on an existing code_map.

    Avoids calling weights_to_sfc (the 31s bottleneck).  Uses
    LACSFCBoundaryQuery (Python, ~0.8ms per query for 100 queries).

    Returns:
        (mean_seg, mean_span_per_cell, mean_page_amp)
    """
    from lacsfcf.metrics.range_query import mean_seg as lacsfcf_mean_seg

    seg = float(lacsfcf_mean_seg(code_map, queries))

    # Compute span and page_amp via boundary scan
    bq = LACSFCBoundaryQuery(code_map=code_map, grid_size=grid_size, d=d)

    total_span = 0
    total_result_count = 0
    total_page_accesses = 0
    n_queries = len(queries)

    gs_minus1 = grid_size - 1
    for q in queries:
        low_cell = tuple(max(0, min(int(c), gs_minus1)) for c in q.lo)
        high_cell = tuple(max(0, min(int(c), gs_minus1)) for c in q.hi)
        intervals = bq.get_intervals(low_cell, high_cell)
        if not intervals:
            continue

        all_codes = []
        for start, end in intervals:
            all_codes.extend(range(start, end + 1))
        if not all_codes:
            continue

        total_span += max(all_codes) - min(all_codes)
        result_count = len(all_codes)
        total_result_count += result_count
        pages_accessed = len(set(c // b for c in all_codes))
        min_pages = max(1, (result_count + b - 1) // b)
        total_page_accesses += pages_accessed / min_pages

    mean_span_per_cell = total_span / total_result_count if total_result_count > 0 else 0.0
    mean_page_amp = total_page_accesses / n_queries if n_queries > 0 else 1.0

    return seg, mean_span_per_cell, mean_page_amp


def compute_quality_mvp(
    grid: Grid,
    weights: np.ndarray,
    queries,
    b: int = 256,
    use_precomputed: bool = False,  # deprecated, ignored
) -> Tuple[float, float, float]:
    """Compute MVP quality metrics (builds SFC from weights first).

    NOTE: This calls weights_to_sfc internally (~31s).  During training
    prefer compute_quality_from_codemap() after caching the code_map.

    The ``use_precomputed`` flag is retained for backward compatibility
    with older callers and is ignored — the legacy precomputed-cpp
    processor has been removed.  All paths now use
    :class:`LACSFCBoundaryQuery`.

    Returns:
        (mean_seg, mean_span_per_cell, mean_page_amp)
    """
    del use_precomputed  # accepted for API parity, no longer routed

    code_map = _build_code_map(grid, weights)
    grid_size = 2 ** grid.K

    bq = LACSFCBoundaryQuery(code_map=code_map, grid_size=grid_size, d=grid.d)

    def _get_intervals_boundary(q):
        low_cell = tuple(max(0, min(int(c), grid_size - 1)) for c in q.lo)
        high_cell = tuple(max(0, min(int(c), grid_size - 1)) for c in q.hi)
        return bq.get_intervals(low_cell, high_cell)

    get_intervals = _get_intervals_boundary

    total_seg = 0
    total_span = 0
    total_result_count = 0
    total_page_accesses = 0
    n_queries = len(queries)

    for q in queries:
        intervals = get_intervals(q)
        if not intervals:
            continue

        total_seg += len(intervals)

        all_codes = []
        for start, end in intervals:
            all_codes.extend(range(start, end + 1))
        if not all_codes:
            continue

        span = max(all_codes) - min(all_codes)
        result_count = len(all_codes)
        total_span += span
        total_result_count += result_count
        pages_accessed = len(set(c // b for c in all_codes))
        min_pages = max(1, (result_count + b - 1) // b)
        total_page_accesses += pages_accessed / min_pages

    mean_seg = total_seg / n_queries if n_queries > 0 else 0.0
    mean_span_per_cell = total_span / total_result_count if total_result_count > 0 else 0.0
    mean_page_amp = total_page_accesses / n_queries if n_queries > 0 else 1.0

    return mean_seg, mean_span_per_cell, mean_page_amp


class AdaptiveSampler:
    """Adaptive sampling strategy selector.
    
    Dynamically adjusts strategy probabilities based on historical performance.
    """
    
    def __init__(self, strategies, initial_probs=None, alpha=0.1):
        """
        Args:
            strategies: List of (strategy_name, initial_prob)
            initial_probs: Optional initial probabilities
            alpha: Learning rate for probability update
        """
        self.strategy_names = [s[0] for s in strategies]
        self.n_strategies = len(self.strategy_names)
        
        if initial_probs is None:
            self.probs = np.array([s[1] for s in strategies], dtype=np.float32)
        else:
            self.probs = np.array(initial_probs, dtype=np.float32)
        
        # Normalize
        self.probs = self.probs / self.probs.sum()
        
        self.alpha = alpha
        
        # Track performance
        self.strategy_rewards = {name: [] for name in self.strategy_names}
        self.strategy_counts = {name: 0 for name in self.strategy_names}
    
    def sample(self, rng):
        """Sample a strategy based on current probabilities."""
        idx = rng.choice(self.n_strategies, p=self.probs)
        return self.strategy_names[idx], idx
    
    def update(self, strategy_name, reward):
        """Update probabilities based on reward.
        
        Args:
            strategy_name: Name of the strategy used
            reward: Reward obtained (negative quality metric)
        """
        self.strategy_rewards[strategy_name].append(reward)
        self.strategy_counts[strategy_name] += 1
    
    def adapt_probabilities(self, verbose=False):
        """Adapt probabilities based on accumulated rewards."""
        # Compute average reward for each strategy
        avg_rewards = []
        for name in self.strategy_names:
            if len(self.strategy_rewards[name]) > 0:
                avg_reward = np.mean(self.strategy_rewards[name][-10:])  # Last 10
            else:
                avg_reward = 0.0
            avg_rewards.append(avg_reward)
        
        avg_rewards = np.array(avg_rewards)
        
        # Convert rewards to scores (higher is better)
        # Since reward is negative quality, we negate it
        scores = -avg_rewards
        
        # Softmax to get new probabilities
        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        
        new_probs = np.exp(scores) / np.exp(scores).sum()
        
        # Smooth update
        self.probs = (1 - self.alpha) * self.probs + self.alpha * new_probs
        self.probs = self.probs / self.probs.sum()
        
        if verbose:
            print("  [Adaptive Sampling] Updated probabilities:")
            for name, prob in zip(self.strategy_names, self.probs):
                count = self.strategy_counts[name]
                avg_r = np.mean(self.strategy_rewards[name][-10:]) if len(self.strategy_rewards[name]) > 0 else 0.0
                print(f"    {name:<25} prob={prob:.3f}  count={count:3d}  avg_reward={avg_r:.2f}")
    
    def get_probabilities(self):
        """Get current probabilities."""
        return self.probs.copy()


class ExperienceReplay:
    """Experience replay buffer for high-quality weights.
    
    Stores historical high-quality weights and replays them during training.
    """
    
    def __init__(self, capacity=100, top_k=10):
        """
        Args:
            capacity: Maximum buffer size
            top_k: Number of top samples to keep
        """
        self.capacity = capacity
        self.top_k = top_k
        self.buffer = deque(maxlen=capacity)
        self.top_samples = []  # (quality, weights, feats, queries)
    
    def add(self, weights, feats, queries, quality):
        """Add a sample to the buffer.
        
        Args:
            weights: Edge weights
            feats: Scene features
            queries: Query workload
            quality: Quality metric (lower is better)
        """
        sample = (quality, weights.copy(), feats.copy(), queries)
        self.buffer.append(sample)
        
        # Update top samples
        self.top_samples.append(sample)
        self.top_samples.sort(key=lambda x: x[0])  # Sort by quality
        self.top_samples = self.top_samples[:self.top_k]  # Keep top-k
    
    def sample_top(self, n=1, rng=None):
        """Sample from top-k high-quality samples."""
        if len(self.top_samples) == 0:
            return []
        
        if rng is None:
            rng = np.random.default_rng()
        
        n = min(n, len(self.top_samples))
        indices = rng.choice(len(self.top_samples), size=n, replace=False)
        
        return [self.top_samples[i] for i in indices]
    
    def get_best(self):
        """Get the best sample."""
        if len(self.top_samples) > 0:
            return self.top_samples[0]
        return None
    
    def __len__(self):
        return len(self.buffer)


class MultiObjectiveOptimizer:
    """Multi-objective optimizer for MVP metrics.
    
    Optimizes three metrics with adaptive weights:
    - mean_seg: average number of segments (highest priority)
    - mean_span_per_cell: average span per cell (medium priority)
    - mean_page_amp: page amplification factor (lowest priority)
    
    Weight hierarchy: mean_seg > mean_span_per_cell > mean_page_amp
    """
    
    def __init__(
        self, 
        seg_weight=0.6, 
        span_per_cell_weight=0.3, 
        page_amp_weight=0.1,
        adaptive=True,
        alpha=0.05
    ):
        """
        Args:
            seg_weight: Initial weight for mean_seg (highest priority)
            span_per_cell_weight: Initial weight for mean_span_per_cell (medium priority)
            page_amp_weight: Initial weight for mean_page_amp (lowest priority)
            adaptive: Whether to adapt weights based on improvement rates
            alpha: Learning rate for weight adaptation
        """
        self.seg_weight = seg_weight
        self.span_per_cell_weight = span_per_cell_weight
        self.page_amp_weight = page_amp_weight
        self.adaptive = adaptive
        self.alpha = alpha
        
        # Normalize weights
        total = self.seg_weight + self.span_per_cell_weight + self.page_amp_weight
        self.seg_weight /= total
        self.span_per_cell_weight /= total
        self.page_amp_weight /= total
        
        # Track metrics history
        self.seg_history = []
        self.span_per_cell_history = []
        self.page_amp_history = []
        self.span_history = []
    
    def compute_combined_quality(
        self, 
        mean_seg: float, 
        mean_span_per_cell: float, 
        mean_page_amp: float
    ) -> float:
        """Compute combined quality metric using MVP metrics.
        
        Args:
            mean_seg: Mean number of segments (lower is better)
            mean_span_per_cell: Mean span per cell (lower is better)
            mean_page_amp: Page amplification factor (lower is better)
        
        Returns:
            Combined quality (lower is better)
        """
        # Weighted sum of normalized metrics
        combined = (
            self.seg_weight * mean_seg +
            self.span_per_cell_weight * mean_span_per_cell +
            self.page_amp_weight * mean_page_amp
        )
        
        # Track history
        self.seg_history.append(mean_seg)
        self.span_per_cell_history.append(mean_span_per_cell)
        self.page_amp_history.append(mean_page_amp)
        
        return combined
    
    def adapt_weights(self, verbose=False):
        """Adapt weights based on recent improvement rates.
        
        Maintains hierarchy: mean_seg > mean_span_per_cell > mean_page_amp
        """
        if not self.adaptive or len(self.seg_history) < 10:
            return
        
        # Compute improvement rates over last 10 samples
        recent_seg = self.seg_history[-10:]
        recent_span_per_cell = self.span_per_cell_history[-10:]
        recent_page_amp = self.page_amp_history[-10:]
        
        # Calculate relative improvement (higher is better)
        seg_improvement = (recent_seg[0] - recent_seg[-1]) / (recent_seg[0] + 1e-8)
        span_improvement = (recent_span_per_cell[0] - recent_span_per_cell[-1]) / (recent_span_per_cell[0] + 1e-8)
        page_improvement = (recent_page_amp[0] - recent_page_amp[-1]) / (recent_page_amp[0] + 1e-8)
        
        # Compute new weights based on improvement rates
        # Metrics with slower improvement get higher weights
        seg_need = max(0.0, -seg_improvement + 0.5)  # Higher if not improving
        span_need = max(0.0, -span_improvement + 0.3)
        page_need = max(0.0, -page_improvement + 0.2)
        
        total_need = seg_need + span_need + page_need + 1e-8
        
        new_seg_weight = seg_need / total_need
        new_span_weight = span_need / total_need
        new_page_weight = page_need / total_need
        
        # Enforce hierarchy: seg_weight >= span_weight >= page_weight
        # Adjust to maintain minimum ratios
        min_seg_ratio = 0.4  # seg should be at least 40%
        min_span_ratio = 0.2  # span should be at least 20%
        
        new_seg_weight = max(new_seg_weight, min_seg_ratio)
        new_span_weight = max(new_span_weight, min_span_ratio)
        
        # Ensure seg > span > page
        if new_seg_weight < new_span_weight:
            new_seg_weight, new_span_weight = new_span_weight, new_seg_weight
        if new_span_weight < new_page_weight:
            new_span_weight, new_page_weight = new_page_weight, new_span_weight
        if new_seg_weight < new_span_weight:
            new_seg_weight, new_span_weight = new_span_weight, new_seg_weight
        
        # Normalize
        total = new_seg_weight + new_span_weight + new_page_weight
        new_seg_weight /= total
        new_span_weight /= total
        new_page_weight /= total
        
        # Smooth update
        self.seg_weight = (1 - self.alpha) * self.seg_weight + self.alpha * new_seg_weight
        self.span_per_cell_weight = (1 - self.alpha) * self.span_per_cell_weight + self.alpha * new_span_weight
        self.page_amp_weight = (1 - self.alpha) * self.page_amp_weight + self.alpha * new_page_weight
        
        # Final normalization
        total = self.seg_weight + self.span_per_cell_weight + self.page_amp_weight
        self.seg_weight /= total
        self.span_per_cell_weight /= total
        self.page_amp_weight /= total
        
        if verbose:
            print(f"  [Multi-Objective] Adapted weights: seg={self.seg_weight:.3f}, span_per_cell={self.span_per_cell_weight:.3f}, page_amp={self.page_amp_weight:.3f}")
    
    def get_weights(self):
        """Get current weights."""
        return self.seg_weight, self.span_per_cell_weight, self.page_amp_weight


def finetune_evaluator_online(
    evaluator: QueryEvaluator,
    generator: EdgeWeightGenerator,
    dataset: SceneDataset,
    grid: Grid,
    n_samples: int = 32,
    lr: float = 1e-4,
    device: str = 'cpu',
    verbose: bool = False,
):
    """Online fine-tuning of Evaluator to track Generator evolution.
    
    This is critical for maintaining Evaluator accuracy as Generator improves.
    
    Args:
        evaluator: Evaluator model to fine-tune
        generator: Current Generator model
        dataset: Training dataset
        grid: Grid object
        n_samples: Number of samples for fine-tuning
        lr: Learning rate for fine-tuning
        device: Device to use
        verbose: Whether to print progress
    """
    device = torch.device(device)
    evaluator.train()
    generator.eval()
    
    optimizer = optim.Adam(evaluator.parameters(), lr=lr)
    
    if verbose:
        print("  [Online Finetune] Calibrating Evaluator on current Generator outputs...")
    
    losses = []
    
    # Sample random scenes for fine-tuning
    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)
    
    for idx in indices:
        feats, queries, _ = dataset[idx]
        t_feats = torch.tensor(feats, dtype=torch.float32, device=device)
        
        # Generate weights from current Generator
        with torch.no_grad():
            weights_gen = generator(t_feats).cpu().numpy()
        
        # Compute true quality — build SFC once, then fast metric on code_map
        path_gen = weights_to_sfc(grid, weights_gen)
        code_map_gen = grid.code_map_from_path(path_gen)
        seg, span_per_cell, page_amp = compute_quality_from_codemap(
            code_map_gen, queries, grid_size=2**grid.K, d=grid.d, b=256
        )
        quality_true = 0.60 * seg + 0.15 * span_per_cell + 0.25 * page_amp
        
        # Fine-tune Evaluator
        optimizer.zero_grad()
        t_weights = torch.tensor(weights_gen, dtype=torch.float32, device=device)
        pred_quality = evaluator(t_feats, t_weights)
        target = torch.tensor(quality_true, dtype=torch.float32, device=device)
        
        loss = (pred_quality - target) ** 2
        loss.backward()
        optimizer.step()
        
        losses.append(float(loss.detach()))
    
    if verbose:
        avg_loss = np.mean(losses) if losses else 0.0
        print(f"  [Online Finetune] Calibration loss: {avg_loss:.4f}")


def sample_target_weights(base_weights: np.ndarray, strategy: str, rng=None, generator=None, feats=None, device=None):
    """Sample target weights with different strategies.
    
    Args:
        base_weights: Base heuristic weights (Dafner-like)
        strategy: Sampling strategy name
        rng: Random number generator
        generator: Optional Generator model for residual connection
        feats: Optional features for Generator
        device: Optional device for Generator
    """
    if rng is None:
        rng = np.random.default_rng()
    
    weights = base_weights.astype(np.float32)
    
    if strategy == 'dafner_baseline':
        # Pure Dafner baseline (no perturbation)
        return weights
    
    elif strategy == 'heuristic_noise':
        noise = rng.normal(0, 0.1, size=weights.shape).astype(np.float32)
        return weights + noise
    
    elif strategy == 'heuristic_flip':
        # 50% probability to flip sign
        if rng.random() > 0.5:
            weights = -weights
        return weights
    
    elif strategy == 'heuristic_mix':
        # Random mix with noise
        mix_ratio = rng.random()
        random_weights = rng.normal(0, 0.5, size=weights.shape).astype(np.float32)
        return mix_ratio * weights + (1 - mix_ratio) * random_weights
    
    elif strategy == 'random_uniform':
        return rng.uniform(-1.0, 1.0, size=weights.shape).astype(np.float32)
    
    elif strategy == 'random_normal':
        return rng.normal(0.0, 0.5, size=weights.shape).astype(np.float32)
    
    elif strategy == 'residual_connection' and generator is not None:
        # Generator + residual from base weights
        with torch.no_grad():
            t_feats = torch.tensor(feats, dtype=torch.float32, device=device)
            weights_gen = generator(t_feats).cpu().numpy()
        return weights_gen + 0.1 * weights
    
    elif strategy == 'flip_and_mix':
        # Combine flip and mix
        if rng.random() > 0.5:
            weights = -weights
        if rng.random() > 0.5:
            mix_ratio = rng.random()
            random_weights = rng.normal(0, 0.5, size=weights.shape).astype(np.float32)
            weights = mix_ratio * weights + (1 - mix_ratio) * random_weights
        return weights
    
    else:
        return weights


def joint_train(
    generator: EdgeWeightGenerator,
    evaluator: QueryEvaluator,
    dataset: SceneDataset,
    grid: Grid,
    epochs: int = 30,
    batch_size: int = 8,
    lr: float = 1e-3,
    device: str = 'cpu',
    verbose: bool = True,
    use_adaptive_sampling: bool = True,
    use_experience_replay: bool = True,
    use_multi_objective: bool = True,
    ranking_margin: float = 0.0,
    ranking_weight: float = 0.8,
):
    """Joint training with advanced optimizations (v5).
    
    Features:
      1. Adaptive sampling: Dynamic strategy probability adjustment
      2. Experience replay: Store and replay high-quality weights
      3. Multi-objective: Adaptive weighting of mean_seg and mean_span
      4. Online Evaluator fine-tuning: Track Generator evolution
      5. Pairwise ranking loss: Evaluator trained to rank pairs correctly
    
    Args:
        use_adaptive_sampling: Enable adaptive strategy sampling
        use_experience_replay: Enable experience replay buffer
        use_multi_objective: Enable multi-objective optimization
        ranking_margin: Margin in pairwise ranking loss (default 0.0)
        ranking_weight: Weight of ranking loss vs MSE loss (0~1, default 0.8)
                        1.0 = pure ranking, 0.0 = pure MSE
    """
    device = torch.device(device)
    generator.to(device)
    evaluator.to(device)
    
    opt_g = optim.Adam(generator.parameters(), lr=lr)
    opt_e = optim.Adam(evaluator.parameters(), lr=lr * 0.1)
    
    if verbose:
        print("=== Joint Evaluator-Generator Training (v5 Advanced) ===")
        print(f"Epochs: {epochs}, Batch size: {batch_size}")
        print(f"Advanced Features:")
        print(f"  - Adaptive Sampling: {use_adaptive_sampling}")
        print(f"  - Experience Replay: {use_experience_replay}")
        print(f"  - Multi-Objective: {use_multi_objective}")
        print(f"  - Online Calibration: True")
        print(f"  - Ranking Loss: weight={ranking_weight:.2f}, margin={ranking_margin:.3f} (0=pure MSE, 1=pure ranking)")
        print()
    
    # Enhanced strategies with probabilities
    strategies = [
        ('dafner_baseline', 0.15),      # Pure baseline
        ('heuristic_noise', 0.15),      # Baseline + noise
        ('heuristic_flip', 0.10),       # Flip sign
        ('heuristic_mix', 0.15),        # Mix with random
        ('residual_connection', 0.20),  # Generator + baseline
        ('flip_and_mix', 0.15),         # Combined perturbation
        ('random_normal', 0.10),        # Pure exploration
    ]
    
    rng = np.random.default_rng()
    
    # Initialize advanced components
    adaptive_sampler = AdaptiveSampler(strategies, alpha=0.1) if use_adaptive_sampling else None
    experience_replay = ExperienceReplay(capacity=100, top_k=10) if use_experience_replay else None
    multi_obj = MultiObjectiveOptimizer(
        seg_weight=0.6, 
        span_per_cell_weight=0.3, 
        page_amp_weight=0.1,
        adaptive=True
    ) if use_multi_objective else None
    
    for epoch in range(epochs):
        evaluator.train()
        generator.train()
        
        e_losses = []
        g_losses = []
        
        batch_weights_gen = []
        batch_feats = []
        
        for scene_idx, (feats, queries, base_weights) in enumerate(dataset):
            t_feats = torch.tensor(feats, dtype=torch.float32, device=device)
            
            # Step 1: Generate weights_gen
            with torch.no_grad():
                weights_gen_tensor = generator(t_feats)
                weights_gen = weights_gen_tensor.cpu().numpy()
            
            batch_weights_gen.append(weights_gen)
            batch_feats.append(t_feats)
            
            # Step 2: Sample weights_tgt with adaptive or probabilistic strategy selection
            if adaptive_sampler:
                strategy_name, _ = adaptive_sampler.sample(rng)
            else:
                strategy_probs = [p for _, p in strategies]
                strategy_idx = rng.choice(len(strategies), p=strategy_probs)
                strategy_name, _ = strategies[strategy_idx]
            
            weights_tgt = sample_target_weights(
                base_weights, 
                strategy_name,
                rng=rng,
                generator=generator if strategy_name == 'residual_connection' else None,
                feats=feats,
                device=device
            )
            
            # Step 3: Build SFC once per weights set, then fast metric on code_map
            grid_size = 2 ** grid.K

            path_tgt = weights_to_sfc(grid, weights_tgt)
            code_map_tgt = grid.code_map_from_path(path_tgt)
            seg_tgt, span_per_cell_tgt, page_amp_tgt = compute_quality_from_codemap(
                code_map_tgt, queries, grid_size=grid_size, d=grid.d, b=256
            )

            path_gen = weights_to_sfc(grid, weights_gen)
            code_map_gen = grid.code_map_from_path(path_gen)
            seg_gen, span_per_cell_gen, page_amp_gen = compute_quality_from_codemap(
                code_map_gen, queries, grid_size=grid_size, d=grid.d, b=256
            )

            if multi_obj:
                quality_tgt = multi_obj.compute_combined_quality(seg_tgt, span_per_cell_tgt, page_amp_tgt)
                quality_gen = multi_obj.compute_combined_quality(seg_gen, span_per_cell_gen, page_amp_gen)
            else:
                quality_tgt = 0.60 * seg_tgt + 0.15 * span_per_cell_tgt + 0.25 * page_amp_tgt
                quality_gen = 0.60 * seg_gen + 0.15 * span_per_cell_gen + 0.25 * page_amp_gen
            
            # Update adaptive sampler
            if adaptive_sampler:
                reward = -quality_tgt  # Negative quality as reward
                adaptive_sampler.update(strategy_name, reward)
            
            # Add to experience replay
            if experience_replay:
                experience_replay.add(weights_tgt, feats, queries, quality_tgt)
                experience_replay.add(weights_gen, feats, queries, quality_gen)
            
            # Step 4: Train Evaluator — pairwise ranking loss + auxiliary MSE
            # Ranking loss: pred(worse) - pred(better) + margin > 0 is penalised
            # This only requires relative ordering to be correct, not absolute values
            evaluator.train()
            opt_e.zero_grad()
            
            t_weights_tgt = torch.tensor(weights_tgt, dtype=torch.float32, device=device)
            pred_tgt = evaluator(t_feats, t_weights_tgt)
            target_tgt = torch.tensor(quality_tgt, dtype=torch.float32, device=device)
            
            t_weights_gen = torch.tensor(weights_gen, dtype=torch.float32, device=device)
            pred_gen = evaluator(t_feats, t_weights_gen.detach())
            target_gen = torch.tensor(quality_gen, dtype=torch.float32, device=device)
            
            # Strong-contrast pair: sample a pure-random weight as the "bad" anchor.
            # This guarantees a large quality gap so ranking loss gets a real gradient signal.
            # base_weights is LACSFC-H (good); random weight is likely much worse.
            if ranking_weight > 0.0:
                w_rand = rng.standard_normal(base_weights.shape).astype(np.float32)
                path_rand = weights_to_sfc(grid, w_rand)
                cm_rand   = grid.code_map_from_path(path_rand)
                seg_r, span_r, pa_r = compute_quality_from_codemap(
                    cm_rand, queries, grid_size=2**grid.K, d=grid.d, b=256)
                quality_rand = (
                    0.6 * seg_r + 0.3 * span_r + 0.1 * pa_r
                    if multi_obj is None
                    else multi_obj.compute_combined_quality(seg_r, span_r, pa_r)
                )
                t_w_rand  = torch.tensor(w_rand, dtype=torch.float32, device=device)
                pred_rand = evaluator(t_feats, t_w_rand)
                target_rand = torch.tensor(quality_rand, dtype=torch.float32, device=device)
                # rank: base_weights (tgt) should score lower (better) than random
                rank_loss_strong = torch.clamp(pred_tgt - pred_rand + ranking_margin, min=0.0)
                # soft pair between tgt and gen
                if quality_tgt < quality_gen:
                    rank_loss_soft = torch.clamp(pred_tgt - pred_gen + ranking_margin, min=0.0)
                else:
                    rank_loss_soft = torch.clamp(pred_gen - pred_tgt + ranking_margin, min=0.0)
                rank_loss = 0.7 * rank_loss_strong + 0.3 * rank_loss_soft
                mse_loss  = ((pred_tgt - target_tgt) ** 2
                             + (pred_gen - target_gen) ** 2
                             + (pred_rand - target_rand) ** 2)
            else:
                # Pure MSE path (ranking_weight == 0)
                if quality_tgt < quality_gen:
                    rank_loss = torch.clamp(pred_tgt - pred_gen + ranking_margin, min=0.0)
                else:
                    rank_loss = torch.clamp(pred_gen - pred_tgt + ranking_margin, min=0.0)
                mse_loss = (pred_tgt - target_tgt) ** 2 + (pred_gen - target_gen) ** 2
            
            e_loss = ranking_weight * rank_loss + (1.0 - ranking_weight) * mse_loss
            e_loss.backward()
            opt_e.step()
            e_losses.append(float(e_loss.detach()))
            
            # Step 5: Train Generator (every batch_size scenes)
            if (scene_idx + 1) % batch_size == 0 and len(batch_weights_gen) > 0:
                weights_gen_avg = np.mean(batch_weights_gen, axis=0).astype(np.float32)
                
                generator.train()
                evaluator.eval()
                opt_g.zero_grad()
                
                weights_gen_new = generator(t_feats)
                pred_quality = evaluator(t_feats, weights_gen_new)
                
                t_weights_avg = torch.tensor(weights_gen_avg, dtype=torch.float32, device=device)
                pred_avg = evaluator(t_feats, t_weights_avg)
                
                g_loss = pred_quality + 0.5 * pred_avg
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
                opt_g.step()
                g_losses.append(float(g_loss.detach()))
                
                batch_weights_gen = []
                batch_feats = []
        
        # Advanced optimizations at epoch milestones
        if (epoch + 1) % 5 == 0:
            # 1. Online fine-tuning of Evaluator
            finetune_evaluator_online(
                evaluator=evaluator,
                generator=generator,
                dataset=dataset,
                grid=grid,
                n_samples=32,
                lr=lr * 0.01,
                device=device,
                verbose=verbose
            )
            
            # 2. Adapt sampling probabilities
            if adaptive_sampler:
                adaptive_sampler.adapt_probabilities(verbose=verbose)
            
            # 3. Adapt multi-objective weights
            if multi_obj:
                multi_obj.adapt_weights(verbose=verbose)
            
            # 4. Experience replay: Train on best samples
            if experience_replay and len(experience_replay) > 0:
                if verbose:
                    print(f"  [Experience Replay] Replaying top-{experience_replay.top_k} samples...")
                
                top_samples = experience_replay.sample_top(n=min(5, len(experience_replay.top_samples)), rng=rng)
                
                for quality, weights, feats, queries in top_samples:
                    t_feats = torch.tensor(feats, dtype=torch.float32, device=device)
                    t_weights = torch.tensor(weights, dtype=torch.float32, device=device)
                    
                    # Train Evaluator on high-quality samples
                    evaluator.train()
                    opt_e.zero_grad()
                    pred = evaluator(t_feats, t_weights)
                    target = torch.tensor(quality, dtype=torch.float32, device=device)
                    loss = (pred - target) ** 2
                    loss.backward()
                    opt_e.step()
                
                if verbose:
                    best = experience_replay.get_best()
                    if best:
                        print(f"  [Experience Replay] Best quality so far: {best[0]:.4f}")
        
        if verbose:
            avg_e_loss = np.mean(e_losses) if e_losses else 0.0
            avg_g_loss = np.mean(g_losses) if g_losses else 0.0
            print(f"  Epoch {epoch+1:3d}/{epochs}  "
                  f"E_loss = {avg_e_loss:.4f}  G_loss = {avg_g_loss:.4f}",
                  flush=True)
    
    if verbose:
        print()


def train_lacsfc_neural(
    grid: Grid,
    training_queries: List = None,
    epochs: int = 30,
    dataset_size: int = 128,
    batch_size: int = 8,
    lr: float = 1e-3,
    gen_hidden: int = 128,
    gen_layers: int = 3,
    eval_hidden: int = 128,
    eval_layers: int = 3,
    device: str = 'cpu',
    verbose: bool = True,
    train_K: int = 4,
) -> Tuple[EdgeWeightGenerator, QueryEvaluator]:
    """Train LACSFC neural network models.

    Args:
        grid: Inference grid (e.g. K=8).
        training_queries: Optional list of training queries.
        epochs: Number of training epochs.
        dataset_size: Number of scenes in dataset.
        batch_size: Batch size for training.
        lr: Learning rate.
        gen_hidden: Generator hidden dimension.
        gen_layers: Generator number of layers.
        eval_hidden: Evaluator hidden dimension.
        eval_layers: Evaluator number of layers.
        device: Device to train on ('cpu' or 'cuda').
        verbose: Whether to print training progress.
        train_K: Grid resolution used *during training* (default 4 = 16x16).
            weights_to_sfc on a K=4 grid takes ~0.003s vs ~31s for K=8,
            giving a ~10000x speedup per quality evaluation.  The Generator
            learns edge-feature -> weight mappings; since SceneEncoder.feat_dim
            is the same for all K values the trained model transfers directly
            to the inference grid.

    Returns:
        (generator, evaluator): Trained models, ready for inference on `grid`.
    """
    train_grid = Grid(d=grid.d, K=train_K)
    if verbose:
        print(f"Training on K={train_K} ({2**train_K}x{2**train_K} grid) "
              f"for speed; inference on K={grid.K} ({2**grid.K}x{2**grid.K})")
        print(f"Building dataset ({dataset_size} scenes) ...")

    t0 = time.time()
    train_encoder = SceneEncoder(train_grid)

    queries_list: List = []
    if training_queries:
        queries_list = [training_queries] * dataset_size

    dataset = SceneDataset(train_grid, train_encoder, queries_list, size=dataset_size)

    if verbose:
        print(f"Done in {time.time()-t0:.1f}s\n")

    generator = EdgeWeightGenerator(
        input_dim=train_encoder.feat_dim,
        hidden_dim=gen_hidden,
        num_layers=gen_layers,
    )
    evaluator = QueryEvaluator(
        input_dim=train_encoder.feat_dim,
        hidden_dim=eval_hidden,
        num_layers=eval_layers,
    )

    joint_train(
        generator=generator,
        evaluator=evaluator,
        dataset=dataset,
        grid=train_grid,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        verbose=verbose,
    )
    
    return generator, evaluator


def optimize_weights_es(
    grid: Grid,
    workload,
    init_weights: np.ndarray = None,
    epochs: int = 30,
    n_perturbations: int = 32,
    sigma: float = 0.1,
    lr: float = 0.3,
    verbose: bool = True,
    points: np.ndarray = None,
    b: int = 256,
) -> np.ndarray:
    """Directly optimise SFC edge weights for a single workload using ES.

    This is an online (per-inference) approach: no Generator is trained.
    Starting from heuristic weights, ES perturbs the weight vector and
    uses true weights_to_sfc quality to estimate gradient direction.

    When ``points`` is provided, quality is measured with the real data
    distribution (consistent with benchmark evaluation).  Otherwise falls
    back to the uniform-distribution proxy (mean_seg only).

    Args:
        grid: The inference grid.
        workload: QueryWorkload (or object with .queries and .dual_edge_weights()).
        init_weights: Starting weights. If None, uses heuristic dual_edge_weights.
        epochs: Number of ES iterations.
        n_perturbations: Antithetic perturbation pairs per epoch.
        sigma: Perturbation scale.
        lr: Step size (gradient descent learning rate).
        verbose: Print progress.
        points: Optional real data points (n, d) for accurate page_amp evaluation.
        b: Page size for page_amp computation (should match benchmark b).

    Returns:
        Optimised weight vector.
    """
    from lacsfcf.metrics.range_query import mean_seg as lacsfcf_mean_seg

    queries = workload.queries if hasattr(workload, 'queries') else workload
    grid_size = 2 ** grid.K

    if init_weights is None:
        from lacsfcf.data.query_workload import QueryWorkload as _QWL
        wl_obj = workload if hasattr(workload, 'dual_edge_weights') else _QWL(queries, grid)
        w = wl_obj.dual_edge_weights().astype(np.float32)
    else:
        w = init_weights.astype(np.float32)

    n_edges = len(w)
    rng = np.random.default_rng(0)

    # Pre-quantise real data points to grid cells (fixed, code-map-independent)
    if points is not None:
        pts_norm = np.clip(np.asarray(points, dtype=np.float64), 0.0, 1.0 - 1e-9)
        pts_cell = np.clip((pts_norm * grid_size).astype(np.int32), 0, grid_size - 1)
        # Pre-compute query cell bounds for fast reuse
        gs_m1 = grid_size - 1
        q_los = [tuple(max(0, min(int(c), gs_m1)) for c in q.lo) for q in queries]
        q_his = [tuple(max(0, min(int(c), gs_m1)) for c in q.hi) for q in queries]

        def quality(w_np: np.ndarray) -> float:
            path = weights_to_sfc(grid, w_np.astype(np.float32))
            cm = grid.code_map_from_path(path)

            # Assign each real data point its SFC code, then sort (= disk order)
            pt_codes = cm[tuple(pts_cell[:, i] for i in range(grid.d))]
            order = np.argsort(pt_codes, kind='mergesort')
            pt_codes_sorted = pt_codes[order]  # physical disk layout

            # page_start_codes: first code of every page of b records
            n_pts = len(pt_codes_sorted)
            page_start_codes = pt_codes_sorted[::b]  # shape (n_pages,)

            seg_val = float(lacsfcf_mean_seg(cm, queries))

            # Compute page_amp with the exact BoundaryIndex logic
            bq = LACSFCBoundaryQuery(code_map=cm, grid_size=grid_size, d=grid.d)
            total_pa, n_q = 0.0, 0
            for qi, q in enumerate(queries):
                intervals = bq.get_intervals(q_los[qi], q_his[qi])
                if not intervals:
                    continue
                # result_count: points whose code falls in any interval
                result_count = 0
                pages_set = set()
                for s, e in intervals:
                    mask = (pt_codes_sorted >= s) & (pt_codes_sorted <= e)
                    result_count += int(mask.sum())
                    # BoundaryIndex: pages whose start_code <= e and
                    # the previous page's start_code <= s
                    i_start = int(np.searchsorted(page_start_codes, s, side='left')) - 1
                    i_start = max(0, i_start)
                    i_end = int(np.searchsorted(page_start_codes, e, side='right')) - 1
                    if i_start <= i_end:
                        pages_set.update(range(i_start, i_end + 1))
                if result_count == 0:
                    continue
                pages_accessed = len(pages_set)
                min_pages = max(1, (result_count + b - 1) // b)
                total_pa += pages_accessed / min_pages
                n_q += 1

            page_amp = total_pa / n_q if n_q > 0 else 1.0
            # Weight: seg and page_amp have different scales; normalise roughly
            return seg_val + page_amp

    else:
        # Fallback: use mean_seg only (stable proxy, no real distribution needed)
        def quality(w_np: np.ndarray) -> float:
            path = weights_to_sfc(grid, w_np.astype(np.float32))
            cm = grid.code_map_from_path(path)
            return float(lacsfcf_mean_seg(cm, queries))

    q_init = quality(w)
    if verbose:
        print(f"[ES-weights] n_edges={n_edges}, init_Q={q_init:.4f}, b={b}, "
              f"epochs={epochs}, n_pert={n_perturbations}, sigma={sigma}, "
              f"real_pts={'yes' if points is not None else 'no'}")

    for epoch in range(epochs):
        grad = np.zeros(n_edges, dtype=np.float32)
        for _ in range(n_perturbations // 2):
            eps = rng.standard_normal(n_edges).astype(np.float32)
            q_plus  = quality(w + sigma * eps)
            q_minus = quality(w - sigma * eps)
            grad += (q_plus - q_minus) * eps
        grad /= (n_perturbations // 2) * sigma
        w -= lr * grad

        if verbose and (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}  Q={quality(w):.4f}  "
                  f"|grad|={np.linalg.norm(grad):.4f}")

    q_final = quality(w)
    if verbose:
        print(f"[ES-weights] final Q={q_final:.4f}  "
              f"improvement={((q_init-q_final)/max(q_init,1e-6))*100:+.1f}%")
    return w


def optimize_weights_cmaes(
    grid: Grid,
    workload,
    points: np.ndarray,
    b: int = 256,
    init_weights: np.ndarray = None,
    sigma0: float = 0.5,
    max_evals: int = 5000,
    popsize: int = None,
    verbose: bool = True,
    query_windows=None,
    quantizer=None,
    reg_lambda: float = 0.0,
    mode: str = "block_space",
    coarsen_factor: int = 2,
    func_space_M: int = 200,
    cover_weight: float = 0.0,
    runtime_surrogate: str = "none",
    runtime_weight: float = 0.0,
    runtime_normalize: str = "raw",
    runtime_interval_weight: float = 0.0,
    objective_mode: str = "page_amp",
    learn_obj_weights: bool = True,
    init_obj_weights: tuple = (0.60, 0.15, 0.25),
    init_strategy: str = "heuristic",
    init_seed: int = 0,
) -> np.ndarray:
    """Optimise SFC edge weights with CMA-ES using exact benchmark page_amp.

    CMA-ES adapts a full covariance matrix which makes it far more effective
    than vanilla ES for correlated, noisy objectives.  The objective is the
    *exact* page_amp as measured by the benchmark (BoundaryIndex on real
    physical pages built from sorted SFC codes).

    Key improvements over optimize_weights_es:
      - CMA-ES self-adapts step size and search directions automatically.
      - Objective uses precise BoundaryIndex paging (matches benchmark).
      - Warm-starts from LACSFC-H heuristic weights.

    Args:
        grid: Inference grid.
        workload: QueryWorkload with .queries and .dual_edge_weights().
        points: Real data points (n, d) in [0,1]^d.
        b: Page size (records per page), must match benchmark.
        init_weights: Starting weights. If None, uses heuristic.
        sigma0: Initial CMA-ES step size.
        max_evals: Maximum number of objective evaluations.
        popsize: CMA-ES population size. If None, uses CMA default.
        verbose: Print progress.
        query_windows: Optional list of QueryWindow objects (benchmark queries).
            If provided, used instead of workload.queries for objective evaluation.
            These should be the *same* queries passed to the benchmark engine.
        quantizer: Quantizer used to map QueryWindow -> cell bounds. Required
            when query_windows is provided.
        mode: Optimisation search space:
            - "block_space": Optimise per-block density values (n_blocks dims,
              sep-CMA). Edge w(b1,b2) = exp(bv[b1]) + exp(bv[b2]).  Faster
              convergence for high-d grids (recommended for d>=4).
            - "hierarchical": Two-phase METIS-style coarse-to-fine CMA-ES.
              Phase 1 groups blocks into super-blocks (n_blocks/coarsen_factor^d
              dims) and searches the coarse space; Phase 2 hot-starts the full
              block_space CMA-ES from the interpolated coarse solution.  Reduces
              search space by coarsen_factor^d (e.g. 32x for d=5, r=2).  Best
              for d>=5 where block_space alone fails to converge.
            - "func_space": RBF function-space parameterisation.  Optimises
              theta ∈ R^M where bv = Phi @ theta and Phi is a fixed RBF
              feature matrix (M Gaussian kernels on a uniform grid,
              sigma = 2/block_side).  Compresses the search space from
              n_blocks dims to M dims while keeping the objective exact.
              Recommended for d>=4 (best results with M=200).
            - "edge_space": Optimise edge weights directly (n_edges dims).
              Original behaviour.
        coarsen_factor: Spatial coarsening factor per dimension for
            "hierarchical" mode (default 2).  n_superblocks = n_blocks / r^d.
        func_space_M: Number of RBF kernel centres for "func_space" mode
            (default 200).  Larger M = finer density representation.

    Returns:
        Optimised weight vector (np.float32).
    """
    try:
        import cma
    except ImportError:
        raise ImportError("CMA-ES requires the 'cma' package: pip install cma")

    lacs_queries = workload.queries if hasattr(workload, 'queries') else workload
    grid_size = 2 ** grid.K

    if init_weights is None:
        from lacsfcf.data.query_workload import QueryWorkload as _QWL
        wl_obj = workload if hasattr(workload, 'dual_edge_weights') else _QWL(lacs_queries, grid)
        w0 = wl_obj.dual_edge_weights().astype(np.float64)
    else:
        w0 = init_weights.astype(np.float64)

    n_edges = len(w0)

    # Pre-quantise points to grid cells (fixed, code-map-independent)
    pts_np = np.asarray(points, dtype=np.float64)
    pts_norm = np.clip(pts_np, 0.0, 1.0 - 1e-9)
    pts_cell = np.clip((pts_norm * grid_size).astype(np.int32), 0, grid_size - 1)

    # Pre-compute per-query hit masks using EXACT point-in-window test.
    # These are fixed (do not depend on SFC weights), so we compute once.
    gs_m1 = grid_size - 1
    if query_windows is not None and quantizer is not None:
        qw_list = list(query_windows)
    else:
        # Build QueryWindow proxies from lacsfcf queries (coarse cell-to-float)
        cell_width = 1.0 / grid_size
        qw_list = []
        for q in lacs_queries:
            lo = tuple(c * cell_width for c in q.lo)
            hi = tuple((c + 1) * cell_width for c in q.hi)
            from src.mvp.types import QueryWindow as _QW
            qw_list.append(_QW(low=lo, high=hi))

    # Exact point-in-window: lo <= x < hi (same semantics as QueryEngine)
    hit_masks = []
    hit_counts = []
    for qw in qw_list:
        lo_arr = np.asarray(qw.low, dtype=np.float64)
        hi_arr = np.asarray(qw.high, dtype=np.float64)
        lo_c = np.minimum(lo_arr, hi_arr)
        hi_c = np.maximum(lo_arr, hi_arr)
        mask = np.ones(len(pts_np), dtype=bool)
        for dim in range(grid.d):
            mask &= (pts_np[:, dim] >= lo_c[dim]) & (pts_np[:, dim] <= hi_c[dim])
        hit_masks.append(mask)
        hit_counts.append(int(mask.sum()))

    # Pre-compute query cell bounds for LACSFCBoundaryQuery (from quantizer or heuristic)
    if query_windows is not None and quantizer is not None:
        qq_list = [quantizer.quantize_window(qw) for qw in qw_list]
        q_los = [tuple(int(c) for c in qq.low_cell)  for qq in qq_list]
        q_his = [tuple(int(c) for c in qq.high_cell) for qq in qq_list]
    else:
        q_los = [tuple(max(0, min(int(c), gs_m1)) for c in q.lo) for q in lacs_queries]
        q_his = [tuple(max(0, min(int(c), gs_m1)) for c in q.hi) for q in lacs_queries]

    n_queries = len(q_los)
    eval_count = [0]

    # L2 正则化的基准权重（init 点，防止离线场景过拟合）
    w0_reg = w0.copy() if reg_lambda > 0.0 else None

    # Optional multi-objective (seg + span + page_amp) path.
    # Default remains page_amp-only to preserve historical behavior.
    multi_obj = None
    if objective_mode == "weighted_mvp":
        multi_obj = MultiObjectiveOptimizer(
            seg_weight=float(init_obj_weights[0]),
            span_per_cell_weight=float(init_obj_weights[1]),
            page_amp_weight=float(init_obj_weights[2]),
            adaptive=bool(learn_obj_weights),
        )

    runtime_surrogate = str(runtime_surrogate)
    runtime_weight = float(runtime_weight)
    runtime_interval_weight = float(runtime_interval_weight)
    if runtime_surrogate not in {"none", "empty_slice", "page_interval", "candidate_slice", "candidate_record"}:
        raise ValueError(f"Unknown runtime_surrogate={runtime_surrogate!r}")
    runtime_normalize = str(runtime_normalize)
    if runtime_normalize not in {"raw", "relative"}:
        raise ValueError(f"Unknown runtime_normalize={runtime_normalize!r}")

    def objective_components(w_arr: np.ndarray) -> tuple:
        w_f32 = np.asarray(w_arr, dtype=np.float32)
        path = weights_to_sfc(grid, w_f32)
        cm = np.zeros(grid.grid_shape, dtype=np.int64)
        for code, cell in enumerate(path):
            cm[tuple(cell)] = int(code)

        # Physical disk layout: sort real point codes
        pt_codes = cm[tuple(pts_cell[:, i] for i in range(grid.d))]
        pt_codes_sorted = np.sort(pt_codes)
        page_start_codes = pt_codes_sorted[::b]

        bq = LACSFCBoundaryQuery(code_map=cm, grid_size=grid_size, d=grid.d)
        total_pa, n_q = 0.0, 0
        total_cover_norm = 0.0  # sum over queries of (cover_len / min_p)
        total_runtime_norm = 0.0
        total_interval_norm = 0.0
        for qi in range(n_queries):
            result_count = hit_counts[qi]   # exact, SFC-independent
            if result_count == 0:
                continue
            intervals = bq.get_intervals(q_los[qi], q_his[qi])
            if not intervals:
                continue
            pages_set = set()
            cover_len = 0
            for s, e in intervals:
                cover_len += int(e) - int(s) + 1
                i_st = max(0, int(np.searchsorted(page_start_codes, s, side='left')) - 1)
                i_end = int(np.searchsorted(page_start_codes, e, side='right')) - 1
                if i_st <= i_end:
                    pages_set.update(range(i_st, i_end + 1))
            min_p = max(1, (result_count + b - 1) // b)
            total_pa += len(pages_set) / min_p
            # Normalise cover_len by b so it is on the same scale as pa.
            total_cover_norm += cover_len / (min_p * b)
            total_interval_norm += len(intervals)
            if runtime_weight != 0.0 and runtime_surrogate != "none":
                candidate_slice_count = 0
                candidate_record_count = 0
                empty_slice_count = 0
                for pid in pages_set:
                    p0 = int(pid) * int(b)
                    p1 = min(len(pt_codes_sorted), p0 + int(b))
                    page_codes = pt_codes_sorted[p0:p1]
                    if len(page_codes) == 0:
                        continue
                    for s, e in intervals:
                        candidate_slice_count += 1
                        start_idx = int(np.searchsorted(page_codes, s, side="left"))
                        end_idx = int(np.searchsorted(page_codes, e, side="right"))
                        candidate_record_count += end_idx - start_idx
                        if start_idx == end_idx:
                            empty_slice_count += 1
                if runtime_surrogate == "empty_slice":
                    runtime_value = empty_slice_count
                elif runtime_surrogate == "candidate_record":
                    runtime_value = candidate_record_count
                else:
                    runtime_value = candidate_slice_count
                total_runtime_norm += runtime_value / min_p
            n_q += 1

        pa = total_pa / n_q if n_q > 0 else 1.0
        cover = total_cover_norm / n_q if n_q > 0 else 0.0
        runtime_penalty = total_runtime_norm / n_q if n_q > 0 else 0.0
        interval_penalty = total_interval_norm / n_q if n_q > 0 else 0.0
        return pa, cover, runtime_penalty, interval_penalty, cm

    pa_ref, cover_ref, runtime_ref, interval_ref, _ = objective_components(w0)
    pa_ref = max(float(pa_ref), 1e-9)
    cover_ref = max(float(cover_ref), 1e-9)
    runtime_ref = max(float(runtime_ref), 1e-9)
    interval_ref = max(float(interval_ref), 1e-9)

    def objective(w_arr: np.ndarray) -> float:
        eval_count[0] += 1
        pa, cover, runtime_penalty, interval_penalty, cm = objective_components(w_arr)
        if runtime_normalize == "relative":
            loss = pa / pa_ref
            loss += cover_weight * (cover / cover_ref)
            loss += runtime_interval_weight * (interval_penalty / interval_ref)
            if runtime_weight != 0.0 and runtime_surrogate != "none":
                loss += runtime_weight * (runtime_penalty / runtime_ref)
        else:
            loss = pa + cover_weight * cover
            loss += runtime_interval_weight * interval_penalty
            loss += runtime_weight * runtime_penalty

        # weighted_mvp: add seg + span_per_cell on top of (pa + cover_weight*cover).
        if multi_obj is not None:
            seg_val, span_per_cell_val, _ = compute_quality_from_codemap(
                code_map=cm,
                queries=lacs_queries,
                grid_size=grid_size,
                d=grid.d,
                b=b,
            )
            loss = multi_obj.compute_combined_quality(
                mean_seg=seg_val,
                mean_span_per_cell=span_per_cell_val,
                mean_page_amp=loss,  # feed (pa + cover_weight*cover) as the pa slot
            )
            if learn_obj_weights and eval_count[0] % 10 == 0:
                multi_obj.adapt_weights(verbose=False)

        # L2 正则项：惩罚权重偏离初始点，防止离线场景过度特化
        if reg_lambda > 0.0 and w0_reg is not None:
            w_arr_np = np.asarray(w_arr, dtype=np.float64)
            reg = reg_lambda * float(np.mean((w_arr_np - w0_reg) ** 2))
            return loss + reg

        return loss

    q_init = objective(w0)
    if verbose:
        print(f"[CMA-ES] mode={mode}, n_edges={n_edges}, b={b}, n_queries={n_queries}, "
              f"init_objective={q_init:.4f}, sigma0={sigma0}, max_evals={max_evals}")

    # ------------------------------------------------------------------
    # block_space mode: optimise per-block log-density values (n_blocks
    # dims) instead of raw edge weights (n_edges dims).  For high-d grids
    # (d>=4) this reduces the search space by ~3-4x and converges faster.
    # edge_weight(b1,b2) = exp(bv[b1]) + exp(bv[b2]), then min-max norm.
    # ------------------------------------------------------------------
    if mode == "block_space":
        from lacsfcf.data.query_workload import QueryWorkload as _QWL
        wl_obj = workload if hasattr(workload, 'dual_edge_weights') else _QWL(lacs_queries, grid)
        bd = wl_obj.block_density  # shape = grid.block_shape

        block_shape = grid.block_shape
        n_blocks = grid.num_blocks
        bd_flat = bd.flatten().astype(np.float64)
        bv0 = np.log(bd_flat + 1e-3)  # log-space initial block values (heuristic warm-start)
        # P1 warm-start ablation: override bv0 for causal test of whether the
        # LACSFC-H heuristic is a floor or a ceiling for LACSFC-Neural.
        if init_strategy == "random":
            _rng_init = np.random.default_rng(int(init_seed))
            # Match the scale of log(density) roughly: std ~ 2 covers typical range.
            bv0 = _rng_init.normal(loc=float(bv0.mean()), scale=2.0, size=n_blocks)
        elif init_strategy == "uniform":
            bv0 = np.zeros(n_blocks, dtype=np.float64)
        elif init_strategy != "heuristic":
            raise ValueError(f"Unknown init_strategy={init_strategy!r}")

        # Pre-compute edge → block index mapping
        edge_b1_idx = np.array(
            [int(np.ravel_multi_index(b1, block_shape)) for b1, b2, _ in grid.dual_edges],
            dtype=np.int32)
        edge_b2_idx = np.array(
            [int(np.ravel_multi_index(b2, block_shape)) for b1, b2, _ in grid.dual_edges],
            dtype=np.int32)

        def _bv_to_weights(bv: np.ndarray) -> np.ndarray:
            ev = np.exp(np.asarray(bv, dtype=np.float32))
            w = ev[edge_b1_idx] + ev[edge_b2_idx]
            mn, mx = w.min(), w.max()
            if mx > mn:
                w = (w - mn) / (mx - mn)
            return w

        def objective_block(bv: np.ndarray) -> float:
            return objective(_bv_to_weights(bv))

        q_block_init = objective_block(bv0)
        if verbose:
            print(f"[CMA-ES] block_space n_blocks={n_blocks}, "
                  f"block_init_objective={q_block_init:.4f}")

        opts_b = cma.CMAOptions()
        opts_b['maxfevals'] = max_evals
        opts_b['verbose'] = -9
        opts_b['tolfun'] = 1e-5
        opts_b['tolx'] = 1e-6
        opts_b['CMA_diagonal'] = True   # sep-CMA for high-dim efficiency
        _pop = popsize if popsize is not None else max(20, int(4 + 3 * np.log(n_blocks)))
        opts_b['popsize'] = _pop

        es_b = cma.CMAEvolutionStrategy(bv0.tolist(), sigma0, opts_b)

        best_bv = bv0.copy()
        best_q_b = q_block_init
        report_every_b = max(1, max_evals // (10 * _pop))

        while not es_b.stop():
            solutions_b = es_b.ask()
            fitnesses_b = [objective_block(x) for x in solutions_b]
            es_b.tell(solutions_b, fitnesses_b)

            gen_best_b = min(fitnesses_b)
            if gen_best_b < best_q_b:
                best_q_b = gen_best_b
                best_bv = np.array(solutions_b[fitnesses_b.index(gen_best_b)], dtype=np.float64)

            if verbose and eval_count[0] % report_every_b < _pop:
                print(f"  evals={eval_count[0]:4d}  best_objective={best_q_b:.4f}  "
                      f"sigma={es_b.sigma:.4f}")

        if verbose:
            print(f"[CMA-ES] done: evals={eval_count[0]}, final_objective={best_q_b:.4f}, "
                  f"improvement={((q_block_init-best_q_b)/max(q_block_init,1e-6))*100:+.1f}%")

        return _bv_to_weights(best_bv).astype(np.float32)

    # ------------------------------------------------------------------
    # hierarchical mode: two-phase coarse-to-fine CMA-ES.
    # Phase 1: optimise n_superblocks-dim bv_coarse (fast, low-dim).
    # Phase 2: hot-start n_blocks-dim bv_fine from interpolated coarse solution.
    # ------------------------------------------------------------------
    if mode == "hierarchical":
        from lacsfcf.data.query_workload import QueryWorkload as _QWL
        wl_obj = workload if hasattr(workload, 'dual_edge_weights') else _QWL(lacs_queries, grid)
        bd = wl_obj.block_density

        block_shape = grid.block_shape
        n_blocks = grid.num_blocks
        bd_flat = bd.flatten().astype(np.float64)
        bv0 = np.log(bd_flat + 1e-3)
        if init_strategy == "random":
            _rng_init = np.random.default_rng(int(init_seed))
            bv0 = _rng_init.normal(loc=float(bv0.mean()), scale=2.0, size=n_blocks)
        elif init_strategy == "uniform":
            bv0 = np.zeros(n_blocks, dtype=np.float64)
        elif init_strategy != "heuristic":
            raise ValueError(f"Unknown init_strategy={init_strategy!r}")

        edge_b1_idx = np.array(
            [int(np.ravel_multi_index(b1, block_shape)) for b1, b2, _ in grid.dual_edges],
            dtype=np.int32)
        edge_b2_idx = np.array(
            [int(np.ravel_multi_index(b2, block_shape)) for b1, b2, _ in grid.dual_edges],
            dtype=np.int32)

        def _bv_to_weights_h(bv: np.ndarray) -> np.ndarray:
            ev = np.exp(np.asarray(bv, dtype=np.float32))
            w = ev[edge_b1_idx] + ev[edge_b2_idx]
            mn, mx = w.min(), w.max()
            if mx > mn:
                w = (w - mn) / (mx - mn)
            return w

        def objective_block_h(bv: np.ndarray) -> float:
            return objective(_bv_to_weights_h(bv))

        # ── Build coarse-to-fine mapping ─────────────────────────────────
        r = coarsen_factor
        super_block_shape = tuple(max(1, s // r) for s in block_shape)
        n_superblocks = int(np.prod(super_block_shape))

        fine_indices = np.arange(n_blocks, dtype=np.int64)
        fine_coords = np.stack(np.unravel_index(fine_indices, block_shape), axis=1)  # (n_blocks, d)
        super_coords = np.clip(fine_coords // r, 0,
                               np.array(super_block_shape, dtype=np.int64) - 1)
        mapping = np.ravel_multi_index(super_coords.T, super_block_shape).astype(np.int64)

        # Coarse bv0: mean of fine bv0 values per superblock
        bv0_coarse = np.zeros(n_superblocks, dtype=np.float64)
        np.add.at(bv0_coarse, mapping, bv0)
        counts = np.bincount(mapping, minlength=n_superblocks).astype(np.float64)
        bv0_coarse /= np.maximum(counts, 1.0)

        reduction = n_blocks // max(1, n_superblocks)
        if verbose:
            print(f"[CMA-ES] hierarchical: n_blocks={n_blocks} → n_superblocks={n_superblocks} "
                  f"(r={r}, {reduction}x reduction)")

        evals_c = max(10, min(int(max_evals * 0.35), max_evals - 50))
        evals_f = max_evals - evals_c

        # ── Phase 1: Coarse CMA-ES ───────────────────────────────────────
        def objective_coarse(bv_c: np.ndarray) -> float:
            return objective_block_h(bv_c[mapping])

        q_coarse_init = objective_coarse(bv0_coarse)
        pop_c = popsize if popsize is not None else max(10, int(4 + 3 * np.log(n_superblocks)))
        if verbose:
            print(f"[CMA-ES] Phase 1 (coarse): dims={n_superblocks}, evals={evals_c}, "
                  f"popsize={pop_c}, init_pa={q_coarse_init:.4f}")

        opts_c = cma.CMAOptions()
        opts_c['maxfevals'] = evals_c
        opts_c['verbose'] = -9
        opts_c['tolfun'] = 1e-5
        opts_c['tolx'] = 1e-6
        opts_c['CMA_diagonal'] = True
        opts_c['popsize'] = pop_c

        es_c = cma.CMAEvolutionStrategy(bv0_coarse.tolist(), sigma0, opts_c)
        best_bv_c = bv0_coarse.copy()
        best_q_c = q_coarse_init

        while not es_c.stop():
            sols_c = es_c.ask()
            fits_c = [objective_coarse(x) for x in sols_c]
            es_c.tell(sols_c, fits_c)
            gen_best_c = min(fits_c)
            if gen_best_c < best_q_c:
                best_q_c = gen_best_c
                best_bv_c = np.array(sols_c[fits_c.index(gen_best_c)], dtype=np.float64)
            if verbose and eval_count[0] % max(pop_c, evals_c // 5) < pop_c:
                print(f"  [P1] evals={eval_count[0]:4d}  best_pa={best_q_c:.4f}")

        if verbose:
            print(f"[CMA-ES] Phase 1 done: evals={eval_count[0]}, pa={best_q_c:.4f}, "
                  f"vs_init={((q_coarse_init - best_q_c) / max(q_coarse_init, 1e-6)) * 100:+.1f}%")

        # ── Phase 2: Fine CMA-ES (hot-start from coarse solution) ────────
        bv_f_init = best_bv_c[mapping]          # constant interpolation → (n_blocks,)
        sigma_refine = max(0.05, sigma0 * 0.2)  # tighter step for local refinement

        q_fine_init = objective_block_h(bv_f_init)
        pop_f = popsize if popsize is not None else max(20, int(4 + 3 * np.log(n_blocks)))
        if verbose:
            print(f"[CMA-ES] Phase 2 (fine): dims={n_blocks}, evals={evals_f}, "
                  f"popsize={pop_f}, sigma={sigma_refine:.3f}, init_pa={q_fine_init:.4f}")

        opts_f = cma.CMAOptions()
        opts_f['maxfevals'] = evals_f
        opts_f['verbose'] = -9
        opts_f['tolfun'] = 1e-5
        opts_f['tolx'] = 1e-6
        opts_f['CMA_diagonal'] = True
        opts_f['popsize'] = pop_f

        es_f = cma.CMAEvolutionStrategy(bv_f_init.tolist(), sigma_refine, opts_f)
        best_bv_f = bv_f_init.copy()
        best_q_f = q_fine_init
        report_every_f = max(1, evals_f // (5 * pop_f))

        while not es_f.stop():
            sols_f = es_f.ask()
            fits_f = [objective_block_h(x) for x in sols_f]
            es_f.tell(sols_f, fits_f)
            gen_best_f = min(fits_f)
            if gen_best_f < best_q_f:
                best_q_f = gen_best_f
                best_bv_f = np.array(sols_f[fits_f.index(gen_best_f)], dtype=np.float64)
            if verbose and eval_count[0] % max(pop_f, report_every_f * pop_f) < pop_f:
                print(f"  [P2] evals={eval_count[0]:4d}  best_pa={best_q_f:.4f}  "
                      f"sigma={es_f.sigma:.4f}")

        if verbose:
            print(f"[CMA-ES] done: total_evals={eval_count[0]}, final_pa={best_q_f:.4f}, "
                  f"overall_improvement={((q_init - best_q_f) / max(q_init, 1e-6)) * 100:+.1f}%")

        return _bv_to_weights_h(best_bv_f).astype(np.float32)

    # ------------------------------------------------------------------
    # func_space mode: RBF function-space parameterisation.
    # bv = Phi @ theta, optimise theta ∈ R^M (M << n_blocks).
    # ------------------------------------------------------------------
    if mode == "func_space":
        from lacsfcf.data.query_workload import QueryWorkload as _QWL
        wl_obj = workload if hasattr(workload, 'dual_edge_weights') else _QWL(lacs_queries, grid)
        bd = wl_obj.block_density

        block_shape = grid.block_shape
        n_blocks = grid.num_blocks
        bd_flat = bd.flatten().astype(np.float64)
        bv0_fs = np.log(bd_flat + 1e-3)
        if init_strategy == "random":
            _rng_init = np.random.default_rng(int(init_seed))
            bv0_fs = _rng_init.normal(loc=float(bv0_fs.mean()), scale=2.0, size=n_blocks)
        elif init_strategy == "uniform":
            bv0_fs = np.zeros(n_blocks, dtype=np.float64)
        elif init_strategy != "heuristic":
            raise ValueError(f"Unknown init_strategy={init_strategy!r}")

        edge_b1_idx_fs = np.array(
            [int(np.ravel_multi_index(b1, block_shape)) for b1, b2, _ in grid.dual_edges],
            dtype=np.int32)
        edge_b2_idx_fs = np.array(
            [int(np.ravel_multi_index(b2, block_shape)) for b1, b2, _ in grid.dual_edges],
            dtype=np.int32)

        def _bv_to_weights_fs(bv: np.ndarray) -> np.ndarray:
            ev = np.exp(np.asarray(bv, dtype=np.float32))
            w = ev[edge_b1_idx_fs] + ev[edge_b2_idx_fs]
            mn, mx = w.min(), w.max()
            if mx > mn:
                w = (w - mn) / (mx - mn)
            return w

        # ── Build RBF feature matrix Phi ─────────────────────────────────
        M_fs = func_space_M
        block_shape_arr = np.array(block_shape, dtype=np.float64)
        fine_idx = np.arange(n_blocks, dtype=np.int64)
        fine_coords = np.stack(np.unravel_index(fine_idx, block_shape), axis=1).astype(np.float64)
        X_fs = fine_coords / block_shape_arr               # (n_blocks, d) in [0,1)

        _rng_rbf = np.random.default_rng(42)
        ctr_idx = _rng_rbf.choice(n_blocks, size=min(M_fs, n_blocks), replace=False)
        centers_fs = X_fs[ctr_idx]                         # (M, d)
        sigma_fs = 2.0 / float(block_shape_arr.min())
        diff_fs = X_fs[:, np.newaxis, :] - centers_fs[np.newaxis, :, :]  # (n_blocks, M, d)
        dist2_fs = np.sum(diff_fs ** 2, axis=-1)           # (n_blocks, M)
        Phi_fs = np.exp(-dist2_fs / (2.0 * sigma_fs ** 2))               # (n_blocks, M)

        # θ initialisation: LS fit bv0 ≈ Phi @ theta0
        theta0_fs, _, _, _ = np.linalg.lstsq(Phi_fs, bv0_fs, rcond=None)

        def objective_fs(theta: np.ndarray) -> float:
            bv = Phi_fs @ theta
            return objective(_bv_to_weights_fs(bv))

        q_fs_init = objective_fs(theta0_fs)
        if verbose:
            print(f"[CMA-ES] func_space: n_blocks={n_blocks}, M={M_fs}, "
                  f"sigma={sigma_fs:.3f}, init_page_amp={q_fs_init:.4f}")

        _pop_fs = popsize if popsize is not None else max(20, int(4 + 3 * np.log(M_fs)))
        opts_fs = cma.CMAOptions()
        opts_fs['maxfevals'] = max_evals
        opts_fs['verbose'] = -9
        opts_fs['tolfun'] = 1e-5
        opts_fs['tolx'] = 1e-6
        opts_fs['popsize'] = _pop_fs

        es_fs = cma.CMAEvolutionStrategy(theta0_fs.tolist(), sigma0, opts_fs)
        best_theta_fs = theta0_fs.copy()
        best_q_fs = q_fs_init
        report_every_fs = max(1, max_evals // (10 * _pop_fs))

        while not es_fs.stop():
            sols_fs = es_fs.ask()
            fits_fs = [objective_fs(x) for x in sols_fs]
            es_fs.tell(sols_fs, fits_fs)
            gen_best_fs = min(fits_fs)
            if gen_best_fs < best_q_fs:
                best_q_fs = gen_best_fs
                best_theta_fs = np.array(
                    sols_fs[fits_fs.index(gen_best_fs)], dtype=np.float64)
            if verbose and eval_count[0] % report_every_fs < _pop_fs:
                print(f"  evals={eval_count[0]:4d}  best_page_amp={best_q_fs:.4f}  "
                      f"sigma={es_fs.sigma:.4f}")

        if verbose:
            print(f"[CMA-ES] done: evals={eval_count[0]}, final_page_amp={best_q_fs:.4f}, "
                  f"improvement={((q_fs_init - best_q_fs) / max(q_fs_init, 1e-6)) * 100:+.1f}%")

        best_bv_fs = Phi_fs @ best_theta_fs
        return _bv_to_weights_fs(best_bv_fs).astype(np.float32)

    # ------------------------------------------------------------------
    # edge_space mode (original): optimise edge weights directly.
    # ------------------------------------------------------------------
    opts = cma.CMAOptions()
    opts['maxfevals'] = max_evals
    opts['verbose'] = -9 if not verbose else -3  # suppress per-iter output; we print manually
    opts['tolfun'] = 1e-4
    opts['tolx'] = 1e-5
    if popsize is not None:
        opts['popsize'] = popsize

    es = cma.CMAEvolutionStrategy(w0.tolist(), sigma0, opts)

    best_w = w0.copy()
    best_q = q_init
    report_every = max(1, max_evals // (10 * (popsize or 10)))

    while not es.stop():
        solutions = es.ask()
        fitnesses = [objective(x) for x in solutions]
        es.tell(solutions, fitnesses)

        gen_best = min(fitnesses)
        if gen_best < best_q:
            best_q = gen_best
            best_w = np.array(solutions[fitnesses.index(gen_best)], dtype=np.float64)

        if verbose and eval_count[0] % report_every < (popsize or 10):
            print(f"  evals={eval_count[0]:4d}  best_page_amp={best_q:.4f}  "
                  f"sigma={es.sigma:.4f}")

    if verbose:
        print(f"[CMA-ES] done: evals={eval_count[0]}, final_page_amp={best_q:.4f}, "
              f"improvement={((q_init-best_q)/max(q_init,1e-6))*100:+.1f}%")

    return best_w.astype(np.float32)


def train_lacsfc_es(
    grid: Grid,
    training_queries: List = None,
    epochs: int = 50,
    dataset_size: int = 64,
    batch_size: int = 8,
    lr: float = 1e-3,
    sigma: float = 0.1,
    n_perturbations: int = 16,
    gen_hidden: int = 128,
    gen_layers: int = 3,
    device: str = 'cpu',
    verbose: bool = True,
    train_K: int = 4,
    learn_obj_weights: bool = True,
    init_obj_weights: tuple = (0.60, 0.15, 0.25),
) -> tuple:
    """Train Generator with Evolution Strategy (ES) using true quality feedback.

    Instead of a differentiable Evaluator proxy (which fails due to the
    discontinuous weights_to_sfc step), this function directly uses true
    quality metrics to compute ES gradient estimates.

    For each scene and each perturbation epsilon_i ~ N(0, sigma^2 * I):
        theta+ = theta + sigma * epsilon_i
        theta- = theta - sigma * epsilon_i
        reward_i = quality(theta-) - quality(theta+)  [antithetic pairs]
        grad_approx = mean(reward_i * epsilon_i) / sigma

    This is equivalent to OpenAI ES (Salimans et al. 2017).

    Returns:
        generator: Trained EdgeWeightGenerator ready for inference.
    """
    dev = torch.device(device)

    train_grid = Grid(d=grid.d, K=train_K)
    train_encoder = SceneEncoder(train_grid)
    grid_size_tr = 2 ** train_K
    # b scaled so there are ~4 pages per grid
    train_b = max(16, grid_size_tr ** train_grid.d // 4)

    if verbose:
        print(f"[ES] Training on K={train_K} ({grid_size_tr}x{grid_size_tr}), "
              f"inference on K={grid.K} ({2**grid.K}x{2**grid.K}), "
              f"train_b={train_b}")
        print(f"[ES] Building dataset ({dataset_size} scenes)...")

    t0 = time.time()
    queries_list = [training_queries] * dataset_size if training_queries else []
    dataset = SceneDataset(train_grid, train_encoder, queries_list, size=dataset_size)
    if verbose:
        print(f"[ES] Dataset ready in {time.time()-t0:.1f}s\n")

    generator = EdgeWeightGenerator(
        input_dim=train_encoder.feat_dim,
        hidden_dim=gen_hidden,
        num_layers=gen_layers,
    )
    generator.to(dev)
    n_gen_params = torch.nn.utils.parameters_to_vector(list(generator.parameters())).numel()

    opt = torch.optim.Adam(generator.parameters(), lr=lr)
    if learn_obj_weights:
        init_logits = torch.log(torch.tensor(list(init_obj_weights), dtype=torch.float32, device=dev))
        obj_logits_param = torch.nn.Parameter(init_logits.clone())
        obj_opt = torch.optim.Adam([obj_logits_param], lr=lr)
    rng = np.random.default_rng(0)

    def true_quality(w_np: np.ndarray, queries, obj_w=None) -> float:
        path = weights_to_sfc(train_grid, w_np)
        code_map = train_grid.code_map_from_path(path)
        seg, span_pc, page_amp = compute_quality_from_codemap(
            code_map, queries, grid_size=grid_size_tr, d=train_grid.d, b=train_b)
        if obj_w is None:
            obj_w = np.array([0.60, 0.15, 0.25])
        return float(obj_w[0]) * seg + float(obj_w[1]) * span_pc + float(obj_w[2]) * page_amp

    # Collect a moving baseline (mean quality) per scene for variance reduction
    baseline = {}

    if verbose:
        print("=== ES Training ===")

    import random as _random
    indices = list(range(len(dataset)))

    for epoch in range(epochs):
        epoch_rewards = []
        # collect (param_vec, grad_estimate) pairs
        all_grads = []

        _random.shuffle(indices)
        batch_idxs = indices[:batch_size]

        # flatten current generator params (+ obj_logits if learning weights)
        gen_vec = torch.nn.utils.parameters_to_vector(generator.parameters()).detach()
        if learn_obj_weights:
            param_vec = torch.cat([gen_vec, obj_logits_param.detach()])
        else:
            param_vec = gen_vec
        n_params = param_vec.numel()

        batch_grad = torch.zeros(n_params, device=dev)
        batch_count = 0

        for idx in batch_idxs:
            feats, queries, _ = dataset[idx]
            t_feats = torch.tensor(feats, dtype=torch.float32, device=dev)

            # ES: antithetic perturbations
            rewards = []
            epsilons = []
            for _ in range(n_perturbations // 2):
                eps = torch.randn(n_params, device=dev)
                epsilons.append(eps)

                # theta+ and theta-
                for sign, w_sign in [(+1, eps), (-1, -eps)]:
                    perturbed = param_vec + sigma * w_sign
                    torch.nn.utils.vector_to_parameters(perturbed[:n_gen_params], generator.parameters())
                    if learn_obj_weights:
                        logits_p = perturbed[n_gen_params:].cpu().numpy()
                        exp_l = np.exp(logits_p - logits_p.max())
                        obj_w = exp_l / exp_l.sum()
                    else:
                        obj_w = np.array(list(init_obj_weights), dtype=np.float32)
                    with torch.no_grad():
                        w_gen = generator(t_feats).cpu().numpy()
                    q = true_quality(w_gen, queries, obj_w)
                    rewards.append((sign, float(q), eps))

            # Restore original generator params
            torch.nn.utils.vector_to_parameters(param_vec[:n_gen_params], generator.parameters())

            # Compute baseline (moving average)
            all_q = [r[1] for r in rewards]
            bl = baseline.get(idx, float(np.mean(all_q)))
            baseline[idx] = 0.9 * bl + 0.1 * float(np.mean(all_q))

            # ES gradient for minimisation:
            # grad ~ (1/sigma) * mean((q_+ - q_-) * eps)
            # Adam step = -lr * grad, so when q_+ < q_- (theta+ is better),
            # grad < 0 -> step > 0 -> theta moves toward +eps direction. Correct.
            scene_grad = torch.zeros(n_params, device=dev)
            for i, eps in enumerate(epsilons):
                q_plus  = rewards[2*i][1]    # quality at theta + sigma*eps
                q_minus = rewards[2*i+1][1]  # quality at theta - sigma*eps
                scene_grad += (q_plus - q_minus) * eps
                epoch_rewards.append((q_plus + q_minus) / 2)

            scene_grad /= (n_perturbations // 2) * sigma
            batch_grad += scene_grad
            batch_count += 1

        # Apply gradient
        if batch_count > 0:
            batch_grad /= batch_count
            # Set gradient manually (generator params)
            opt.zero_grad()
            idx_offset = 0
            for p in generator.parameters():
                n = p.numel()
                p.grad = batch_grad[idx_offset:idx_offset + n].view(p.shape)
                idx_offset += n
            opt.step()

            # Update objective weight logits
            if learn_obj_weights:
                obj_opt.zero_grad()
                obj_logits_param.grad = batch_grad[n_gen_params:].view(3).clone()
                obj_opt.step()

            # Update gen_vec (obj_logits updated via obj_opt above)
            param_vec = torch.nn.utils.parameters_to_vector(generator.parameters()).detach()

        if verbose and (epoch + 1) % max(1, epochs // 5) == 0:
            avg_q = float(np.mean(epoch_rewards)) if epoch_rewards else float('nan')
            if learn_obj_weights:
                obj_w_cur = torch.softmax(obj_logits_param.detach(), dim=0).cpu().numpy()
                print(f"  Epoch {epoch+1:3d}/{epochs}  mean_quality = {avg_q:.4f}  "
                      f"obj_w=[{obj_w_cur[0]:.2f},{obj_w_cur[1]:.2f},{obj_w_cur[2]:.2f}]")
            else:
                print(f"  Epoch {epoch+1:3d}/{epochs}  mean_quality = {avg_q:.4f}")

    if learn_obj_weights:
        final_weights = torch.softmax(obj_logits_param.detach(), dim=0).cpu().numpy()
        if verbose:
            print(f"[ES] Learned obj weights: seg={final_weights[0]:.3f}, span={final_weights[1]:.3f}, pa={final_weights[2]:.3f}\n")
        return generator, final_weights
    else:
        if verbose:
            print()
        return generator, None


def train_lacsfc_two_phase(
    grid: Grid,
    training_queries: List = None,
    pretrain_epochs: int = 15,
    gen_epochs: int = 30,
    dataset_size: int = 128,
    samples_per_scene: int = 8,
    batch_size: int = 16,
    lr: float = 1e-3,
    gen_hidden: int = 128,
    gen_layers: int = 3,
    eval_hidden: int = 128,
    eval_layers: int = 3,
    device: str = 'cpu',
    verbose: bool = True,
    train_K: int = 4,
) -> Tuple[EdgeWeightGenerator, QueryEvaluator]:
    """Two-phase training for LACSFC (mirrors original lacsfcf/train.py).

    Phase 1: Pretrain Evaluator via supervised regression on random weights.
             Uses weights_to_sfc to compute true quality labels.
    Phase 2: Freeze Evaluator, train Generator to minimise predicted quality.
             Online Evaluator fine-tuning every 10 epochs to track Generator.

    This avoids the adversarial collapse seen in joint training (v5), where
    Generator learns to output near-constant weights that fool an under-trained
    Evaluator rather than producing genuinely good SFC paths.

    Args:
        grid: Inference grid (e.g. K=6 or K=8).
        training_queries: Optional query list to bias the workload.
        pretrain_epochs: Evaluator pre-training epochs (Phase 1).
        gen_epochs: Generator training epochs (Phase 2).
        dataset_size: Number of scenes.
        samples_per_scene: Weight samples per scene for Evaluator pre-training.
        batch_size: Scenes per Generator update step.
        lr: Learning rate.
        train_K: Grid resolution used during training (default 4 = 16×16).
            weights_to_sfc on K=4 takes ~0.003 s vs ~31 s for K=8.
            Generator weights transfer because feat_dim = d+15 is K-independent.
    Returns:
        Tuple of (generator, evaluator) trained models.
    """
    train_grid = Grid(d=grid.d, K=train_K)
    enc = SceneEncoder(train_grid)
    feat_dim = enc.feat_dim

    generator = EdgeWeightGenerator(
        feat_dim=feat_dim, hidden_dim=gen_hidden,
        n_layers=gen_layers, n_edges=train_grid.n_edges,
    ).to(device)
    evaluator = QueryEvaluator(
        feat_dim=feat_dim, hidden_dim=eval_hidden,
        n_layers=eval_layers, n_edges=train_grid.n_edges,
    ).to(device)

    dataset = SceneDataset(train_grid, enc, training_queries or [], size=dataset_size)
    opt_e = torch.optim.Adam(evaluator.parameters(), lr=lr)
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr)
    rng = np.random.default_rng(0)

    if verbose:
        print(f"[two-phase] Phase 1: Evaluator pretraining ({pretrain_epochs} epochs)...")
    for epoch in range(pretrain_epochs):
        losses = []
        for scene in dataset.scenes:
            feats = scene['feats'].astype(np.float32)
            t_feats = torch.tensor(feats, dtype=torch.float32, device=device)
            wl = scene['workload']
            for _ in range(samples_per_scene):
                w_rand = np.abs(rng.standard_normal(train_grid.n_edges).astype(np.float32) * 0.5)
                path = weights_to_sfc(train_grid, w_rand)
                cm = train_grid.code_map_from_path(path)
                from lacsfcf.metrics.range_query import mean_seg as _ms
                q_val = float(_ms(cm, wl.queries))
                t_w = torch.tensor(w_rand, dtype=torch.float32, device=device)
                evaluator.train(); opt_e.zero_grad()
                pred = evaluator(t_feats, t_w)
                loss = (pred - torch.tensor(q_val, dtype=torch.float32, device=device)) ** 2
                loss.backward(); opt_e.step()
                losses.append(float(loss.detach()))
        if verbose and (epoch + 1) % max(1, pretrain_epochs // 3) == 0:
            print(f"  Pretrain epoch {epoch+1}/{pretrain_epochs}  loss={np.mean(losses):.4f}")

    if verbose:
        print(f"[two-phase] Phase 2: Generator training ({gen_epochs} epochs)...")
    for epoch in range(gen_epochs):
        g_losses = []
        idxs = list(range(len(dataset.scenes)))
        rng.shuffle(idxs)
        for i in range(0, len(idxs), batch_size):
            batch_scenes = [dataset.scenes[j] for j in idxs[i:i+batch_size]]
            opt_g.zero_grad()
            bloss = torch.tensor(0.0, device=device)
            for scene in batch_scenes:
                t_f = torch.tensor(scene['feats'].astype(np.float32), dtype=torch.float32, device=device)
                generator.train(); evaluator.eval()
                bloss = bloss + evaluator(t_f, generator(t_f))
            bloss = bloss / max(len(batch_scenes), 1)
            bloss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            opt_g.step(); g_losses.append(float(bloss.detach()))
        if verbose and (epoch + 1) % max(1, gen_epochs // 5) == 0:
            print(f"  Gen epoch {epoch+1}/{gen_epochs}  g_loss={np.mean(g_losses):.4f}")

    if verbose:
        print("[two-phase] Training complete.")
    return generator, evaluator
