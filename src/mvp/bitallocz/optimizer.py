"""
BitAllocLZ Optimizers: Bayesian Optimization and Gradient Descent.
Adapted from bitallocLZ/src/uc_optimize.py and uc_gradient_optimizer.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .curve import BitAllocZCurve, _alloc_from_weights


@dataclass(frozen=True)
class LearnResult:
    alloc: Tuple[int, ...]
    cost: float
    n_evals: int


def _weights_to_alloc(weights: np.ndarray, total_bits: int, k: int) -> np.ndarray:
    """Convert raw weight vector to integer alloc using the verified _alloc_from_weights."""
    alloc = _alloc_from_weights(
        weights=[float(w) for w in weights],
        total_bits=int(total_bits),
        k=int(k),
    )
    return np.array(alloc, dtype=np.int64)


def _eval_cost(
    *,
    curve: BitAllocZCurve,
    points: np.ndarray,
    queries,
    b: int,
    max_cells: int,
    w_seg: float,
    w_pa: float,
    w_cand: float,
    quantizer,
) -> float:
    """Evaluate workload cost for a given curve configuration."""
    from .query_eval import eval_workload_cost
    return eval_workload_cost(
        curve=curve,
        points=points,
        queries=queries,
        b=b,
        max_cells=max_cells,
        w_seg=w_seg,
        w_pa=w_pa,
        w_cand=w_cand,
        quantizer=quantizer,
    )


def learn_alloc_bo(
    *,
    points: np.ndarray,
    train_queries: Sequence,
    d: int,
    k: int,
    b: int,
    total_bits: int,
    max_cells: int = 2_000_000,
    bo_init: int = 5,
    bo_iters: int = 80,
    w_seg: float = 1.0,
    w_pa: float = 1.0,
    w_cand: float = 0.0,
    seed: int = 0,
    quantizer=None,
    verbose: bool = False,
) -> LearnResult:
    """Learn bit allocation using Bayesian Optimization.

    Parameters
    ----------
    bo_init : int
        Number of random exploration points before BO.
    bo_iters : int
        Number of BO acquisition iterations.
    """
    try:
        from bayes_opt import BayesianOptimization
    except ImportError as e:
        raise ImportError(
            "bayes_opt is required for BO optimizer. "
            "Install with: pip install bayesian-optimization"
        ) from e

    n_evals = [0]
    pbounds = {f"p{i}": (-3.0, 3.0) for i in range(int(d))}

    def objective(**kwargs: float) -> float:
        weights = [float(kwargs[f"p{i}"]) for i in range(int(d))]
        curve = BitAllocZCurve.from_weights(
            d=int(d), k=int(k), weights=weights, total_bits=int(total_bits)
        )
        cost = _eval_cost(
            curve=curve, points=points, queries=train_queries,
            b=b, max_cells=max_cells,
            w_seg=w_seg, w_pa=w_pa, w_cand=w_cand,
            quantizer=quantizer,
        )
        n_evals[0] += 1
        if verbose:
            print(f"  [BO eval {n_evals[0]}] alloc={list(curve.alloc)}  cost={cost:.4f}")
        return -cost

    opt = BayesianOptimization(
        f=objective, pbounds=pbounds, random_state=int(seed), verbose=0
    )
    opt.maximize(init_points=int(bo_init), n_iter=int(bo_iters))

    best = opt.max
    best_weights = [float(best["params"][f"p{i}"]) for i in range(int(d))]
    best_curve = BitAllocZCurve.from_weights(
        d=int(d), k=int(k), weights=best_weights, total_bits=int(total_bits)
    )

    if verbose:
        print(f"  [BO done] best alloc={list(best_curve.alloc)}  cost={float(-best['target']):.4f}")

    return LearnResult(
        alloc=tuple(int(x) for x in best_curve.alloc),
        cost=float(-best["target"]),
        n_evals=n_evals[0],
    )


def learn_alloc_gd(
    *,
    points: np.ndarray,
    train_queries: Sequence,
    d: int,
    k: int,
    b: int,
    total_bits: int,
    max_cells: int = 2_000_000,
    n_iterations: int = 100,
    learning_rate: float = 0.1,
    epsilon: float = 1e-4,
    w_seg: float = 1.0,
    w_pa: float = 1.0,
    w_cand: float = 0.0,
    seed: int = 0,
    quantizer=None,
    verbose: bool = False,
) -> LearnResult:
    """Learn bit allocation using gradient descent with numerical gradients."""
    rng = np.random.default_rng(int(seed))
    weights = rng.standard_normal(int(d)).astype(np.float64)

    n_evals = [0]

    def compute_cost(w: np.ndarray) -> float:
        alloc = _weights_to_alloc(w, int(total_bits), int(k))
        curve = BitAllocZCurve(
            d=int(d), k=int(k), alloc=tuple(int(x) for x in alloc), total_bits=int(total_bits)
        )
        cost = _eval_cost(
            curve=curve, points=points, queries=train_queries,
            b=b, max_cells=max_cells,
            w_seg=w_seg, w_pa=w_pa, w_cand=w_cand,
            quantizer=quantizer,
        )
        n_evals[0] += 1
        return float(cost)

    best_cost = float("inf")
    best_weights = weights.copy()

    for it in range(int(n_iterations)):
        cost = compute_cost(weights)

        alloc_now = _weights_to_alloc(weights, int(total_bits), int(k))
        if verbose:
            print(f"  [GD iter {it}] alloc={list(alloc_now)}  cost={cost:.4f}")

        if cost < best_cost:
            best_cost = cost
            best_weights = weights.copy()

        if it < int(n_iterations) - 1:
            grad = np.zeros_like(weights)
            for i in range(len(weights)):
                wp = weights.copy(); wp[i] += float(epsilon)
                wm = weights.copy(); wm[i] -= float(epsilon)
                grad[i] = (compute_cost(wp) - compute_cost(wm)) / (2.0 * float(epsilon))
            weights = weights - float(learning_rate) * grad

    final_alloc = _weights_to_alloc(best_weights, int(total_bits), int(k))

    if verbose:
        print(f"  [GD done] best alloc={list(final_alloc)}  cost={best_cost:.4f}")

    return LearnResult(
        alloc=tuple(int(x) for x in final_alloc),
        cost=float(best_cost),
        n_evals=n_evals[0],
    )
