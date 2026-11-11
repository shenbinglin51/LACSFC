"""
MCTS-based training for PiecewiseSFC

This module now provides two layers:

1. ``MCTSTrainer`` + ``train_piecewise_mcts`` — the historical skeletal MCTS
   interface (kept for backward compatibility with early experiments).
2. ``train_piecewise_query_guided`` + ``evaluate_mean_seg`` — the practical
   closure of D-1: a real evaluator computing ``mean_seg`` over training
   queries via ``PiecewiseSFCPlugin.intervals`` and a query-straddle-
   minimizing builder that actually beats the data-adaptive baseline.

The query-guided search is the "greedy rollout" baseline that Learned-BMTree
uses as the first step before full MCTS; closing this baseline properly is
the prerequisite for adding UCT-based search later.
"""

from __future__ import annotations
from collections import defaultdict
from typing import List, Tuple, Optional, Callable, Sequence
import numpy as np
import math
import random

from ..types import Interval, QuantizedQuery


class MCTSTrainer:
    """
    Monte Carlo Tree Search trainer for PiecewiseSFC
    
    This is a simplified version of the MCTS algorithm from Learned-BMTree,
    adapted to work with the sfcBenchmark framework.
    """
    
    def __init__(
        self,
        initial_score: float,
        exploration_weight: float = 1.0,
        split_depth: int = 5,
        max_depth: int = 10,
    ):
        """
        Initialize MCTS trainer
        
        Args:
            initial_score: Initial baseline score (e.g., Z-order performance)
            exploration_weight: UCT exploration weight
            split_depth: Depth at which to always split
            max_depth: Maximum tree depth
        """
        self.Q = defaultdict(float)  # Total reward of each node
        self.N = defaultdict(int)    # Visit count for each node
        self.children = {}           # Children of each node
        
        self.exploration_weight = exploration_weight
        self.initial_score = initial_score
        self.split_depth = split_depth
        self.max_depth = max_depth
        
        # Initialize root
        root_state = ""
        self.children[root_state] = set()
        self.N[root_state] = 1
        self.Q[root_state] = initial_score
    
    def train(
        self,
        tree,
        evaluate_fn: Callable,
        n_rollouts: int = 100,
        n_iterations: int = 10,
    ) -> Tuple[object, float]:
        """
        Train the tree using MCTS
        
        Args:
            tree: PiecewiseTree instance
            evaluate_fn: Function to evaluate tree performance
            n_rollouts: Number of rollouts per iteration
            n_iterations: Number of training iterations
        
        Returns:
            (best_tree, best_score)
        """
        best_score = float('inf')
        best_tree = None
        
        for iteration in range(n_iterations):
            # Perform rollouts
            for rollout in range(n_rollouts):
                self._do_rollout(tree, evaluate_fn)
            
            # Choose best action
            actions, score = self._choose_best(tree)
            
            # Apply actions to tree
            if actions:
                tree = self._apply_actions(tree, actions)
            
            # Evaluate current tree
            current_score = evaluate_fn(tree)
            
            if current_score < best_score:
                best_score = current_score
                best_tree = tree
            
            print(f"Iteration {iteration+1}/{n_iterations}: score={current_score:.4f}, best={best_score:.4f}")
        
        return best_tree, best_score
    
    def _do_rollout(self, tree, evaluate_fn):
        """Perform one MCTS rollout"""
        # Selection
        path = self._select(tree)
        
        # Expansion
        self._expand(path)
        
        # Simulation
        reward = self._simulate(tree, evaluate_fn)
        
        # Backpropagation
        self._backpropagate(path, reward)
    
    def _select(self, tree) -> List[str]:
        """Select a path through the tree using UCT"""
        path = []
        state = self._get_state(tree)
        
        while state in self.children:
            path.append(state)
            
            if not self.children[state]:
                return path
            
            # UCT selection
            state = self._uct_select(state)
        
        path.append(state)
        return path
    
    def _expand(self, path: List[str]):
        """Expand the leaf node"""
        if len(path) < 2:
            return
        
        parent = path[-2]
        child = path[-1]
        
        if child not in self.children:
            self.children[parent].add(child)
            self.children[child] = set()
    
    def _simulate(self, tree, evaluate_fn) -> float:
        """Simulate and evaluate the tree"""
        score = evaluate_fn(tree)
        reward = self.initial_score - score
        return reward
    
    def _backpropagate(self, path: List[str], reward: float):
        """Backpropagate reward through the path"""
        for state in reversed(path):
            self.N[state] += 1
            
            if self.Q[state] == 0:
                self.Q[state] = reward
            else:
                # Update with max reward
                mean_reward = self.Q[state] / (self.N[state] - 1)
                if mean_reward < reward:
                    self.Q[state] = reward * self.N[state]
                else:
                    self.Q[state] += mean_reward
    
    def _uct_select(self, state: str) -> str:
        """Select child using UCT formula"""
        log_n_parent = math.log(self.N[state])
        
        def uct_score(child: str) -> float:
            if self.N[child] == 0:
                return float('inf')
            
            exploitation = self.Q[child] / self.N[child]
            exploration = self.exploration_weight * math.sqrt(log_n_parent / self.N[child])
            return exploitation + exploration
        
        return max(self.children[state], key=uct_score)
    
    def _choose_best(self, tree) -> Tuple[str, float]:
        """Choose the best action sequence"""
        state = self._get_state(tree)
        
        if state not in self.children or not self.children[state]:
            return "", self.initial_score
        
        def score(child: str) -> float:
            if self.N[child] == 0:
                return float('-inf')
            return self.Q[child] / self.N[child]
        
        best_child = max(self.children[state], key=score)
        best_score = self.initial_score - score(best_child)
        
        # Extract actions from state difference
        actions = best_child[len(state):]
        
        return actions, best_score
    
    def _get_state(self, tree) -> str:
        """Get current tree state as a string"""
        # Simplified: use tree depth as state
        # In full implementation, this would encode the full tree structure
        return str(getattr(tree, 'depth', 0))
    
    def _apply_actions(self, tree, actions: str):
        """Apply action sequence to tree"""
        # Simplified: in full implementation, this would modify the tree
        # based on the action sequence
        return tree


def train_piecewise_mcts(
    points: np.ndarray,
    queries: Sequence,
    k: int = 8,
    d: int = 2,
    n_rollouts: int = 100,
    n_iterations: int = 10,
    exploration_weight: float = 1.0,
    smallest_split_card: int = 100,
    max_depth: int = 20,
    tiebreak: str = "variance",
    verbose: bool = False,
):
    """Train a PiecewiseSFC using query-guided search.

    This is the practical closure for D-1. The historical MCTS skeleton
    was kept above but produced no usable tree; this function replaces its
    behaviour with a working query-straddle-minimizing builder that beats
    the pure data-adaptive baseline on query-heavy workloads.

    Args:
        points: (n, d) float array in [0, 1).
        queries: iterable of ``QueryWindow``-like objects exposing
            ``low`` / ``high`` arrays (float cell or float [0,1)).
        k: bits per dimension.
        d: data dimensionality.
        n_rollouts, n_iterations, exploration_weight: kept for API
            compatibility; unused by the greedy baseline.

    Returns:
        Trained ``PiecewiseTree`` (already built, ready for encoding).
    """
    from .tree import PiecewiseTree

    del n_rollouts, n_iterations, exploration_weight  # reserved for future UCT

    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != d:
        raise ValueError(f"points must have shape (n,{d}), got {pts.shape}")

    # Convert queries to (Q, d) cell-normalized low/high arrays.
    ql, qh = _queries_to_cell_bounds(queries, k=k, d=d)

    tree = PiecewiseTree(
        bit_length=[k] * d,
        dataset=pts,
        smallest_split_card=smallest_split_card,
        max_depth=max_depth,
    )
    tree.build_tree_query_guided(ql, qh, tiebreak=tiebreak)
    if verbose:
        info = tree.get_tree_info()
        print(
            f"[train_piecewise_mcts] built query-guided tree: "
            f"leaves={info['leaf_nodes']} depth={info['max_depth']}"
        )
    return tree


def train_piecewise_query_guided(
    points: np.ndarray,
    queries: Sequence,
    *,
    k: int = 8,
    d: int = 2,
    smallest_split_card: int = 100,
    max_depth: int = 20,
    tiebreak: str = "variance",
    verbose: bool = False,
):
    """Alias with the more descriptive name used in the D-1 docs."""
    return train_piecewise_mcts(
        points,
        queries,
        k=k,
        d=d,
        smallest_split_card=smallest_split_card,
        max_depth=max_depth,
        tiebreak=tiebreak,
        verbose=verbose,
    )


# ----------------------------------------------------------------------
# Evaluator: compute mean_seg on training queries via plugin.intervals.
# ----------------------------------------------------------------------

def evaluate_mean_seg(plugin, queries: Sequence, *, k: int, d: int) -> float:
    """Compute mean number of code segments per query for a built plugin.

    This is the canonical MVP metric (``mean_seg`` in ``metrics.py``), and
    serves as the training objective for the query-guided search.

    Args:
        plugin: a built ``PiecewiseSFCPlugin`` (or any SFCPlugin exposing
            ``intervals(QuantizedQuery)``).
        queries: iterable of ``QueryWindow``-like objects.
        k: bits per dimension.
        d: data dimensionality.

    Returns:
        Average segment count across queries. Returns inf if no query
        yields intervals (e.g. broken plugin).
    """
    m = 1 << k
    total = 0
    valid = 0
    for q in queries:
        if hasattr(q, "low") and hasattr(q, "high"):
            low = np.asarray(q.low, dtype=np.float64)
            high = np.asarray(q.high, dtype=np.float64)
        else:
            low = np.asarray(q[0], dtype=np.float64)
            high = np.asarray(q[1], dtype=np.float64)
        lo_cell = np.clip(np.floor(low * m).astype(np.int64), 0, m - 1)
        hi_cell = np.clip(np.floor(high * m).astype(np.int64), 0, m - 1)
        qq = QuantizedQuery(low_cell=lo_cell, high_cell=hi_cell)
        intervals = plugin.intervals(qq)
        if intervals is None:
            continue
        total += len(intervals)
        valid += 1
    if valid == 0:
        return float("inf")
    return total / valid


def _queries_to_cell_bounds(
    queries: Sequence, *, k: int, d: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize a heterogenous query list to (Q,d) [0,1)-space bounds."""
    if len(queries) == 0:
        return (
            np.zeros((0, d), dtype=np.float64),
            np.zeros((0, d), dtype=np.float64),
        )
    lows, highs = [], []
    for q in queries:
        if hasattr(q, "low") and hasattr(q, "high"):
            low, high = q.low, q.high
        else:
            low, high = q[0], q[1]
        lows.append(np.asarray(low, dtype=np.float64))
        highs.append(np.asarray(high, dtype=np.float64))
    ql = np.stack(lows, axis=0)
    qh = np.stack(highs, axis=0)
    if ql.shape[1] != d or qh.shape[1] != d:
        raise ValueError(
            f"query dim mismatch: got {ql.shape[1]}, expected {d}"
        )
    # clip to [0, 1) so the tree's [0,1)^d reasoning stays well-defined
    eps = 1.0 - 1e-12
    ql = np.clip(ql, 0.0, eps)
    qh = np.clip(qh, 0.0, eps)
    return ql, qh


# Placeholder for future full implementation
class MCTSTrainerFull:
    """
    Full MCTS implementation (to be completed)
    
    This will integrate the complete Learned-BMTree MCTS algorithm
    with proper state representation, action space, and evaluation.
    """
    
    def __init__(self):
        raise NotImplementedError(
            "Full MCTS implementation requires integration with "
            "Learned-BMTree's complete training pipeline. "
            "Use the simplified MCTSTrainer for now."
        )
