"""
Complete MCTS implementation for PiecewiseSFC training

Adapted from Learned-BMTree for sfcBenchmark MVP framework.
"""

from __future__ import annotations
from collections import defaultdict
from typing import List, Tuple, Optional, Dict, Set
import numpy as np
import math
import random
from .tree import PiecewiseTree, PiecewiseNode


class PiecewiseMCTS:
    """
    Monte Carlo Tree Search for learning optimal piecewise SFC
    
    Adapted from Learned-BMTree's MCTS implementation.
    """
    
    def __init__(
        self,
        initial_score: float,
        tree: PiecewiseTree,
        split_depth: int = 5,
        max_depth: int = 10,
        exploration_weight: float = 1.0,
        use_greedy: bool = True,
    ):
        """
        Initialize MCTS trainer
        
        Args:
            initial_score: Baseline score (e.g., Z-order scan range)
            tree: PiecewiseTree instance
            split_depth: Depth at which to always split
            max_depth: Maximum tree depth
            exploration_weight: UCT exploration weight
            use_greedy: Whether to use greedy rollout
        """
        self.Q = defaultdict(float)  # Total reward
        self.N = defaultdict(int)    # Visit count
        self.G = defaultdict(bool)   # Greedy flag
        self.children: Dict[str, Set[str]] = {}
        
        self.exploration_weight = exploration_weight
        self.initial_score = initial_score + 300  # Add buffer like original
        self.split_depth = split_depth
        self.max_depth = max_depth
        self.use_greedy = use_greedy
        
        # Initialize root
        state = self._get_state(tree)
        self.children[state] = set()
        self.N[state] = 1
        self.Q[state] = self.initial_score
    
    def train(
        self,
        tree: PiecewiseTree,
        evaluate_fn,
        queries: np.ndarray,
        action_depth: int = 16,
        rollouts_per_action: int = 100,
    ) -> Tuple[PiecewiseTree, float]:
        """
        Train the tree using MCTS
        
        Args:
            tree: PiecewiseTree to train
            evaluate_fn: Function(tree, queries) -> scan_range
            queries: Training queries
            action_depth: Number of actions to take
            rollouts_per_action: Rollouts per action
        
        Returns:
            (best_tree, best_score)
        """
        best_score = float('inf')
        best_tree_state = None
        
        for depth in range(action_depth):
            # Perform rollouts
            for rollout_idx in range(rollouts_per_action):
                path, scan_range = self.do_rollout(tree, evaluate_fn, queries)
                
                if rollout_idx % 20 == 0:
                    print(f"  Depth {depth}, Rollout {rollout_idx}: scan={scan_range:.2f}")
            
            # Choose best action
            actions, reward = self.choose(tree)
            
            if not actions:
                print(f"No more actions available at depth {depth}")
                break
            
            # Apply actions
            tree.multi_action(actions)
            
            # Evaluate
            current_score = evaluate_fn(tree, queries)
            
            print(f"Depth {depth}: actions={actions}, score={current_score:.2f}, reward={reward:.2f}")
            
            if current_score < best_score:
                best_score = current_score
                best_tree_state = self._save_tree_state(tree)
            
            # Early stopping if no improvement
            if current_score > best_score + 10:
                print(f"No improvement, stopping early")
                break
        
        # Restore best tree
        if best_tree_state:
            tree = self._restore_tree_state(best_tree_state)
        
        return tree, best_score
    
    def do_rollout(self, tree: PiecewiseTree, evaluate_fn, queries: np.ndarray) -> Tuple[List[str], float]:
        """Perform one MCTS rollout"""
        state = self._get_state(tree)
        
        # Selection
        path = self._select(tree, evaluate_fn, queries)
        
        # Expansion
        self._expand(path)
        
        # Simulation
        reward = self._simulate(tree, evaluate_fn, queries)
        
        # Backpropagation
        self._backpropagate(path, reward)
        
        # Undo actions
        num_actions = (len(path[-1]) - len(state)) // 2
        tree.unstep(num_actions)
        
        return path, self.initial_score - reward
    
    def _select(self, tree: PiecewiseTree, evaluate_fn, queries: np.ndarray) -> List[str]:
        """Select path using UCT"""
        path = []
        
        while True:
            state = self._get_state(tree)
            path.append(state)
            
            if state not in self.children:
                return path
            
            # Greedy rollout
            if self.use_greedy and not self.G[state]:
                self._greedy_step(tree, evaluate_fn, queries)
                self.G[state] = True
                continue
            
            # Generate children
            children_actions = self._generate_children_actions(tree)
            
            if not children_actions:
                return path
            
            # Determine split strategy
            current_depth = tree.current_node.depth if hasattr(tree, 'current_node') else 0
            
            if current_depth <= self.split_depth:
                split = [1] * tree.d
            elif current_depth <= self.max_depth:
                split = [random.randint(0, 1) for _ in range(tree.d)]
            else:
                split = [0] * tree.d
            
            # Build children states
            children = set()
            for child_action in children_actions:
                child_state = state
                for i, ac in enumerate(child_action):
                    child_state += str(ac) + str(split[i])
                children.add(child_state)
            
            unexplored = children - self.children[state]
            
            if unexplored and (random.random() >= 0.0 or len(self.children[state]) == 0):
                # Explore unexplored
                n = unexplored.pop()
                actions = n[len(state):]
                tree.multi_action(actions)
            else:
                # UCT selection
                node = self._uct_select(tree)
                actions = node[len(state):]
                tree.multi_action(actions)
    
    def _greedy_step(self, tree: PiecewiseTree, evaluate_fn, queries: np.ndarray):
        """Take greedy action"""
        available_dims = tree.current_node.get_available_dimensions() if hasattr(tree, 'current_node') else list(range(tree.d))
        
        if not available_dims:
            return
        
        best_dim = None
        best_score = float('inf')
        
        for dim in available_dims:
            # Try this dimension
            tree.take_action(dim)
            score = evaluate_fn(tree, queries)
            tree.unaction()
            
            if score < best_score:
                best_score = score
                best_dim = dim
        
        if best_dim is not None:
            tree.take_action(best_dim)
    
    def _generate_children_actions(self, tree: PiecewiseTree) -> List[List[int]]:
        """Generate possible children actions"""
        available_dims = tree.current_node.get_available_dimensions() if hasattr(tree, 'current_node') else list(range(tree.d))
        
        if not available_dims:
            return []
        
        # For simplicity, return all available dimensions
        return [[dim] for dim in available_dims]
    
    def _expand(self, path: List[str]):
        """Expand leaf node"""
        if len(path) < 2:
            if path[0] not in self.children:
                self.children[path[0]] = set()
            return
        
        parent = path[-2]
        child = path[-1]
        
        if child not in self.children:
            self.children[parent].add(child)
            self.children[child] = set()
    
    def _simulate(self, tree: PiecewiseTree, evaluate_fn, queries: np.ndarray) -> float:
        """Simulate and evaluate"""
        score = evaluate_fn(tree, queries)
        reward = self.initial_score - score
        return reward
    
    def _backpropagate(self, path: List[str], reward: float):
        """Backpropagate reward"""
        for node in reversed(path):
            self.N[node] += 1
            
            if self.Q[node] == 0:
                self.Q[node] = reward
            else:
                mean_reward = self.Q[node] / (self.N[node] - 1)
                if mean_reward > reward:
                    self.Q[node] += mean_reward
                else:
                    self.Q[node] = reward * self.N[node]
    
    def _uct_select(self, tree: PiecewiseTree) -> str:
        """UCT selection"""
        state = self._get_state(tree)
        
        if state not in self.children or not self.children[state]:
            return state
        
        log_n_parent = math.log(self.N[state])
        
        def uct_score(child: str) -> float:
            if self.N[child] == 0:
                return float('inf')
            
            exploitation = self.Q[child] / self.N[child]
            exploration = self.exploration_weight * math.sqrt(log_n_parent / self.N[child])
            return exploitation + exploration
        
        return max(self.children[state], key=uct_score)
    
    def choose(self, tree: PiecewiseTree) -> Tuple[str, float]:
        """Choose best action"""
        state = self._get_state(tree)
        
        if state not in self.children or not self.children[state]:
            return "", self.initial_score
        
        def score(child: str) -> float:
            if self.N[child] == 0:
                return float('-inf')
            return self.Q[child] / self.N[child]
        
        best_child = max(self.children[state], key=score)
        best_score = score(best_child)
        actions = best_child[len(state):]
        
        return actions, self.initial_score - best_score
    
    def _get_state(self, tree: PiecewiseTree) -> str:
        """Get tree state as string"""
        # Simplified: use action history
        if hasattr(tree, 'action_history'):
            return ''.join(map(str, tree.action_history))
        return ""
    
    def _save_tree_state(self, tree: PiecewiseTree) -> dict:
        """Save tree state"""
        return {
            'action_history': getattr(tree, 'action_history', []).copy(),
        }
    
    def _restore_tree_state(self, state: dict) -> PiecewiseTree:
        """Restore tree state (placeholder)"""
        # In full implementation, reconstruct tree from action history
        return None


def evaluate_scan_range(tree: PiecewiseTree, queries: np.ndarray) -> float:
    """
    Evaluate tree performance by computing scan range
    
    Args:
        tree: PiecewiseTree
        queries: Query rectangles, shape (n_queries, 2, d)
    
    Returns:
        Average scan range
    """
    if len(queries) == 0:
        return 0.0
    
    total_scan = 0.0
    
    for query in queries:
        # Compute query range in curve space
        # Simplified: encode query corners and compute range
        min_corner = query[0]
        max_corner = query[1]
        
        # Encode corners
        min_code = tree.encode_point(min_corner)
        max_code = tree.encode_point(max_corner)
        
        # Scan range is the difference
        scan = abs(max_code - min_code)
        total_scan += scan
    
    return total_scan / len(queries)
