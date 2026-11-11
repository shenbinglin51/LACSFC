"""
Simplified PiecewiseTree implementation for sfcBenchmark MVP

This is a simplified version of the Learned-BMTree algorithm,
adapted to work with the sfcBenchmark framework.
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Sequence
import json
import numpy as np


class PiecewiseNode:
    """Node in the piecewise SFC tree"""
    
    def __init__(
        self,
        node_id: int,
        depth: int,
        dimension_choose: List[int],
        space_range: List[List[int]],
        curve_range: Tuple[int, int],
        bit_length: List[int],
    ):
        self.node_id = node_id
        self.depth = depth
        self.dimension_choose = dimension_choose  # 每个维度已选择的位数
        self.space_range = space_range  # 空间范围 [[0,0,...], [1,1,...], ...]
        self.curve_range = curve_range  # 曲线值范围 (start, end)
        self.bit_length = bit_length
        
        self.is_leaf = False
        self.left_child: Optional[PiecewiseNode] = None
        self.right_child: Optional[PiecewiseNode] = None
        self.chosen_dimension: Optional[int] = None
    
    def get_remaining_bits(self) -> int:
        """获取剩余需要编码的位数"""
        total_bits = sum(self.bit_length)
        used_bits = sum(self.dimension_choose)
        return total_bits - used_bits
    
    def get_available_dimensions(self) -> List[int]:
        """获取还可以选择的维度"""
        available = []
        for dim in range(len(self.bit_length)):
            if self.dimension_choose[dim] < self.bit_length[dim]:
                available.append(dim)
        return available


class PiecewiseTree:
    """
    Piecewise Space-Filling Curve Tree
    
    This is a simplified implementation that uses a heuristic strategy
    (similar to Z-order) to build the tree structure.
    """
    
    def __init__(
        self,
        bit_length: List[int],
        dataset: np.ndarray,
        smallest_split_card: int = 100,
        max_depth: int = 20,
    ):
        """
        Initialize PiecewiseTree
        
        Args:
            bit_length: 每个维度的位数 [bits_dim0, bits_dim1, ...]
            dataset: 数据集，shape (n, d)
            smallest_split_card: 最小分裂基数
            max_depth: 最大深度
        """
        self.bit_length = bit_length
        self.dataset = dataset
        self.smallest_split_card = smallest_split_card
        self.max_depth = max_depth
        
        self.dimensions = len(bit_length)
        self.node_count = 0
        
        # 初始化空间范围
        space_range = []
        for i in range(self.dimensions):
            space_range.append([0] * self.bit_length[i])  # 下界
            space_range.append([1] * self.bit_length[i])  # 上界
        
        # 创建根节点
        curve_range = (0, len(dataset) - 1)
        self.root = PiecewiseNode(
            node_id=self._gen_node_id(),
            depth=0,
            dimension_choose=[0] * self.dimensions,
            space_range=space_range,
            curve_range=curve_range,
            bit_length=bit_length,
        )
        
        self.nodes = [self.root]
        
        # MCTS 支持
        self.current_node = self.root
        self.action_history = []
        self.d = len(bit_length)  # 维度数
    
    def _gen_node_id(self) -> int:
        """生成节点 ID"""
        node_id = self.node_count
        self.node_count += 1
        return node_id
    
    def take_action(self, dimension: int) -> Tuple[int, int, int, Tuple[int, int]]:
        """
        执行一个动作（选择一个维度进行分裂）
        
        Args:
            dimension: 选择的维度
        
        Returns:
            (node_id, depth, dimension, curve_range)
        """
        # 检查是否还有可用维度
        available_dims = self.current_node.get_available_dimensions()
        if not available_dims or dimension not in available_dims:
            # 如果没有可用维度或维度无效，返回当前节点信息
            return (
                self.current_node.node_id,
                self.current_node.depth,
                dimension,
                self.current_node.curve_range
            )
        
        # 记录动作
        self.action_history.append(dimension)
        
        # 创建或获取子节点
        if self.current_node.chosen_dimension is None:
            self.current_node.chosen_dimension = dimension
            self._create_children(self.current_node, dimension)
        
        # 移动到左子节点（简化实现）
        if self.current_node.left_child is not None:
            self.current_node = self.current_node.left_child
        
        return (
            self.current_node.node_id,
            self.current_node.depth,
            dimension,
            self.current_node.curve_range
        )
    
    def unaction(self):
        """撤销上一个动作"""
        if not self.action_history:
            return
        
        self.action_history.pop()
        
        # 回到父节点（简化实现：重建路径）
        self.current_node = self.root
        for action in self.action_history:
            if self.current_node.left_child:
                self.current_node = self.current_node.left_child
    
    def multi_action(self, actions: str):
        """
        执行多个动作
        
        Args:
            actions: 动作序列字符串，如 "0101" 表示维度0,1,0,1
        """
        for i in range(0, len(actions), 2):
            if i >= len(actions):
                break
            dim = int(actions[i])
            # actions[i+1] 是 split 标志，这里简化处理
            if dim < self.d:
                try:
                    self.take_action(dim)
                except:
                    break
    
    def unstep(self, num_steps: int):
        """
        撤销多个步骤
        
        Args:
            num_steps: 要撤销的步骤数
        """
        for _ in range(num_steps):
            self.unaction()
    
    def get_all_states(self) -> Tuple[str, List[List[int]], int]:
        """
        获取当前状态
        
        Returns:
            (action_string, masks, depth)
        """
        action_string = ''.join(map(str, self.action_history))
        
        # 生成 masks（可用维度）
        available_dims = self.current_node.get_available_dimensions()
        masks = []
        for _ in range(self.d):
            mask = [1 if dim in available_dims else 0 for dim in range(self.d)]
            masks.append(mask)
        
        return action_string, masks, self.current_node.depth
    
    def _create_children(self, node: PiecewiseNode, chosen_dim: int):
        """创建子节点"""
        dimension_choose_new = node.dimension_choose.copy()
        dimension_choose_new[chosen_dim] += 1
        
        # 左子节点
        space_range_left = [row.copy() for row in node.space_range]
        space_range_left[chosen_dim * 2 + 1][node.dimension_choose[chosen_dim]] = 0
        
        curve_start, curve_end = node.curve_range
        mid = (curve_start + curve_end) // 2
        
        node.left_child = PiecewiseNode(
            node_id=self._gen_node_id(),
            depth=node.depth + 1,
            dimension_choose=dimension_choose_new,
            space_range=space_range_left,
            curve_range=(curve_start, mid),
            bit_length=self.bit_length,
        )
        
        # 右子节点
        space_range_right = [row.copy() for row in node.space_range]
        space_range_right[chosen_dim * 2][node.dimension_choose[chosen_dim]] = 1
        
        node.right_child = PiecewiseNode(
            node_id=self._gen_node_id(),
            depth=node.depth + 1,
            dimension_choose=dimension_choose_new,
            space_range=space_range_right,
            curve_range=(mid + 1, curve_end),
            bit_length=self.bit_length,
        )
        
        self.nodes.extend([node.left_child, node.right_child])
    
    def build_tree_zorder_heuristic(self):
        """使用 Z-order 启发式策略构建树"""
        self._build_recursive(self.root, parent_action=0)

    def build_tree_data_adaptive(self) -> None:
        """数据自适应构建：按当前子空间数据方差选择分裂维度。

        与 Z-order 启发式不同，该方法在每个节点处选择数据
        分布最分散（方差最大）的维度进行分裂，使曲线顺序
        更贴近实际数据分布，减少跨页面的无效扫描。
        """
        pts = np.asarray(self.dataset, dtype=np.float64)
        indices = np.arange(len(pts))
        lo = np.zeros(self.dimensions, dtype=np.float64)
        hi = np.ones(self.dimensions, dtype=np.float64)
        self._build_recursive_adaptive(self.root, pts, indices, lo, hi)

    def _build_recursive_adaptive(
        self,
        node: 'PiecewiseNode',
        pts: np.ndarray,
        indices: np.ndarray,
        lo: np.ndarray,
        hi: np.ndarray,
    ) -> None:
        """数据自适应递归构建"""
        remaining_bits = node.get_remaining_bits()
        if remaining_bits == 0 or node.depth >= self.max_depth:
            node.is_leaf = True
            return

        curve_start, curve_end = node.curve_range
        cardinality = curve_end - curve_start + 1
        if cardinality < self.smallest_split_card:
            node.is_leaf = True
            return

        available_dims = node.get_available_dimensions()
        if not available_dims:
            node.is_leaf = True
            return

        # 选择方差最大的维度；无数据时退化到 Z-order 轮询
        if len(indices) >= 2:
            pts_here = pts[indices]
            chosen_dim = available_dims[
                int(np.argmax([np.var(pts_here[:, d]) for d in available_dims]))
            ]
        else:
            chosen_dim = self._zorder_heuristic(
                node.depth % self.dimensions, available_dims
            )

        node.chosen_dimension = chosen_dim
        bit_level = node.dimension_choose[chosen_dim]

        # 沿 chosen_dim 在当前范围中点处分裂
        split_val = 0.5 * (lo[chosen_dim] + hi[chosen_dim])

        if len(indices) > 0:
            left_mask = pts[indices, chosen_dim] < split_val
            left_indices = indices[left_mask]
            right_indices = indices[~left_mask]
        else:
            left_indices = right_indices = np.empty(0, dtype=np.int64)

        # 创建子节点（结构与 _build_recursive 一致）
        dimension_choose_new = node.dimension_choose.copy()
        dimension_choose_new[chosen_dim] += 1

        mid = (curve_start + curve_end) // 2

        space_range_left = [row.copy() for row in node.space_range]
        space_range_left[chosen_dim * 2 + 1][bit_level] = 0

        space_range_right = [row.copy() for row in node.space_range]
        space_range_right[chosen_dim * 2][bit_level] = 1

        node.left_child = PiecewiseNode(
            node_id=self._gen_node_id(),
            depth=node.depth + 1,
            dimension_choose=dimension_choose_new,
            space_range=space_range_left,
            curve_range=(curve_start, mid),
            bit_length=self.bit_length,
        )
        node.right_child = PiecewiseNode(
            node_id=self._gen_node_id(),
            depth=node.depth + 1,
            dimension_choose=dimension_choose_new,
            space_range=space_range_right,
            curve_range=(mid + 1, curve_end),
            bit_length=self.bit_length,
        )
        self.nodes.extend([node.left_child, node.right_child])

        hi_left = hi.copy()
        hi_left[chosen_dim] = split_val
        lo_right = lo.copy()
        lo_right[chosen_dim] = split_val

        self._build_recursive_adaptive(
            node.left_child, pts, left_indices, lo, hi_left
        )
        self._build_recursive_adaptive(
            node.right_child, pts, right_indices, lo_right, hi
        )
    
    def _build_recursive(self, node: PiecewiseNode, parent_action: int):
        """递归构建树"""
        # 检查终止条件
        remaining_bits = node.get_remaining_bits()
        if remaining_bits == 0 or node.depth >= self.max_depth:
            node.is_leaf = True
            return
        
        # 检查数据基数
        curve_start, curve_end = node.curve_range
        cardinality = curve_end - curve_start + 1
        if cardinality < self.smallest_split_card:
            node.is_leaf = True
            return
        
        # 获取可用维度
        available_dims = node.get_available_dimensions()
        if not available_dims:
            node.is_leaf = True
            return
        
        # Z-order 启发式：轮询选择维度
        chosen_dim = self._zorder_heuristic(parent_action, available_dims)
        node.chosen_dimension = chosen_dim
        
        # 计算子节点的空间范围和曲线范围
        dimension_choose_new = node.dimension_choose.copy()
        dimension_choose_new[chosen_dim] += 1
        
        # 左子节点（该维度的位为 0）
        space_range_left = [row.copy() for row in node.space_range]
        space_range_left[chosen_dim * 2 + 1][node.dimension_choose[chosen_dim]] = 0
        
        # 右子节点（该维度的位为 1）
        space_range_right = [row.copy() for row in node.space_range]
        space_range_right[chosen_dim * 2][node.dimension_choose[chosen_dim]] = 1
        
        # 分割曲线范围（简单二分）
        mid = (curve_start + curve_end) // 2
        
        # 创建左子节点
        node.left_child = PiecewiseNode(
            node_id=self._gen_node_id(),
            depth=node.depth + 1,
            dimension_choose=dimension_choose_new,
            space_range=space_range_left,
            curve_range=(curve_start, mid),
            bit_length=self.bit_length,
        )
        self.nodes.append(node.left_child)
        
        # 创建右子节点
        node.right_child = PiecewiseNode(
            node_id=self._gen_node_id(),
            depth=node.depth + 1,
            dimension_choose=dimension_choose_new,
            space_range=space_range_right,
            curve_range=(mid + 1, curve_end),
            bit_length=self.bit_length,
        )
        self.nodes.append(node.right_child)
        
        # 递归构建子树
        self._build_recursive(node.left_child, chosen_dim)
        self._build_recursive(node.right_child, chosen_dim)
    
    def _zorder_heuristic(self, parent_action: int, available_dims: List[int]) -> int:
        """Z-order 启发式策略：轮询选择维度"""
        # 从上一个动作的下一个维度开始
        next_dim = (parent_action + 1) % self.dimensions
        
        # 找到第一个可用的维度
        for _ in range(self.dimensions):
            if next_dim in available_dims:
                return next_dim
            next_dim = (next_dim + 1) % self.dimensions
        
        # 如果没有找到，返回第一个可用维度
        return available_dims[0]
    
    def encode_point(self, point: np.ndarray) -> int:
        """
        编码单个点到曲线值
        
        Args:
            point: 点坐标，shape (d,)，值在 [0, 1) 范围内
        
        Returns:
            曲线值（整数）
        """
        # 将浮点坐标转换为整数坐标
        int_coords = np.floor(point * np.array([2**b for b in self.bit_length])).astype(int)
        int_coords = np.clip(int_coords, 0, [2**b - 1 for b in self.bit_length])
        
        # 转换为二进制表示
        binary_coords = []
        for dim, coord in enumerate(int_coords):
            bits = []
            for _ in range(self.bit_length[dim]):
                bits.append(coord & 1)
                coord >>= 1
            binary_coords.append(list(reversed(bits)))
        
        # 遍历树找到对应的叶子节点
        return self._encode_recursive(self.root, binary_coords)
    
    def _encode_recursive(self, node: PiecewiseNode, binary_coords: List[List[int]]) -> int:
        """递归编码"""
        if node.is_leaf:
            # 叶子节点，返回曲线范围的中点
            return (node.curve_range[0] + node.curve_range[1]) // 2
        
        # 检查选择的维度的下一位
        chosen_dim = node.chosen_dimension
        
        # 如果节点还没有选择维度（MCTS 训练中可能发生），返回曲线范围中点
        if chosen_dim is None:
            return (node.curve_range[0] + node.curve_range[1]) // 2
        
        # 检查是否有子节点
        if node.left_child is None or node.right_child is None:
            return (node.curve_range[0] + node.curve_range[1]) // 2
        
        bit_index = node.dimension_choose[chosen_dim]
        bit_value = binary_coords[chosen_dim][bit_index]
        
        # 根据位值选择子节点
        if bit_value == 0:
            return self._encode_recursive(node.left_child, binary_coords)
        else:
            return self._encode_recursive(node.right_child, binary_coords)
    
    def get_tree_info(self) -> dict:
        """获取树的统计信息"""
        leaf_count = sum(1 for node in self.nodes if node.is_leaf)
        max_depth = max(node.depth for node in self.nodes)
        
        return {
            "total_nodes": len(self.nodes),
            "leaf_nodes": leaf_count,
            "internal_nodes": len(self.nodes) - leaf_count,
            "max_depth": max_depth,
            "dimensions": self.dimensions,
            "bit_length": self.bit_length,
        }

    # ------------------------------------------------------------------
    # Query-guided adaptive build (D-1: PiecewiseSFC MCTS/search closure)
    # ------------------------------------------------------------------

    def build_tree_query_guided(
        self,
        queries_low: np.ndarray,
        queries_high: np.ndarray,
        *,
        tiebreak: str = "variance",
    ) -> None:
        """Build tree using query-straddle minimization per node.

        At each internal node we evaluate every available dimension and pick
        the one whose axis-midpoint split is straddled by the fewest training
        queries (queries whose ranges cross the split in that dimension).
        Ties are broken by data variance (default) or by dimension index.

        Intuition: a query that straddles a split at some depth introduces a
        code gap (+1 to mean_seg). Greedily minimizing straddle counts per
        node is a first-order proxy for reducing the final mean_seg.

        Args:
            queries_low: (Q, d) float cell-centers low corners in [0, M).
            queries_high: (Q, d) float cell-centers high corners in [0, M).
            tiebreak: 'variance' (default) or 'dim' (smaller index wins).
        """
        if tiebreak not in {"variance", "dim"}:
            raise ValueError(f"unknown tiebreak: {tiebreak}")
        pts = np.asarray(self.dataset, dtype=np.float64)
        indices = np.arange(len(pts))
        lo = np.zeros(self.dimensions, dtype=np.float64)
        hi = np.ones(self.dimensions, dtype=np.float64)
        ql = np.asarray(queries_low, dtype=np.float64)
        qh = np.asarray(queries_high, dtype=np.float64)
        if ql.ndim != 2 or qh.ndim != 2 or ql.shape != qh.shape:
            raise ValueError("queries_low and queries_high must have same 2D shape")
        if ql.shape[1] != self.dimensions:
            raise ValueError(
                f"query dim {ql.shape[1]} != tree dim {self.dimensions}"
            )
        q_indices = np.arange(ql.shape[0])
        self._build_recursive_query(
            self.root, pts, indices, lo, hi, ql, qh, q_indices, tiebreak
        )

    def _build_recursive_query(
        self,
        node: "PiecewiseNode",
        pts: np.ndarray,
        indices: np.ndarray,
        lo: np.ndarray,
        hi: np.ndarray,
        ql: np.ndarray,
        qh: np.ndarray,
        q_indices: np.ndarray,
        tiebreak: str,
    ) -> None:
        remaining_bits = node.get_remaining_bits()
        if remaining_bits == 0 or node.depth >= self.max_depth:
            node.is_leaf = True
            return

        curve_start, curve_end = node.curve_range
        cardinality = curve_end - curve_start + 1
        if cardinality < self.smallest_split_card:
            node.is_leaf = True
            return

        available_dims = node.get_available_dimensions()
        if not available_dims:
            node.is_leaf = True
            return

        # Score each available dim by query straddle count at midpoint split.
        best_dim = available_dims[0]
        best_straddle = None
        best_tiebreak_val = None
        for d in available_dims:
            split_val = 0.5 * (lo[d] + hi[d])
            if q_indices.size > 0:
                qlo = ql[q_indices, d]
                qhi = qh[q_indices, d]
                straddle = int(np.sum((qlo < split_val) & (qhi >= split_val)))
            else:
                straddle = 0
            if tiebreak == "variance":
                if indices.size >= 2:
                    tb = -float(np.var(pts[indices, d]))  # larger var = better
                else:
                    tb = 0.0
            else:  # 'dim'
                tb = float(d)
            if (
                best_straddle is None
                or straddle < best_straddle
                or (straddle == best_straddle and tb < best_tiebreak_val)
            ):
                best_straddle = straddle
                best_tiebreak_val = tb
                best_dim = d

        chosen_dim = best_dim
        node.chosen_dimension = chosen_dim
        bit_level = node.dimension_choose[chosen_dim]
        split_val = 0.5 * (lo[chosen_dim] + hi[chosen_dim])

        if indices.size > 0:
            left_mask = pts[indices, chosen_dim] < split_val
            left_indices = indices[left_mask]
            right_indices = indices[~left_mask]
        else:
            left_indices = right_indices = np.empty(0, dtype=np.int64)

        # Partition queries by their relationship to the split for recursion.
        if q_indices.size > 0:
            q_lo_here = ql[q_indices, chosen_dim]
            q_hi_here = qh[q_indices, chosen_dim]
            left_q_mask = q_lo_here < split_val
            right_q_mask = q_hi_here >= split_val
            left_q = q_indices[left_q_mask]
            right_q = q_indices[right_q_mask]
        else:
            left_q = right_q = np.empty(0, dtype=np.int64)

        dimension_choose_new = node.dimension_choose.copy()
        dimension_choose_new[chosen_dim] += 1

        mid = (curve_start + curve_end) // 2

        space_range_left = [row.copy() for row in node.space_range]
        space_range_left[chosen_dim * 2 + 1][bit_level] = 0
        space_range_right = [row.copy() for row in node.space_range]
        space_range_right[chosen_dim * 2][bit_level] = 1

        node.left_child = PiecewiseNode(
            node_id=self._gen_node_id(),
            depth=node.depth + 1,
            dimension_choose=dimension_choose_new,
            space_range=space_range_left,
            curve_range=(curve_start, mid),
            bit_length=self.bit_length,
        )
        node.right_child = PiecewiseNode(
            node_id=self._gen_node_id(),
            depth=node.depth + 1,
            dimension_choose=dimension_choose_new,
            space_range=space_range_right,
            curve_range=(mid + 1, curve_end),
            bit_length=self.bit_length,
        )
        self.nodes.extend([node.left_child, node.right_child])

        hi_left = hi.copy()
        hi_left[chosen_dim] = split_val
        lo_right = lo.copy()
        lo_right[chosen_dim] = split_val

        self._build_recursive_query(
            node.left_child, pts, left_indices, lo, hi_left, ql, qh, left_q, tiebreak
        )
        self._build_recursive_query(
            node.right_child, pts, right_indices, lo_right, hi, ql, qh, right_q, tiebreak
        )

    # ------------------------------------------------------------------
    # Serialization (save/load tree structure as JSON)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize tree structure (excluding dataset) to a plain dict."""
        return {
            "bit_length": list(self.bit_length),
            "smallest_split_card": int(self.smallest_split_card),
            "max_depth": int(self.max_depth),
            "root": self._node_to_dict(self.root),
        }

    def _node_to_dict(self, node: PiecewiseNode) -> dict:
        return {
            "depth": int(node.depth),
            "dimension_choose": list(node.dimension_choose),
            "space_range": [list(r) for r in node.space_range],
            "curve_range": [int(node.curve_range[0]), int(node.curve_range[1])],
            "is_leaf": bool(node.is_leaf),
            "chosen_dimension": (
                None if node.chosen_dimension is None else int(node.chosen_dimension)
            ),
            "left": self._node_to_dict(node.left_child) if node.left_child else None,
            "right": self._node_to_dict(node.right_child) if node.right_child else None,
        }

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, data: dict, dataset: np.ndarray) -> "PiecewiseTree":
        tree = cls(
            bit_length=list(data["bit_length"]),
            dataset=dataset,
            smallest_split_card=int(data["smallest_split_card"]),
            max_depth=int(data["max_depth"]),
        )
        # Reset: root is created by __init__, but we need to rebuild it from dict.
        tree.nodes = []
        tree.node_count = 0
        tree.root = tree._node_from_dict(data["root"])
        tree.current_node = tree.root
        tree.action_history = []
        return tree

    @classmethod
    def load_json(cls, path: str, dataset: np.ndarray) -> "PiecewiseTree":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data, dataset)

    def _node_from_dict(self, data: dict) -> PiecewiseNode:
        node = PiecewiseNode(
            node_id=self._gen_node_id(),
            depth=int(data["depth"]),
            dimension_choose=list(data["dimension_choose"]),
            space_range=[list(r) for r in data["space_range"]],
            curve_range=(int(data["curve_range"][0]), int(data["curve_range"][1])),
            bit_length=self.bit_length,
        )
        node.is_leaf = bool(data["is_leaf"])
        node.chosen_dimension = data.get("chosen_dimension")
        self.nodes.append(node)
        if data.get("left") is not None:
            node.left_child = self._node_from_dict(data["left"])
        if data.get("right") is not None:
            node.right_child = self._node_from_dict(data["right"])
        return node
