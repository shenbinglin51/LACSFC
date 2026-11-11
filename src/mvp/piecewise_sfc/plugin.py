"""
PiecewiseSFC Plugin for sfcBenchmark MVP

Adapts PiecewiseTree to work with the sfcBenchmark framework.
"""

from __future__ import annotations
from typing import Optional, List, Sequence
import numpy as np

from ..types import SFCPlugin, QuantizedQuery, Interval
from .tree import PiecewiseTree, PiecewiseNode


class PiecewiseSFCPlugin(SFCPlugin):
    """
    PiecewiseSFC (Learned-BMTree) plugin for sfcBenchmark
    
    This plugin implements a piecewise space-filling curve using a
    binary tree structure with learned/heuristic dimension selection.
    """
    
    def __init__(
        self,
        k: int,
        d: int = 2,
        tree: Optional[PiecewiseTree] = None,
        smallest_split_card: int = 100,
        max_depth: int = 20,
        build_mode: str = "data_adaptive",
    ):
        """
        Initialize PiecewiseSFC plugin
        
        Args:
            k: 每个维度的位数
            d: 维度数（默认 2）
            tree: 预训练的树（可选）
            smallest_split_card: 最小分裂基数
            max_depth: 最大树深度
            build_mode: 'data_adaptive' (default, variance-max per node),
                'query_guided' (requires queries in build), or 'zorder'.
        """
        self.k = k
        self.d = d
        self.bit_length = [k] * d
        self.tree = tree
        self.smallest_split_card = smallest_split_card
        self.max_depth = max_depth
        if build_mode not in {"data_adaptive", "query_guided", "zorder"}:
            raise ValueError(f"unknown build_mode: {build_mode}")
        self.build_mode = build_mode
        
        # 如果没有提供树，将在 build 时创建
        self._is_built = tree is not None
        self._pending_points = None  # 用于延迟构建
        
        # 预计算 Z-order masks 以提升性能
        from ..lmsfc.masks import standard_morton_masks
        self._zorder_masks = standard_morton_masks(self.d)
        
        # 创建内部编码器用于批量编码
        self._setup_encoder()
    
    def build(self, points: np.ndarray, queries: Optional[List] = None) -> None:
        """Build the PiecewiseSFC tree.

        Args:
            points: (n, d) float array in [0, 1).
            queries: Optional training queries (list of ``QueryWindow``).
                Required when ``build_mode == 'query_guided'`` and ignored
                otherwise.
        """
        if points.ndim != 2:
            raise ValueError(f"points must be 2D, got shape {points.shape}")
        
        if points.shape[1] != self.d:
            raise ValueError(f"points dimension mismatch: expected {self.d}, got {points.shape[1]}")
        
        # 创建树
        self.tree = PiecewiseTree(
            bit_length=self.bit_length,
            dataset=points,
            smallest_split_card=self.smallest_split_card,
            max_depth=self.max_depth,
        )

        if self.build_mode == "query_guided":
            if queries is None or len(queries) == 0:
                raise ValueError(
                    "PiecewiseSFCPlugin(build_mode='query_guided') requires "
                    "non-empty `queries` in build()"
                )
            from .training_mcts import _queries_to_cell_bounds
            ql, qh = _queries_to_cell_bounds(list(queries), k=self.k, d=self.d)
            self.tree.build_tree_query_guided(ql, qh)
        elif self.build_mode == "zorder":
            self.tree.build_tree_zorder_heuristic()
        else:  # data_adaptive
            self.tree.build_tree_data_adaptive()
        
        self._is_built = True
        # 确保 build_artifacts 使用 encode_cell(tree-code) 而非 ZCurve
        self._encoder = None
    
    def encode_cell(self, cell: np.ndarray) -> int:
        """
        编码单个网格单元到曲线值
        
        Args:
            cell: 网格单元坐标，shape (d,)，整数坐标
        
        Returns:
            曲线值（整数）
        """
        if not self._is_built:
            # 如果树未构建，使用简单的 Z-order 编码作为后备
            return self._zorder_encode_cell(cell)
        
        # 将整数网格坐标转换为归一化坐标 [0, 1)
        grid_size = 2 ** self.k
        point = (cell.astype(np.float64) + 0.5) / grid_size
        
        return self.tree.encode_point(point)
    
    def _setup_encoder(self):
        """设置内部编码器用于批量编码"""
        from src.sfcs.traditional.z_curve import ZCurve
        
        m = 1 << self.k
        grid_size = tuple([m] * self.d)
        self._zorder_encoder = ZCurve(dimensions=self.d, grid_size=grid_size)
        # _encoder 必须为 None：build_artifacts 优先使用 _encoder，
        # 若设为 ZCurve 会导致 layout 与 intervals() 使用不同 code 系统（漏检 Bug）
        self._encoder = None
    
    def _zorder_encode_cell(self, cell: np.ndarray) -> int:
        """使用 Z-order 编码作为后备方案（仅 _is_built=False 时使用）"""
        from ..lmsfc.zaddress import cal_z_address
        
        cell = np.asarray(cell, dtype=np.uint64)
        return int(cal_z_address(cell, self._zorder_masks))
    
    def encode(self, point: np.ndarray) -> int:
        """
        编码单个点到曲线值
        
        Args:
            point: 点坐标，shape (d,)，值在 [0, 1) 范围内
        
        Returns:
            曲线值（整数）
        """
        if not self._is_built:
            raise RuntimeError("PiecewiseSFC tree not built. Call build() first.")
        
        return self.tree.encode_point(point)
    
    def encode_batch(self, points: np.ndarray) -> np.ndarray:
        """
        批量编码点到曲线值
        
        Args:
            points: 点坐标，shape (n, d)，值在 [0, 1) 范围内
        
        Returns:
            曲线值数组，shape (n,)
        """
        if not self._is_built:
            raise RuntimeError("PiecewiseSFC tree not built. Call build() first.")
        
        n = points.shape[0]
        codes = np.empty(n, dtype=np.int64)
        
        for i in range(n):
            codes[i] = self.tree.encode_point(points[i])
        
        return codes
    
    def intervals(self, q: QuantizedQuery) -> Optional[List[Interval]]:
        """
        高效计算查询窗口对应的 SFC code 区间
        
        这是 PiecewiseSFC 的核心优化方法，通过递归遍历树结构
        来计算查询窗口覆盖的 code 区间，避免枚举所有 cells。
        
        Args:
            q: 量化后的查询窗口
        
        Returns:
            SFC code 区间列表，已排序和合并
        """
        if not self._is_built or self.tree is None:
            return None
        
        # 收集所有相交的区间
        intervals = []
        self._collect_intervals_recursive(
            node=self.tree.root,
            query_low=q.low_cell,
            query_high=q.high_cell,
            intervals=intervals
        )
        
        # 排序并合并区间
        if not intervals:
            return []
        
        intervals.sort(key=lambda it: it.start)
        return self._merge_intervals(intervals)
    
    def _collect_intervals_recursive(
        self,
        node: PiecewiseNode,
        query_low: np.ndarray,
        query_high: np.ndarray,
        intervals: List[Interval]
    ) -> None:
        """
        递归收集与查询窗口相交的 code 区间
        
        核心思想：
        1. 检查节点的空间范围是否与查询窗口相交
        2. 如果完全包含，直接添加整个曲线范围
        3. 如果部分相交，递归检查子节点
        4. 如果不相交，剪枝
        """
        # 检查节点空间范围与查询窗口的关系
        relation = self._check_intersection(node, query_low, query_high)
        
        if relation == 'disjoint':
            # 不相交，剪枝
            return
        
        if relation == 'contained':
            # 节点完全在查询窗口内，添加整个曲线范围
            intervals.append(Interval(
                start=node.curve_range[0],
                end=node.curve_range[1]
            ))
            return
        
        # 部分相交
        if node.is_leaf:
            # 叶子节点，添加曲线范围
            intervals.append(Interval(
                start=node.curve_range[0],
                end=node.curve_range[1]
            ))
            return
        
        # 内部节点，递归检查子节点
        if node.left_child is not None:
            self._collect_intervals_recursive(
                node.left_child, query_low, query_high, intervals
            )
        
        if node.right_child is not None:
            self._collect_intervals_recursive(
                node.right_child, query_low, query_high, intervals
            )
    
    def _check_intersection(
        self,
        node: PiecewiseNode,
        query_low: np.ndarray,
        query_high: np.ndarray
    ) -> str:
        """
        检查节点空间范围与查询窗口的关系
        
        Returns:
            'disjoint': 不相交
            'contained': 节点完全在查询窗口内
            'intersect': 部分相交
        """
        # 将节点的 space_range 转换为整数坐标范围
        node_low = np.zeros(self.d, dtype=np.int64)
        node_high = np.zeros(self.d, dtype=np.int64)
        
        for dim in range(self.d):
            # space_range 格式: [dim0_low_bits, dim0_high_bits, dim1_low_bits, ...]
            low_bits = node.space_range[dim * 2]
            high_bits = node.space_range[dim * 2 + 1]
            
            # 将二进制位转换为整数
            low_val = 0
            high_val = 0
            
            for bit_idx in range(len(low_bits)):
                low_val = (low_val << 1) | low_bits[bit_idx]
                high_val = (high_val << 1) | high_bits[bit_idx]
            
            # 补齐剩余位
            remaining_bits = self.bit_length[dim] - len(low_bits)
            if remaining_bits > 0:
                low_val <<= remaining_bits
                high_val <<= remaining_bits
                high_val |= (1 << remaining_bits) - 1
            
            node_low[dim] = low_val
            node_high[dim] = high_val
        
        # 检查是否相交
        if np.any(node_high < query_low) or np.any(node_low > query_high):
            return 'disjoint'
        
        # 检查是否完全包含
        if np.all(node_low >= query_low) and np.all(node_high <= query_high):
            return 'contained'
        
        return 'intersect'
    
    def _merge_intervals(self, intervals: List[Interval]) -> List[Interval]:
        """
        合并重叠或相邻的区间
        
        Args:
            intervals: 已排序的区间列表
        
        Returns:
            合并后的区间列表
        """
        if not intervals:
            return []
        
        merged = [intervals[0]]
        
        for current in intervals[1:]:
            last = merged[-1]
            
            # 如果当前区间与上一个区间重叠或相邻
            if current.start <= last.end + 1:
                # 合并区间
                merged[-1] = Interval(
                    start=last.start,
                    end=max(last.end, current.end)
                )
            else:
                # 添加新区间
                merged.append(current)
        
        return merged
    
    def get_info(self) -> dict:
        """获取插件信息"""
        info = {
            "name": "PiecewiseSFC",
            "k": self.k,
            "d": self.d,
            "smallest_split_card": self.smallest_split_card,
            "max_depth": self.max_depth,
            "is_built": self._is_built,
        }
        
        if self._is_built and self.tree is not None:
            info.update(self.tree.get_tree_info())
        
        return info


def make_piecewise_sfc(
    k: int,
    d: int = 2,
    smallest_split_card: int = 100,
    max_depth: int = 20,
) -> PiecewiseSFCPlugin:
    """
    创建 PiecewiseSFC 插件的工厂函数
    
    Args:
        k: 每个维度的位数
        d: 维度数
        smallest_split_card: 最小分裂基数
        max_depth: 最大树深度
    
    Returns:
        PiecewiseSFC 插件实例
    """
    return PiecewiseSFCPlugin(
        k=k,
        d=d,
        smallest_split_card=smallest_split_card,
        max_depth=max_depth,
    )
