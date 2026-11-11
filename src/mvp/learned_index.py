"""
学习型索引的统一接口

为学习型索引（如 AdaCurve）提供独立的查询引擎接口，
不强制适配为 SFC，保持其原生的查询机制。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class LearnedIndexQueryResult:
    """
    学习型索引的查询结果
    
    与 sfcBenchmark 的 QueryResult 保持一致的字段，便于对比
    """
    # 基础统计
    candidates: int  # 候选点数（扫描的总点数）
    true_hits: int   # 真实命中数
    
    # 时间统计
    query_time_ms: float  # 查询时间（毫秒）
    
    # 学习型索引特有的统计
    blocks_checked: int   # 检查的 block 数量
    blocks_scanned: int   # 实际扫描的 block 数量
    
    # 可选：详细的性能指标
    cells_checked: int = 0
    cells_hit: int = 0
    intervals: int = 0
    
    @property
    def selectivity(self) -> float:
        """选择率：真实命中数 / 候选点数"""
        return self.true_hits / self.candidates if self.candidates > 0 else 0.0
    
    @property
    def block_skip_ratio(self) -> float:
        """Block 跳过率：跳过的 blocks / 总 blocks"""
        return (self.blocks_checked - self.blocks_scanned) / self.blocks_checked if self.blocks_checked > 0 else 0.0


class LearnedIndexEngine(ABC):
    """
    学习型索引查询引擎的抽象基类
    
    设计原则：
    1. 不强制适配为 SFC
    2. 保持原生的查询机制
    3. 提供统一的输入输出接口
    """
    
    @abstractmethod
    def build(self, points: np.ndarray, training_queries: Optional[List] = None) -> None:
        """
        构建学习型索引
        
        Args:
            points: (n, d) 数组，范围 [0, 1)
            training_queries: 训练查询列表（可选）
        """
        pass
    
    @abstractmethod
    def query(self, query_window) -> LearnedIndexQueryResult:
        """
        执行范围查询
        
        Args:
            query_window: QueryWindow 对象，包含 low 和 high
        
        Returns:
            LearnedIndexQueryResult 对象
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """索引名称"""
        pass
    
    @property
    def is_built(self) -> bool:
        """是否已构建"""
        return False


@dataclass
class IndexComparisonResult:
    """
    索引对比结果
    
    用于统一对比 SFC 和学习型索引的性能
    """
    index_name: str
    
    # 查询性能指标
    mean_candidates: float
    mean_true_hits: float
    mean_selectivity: float
    mean_query_time_ms: float
    
    # SFC 特有指标（如果适用）
    mean_seg: Optional[float] = None
    mean_span: Optional[float] = None
    mean_span_per_cell: Optional[float] = None
    mean_page_amp: Optional[float] = None
    
    # 学习型索引特有指标（如果适用）
    mean_blocks_checked: Optional[float] = None
    mean_blocks_scanned: Optional[float] = None
    mean_block_skip_ratio: Optional[float] = None
    
    # 构建时间
    build_time_s: float = 0.0
    
    def __str__(self) -> str:
        """格式化输出"""
        lines = [
            f"Index: {self.index_name}",
            f"  Query Performance:",
            f"    - Candidates: {self.mean_candidates:.1f}",
            f"    - True Hits: {self.mean_true_hits:.1f}",
            f"    - Selectivity: {self.mean_selectivity:.4f}",
            f"    - Query Time: {self.mean_query_time_ms:.2f} ms",
        ]
        
        if self.mean_seg is not None:
            lines.append(f"  SFC Metrics:")
            lines.append(f"    - mean_seg: {self.mean_seg:.2f}")
            lines.append(f"    - mean_span: {self.mean_span:.1f}")
            lines.append(f"    - span_per_cell: {self.mean_span_per_cell:.4f}")
            lines.append(f"    - page_amp: {self.mean_page_amp:.4f}")
        
        if self.mean_blocks_checked is not None:
            lines.append(f"  Block Metrics:")
            lines.append(f"    - Blocks Checked: {self.mean_blocks_checked:.1f}")
            lines.append(f"    - Blocks Scanned: {self.mean_blocks_scanned:.1f}")
            lines.append(f"    - Block Skip Ratio: {self.mean_block_skip_ratio:.4f}")
        
        lines.append(f"  Build Time: {self.build_time_s:.2f} s")
        
        return "\n".join(lines)
