"""
AdaCurve 磁盘模拟引擎

模拟从磁盘加载 block 数据，与其他 SFC 方法保持一致的测试条件
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional
import time

import numpy as np

# 添加 Adacurve 项目路径
from .._paths import adacurve_root
ADACURVE_PATH = adacurve_root()
if str(ADACURVE_PATH) not in sys.path:
    sys.path.insert(0, str(ADACURVE_PATH))

from Adacurve.data import Dataset
from Adacurve.methods.adacurve_paper import AdaCurvePaperLike
from Adacurve.queries import WindowQuery, points_in_window

from ..learned_index import LearnedIndexEngine, LearnedIndexQueryResult
from .adapter import convert_to_adacurve_dataset, convert_to_adacurve_queries


class BlockStorage:
    """
    模拟 block 的磁盘存储
    
    每个 block 是一个存储单元，类似于页
    """
    
    def __init__(self, points_sorted: np.ndarray, block_starts: np.ndarray, block_ends: np.ndarray):
        """
        初始化 block 存储
        
        Args:
            points_sorted: 排序后的点数组
            block_starts: 每个 block 的起始位置
            block_ends: 每个 block 的结束位置
        """
        self.n_blocks = len(block_starts)
        self.block_starts = block_starts
        self.block_ends = block_ends
        
        # 将数据按 block 存储（模拟磁盘页）
        self.blocks = []
        for i in range(self.n_blocks):
            start = int(block_starts[i])
            end = int(block_ends[i])
            # 复制数据，模拟从磁盘加载
            block_data = points_sorted[start:end].copy()
            self.blocks.append(block_data)
        
        # 统计信息
        self.blocks_loaded = 0
        self.points_loaded = 0
    
    def load_block(self, block_id: int) -> np.ndarray:
        """
        加载一个 block（模拟磁盘 I/O）
        
        Args:
            block_id: block 的 ID
        
        Returns:
            block 中的点数据
        """
        self.blocks_loaded += 1
        block_data = self.blocks[block_id]
        self.points_loaded += len(block_data)
        
        # 模拟磁盘 I/O 延迟（可选）
        # time.sleep(0.0001)  # 0.1ms per block
        
        return block_data
    
    def reset_stats(self):
        """重置统计信息"""
        self.blocks_loaded = 0
        self.points_loaded = 0


class AdaCurveDiskSimulationEngine(LearnedIndexEngine):
    """
    AdaCurve 磁盘模拟引擎
    
    模拟从磁盘加载 block 数据，与其他 SFC 方法保持一致的测试条件
    
    关键改进：
    1. 将数据按 block 存储（模拟磁盘页）
    2. 查询时显式加载 blocks（模拟磁盘 I/O）
    3. 统计实际加载的 blocks 和 points
    """
    
    def __init__(
        self,
        name: str = "AdaCurve-DiskSim",
        block_size_points: int = 256,
        compress_dim: int = 8,
        epochs: int = 50,
        query_batch_size: int = 64,
        learning_rate: float = 1e-2,
        loss_weight_seg: float = 1.0,
        loss_weight_span: float = 1.0,
        loss_weight_local: float = 0.5,
        loss_weight_recon: float = 0.1,
        warmstart_zorder: bool = True,
        soft_rank_tau: float = 0.5,
        prioritized_sampling: bool = True,
        point_level_filter: bool = True,
        seed: int = 7,
        k: int = 8,
        simulate_disk_io: bool = True,
    ):
        """
        初始化 AdaCurve 磁盘模拟引擎
        
        Args:
            simulate_disk_io: 是否模拟磁盘 I/O 延迟
        """
        self._name = name
        self._block_size_points = block_size_points
        self._compress_dim = compress_dim
        self._epochs = epochs
        self._query_batch_size = query_batch_size
        self._learning_rate = learning_rate
        self._loss_weight_seg = loss_weight_seg
        self._loss_weight_span = loss_weight_span
        self._loss_weight_local = loss_weight_local
        self._loss_weight_recon = loss_weight_recon
        self._warmstart_zorder = warmstart_zorder
        self._soft_rank_tau = soft_rank_tau
        self._prioritized_sampling = prioritized_sampling
        self._point_level_filter = point_level_filter
        self._seed = seed
        self._k = k
        self._simulate_disk_io = simulate_disk_io
        
        self._method: Optional[AdaCurvePaperLike] = None
        self._dataset: Optional[Dataset] = None
        self._storage: Optional[BlockStorage] = None
        self._is_built = False
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def is_built(self) -> bool:
        return self._is_built
    
    def build(self, points: np.ndarray, training_queries: Optional[List] = None) -> None:
        """构建 AdaCurve 索引"""
        points = np.asarray(points, dtype=np.float32)
        n, d = points.shape
        
        # 转换为 AdaCurve Dataset
        self._dataset = convert_to_adacurve_dataset(points, self._k)
        
        # 创建 AdaCurve 方法实例
        self._method = AdaCurvePaperLike(
            block_size_points=self._block_size_points,
            compress_dim=self._compress_dim,
            epochs=self._epochs,
            query_batch_size=self._query_batch_size,
            learning_rate=self._learning_rate,
            loss_weight_seg=self._loss_weight_seg,
            loss_weight_span=self._loss_weight_span,
            loss_weight_local=self._loss_weight_local,
            loss_weight_recon=self._loss_weight_recon,
            warmstart_zorder=self._warmstart_zorder,
            soft_rank_tau=self._soft_rank_tau,
            prioritized_sampling=self._prioritized_sampling,
            point_level_filter=self._point_level_filter,
            seed=self._seed,
        )
        
        # 转换训练查询
        if training_queries is not None and len(training_queries) > 0:
            adacurve_queries = convert_to_adacurve_queries(training_queries)
        else:
            from Adacurve.queries import generate_uniform_selectivity_window_queries
            adacurve_queries = generate_uniform_selectivity_window_queries(
                d=d,
                n_queries=100,
                selectivity_min=0.0001,
                selectivity_max=0.01,
                seed=self._seed,
            )
        
        # 构建 AdaCurve 索引
        self._method.build(self._dataset, adacurve_queries)
        
        # 创建 block 存储（模拟磁盘）
        self._storage = BlockStorage(
            points_sorted=self._method._points_sorted,
            block_starts=self._method._block_start,
            block_ends=self._method._block_end,
        )
        
        self._is_built = True
    
    def query(self, query_window) -> LearnedIndexQueryResult:
        """
        执行范围查询（模拟磁盘加载）
        
        关键改进：
        1. 显式加载每个命中的 block
        2. 统计实际加载的数据量
        """
        if not self._is_built or self._method is None or self._storage is None:
            raise RuntimeError("AdaCurve engine not built. Call build() first.")
        
        # 重置统计
        self._storage.reset_stats()
        
        # 转换为 AdaCurve WindowQuery
        adacurve_query = WindowQuery(
            lower=np.asarray(query_window.low, dtype=np.float32),
            upper=np.asarray(query_window.high, dtype=np.float32),
        )
        
        t0 = time.perf_counter()
        
        # 1. Block-level filtering (SMA)
        inter = np.all(
            (self._method._block_max >= adacurve_query.lower) & 
            (self._method._block_min <= adacurve_query.upper),
            axis=1
        )
        hit_blocks = np.nonzero(inter)[0]
        
        # 2. 加载命中的 blocks 并过滤点
        candidates = 0
        true_hits = 0
        
        for bi in hit_blocks.tolist():
            # 显式加载 block（模拟磁盘 I/O）
            block_data = self._storage.load_block(bi)
            candidates += len(block_data)
            
            # 点级过滤
            if self._point_level_filter:
                mask = points_in_window(block_data, adacurve_query.lower, adacurve_query.upper)
                true_hits += int(mask.sum())
            else:
                true_hits += len(block_data)
        
        t1 = time.perf_counter()
        
        return LearnedIndexQueryResult(
            candidates=candidates,
            true_hits=true_hits,
            query_time_ms=(t1 - t0) * 1000.0,
            blocks_checked=int(len(self._method._block_min)),
            blocks_scanned=int(len(hit_blocks)),
            cells_checked=0,
            cells_hit=0,
            intervals=0,
        )
    
    def get_block_info(self) -> dict:
        """获取 block 信息"""
        if not self._is_built or self._method is None:
            return {}
        
        n_blocks = len(self._method._block_start)
        total_points = len(self._method._points_sorted)
        
        return {
            'n_blocks': n_blocks,
            'total_points': total_points,
            'block_size': self._block_size_points,
            'avg_points_per_block': total_points / n_blocks if n_blocks > 0 else 0,
        }


def make_adacurve_disk_sim(
    block_size_points: int = 256,
    compress_dim: int = 8,
    epochs: int = 50,
    name: str = "AdaCurve-DiskSim",
    k: int = 8,
    simulate_disk_io: bool = True,
) -> AdaCurveDiskSimulationEngine:
    """
    创建 AdaCurve 磁盘模拟引擎
    
    Args:
        block_size_points: 每个 block 的点数
        compress_dim: 压缩维度
        epochs: 训练轮数
        name: 引擎名称
        k: 量化位数
        simulate_disk_io: 是否模拟磁盘 I/O 延迟
    
    Returns:
        AdaCurveDiskSimulationEngine 实例
    """
    return AdaCurveDiskSimulationEngine(
        name=name,
        block_size_points=block_size_points,
        compress_dim=compress_dim,
        epochs=epochs,
        k=k,
        simulate_disk_io=simulate_disk_io,
    )
