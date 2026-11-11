"""
AdaCurve 原生查询引擎

完全复现 AdaCurve 论文的 block-based 查询机制，
不强制适配为 SFC，保持其原生的设计哲学。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import hashlib
import pickle
import numpy as np
import time

# 添加 Adacurve 项目路径到 sys.path
from .._paths import adacurve_root
ADACURVE_PATH = adacurve_root()
if str(ADACURVE_PATH) not in sys.path:
    sys.path.insert(0, str(ADACURVE_PATH))

from Adacurve.data import Dataset
from Adacurve.methods.adacurve_paper import AdaCurvePaperLike
from Adacurve.queries import WindowQuery, points_in_window

from ..learned_index import LearnedIndexEngine, LearnedIndexQueryResult
from .adapter import convert_to_adacurve_dataset, convert_to_adacurve_queries


class AdaCurveNativeEngine(LearnedIndexEngine):
    """
    AdaCurve 原生查询引擎
    
    核心特性：
    1. 100% 复现论文的 block-based 查询算法
    2. 使用 AdaCurve 的原生 build() 和 query() 方法
    3. 不适配为 SFC，保持独立的查询机制
    4. 提供与 sfcBenchmark 一致的输入输出接口
    
    论文对齐：
    - Block 划分：Section 6, 源码 line 270-292
    - Block filtering (SMA)：Section 6.1, 源码 line 298-334
    - 损失函数：Global Cost (公式 5) + Local Cost (SCL)
    """
    
    def __init__(
        self,
        name: str = "AdaCurve-Native",
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
    ):
        """
        初始化 AdaCurve 原生引擎
        
        Args:
            name: 引擎名称
            block_size_points: 每个 block 的点数（论文默认 256）
            compress_dim: 压缩维度（论文默认 8-16）
            epochs: 训练轮数
            query_batch_size: 查询批大小
            learning_rate: 学习率
            loss_weight_seg: Segment loss 权重
            loss_weight_span: Span loss 权重
            loss_weight_local: Local loss 权重（SCL）
            loss_weight_recon: Reconstruction loss 权重
            warmstart_zorder: 是否使用 Z-order 热启动
            soft_rank_tau: Soft rank 温度参数
            prioritized_sampling: 是否使用优先采样
            point_level_filter: 是否使用点级过滤
            seed: 随机种子
            k: 量化位数
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
        
        # AdaCurve 方法实例
        self._method: Optional[AdaCurvePaperLike] = None
        self._dataset: Optional[Dataset] = None
        self._is_built = False
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def is_built(self) -> bool:
        return self._is_built
    
    def build(self, points: np.ndarray, training_queries: Optional[List] = None) -> None:
        """
        构建 AdaCurve 索引
        
        完全使用 AdaCurve 的原生 build() 方法，包括：
        1. 特征化（Featurization）
        2. 神经网络训练（Attention + Ranking Model）
        3. Block 划分（基于学习到的排序）
        4. SMA 计算（每个 block 的 min/max）
        
        Args:
            points: (n, d) 数组，范围 [0, 1)
            training_queries: 训练查询列表（可选）
        """
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
            # 生成默认训练查询
            from Adacurve.queries import generate_uniform_selectivity_window_queries
            adacurve_queries = generate_uniform_selectivity_window_queries(
                d=d,
                n_queries=100,
                selectivity_min=0.0001,
                selectivity_max=0.01,
                seed=self._seed,
            )
        
        # 构建 AdaCurve 索引（训练神经网络）
        # 这是论文的核心：学习一个自适应曲线
        self._method.build(self._dataset, adacurve_queries)
        
        self._is_built = True

    # ── 缓存支持 ────────────────────────────────────────────────────────────

    @staticmethod
    def _cache_key(
        points: np.ndarray,
        training_queries,
        k: int,
        epochs: int,
        block_size_points: int,
        compress_dim: int,
        seed: int,
    ) -> str:
        """生成确定性缓存 key（基于数据 + 查询 + 超参数的 hash）。"""
        h = hashlib.md5()
        h.update(points.tobytes())
        if training_queries is not None:
            for q in training_queries:
                h.update(np.asarray(q.low, dtype=np.float32).tobytes())
                h.update(np.asarray(q.high, dtype=np.float32).tobytes())
        h.update(f"{k}_{epochs}_{block_size_points}_{compress_dim}_{seed}".encode())
        return h.hexdigest()[:16]

    def save(self, cache_path: str) -> None:
        """将训练好的模型状态序列化到磁盘。"""
        import os
        if not self._is_built or self._method is None:
            raise RuntimeError("Engine not built, cannot save.")
        state = {
            "points_sorted": self._method._points_sorted,
            "block_min": self._method._block_min,
            "block_max": self._method._block_max,
            "block_start": self._method._block_start,
            "block_end": self._method._block_end,
            "order_pos_by_cell_index": self._method._order_pos_by_cell_index,
        }
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(state, f, protocol=4)

    def load(self, cache_path: str, dataset=None) -> None:
        """从磁盘加载已训练模型状态，跳过神经网络训练。"""
        with open(cache_path, "rb") as f:
            state = pickle.load(f)
        if self._method is None:
            self._method = AdaCurvePaperLike(
                block_size_points=self._block_size_points,
                compress_dim=self._compress_dim,
                epochs=self._epochs,
                seed=self._seed,
            )
        self._method._points_sorted = state["points_sorted"]
        self._method._block_min = state["block_min"]
        self._method._block_max = state["block_max"]
        self._method._block_start = state["block_start"]
        self._method._block_end = state["block_end"]
        self._method._order_pos_by_cell_index = state["order_pos_by_cell_index"]
        # query() 内部会 assert self._dataset is not None
        if dataset is not None:
            self._method._dataset = dataset
        self._is_built = True

    def build_cached(
        self,
        points: np.ndarray,
        training_queries=None,
        cache_dir: str = "/tmp/adacurve_cache",
    ) -> bool:
        """build 的缓存版本。命中缓存时直接加载，否则训练后保存。

        Returns:
            True 表示命中缓存（已跳过训练），False 表示重新训练。
        """
        import os
        points = np.asarray(points, dtype=np.float32)
        key = self._cache_key(
            points, training_queries,
            self._k, self._epochs, self._block_size_points,
            self._compress_dim, self._seed,
        )
        cache_path = os.path.join(cache_dir, f"adacurve_{key}.pkl")
        if os.path.exists(cache_path):
            dataset = convert_to_adacurve_dataset(points, self._k)
            self._dataset = dataset
            self.load(cache_path, dataset=dataset)
            return True
        self.build(points, training_queries)
        self.save(cache_path)
        return False

    def query(self, query_window) -> LearnedIndexQueryResult:
        """
        执行范围查询
        
        完全使用 AdaCurve 的原生 query() 方法，包括：
        1. Block-level filtering (SMA)
        2. Point-level filtering（可选）
        3. 统计 candidates 和 true_hits
        
        这是论文 Section 6.1 描述的查询算法。
        
        Args:
            query_window: QueryWindow 对象，包含 low 和 high
        
        Returns:
            LearnedIndexQueryResult 对象
        """
        if not self._is_built or self._method is None:
            raise RuntimeError("AdaCurve engine not built. Call build() first.")
        
        # 转换为 AdaCurve WindowQuery
        adacurve_query = WindowQuery(
            lower=np.asarray(query_window.low, dtype=np.float32),
            upper=np.asarray(query_window.high, dtype=np.float32),
        )
        
        # 调用 AdaCurve 的原生 query() 方法
        # 这是论文的核心查询算法（源码 line 298-334）
        query_stats = self._method.query(adacurve_query)
        
        # 转换为统一的结果格式
        return LearnedIndexQueryResult(
            candidates=query_stats.candidates,
            true_hits=query_stats.true_hits,
            query_time_ms=query_stats.time_ms,
            blocks_checked=query_stats.blocks_checked,
            blocks_scanned=query_stats.blocks_scanned,
            cells_checked=query_stats.cells_checked,
            cells_hit=query_stats.cells_hit,
            intervals=query_stats.intervals,
        )
    
    def get_block_info(self) -> dict:
        """
        获取 block 信息（用于调试和分析）
        
        Returns:
            包含 block 统计信息的字典
        """
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


def make_adacurve_native(
    block_size_points: int = 256,
    compress_dim: int = 8,
    epochs: int = 50,
    name: str = "AdaCurve-Native",
    k: int = 8,
) -> AdaCurveNativeEngine:
    """
    创建 AdaCurve 原生查询引擎的便捷函数
    
    Args:
        block_size_points: 每个 block 的点数
        compress_dim: 压缩维度
        epochs: 训练轮数
        name: 引擎名称
        k: 量化位数
    
    Returns:
        AdaCurveNativeEngine 实例
    """
    return AdaCurveNativeEngine(
        name=name,
        block_size_points=block_size_points,
        compress_dim=compress_dim,
        epochs=epochs,
        k=k,
    )
