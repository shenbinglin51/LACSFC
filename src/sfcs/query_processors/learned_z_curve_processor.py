"""
基于LMSFC的学习型Z曲线查询处理器
为LearnedZCurve提供高效的查询处理能力，利用学习到的位模式优化查询性能
"""
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import math

from .query_processor_base import QueryProcessorBase, QueryWindow, Interval
from .monotonic_curve_processor import MonotonicCurveProcessor

@dataclass
class LearnedZCurveConfig:
    """学习型Z曲线的配置参数"""
    page_size: int = 4096
    cache_size: int = 1000  # 缓存大小
    enable_io_optimization: bool = True  # 是否启用I/O优化
    enable_cache: bool = True  # 是否启用查询缓存
    merge_threshold: float = 0.8  # 合并区间的阈值（相对于page_size的比例）


class LearnedZCurveProcessor(MonotonicCurveProcessor):
    """
    学习型Z曲线查询处理器
    继承自MonotonicCurveProcessor，针对学习型Z曲线特性进行了优化，利用学习到的位模式
    提供比标准Z曲线更高效的查询处理能力
    """
    
    def __init__(self, grid_size: int, **kwargs):
        """
        初始化学习型Z曲线查询处理器
        
        参数:
            grid_size: 网格大小
            **kwargs: 其他参数
                - masks: 学习到的位掩码 (List[int])
                - page_size: 页面大小 (默认: 4096)
                - config: 配置对象 (LearnedZCurveConfig)
        """
        super().__init__(grid_size, curve_type='learned_z_curve')
        
        # 解析配置
        self.config = kwargs.get('config', LearnedZCurveConfig())
        if isinstance(self.config, dict):
            self.config = LearnedZCurveConfig(**self.config)
            
        # 学习到的掩码和维度信息
        self.masks = kwargs.get('masks')
        self.dimensions = kwargs.get('dimensions', 2)  # 默认2维
        self.bits_per_dim = kwargs.get('bits_per_dim', [16, 16])  # 默认每维16位
        
        # 设置page_size
        self.page_size = self.config.page_size if hasattr(self.config, 'page_size') else 4096
        
        # 查询缓存
        self._query_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # 预计算位掩码
        self._init_bit_masks()
    
    def _init_bit_masks(self):
        """初始化位掩码，用于快速编码/解码"""
        if self.masks is None:
            # 如果没有提供掩码，使用标准Z曲线掩码
            self.masks = []
            for dim in range(self.dimensions):
                mask = 0
                for bit in range(self.bits_per_dim[dim]):
                    mask |= (1 << (bit * self.dimensions + dim))
                self.masks.append(mask)
        
        # 预计算每个维度的位数
        self.total_bits = sum(self.bits_per_dim)
        
        # 预计算每个维度的掩码
        self.dim_masks = []
        for dim in range(self.dimensions):
            mask = 0
            for bit in range(self.bits_per_dim[dim]):
                mask |= (1 << (bit * self.dimensions + dim))
            self.dim_masks.append(mask)
        
    def process_query(self, query_window: QueryWindow) -> List[Interval]:
        """
        处理查询窗口，返回曲线上的区间列表
        利用学习到的位模式优化查询处理
        
        参数:
            query_window: 查询窗口对象
            
        返回:
            曲线上的区间列表
        """
        # 生成缓存键
        cache_key = self._generate_cache_key(query_window)
        
        # 检查缓存
        if self.config.enable_cache and cache_key in self._query_cache:
            self._cache_hits += 1
            return self._query_cache[cache_key]
            
        self._cache_misses += 1
        
        # 1. 使用继承的方法获取基础区间
        intervals = super().process_query(query_window)
        
        # 2. 如果有学习到的掩码，应用优化
        if self.masks is not None and self.config.enable_io_optimization:
            intervals = self._optimize_with_learned_masks(intervals, query_window)
        
        # 3. 合并相邻区间
        if len(intervals) > 1:
            intervals = self._merge_intervals(intervals)
        
        # 4. 更新缓存
        if self.config.enable_cache and len(self._query_cache) < self.config.cache_size:
            self._query_cache[cache_key] = intervals
        
        return intervals
    
    def _generate_cache_key(self, query_window: QueryWindow) -> str:
        """生成查询窗口的缓存键"""
        # 使用QueryWindow的x, y, width, height属性生成缓存键
        return f"{query_window.x}_{query_window.y}_{query_window.width}_{query_window.height}"
    
    def _optimize_with_learned_masks(self, intervals: List[Interval], 
                                  query_window: QueryWindow) -> List[Interval]:
        """
        使用学习到的掩码优化区间列表
        
        参数:
            intervals: 原始区间列表
            query_window: 查询窗口
            
        返回:
            优化后的区间列表
        """
        if not intervals:
            return []
            
        # 1. 计算查询窗口的特征
        # 使用QueryWindow的width和height属性计算查询窗口大小
        query_size = query_window.width * query_window.height
        
        # 2. 如果查询窗口很小，直接返回原始区间
        if query_size <= (self.page_size // 4) if hasattr(self, 'page_size') else 1024:
            return intervals
            
        # 3. 对区间进行排序和合并
        optimized = []
        current = intervals[0]
        
        for i in range(1, len(intervals)):
            next_interval = intervals[i]
            
            # 计算两个区间之间的距离
            distance = next_interval.start - current.end - 1
            
            # 如果距离小于阈值，合并区间
            if distance <= self.page_size * self.config.merge_threshold:
                current = Interval(current.start, next_interval.end)
            else:
                optimized.append(current)
                current = next_interval
                
        optimized.append(current)
        
        return optimized
    
    def _merge_intervals(self, intervals: List[Interval]) -> List[Interval]:
        """合并相邻或重叠的区间"""
        if not intervals:
            return []
            
        # 按起始位置排序
        sorted_intervals = sorted(intervals, key=lambda x: x.start)
        merged = [sorted_intervals[0]]
        
        for current in sorted_intervals[1:]:
            last = merged[-1]
            
            # 如果当前区间与上一个区间重叠或相邻，合并它们
            if current.start <= last.end + 1:
                # 更新结束位置为两者中较大的
                merged[-1] = Interval(last.start, max(last.end, current.end))
            else:
                merged.append(current)
                
        return merged
    
    def optimize_io_cost(self, intervals: List[Interval]) -> List[Interval]:
        """
        优化I/O成本，合并相近的区间
        这是LMSFC中的一个重要步骤，考虑了页面边界和缓存局部性
        
        参数:
            intervals: 原始区间列表
            
        返回:
            优化后的区间列表
        """
        if not intervals:
            return []
            
        # 1. 按开始位置排序
        sorted_intervals = sorted(intervals, key=lambda x: x.start)
        
        # 2. 计算页面边界
        page_boundaries = set()
        for interval in sorted_intervals:
            start_page = interval.start // self.page_size
            end_page = interval.end // self.page_size
            
            # 添加页面边界
            for page in range(start_page, end_page + 1):
                page_start = page * self.page_size
                page_end = (page + 1) * self.page_size - 1
                
                # 计算区间与页面的交集
                intersect_start = max(interval.start, page_start)
                intersect_end = min(interval.end, page_end)
                
                if intersect_start <= intersect_end:
                    page_boundaries.add((intersect_start, intersect_end))
        
        # 3. 合并相邻的页面区间
        if not page_boundaries:
            return []
            
        sorted_pages = sorted(page_boundaries, key=lambda x: x[0])
        merged = [list(sorted_pages[0])]
        
        for current in sorted_pages[1:]:
            last = merged[-1]
            
            # 如果当前区间与上一个区间相邻或重叠，合并它们
            if current[0] <= last[1] + 1:
                last[1] = max(last[1], current[1])
            else:
                merged.append(list(current))
        
        # 4. 将合并后的页面区间转换回Interval对象
        optimized = [Interval(start, end) for start, end in merged]
        
        return optimized
    
    def get_cache_stats(self) -> Dict[str, int]:
        """
        获取缓存统计信息
        
        返回:
            包含缓存命中率等统计信息的字典
        """
        total = self._cache_hits + self._cache_misses
        hit_ratio = self._cache_hits / total if total > 0 else 0.0
        
        return {
            'hits': self._cache_hits,
            'misses': self._cache_misses,
            'hit_ratio': hit_ratio,
            'size': len(self._query_cache)
        }
