#!/usr/bin/env python3
"""
基础查询处理器
==============

所有SFC查询处理器的基类，定义通用接口和方法。

作者: Cascade AI
日期: 2025-10-04
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Any

class BaseProcessor(ABC):
    """SFC查询处理器基类"""
    
    def __init__(self, grid_size: Tuple[int, int]):
        self.grid_size = grid_size
    
    @abstractmethod
    def process_query(self, query_window, sfc_instance) -> List[int]:
        """
        处理查询窗口，返回索引列表
        
        Args:
            query_window: 查询窗口对象
            sfc_instance: SFC实例
            
        Returns:
            索引列表
        """
        pass
    
    def get_query_segments(self, indices: List[int]) -> List[Tuple[int, int]]:
        """
        将索引列表转换为连续分段
        
        Args:
            indices: 索引列表
            
        Returns:
            分段列表，每个分段为(start, end)元组
        """
        if not indices:
            return []
        
        sorted_indices = sorted(indices)
        segments = []
        start = sorted_indices[0]
        end = sorted_indices[0]
        
        for idx in sorted_indices[1:]:
            if idx == end + 1:
                end = idx
            else:
                segments.append((start, end))
                start = idx
                end = idx
        
        segments.append((start, end))
        return segments
