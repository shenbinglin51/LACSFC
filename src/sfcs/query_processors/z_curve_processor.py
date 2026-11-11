#!/usr/bin/env python3
"""
Z-Curve查询处理器
================

使用四分树最大块分解和基于单调性的优化分段算法处理Z-Curve查询。
与HilbertCurveProcessor类似，但针对Z-Curve的特性进行优化。

作者: Cascade AI
日期: 2025-10-04
"""

import numpy as np
from typing import List, Tuple, Set
from .base_processor import BaseProcessor

class ZCurveProcessor(BaseProcessor):
    """Z-Curve查询处理器，使用四分树分解和单调性优化"""
    
    def __init__(self, grid_size: Tuple[int, int]):
        super().__init__(grid_size)
        self.max_depth = int(np.ceil(np.log2(max(grid_size))))
    
    def process_query(self, query_window, z_curve_instance) -> List[int]:
        """
        处理Z-Curve查询，使用四分树最大块分解
        
        Args:
            query_window: 查询窗口对象
            z_curve_instance: Z-Curve实例，用于编码坐标
            
        Returns:
            查询窗口内的索引列表
        """
        # 获取查询边界
        if hasattr(query_window, 'lower_bounds') and hasattr(query_window, 'upper_bounds'):
            min_bounds = query_window.lower_bounds
            max_bounds = query_window.upper_bounds
        else:
            min_bounds = query_window.get_lower_bounds()
            max_bounds = query_window.get_upper_bounds()
        
        # 转换为网格坐标
        min_x = int(min_bounds[0] * self.grid_size[0])
        max_x = int(max_bounds[0] * self.grid_size[0]) - 1
        min_y = int(min_bounds[1] * self.grid_size[1])
        max_y = int(max_bounds[1] * self.grid_size[1]) - 1
        
        # 边界检查
        min_x = max(0, min(min_x, self.grid_size[0] - 1))
        max_x = max(0, min(max_x, self.grid_size[0] - 1))
        min_y = max(0, min(min_y, self.grid_size[1] - 1))
        max_y = max(0, min(max_y, self.grid_size[1] - 1))
        
        if min_x > max_x:
            min_x, max_x = max_x, min_x
        if min_y > max_y:
            min_y, max_y = max_y, min_y
        
        # 使用四分树分解获取最大块
        blocks = self._decompose_query_region(min_x, max_x, min_y, max_y)
        
        # 为每个块生成索引
        all_indices = []
        for block in blocks:
            block_indices = self._generate_block_indices(block, z_curve_instance)
            all_indices.extend(block_indices)
        
        return sorted(list(set(all_indices)))  # 去重并排序
    
    def _decompose_query_region(self, min_x: int, max_x: int, min_y: int, max_y: int) -> List[dict]:
        """
        使用四分树分解查询区域为最大块
        
        Returns:
            块列表，每个块包含 {'x': int, 'y': int, 'size': int, 'type': str}
        """
        blocks = []
        
        # 计算可能的最大块大小
        width = max_x - min_x + 1
        height = max_y - min_y + 1
        
        # 递归分解区域
        self._recursive_decompose(min_x, min_y, width, height, 
                                min_x, max_x, min_y, max_y, blocks)
        
        return blocks
    
    def _recursive_decompose(self, x: int, y: int, width: int, height: int,
                           query_min_x: int, query_max_x: int, 
                           query_min_y: int, query_max_y: int, 
                           blocks: List[dict]):
        """递归分解区域"""
        
        # 检查当前区域是否与查询窗口相交
        if (x > query_max_x or x + width - 1 < query_min_x or 
            y > query_max_y or y + height - 1 < query_min_y):
            return  # 不相交，跳过
        
        # 检查当前区域是否完全在查询窗口内
        if (x >= query_min_x and x + width - 1 <= query_max_x and
            y >= query_min_y and y + height - 1 <= query_max_y):
            # 完全包含，添加为一个块
            blocks.append({
                'x': x, 'y': y, 'width': width, 'height': height,
                'type': 'complete'
            })
            return
        
        # 如果区域太小，直接处理
        if width <= 2 and height <= 2:
            blocks.append({
                'x': x, 'y': y, 'width': width, 'height': height,
                'type': 'partial'
            })
            return
        
        # 分解为四个子区域
        half_width = width // 2
        half_height = height // 2
        
        # 四个象限
        quadrants = [
            (x, y, half_width, half_height),  # 左下
            (x + half_width, y, width - half_width, half_height),  # 右下
            (x, y + half_height, half_width, height - half_height),  # 左上
            (x + half_width, y + half_height, width - half_width, height - half_height)  # 右上
        ]
        
        for qx, qy, qw, qh in quadrants:
            if qw > 0 and qh > 0:
                self._recursive_decompose(qx, qy, qw, qh, 
                                        query_min_x, query_max_x, 
                                        query_min_y, query_max_y, blocks)
    
    def _generate_block_indices(self, block: dict, z_curve_instance) -> List[int]:
        """为块生成Z-Curve索引"""
        
        indices = []
        x_start = block['x']
        y_start = block['y']
        width = block['width']
        height = block['height']
        
        if block['type'] == 'complete' and width == height and self._is_power_of_2(width):
            # 对于完整的2^n x 2^n块，可以使用优化的Z-order范围
            indices = self._generate_z_order_range(x_start, y_start, width)
        else:
            # 对于部分块或不规则块，枚举所有点
            for x in range(x_start, x_start + width):
                for y in range(y_start, y_start + height):
                    if (x < self.grid_size[0] and y < self.grid_size[1]):
                        point = np.array([[x, y]], dtype=np.int64)
                        try:
                            index = z_curve_instance.encode(point)[0]
                            indices.append(int(index))
                        except:
                            continue
        
        return indices
    
    def _is_power_of_2(self, n: int) -> bool:
        """检查是否为2的幂"""
        return n > 0 and (n & (n - 1)) == 0
    
    def _generate_z_order_range(self, x: int, y: int, size: int) -> List[int]:
        """
        为2^n x 2^n的完整块生成Z-order索引范围
        这是Z-Curve的优化：完整的2^n块在Z-order中是连续的
        """
        # 计算块的Z-order起始值
        start_z = self._interleave_bits(x, y)
        
        # 计算块大小对应的Z-order范围
        # 对于2^n x 2^n的块，Z-order索引是连续的
        num_points = size * size
        
        return list(range(start_z, start_z + num_points))
    
    def _interleave_bits(self, x: int, y: int) -> int:
        """位交错计算Z-order值"""
        z = 0
        for i in range(16):  # 假设坐标不超过16位
            z |= (x & (1 << i)) << i | (y & (1 << i)) << (i + 1)
        return z
    
    def get_query_segments(self, indices: List[int]) -> List[Tuple[int, int]]:
        """
        将索引列表转换为连续分段
        利用Z-Curve的单调性进行优化
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
