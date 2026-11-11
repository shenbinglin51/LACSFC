"""
LMSFC查询处理器
基于原始LMSFC项目的高性能查询处理算法
"""
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

from .query_processor_base import QueryProcessorBase, QueryWindow, Interval
from .monotonic_curve_processor import MonotonicCurveProcessor

@dataclass
class LMSFCConfig:
    """LMSFC配置参数"""
    page_size: int = 1024
    enable_optimization: bool = True
    partition_threshold: int = 500  # 分割阈值
    max_partition_depth: int = 10   # 最大分割深度

class LMSFCProcessor(MonotonicCurveProcessor):
    """
    LMSFC查询处理器
    实现基于LMSFC论文的高效查询分割算法
    """
    
    def __init__(self, grid_size: int, **kwargs):
        """
        初始化LMSFC查询处理器
        
        参数:
            grid_size: 网格大小
            **kwargs: 其他参数
                - masks: 学习到的位掩码
                - config: LMSFC配置
        """
        super().__init__(grid_size, curve_type='lmsfc')
        
        self.config = kwargs.get('config', LMSFCConfig())
        if isinstance(self.config, dict):
            self.config = LMSFCConfig(**self.config)
        
        # 掩码配置
        self.masks = kwargs.get('masks', [
            0x5555555555555555,  # X维度标准掩码
            0xAAAAAAAAAAAAA   # Y维度标准掩码
        ])
        
        self.dimensions = 2  # LMSFC主要支持2D
        self.page_size = self.config.page_size
        
        # 性能统计
        self._query_count = 0
        self._total_segments = 0
        
    def process_query(self, query_window: QueryWindow) -> List[Interval]:
        """
        处理查询窗口，使用LMSFC的递归分割算法
        
        参数:
            query_window: 查询窗口对象
            
        返回:
            曲线上的区间列表
        """
        self._query_count += 1
        
        # 获取查询边界
        min_x, min_y = int(query_window.x), int(query_window.y)
        max_x = min_x + int(query_window.width) - 1
        max_y = min_y + int(query_window.height) - 1
        
        # 确保坐标在有效范围内
        min_x = max(0, min(min_x, self.grid_size - 1))
        min_y = max(0, min(min_y, self.grid_size - 1))
        max_x = max(0, min(max_x, self.grid_size - 1))
        max_y = max(0, min(max_y, self.grid_size - 1))
        
        # 计算查询窗口的Z地址边界
        min_z = self._calculate_z_address(min_x, min_y)
        max_z = self._calculate_z_address(max_x, max_y)
        
        # 使用LMSFC递归分割算法
        segments = []
        self._query_partition(min_x, min_y, max_x, max_y, min_z, max_z, segments, 0)
        
        # 转换为Interval对象
        intervals = [Interval(start, end) for start, end in segments]
        
        # 合并相邻区间
        intervals = self._merge_intervals(intervals)
        
        self._total_segments += len(intervals)
        
        return intervals
    
    def _calculate_z_address(self, x: int, y: int) -> int:
        """
        使用学习到的掩码计算Z地址
        基于LMSFC的calZaddress方法
        """
        z_addr = 0
        
        # 应用X维度掩码
        z_addr |= self._apply_bmi_extract(self.masks[0], x)
        
        # 应用Y维度掩码
        z_addr |= self._apply_bmi_extract(self.masks[1], y)
        
        return z_addr
    
    def _apply_bmi_extract(self, mask: int, value: int) -> int:
        """
        模拟BMI_e指令 (Bit Manipulation Instruction - Extract)
        从value中提取mask指定的位，并紧密排列
        """
        result = 0
        result_pos = 0
        value_pos = 0
        
        while mask > 0:
            if mask & 1:
                if value & (1 << value_pos):
                    result |= (1 << result_pos)
                result_pos += 1
            mask >>= 1
            value_pos += 1
        
        return result
    
    def _query_partition(self, min_x: int, min_y: int, max_x: int, max_y: int,
                        min_z: int, max_z: int, segments: List[Tuple[int, int]], 
                        depth: int):
        """
        LMSFC递归查询分割算法
        基于原始LMSFC论文的QueryPartition方法
        
        参数:
            min_x, min_y: 查询窗口最小坐标
            max_x, max_y: 查询窗口最大坐标
            min_z, max_z: Z地址范围
            segments: 输出段列表
            depth: 递归深度
        """
        # 分割停止条件
        if (max_z - min_z < self.config.partition_threshold or 
            depth >= self.config.max_partition_depth):
            segments.append((min_z, max_z))
            return
        
        # 计算XOR来找到最高有效位差异
        xor_val = min_z ^ max_z
        
        # 找到最高有效位位置
        msb_pos = self._find_msb_position(xor_val)
        
        if msb_pos == -1:
            segments.append((min_z, max_z))
            return
        
        msb_mask = 1 << msb_pos
        
        # 确定MSB属于哪个维度
        if msb_mask & self.masks[0]:  # X维度
            self._partition_by_dimension(0, min_x, min_y, max_x, max_y, 
                                       min_z, max_z, msb_pos, segments, depth)
        elif msb_mask & self.masks[1]:  # Y维度
            self._partition_by_dimension(1, min_x, min_y, max_x, max_y, 
                                       min_z, max_z, msb_pos, segments, depth)
        else:
            # 无法确定维度，直接添加段
            segments.append((min_z, max_z))
    
    def _find_msb_position(self, value: int) -> int:
        """找到最高有效位的位置"""
        if value == 0:
            return -1
        
        pos = 0
        while value > 1:
            value >>= 1
            pos += 1
        return pos
    
    def _partition_by_dimension(self, dim: int, min_x: int, min_y: int, 
                              max_x: int, max_y: int, min_z: int, max_z: int,
                              msb_pos: int, segments: List[Tuple[int, int]], depth: int):
        """
        按指定维度分割查询窗口
        实现LMSFC的bigmin/litmax算法思想
        """
        if dim == 0:  # X维度分割
            # 计算分割位置
            mask_bits = self._count_mask_bits_before_pos(self.masks[0], msb_pos)
            split_pos = ((max_x >> mask_bits) << mask_bits)
            
            if split_pos > min_x and split_pos <= max_x:
                # 左半部分: [min_x, split_pos-1]
                left_max_z = self._calculate_z_address(split_pos - 1, max_y)
                self._query_partition(min_x, min_y, split_pos - 1, max_y,
                                    min_z, left_max_z, segments, depth + 1)
                
                # 右半部分: [split_pos, max_x]
                right_min_z = self._calculate_z_address(split_pos, min_y)
                self._query_partition(split_pos, min_y, max_x, max_y,
                                    right_min_z, max_z, segments, depth + 1)
            else:
                segments.append((min_z, max_z))
                
        else:  # Y维度分割
            # 计算分割位置
            mask_bits = self._count_mask_bits_before_pos(self.masks[1], msb_pos)
            split_pos = ((max_y >> mask_bits) << mask_bits)
            
            if split_pos > min_y and split_pos <= max_y:
                # 下半部分: [min_y, split_pos-1]
                bottom_max_z = self._calculate_z_address(max_x, split_pos - 1)
                self._query_partition(min_x, min_y, max_x, split_pos - 1,
                                    min_z, bottom_max_z, segments, depth + 1)
                
                # 上半部分: [split_pos, max_y]
                top_min_z = self._calculate_z_address(min_x, split_pos)
                self._query_partition(min_x, split_pos, max_x, max_y,
                                    top_min_z, max_z, segments, depth + 1)
            else:
                segments.append((min_z, max_z))
    
    def _count_mask_bits_before_pos(self, mask: int, pos: int) -> int:
        """计算指定位置之前的掩码位数"""
        count = 0
        for i in range(pos):
            if mask & (1 << i):
                count += 1
        return count
    
    def _merge_intervals(self, intervals: List[Interval]) -> List[Interval]:
        """合并相邻或重叠的区间"""
        if not intervals:
            return []
        
        # 按起始位置排序
        sorted_intervals = sorted(intervals, key=lambda x: x.start)
        merged = [sorted_intervals[0]]
        
        for current in sorted_intervals[1:]:
            last = merged[-1]
            
            # 如果当前区间与上一个区间相邻或重叠，合并它们
            if current.start <= last.end + 1:
                merged[-1] = Interval(last.start, max(last.end, current.end))
            else:
                merged.append(current)
        
        return merged
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取处理器统计信息"""
        avg_segments = self._total_segments / self._query_count if self._query_count > 0 else 0
        
        return {
            'query_count': self._query_count,
            'total_segments': self._total_segments,
            'avg_segments_per_query': avg_segments,
            'masks': self.masks,
            'page_size': self.page_size
        }
