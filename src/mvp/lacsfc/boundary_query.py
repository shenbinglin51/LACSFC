"""
LACSFC 边界扫描查询算法

基于 AC 曲线的理论性质实现高效的窗口查询算法。

理论保证：
- Property 1 (Segment Extremes): 所有最大连续分段的端点都在边界上
- Proposition 1 (Boundary attainment): AC 性质保证边界扫描的完备性
- 完全正确性：输出恰好是所有最大连续分段

参考文献：
- AC Space-Filling Curves 理论
- LACSFC 论文 Section 2.5 Properties
"""

from typing import List, Tuple
import numpy as np


class LACSFCBoundaryQuery:
    """
    LACSFC 边界扫描查询算法
    
    核心思想：
    1. AC 曲线的性质保证所有最大连续分段的起点和终点都在查询窗口边界上
    2. 通过枚举边界点找到所有入口点（前驱不在窗口内的点）
    3. 从每个入口点沿着 SFC 扩展到完整的最大连续分段
    
    复杂度：
    - 时间：O(|Q|) 其中 |Q| 是窗口内点数
    - 空间：O(|Q|)
    """
    
    def __init__(self, code_map: np.ndarray, grid_size: int, d: int):
        """
        初始化边界扫描查询处理器
        
        Args:
            code_map: SFC code map, shape = (grid_size,) * d
            grid_size: 每个维度的网格大小
            d: 维度数
        """
        self.code_map = code_map
        self.grid_size = grid_size
        self.d = d
    
    def get_intervals(self, low_cell: Tuple[int, ...], high_cell: Tuple[int, ...]) -> List[Tuple[int, int]]:
        """
        获取查询窗口内的所有最大连续分段（NumPy 向量化实现）
        
        Args:
            low_cell: 查询窗口的最小角坐标
            high_cell: 查询窗口的最大角坐标
        
        Returns:
            List of (start, end) tuples representing maximal contiguous segments
        """
        # Step 1: 向量化提取窗口内所有 codes (NumPy 切片，O(1) 对 2D)
        if self.d == 2:
            window = self.code_map[low_cell[0]:high_cell[0]+1, low_cell[1]:high_cell[1]+1]
            all_codes = window.ravel()
        else:
            slices = tuple(slice(low_cell[i], high_cell[i]+1) for i in range(self.d))
            all_codes = self.code_map[slices].ravel()

        if all_codes.size == 0:
            return []

        # Step 2: 找入口点 —— codes 集合中前驱不在集合内的元素
        # 排序后差分：gap > 1 的位置前面是段终点，后面是新段起点
        sorted_codes = np.sort(all_codes)
        
        # 差分找断点：sorted_codes[i] - sorted_codes[i-1] > 1 → sorted_codes[i] 是新入口
        if sorted_codes.size == 1:
            return [(int(sorted_codes[0]), int(sorted_codes[0]))]
        
        diffs = np.diff(sorted_codes)
        # 第一个元素总是入口；差分 > 1 的后续位置也是入口
        entry_mask = np.empty(sorted_codes.size, dtype=bool)
        entry_mask[0] = True
        entry_mask[1:] = diffs > 1
        
        # 段起点和终点
        starts = sorted_codes[entry_mask]
        # 终点：entry_mask 移位后前一位 (即下一个入口的前一个元素) + 最后一个元素
        end_mask = np.empty(sorted_codes.size, dtype=bool)
        end_mask[:-1] = entry_mask[1:]
        end_mask[-1] = True
        ends = sorted_codes[end_mask]

        return list(zip(starts.tolist(), ends.tolist()))

    def _enumerate_all_points(self, low_cell: Tuple[int, ...], high_cell: Tuple[int, ...]):
        """枚举查询窗口内的所有点（供 verify_correctness 使用）"""
        points = []

        def enumerate_recursive(current, depth):
            if depth == self.d:
                points.append(tuple(current))
                return
            for i in range(low_cell[depth], high_cell[depth] + 1):
                current[depth] = i
                enumerate_recursive(current, depth + 1)

        enumerate_recursive([0] * self.d, 0)
        return points

    def _enumerate_boundary_points(self, low_cell: Tuple[int, ...], high_cell: Tuple[int, ...]):
        """枚举查询窗口边界上的所有点（供 verify_correctness 使用）"""
        boundary = []
        for point in self._enumerate_all_points(low_cell, high_cell):
            for dim in range(self.d):
                if point[dim] == low_cell[dim] or point[dim] == high_cell[dim]:
                    boundary.append(point)
                    break
        return boundary
    
    def verify_correctness(self, low_cell: Tuple[int, ...], high_cell: Tuple[int, ...]) -> dict:
        """
        验证算法的正确性
        
        Returns:
            dict with verification results:
            - intervals: 算法输出的分段
            - all_codes: 窗口内所有 codes（排序后）
            - coverage: 分段覆盖的 codes 集合
            - is_complete: 是否覆盖了所有窗口内的点
            - is_maximal: 是否所有分段都是最大的
        """
        intervals = self.get_intervals(low_cell, high_cell)
        
        # 获取窗口内所有 codes
        all_codes = set()
        for point in self._enumerate_all_points(low_cell, high_cell):
            all_codes.add(int(self.code_map[point]))
        
        # 检查分段覆盖的 codes
        coverage = set()
        for start, end in intervals:
            for code in range(start, end + 1):
                coverage.add(code)
        
        # 验证完备性：所有窗口内的点都被覆盖
        is_complete = (coverage == all_codes)
        
        # 验证最大性：每个分段都是最大的
        is_maximal = True
        for start, end in intervals:
            # 检查左边界
            if start > 0 and (start - 1) in all_codes:
                is_maximal = False
                break
            # 检查右边界
            if (end + 1) in all_codes:
                is_maximal = False
                break
        
        return {
            'intervals': intervals,
            'all_codes': sorted(all_codes),
            'coverage': sorted(coverage),
            'is_complete': is_complete,
            'is_maximal': is_maximal,
            'num_segments': len(intervals),
            'num_points': len(all_codes),
        }
