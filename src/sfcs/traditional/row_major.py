"""
Implementation of Row-major space-filling curve
"""
import numpy as np
from typing import Tuple
from ..sfc_base import TraditionalSFC


class RowMajor(TraditionalSFC):
    """Row-major space-filling curve"""
    
    def __init__(self, dimensions: int, grid_size: Tuple[int, ...]):
        """
        Initialize the Row-major SFC
        
        Args:
            dimensions (int): Number of dimensions
            grid_size (Tuple[int, ...]): Size of the grid in each dimension
        """
        super().__init__(dimensions, grid_size)
        self.name = "RowMajor"
        
        # Precompute multipliers for encoding/decoding
        self.multipliers = np.ones(dimensions, dtype=np.int64)
        for i in range(1, dimensions):
            self.multipliers[i] = self.multipliers[i-1] * grid_size[i-1]
            
    def encode(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Convert multi-dimensional coordinates to 1D indices using row-major order
        
        Args:
            coordinates (np.ndarray): Array of multi-dimensional coordinates
                                     shape: (n_points, dimensions)
        
        Returns:
            np.ndarray: 1D indices, shape: (n_points,)
        """
        # Check if coordinates are within bounds
        for dim in range(self.dimensions):
            if np.any(coordinates[:, dim] < 0) or np.any(coordinates[:, dim] >= self.grid_size[dim]):
                raise ValueError(f"Coordinates out of bounds in dimension {dim}")
        
        # Compute linear indices (row-major order)
        indices = np.zeros(coordinates.shape[0], dtype=np.int64)
        for dim in range(self.dimensions):
            indices += coordinates[:, dim] * self.multipliers[dim]
            
        return indices
    
    def map_query_to_indices(self, query_window):
        """
        将查询窗口映射到一维索引集合，利用行优先曲线的特性进行高效计算
        
        Args:
            query_window: 查询窗口对象，具有 get_lower_bounds() 和 get_upper_bounds() 方法
                         或者 lower_bounds 和 upper_bounds 属性
            
        Returns:
            包含查询窗口内所有索引的列表
        """
        try:
            # 处理 PaperCompliantQuery（属性）和 QueryWindow（方法）两种情况
            if hasattr(query_window, 'lower_bounds') and hasattr(query_window, 'upper_bounds'):
                min_bounds = query_window.lower_bounds
                max_bounds = query_window.upper_bounds
            else:
                min_bounds = query_window.get_lower_bounds()
                max_bounds = query_window.get_upper_bounds()
            
            # 正确转换为网格坐标 - 修复坐标转换错误
            min_x = int(min_bounds[0] * self.grid_size[0])
            max_x = int(max_bounds[0] * self.grid_size[0]) - 1  # 修正边界
            min_y = int(min_bounds[1] * self.grid_size[1])
            max_y = int(max_bounds[1] * self.grid_size[1]) - 1  # 修正边界
            
            # 确保坐标在有效范围内
            min_x = max(0, min(min_x, self.grid_size[0] - 1))
            max_x = max(0, min(max_x, self.grid_size[0] - 1))
            min_y = max(0, min(min_y, self.grid_size[1] - 1))
            max_y = max(0, min(max_y, self.grid_size[1] - 1))
            
            # 确保min <= max
            if min_x > max_x:
                min_x, max_x = max_x, min_x
            if min_y > max_y:
                min_y, max_y = max_y, min_y
            
            indices = []
            
            # 行优先：对每一行，生成连续的索引范围
            # Row Major: index = y * grid_width + x
            for y in range(min_y, max_y + 1):
                start_idx = y * self.grid_size[0] + min_x
                end_idx = y * self.grid_size[0] + max_x
                indices.extend(range(start_idx, end_idx + 1))
            
            return indices
        
        except Exception as e:
            print(f"Error in RowMajor.map_query_to_indices: {str(e)}")
            # 如果出现错误，返回空列表
            return []
            
    def decode(self, indices: np.ndarray) -> np.ndarray:
        """
        Convert 1D indices to multi-dimensional coordinates using row-major order
        
        Args:
            indices (np.ndarray): Array of 1D indices, shape: (n_points,)
            
        Returns:
            np.ndarray: Multi-dimensional coordinates, shape: (n_points, dimensions)
        """
        # Handle both single index and array of indices
        if np.isscalar(indices):
            indices = np.array([indices])
        elif indices.ndim == 0:
            indices = np.array([indices.item()])
        
        total_size = np.prod(self.grid_size)
        if np.any(indices < 0) or np.any(indices >= total_size):
            raise ValueError("Indices out of bounds")
        
        n_points = indices.size
        coordinates = np.zeros((n_points, self.dimensions), dtype=np.int64)
        remaining = indices.copy()
        
        for dim in range(self.dimensions - 1, -1, -1):
            coordinates[:, dim] = remaining // self.multipliers[dim]
            remaining = remaining % self.multipliers[dim]
        
        # Return consistent shape - remove extra dimensions for single point
        if n_points == 1:
            return coordinates[0]
        return coordinates
