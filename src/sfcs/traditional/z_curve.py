"""
Implementation of Z-curve (Z-order) space-filling curve
"""
import numpy as np
from typing import Tuple
from ..sfc_base import TraditionalSFC


class ZCurve(TraditionalSFC):
    """Z-curve (Z-order) space-filling curve"""
    
    def __init__(self, dimensions: int, grid_size: Tuple[int, ...]):
        """
        Initialize the Z-curve SFC
        
        Args:
            dimensions (int): Number of dimensions
            grid_size (Tuple[int, ...]): Size of the grid in each dimension
        """
        super().__init__(dimensions, grid_size)
        self.name = "ZCurve"
        
        # Find the maximum number of bits needed
        self.max_bits = max([int(np.ceil(np.log2(size))) for size in grid_size])
        
    def _interleave_bits(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Interleave bits from coordinates to generate Z-values
        
        Args:
            coordinates (np.ndarray): Array of coordinates, shape: (n_points, dimensions)
            
        Returns:
            np.ndarray: Array of Z-values, shape: (n_points,)
        """
        # Ensure coordinates are integers
        coordinates = coordinates.astype(np.int64)
        n_points = coordinates.shape[0]
        z_values = np.zeros(n_points, dtype=np.int64)
        
        for bit in range(self.max_bits):
            for dim in range(self.dimensions):
                # Extract the bit at position 'bit' from each coordinate
                bit_val = (coordinates[:, dim] >> bit) & 1
                # Place it at the appropriate position in the Z-value
                z_values |= bit_val << (bit * self.dimensions + dim)
                
        return z_values
        
    def encode(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Convert multi-dimensional coordinates to 1D indices using Z-curve
        
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
                
        # Use bit interleaving to compute Z-values
        return self._interleave_bits(coordinates)
    
    def map_query_to_indices(self, query_window):
        """
        Map a query window to a set of indices using Z-Curve processor with quadtree decomposition.
        
        Args:
            query_window: A query window object with get_lower_bounds() and get_upper_bounds() methods
                         or lower_bounds and upper_bounds attributes
            
        Returns:
            A list of indices that fall within the query window
        """
        try:
            # 使用Z-Curve专用处理器
            from ..query_processors.z_curve_processor import ZCurveProcessor
            
            if not hasattr(self, '_processor'):
                self._processor = ZCurveProcessor(self.grid_size)
            
            return self._processor.process_query(query_window, self)
            
        except ImportError:
            # 如果处理器不可用，使用简化的直接方法
            return self._fallback_query_processing(query_window)
        except Exception as e:
            print(f"Error in Z-Curve map_query_to_indices: {str(e)}")
            return self._fallback_query_processing(query_window)
    
    def _fallback_query_processing(self, query_window):
        """备用查询处理方法"""
        try:
            # Handle both PaperCompliantQuery (attributes) and QueryWindow (methods)
            if hasattr(query_window, 'lower_bounds') and hasattr(query_window, 'upper_bounds'):
                min_bounds = query_window.lower_bounds
                max_bounds = query_window.upper_bounds
            else:
                min_bounds = query_window.get_lower_bounds()
                max_bounds = query_window.get_upper_bounds()
            
            # 正确转换为网格坐标
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
            
            # 枚举所有点并编码
            indices = []
            for x in range(min_x, max_x + 1):
                for y in range(min_y, max_y + 1):
                    point = np.array([[x, y]], dtype=np.int64)
                    index = self.encode(point)[0]
                    indices.append(int(index))
            
            return indices
            
        except Exception as e:
            print(f"Error in fallback query processing: {str(e)}")
            return []
    
    def decode(self, indices: np.ndarray) -> np.ndarray:
        """
        Convert 1D indices to multi-dimensional coordinates using Z-curve
        
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
        
        n_points = indices.size
        coordinates = np.zeros((n_points, self.dimensions), dtype=np.int64)
        
        for bit in range(self.max_bits):
            for dim in range(self.dimensions):
                # Extract the bit at position (bit * dimensions + dim) from Z-value
                bit_val = (indices >> (bit * self.dimensions + dim)) & 1
                # Place it at the appropriate position in the coordinate
                coordinates[:, dim] |= bit_val << bit
                
        # Check if coordinates are within bounds
        for dim in range(self.dimensions):
            if np.any(coordinates[:, dim] >= self.grid_size[dim]):
                raise ValueError(f"Decoded coordinates out of bounds in dimension {dim}")
        
        # Return consistent shape - remove extra dimensions for single point
        if n_points == 1:
            return coordinates[0]
        return coordinates
