"""
Implementation of Hilbert space-filling curve using optimized bit operations
"""
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from ..sfc_base import TraditionalSFC


class HilbertCell:
    """ Hilbert Space 单元结构 """
    def __init__(self):
        self.hilbertCode = -1
        self.color = 0
        self.entryDirection = 0
        self.v = [[0] * 32 for _ in range(2)]  # v 数组存储 Hilbert 变换状态


class HilbertCurve(TraditionalSFC):
    """Hilbert space-filling curve with optimized implementation"""
    
    def __init__(self, dimensions: int, grid_size: Tuple[int, ...]):
        """
        Initialize the Hilbert curve SFC
        
        Args:
            dimensions (int): Number of dimensions
            grid_size (Tuple[int, ...]): Size of the grid in each dimension
        """
        super().__init__(dimensions, grid_size)
        self.name = "HilbertCurve"
        
        # Check for 2D only
        if dimensions != 2:
            raise ValueError("Hilbert curve currently only supports 2D")
        
        # Find the number of bits needed and adapt grid_size to power of 2
        max_size = max(grid_size)
        self.order = 0
        power_of_2 = 1
        while power_of_2 < max_size:
            self.order += 1
            power_of_2 *= 2
        
        # Initialize the Hilbert space
        self.hilbertSpace = self._initialize_hilbert_space(self.order, power_of_2)
        self.adjusted_grid_size = (power_of_2, power_of_2)
    
    def _compute_hilbert_code(self, x: int, y: int, r: int, 
                             hilbert_space: List[List[HilbertCell]]) -> None:
        """
        Calculate Hilbert Code for a point, using optimized bit operations
        
        Args:
            x (int): X coordinate
            y (int): Y coordinate
            r (int): Order of Hilbert curve (determines size as 2^r)
            hilbert_space (List[List[HilbertCell]]): Hilbert space grid to update
        """
        h0 = 0
        xBits = [(x >> k) & 1 for k in range(r)]  # Extract binary bits of x
        yBits = [(y >> k) & 1 for k in range(r)]  # Extract binary bits of y

        v = [[0] * 32 for _ in range(2)]  # Initialize v array
        v[1][r - 1] = 0
        v[0][r - 1] = 0

        hOrder = [0] * (2 * r)  # hOrder array stores Hilbert transform sequence
        hilbert_space[x][y].v[1][r - 1] = 0
        hilbert_space[x][y].v[0][r - 1] = 0

        for j in range(r - 1, 0, -1):
            xj_xor_yj = xBits[j] ^ yBits[j]
            not_xj = (~xBits[j]) & 1
            not_yj = (~yBits[j]) & 1
            v1j = v[1][j]
            v0j = v[0][j]
            not_v0j = (~v0j) & 1

            v[1][j - 1] = v1j * xj_xor_yj + (not_xj ^ yBits[j]) * (v0j ^ not_yj)
            v[0][j - 1] = v0j * (v1j ^ not_xj) + not_v0j * (v1j ^ not_yj)

            if (x & 1) == 0 and (y & 1) == 0:
                hilbert_space[x][y].v[1][j - 1] = v[1][j - 1]
                hilbert_space[x][y].v[0][j - 1] = v[0][j - 1]

        for k in range(r):
            v1k = v[1][k]
            v0k = v[0][k]
            not_v0k = (~v0k) & 1
            not_yk = (~yBits[k]) & 1

            hOrder[k * 2] = xBits[k] ^ yBits[k]
            hOrder[k * 2 + 1] = not_v0k * (v1k ^ xBits[k]) + v0k * (v1k ^ not_yk)

        for i in range(2 * r):
            h0 += hOrder[i] << i  # Use bit shift for calculation

        hilbert_space[x][y].hilbertCode = h0
        hilbert_space[x][y].color = 0
        hilbert_space[x][y].entryDirection = 0
    
    def _initialize_hilbert_space(self, r: int, grid_size: int) -> List[List[HilbertCell]]:
        """
        Initialize the Hilbert space
        
        Args:
            r (int): Order of Hilbert curve
            grid_size (int): Size of the grid (2^r)
            
        Returns:
            List[List[HilbertCell]]: Initialized Hilbert space
        """
        hilbert_space = [[HilbertCell() for _ in range(grid_size)] for _ in range(grid_size)]
        for i in range(grid_size):
            for j in range(grid_size):
                self._compute_hilbert_code(i, j, r, hilbert_space)
        return hilbert_space
    
    def encode(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Convert multi-dimensional coordinates to 1D indices using Hilbert order
        
        Args:
            coordinates (np.ndarray): Array of multi-dimensional coordinates
                                     shape: (n_points, dimensions)
        
        Returns:
            np.ndarray: 1D indices, shape: (n_points,)
        """
        # Check if coordinates are within bounds
        for dim in range(self.dimensions):
            if np.any(coordinates[:, dim] < 0) or np.any(coordinates[:, dim] >= self.grid_size[dim]):
                raise ValueError(f"Coordinates must be within bounds [0, {self.grid_size[dim]})")
        
        n_points = coordinates.shape[0]
        indices = np.zeros(n_points, dtype=np.int64)
        
        for i in range(n_points):
            x, y = coordinates[i]
            # Clip to adjusted grid size if necessary
            x_adj = min(int(x), self.adjusted_grid_size[0] - 1)
            y_adj = min(int(y), self.adjusted_grid_size[1] - 1)
            indices[i] = self.hilbertSpace[x_adj][y_adj].hilbertCode
            
        return indices
    
    def map_query_to_indices(self, query_window):
        """
        Map a query window to a set of indices using HilbertCurveProcessor with recursive quadrant decomposition.
        
        Args:
            query_window: A query window object with get_lower_bounds() and get_upper_bounds() methods
            
        Returns:
            A list of indices that fall within the query window
        """
        try:
            # 使用Hilbert Curve专用处理器
            from ..query_processors.hilbert_curve_processor import HilbertCurveProcessor
            
            if not hasattr(self, '_processor'):
                self._processor = HilbertCurveProcessor(self.grid_size)
            
            return self._processor.process_query(query_window, self)
            
        except ImportError:
            # 如果处理器不可用，使用简化的直接方法
            return self._fallback_query_processing(query_window)
        except Exception as e:
            print(f"Error in HilbertCurve map_query_to_indices: {str(e)}")
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
        """Convert 1D indices to multi-dimensional coordinates
        
        Args:
            indices (np.ndarray): 1D indices, shape: (n_points,)
            
        Returns:
            np.ndarray: Array of multi-dimensional coordinates, shape: (n_points, dimensions)
        """
        # Handle both single index and array of indices
        if np.isscalar(indices):
            indices = np.array([indices])
        elif indices.ndim == 0:
            indices = np.array([indices.item()])
        
        # Create mapping from Hilbert code to coordinates
        code_to_coords = {}
        for x in range(self.adjusted_grid_size[0]):
            for y in range(self.adjusted_grid_size[1]):
                code = self.hilbertSpace[x][y].hilbertCode
                code_to_coords[code] = (x, y)
        
        n_points = indices.size
        coordinates = np.zeros((n_points, self.dimensions), dtype=np.int64)
        
        for i in range(n_points):
            # Find the closest valid Hilbert code
            code = indices[i]
            if code in code_to_coords:
                coordinates[i] = code_to_coords[code]
            else:
                # Find the closest code
                valid_codes = np.array(list(code_to_coords.keys()))
                closest_idx = np.argmin(np.abs(valid_codes - code))
                coordinates[i] = code_to_coords[valid_codes[closest_idx]]
        
        # Return consistent shape - remove extra dimensions for single point
        if n_points == 1:
            return coordinates[0]
        return coordinates
