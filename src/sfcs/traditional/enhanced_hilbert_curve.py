"""
Enhanced Hilbert curve implementation using the precise C++ algorithm
"""
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from ..sfc_base import TraditionalSFC
from ..query_processors.enhanced_hilbert_wrapper import EnhancedHilbertProcessor
from ..query_processors.query_processor_base import QueryWindow

class EnhancedHilbertCurve(TraditionalSFC):
    """Enhanced Hilbert space-filling curve with precise C++ implementation"""
    
    def __init__(self, dimensions, grid_size):
        """
        初始化增强Hilbert曲线
        
        参数:
            dimensions: 维度数（目前只支持2D）
            grid_size: 网格大小元组，例如(256, 256)
        """
        super().__init__(dimensions, grid_size)
        
        if dimensions != 2:
            raise ValueError("增强Hilbert曲线目前只支持2维空间")
        
        # 确保网格大小是2的幂次方且为正方形
        if grid_size[0] != grid_size[1]:
            raise ValueError("增强Hilbert曲线要求正方形网格")
        
        size = grid_size[0]
        if size & (size - 1) != 0:
            raise ValueError("网格大小必须是2的幂次方")
        
        self.size = size
        self.order = int(np.log2(size))
        
        # 初始化C++处理器
        try:
            self.processor = EnhancedHilbertProcessor(self.order)
        except FileNotFoundError:
            print("Warning: C++ library not found. Please compile first using 'make' in query_processors directory.")
            self.processor = None
    
    def get_name(self):
        """返回曲线名称"""
        return "EnhancedHilbertCurve"
    
    def encode(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Encode 2D coordinates to Hilbert curve indices
        使用简单的行主序作为fallback，主要功能通过query_segments实现
        """
        if coordinates.shape[1] != 2:
            raise ValueError("Enhanced Hilbert curve only supports 2D coordinates")
        
        indices = np.zeros(coordinates.shape[0], dtype=np.int64)
        
        for i, (x, y) in enumerate(coordinates):
            # 简单的行主序编码作为fallback
            indices[i] = int(y) * self.size + int(x)
        
        return indices
    
    def decode(self, indices: np.ndarray) -> np.ndarray:
        """
        Decode Hilbert curve indices to 2D coordinates
        使用简单的行主序作为fallback
        """
        coordinates = np.zeros((len(indices), 2), dtype=np.int32)
        
        for i, d in enumerate(indices):
            d = int(d)
            y = d // self.size
            x = d % self.size
            coordinates[i] = [x, y]
        
        return coordinates
    
    def get_query_segments(self, query_window: Dict[str, Any]) -> List[Tuple[int, int]]:
        """
        Get query segments using the enhanced C++ implementation
        
        Args:
            query_window: Dictionary with keys 'min_coords' and 'max_coords'
            
        Returns:
            List of (start, end) segment tuples
        """
        if self.processor is None:
            # Fallback to simple implementation
            return self._get_query_segments_fallback(query_window)
        
        min_coords = query_window['min_coords']
        max_coords = query_window['max_coords']
        
        x = int(min_coords[0])
        y = int(min_coords[1])
        width = int(max_coords[0] - min_coords[0] + 1)
        height = int(max_coords[1] - min_coords[1] + 1)
        
        # Create query window object
        qw = QueryWindow(x, y, width, height)
        
        # Get intervals from C++ processor
        intervals = self.processor.process_query(qw)
        
        # Convert to segment tuples
        segments = [(interval.start, interval.end) for interval in intervals]
        
        return segments
    
    def _get_query_segments_fallback(self, query_window: Dict[str, Any]) -> List[Tuple[int, int]]:
        """
        Fallback implementation when C++ library is not available
        """
        min_coords = query_window['min_coords']
        max_coords = query_window['max_coords']
        
        # Simple implementation: collect all points and find segments
        points = []
        for x in range(int(min_coords[0]), int(max_coords[0]) + 1):
            for y in range(int(min_coords[1]), int(max_coords[1]) + 1):
                if 0 <= x < self.actual_grid_size and 0 <= y < self.actual_grid_size:
                    points.append(self._xy2d(x, y))
        
        if not points:
            return []
        
        # Sort points and find consecutive segments
        points.sort()
        segments = []
        start = points[0]
        end = points[0]
        
        for i in range(1, len(points)):
            if points[i] == end + 1:
                end = points[i]
            else:
                segments.append((start, end))
                start = end = points[i]
        
        segments.append((start, end))
        return segments
    
    def get_metadata(self):
        """返回元数据"""
        return {
            "name": self.get_name(),
            "dimensions": self.dimensions,
            "grid_size": self.grid_size,
            "size": self.size,
            "order": self.order,
            "processor_available": self.processor is not None
        }
