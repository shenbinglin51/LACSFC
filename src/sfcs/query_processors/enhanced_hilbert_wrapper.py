"""
Python wrapper for the enhanced C++ Hilbert curve query processor
"""
import ctypes
import os
import numpy as np
from typing import List, Tuple
from .query_processor_base import QueryProcessorBase, QueryWindow, Interval

class EnhancedHilbertProcessor(QueryProcessorBase):
    """Enhanced Hilbert curve query processor using C++ implementation"""
    
    def __init__(self, grid_size: int):
        """
        Initialize the enhanced Hilbert processor
        
        Args:
            grid_size (int): Size of the grid (must be power of 2)
        """
        super().__init__(grid_size)
        self.name = "EnhancedHilbertProcessor"
        
        # Load the C++ library
        lib_path = os.path.join(os.path.dirname(__file__), 'enhanced_hilbert_processor.so')
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"C++ library not found at {lib_path}. Please compile first.")
        
        self.lib = ctypes.CDLL(lib_path)
        
        # Define function signatures
        self.lib.create_processor.argtypes = [ctypes.c_int]
        self.lib.create_processor.restype = ctypes.c_void_p
        
        self.lib.delete_processor.argtypes = [ctypes.c_void_p]
        self.lib.delete_processor.restype = None
        
        self.lib.process_query.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.lib.process_query.restype = ctypes.c_int
        
        self.lib.clear_processor.argtypes = [ctypes.c_void_p]
        self.lib.clear_processor.restype = None
        
        # Create the processor instance
        self.processor = self.lib.create_processor(grid_size)
        
    def __del__(self):
        """Cleanup the C++ processor"""
        if hasattr(self, 'processor') and self.processor:
            self.lib.delete_processor(self.processor)
    
    def process_query(self, query_window: QueryWindow) -> List[Interval]:
        """
        Process a query window using the enhanced C++ implementation
        
        Args:
            query_window (QueryWindow): Query window in the 2D grid
            
        Returns:
            List[Interval]: List of intervals on the Hilbert curve
        """
        # Clear previous results
        self.lib.clear_processor(self.processor)
        
        # Process the query and get segment count
        segment_count = self.lib.process_query(
            self.processor,
            query_window.x,
            query_window.y,
            query_window.width,
            query_window.height
        )
        
        # For now, we return a simplified interval representation
        # In a full implementation, you would need to extend the C++ interface
        # to return the actual segment boundaries
        intervals = []
        for i in range(segment_count):
            # This is a placeholder - you would need to modify the C++ code
            # to return actual segment boundaries
            intervals.append(Interval(i * 100, (i + 1) * 100 - 1))
        
        return intervals
    
    def get_segment_count(self, query_window: QueryWindow) -> int:
        """
        Get the number of segments for a query window
        
        Args:
            query_window (QueryWindow): Query window in the 2D grid
            
        Returns:
            int: Number of segments
        """
        self.lib.clear_processor(self.processor)
        return self.lib.process_query(
            self.processor,
            query_window.x,
            query_window.y,
            query_window.width,
            query_window.height
        )
