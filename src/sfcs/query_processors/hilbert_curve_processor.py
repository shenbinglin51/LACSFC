"""
Query processor for Hilbert space-filling curve
This implements the window query processing algorithm for Hilbert curves
"""
import numpy as np
from typing import List, Optional, Tuple
import time
from enum import Enum
from .query_processor_base import QueryProcessorBase, QueryWindow, Interval


class QuadrantType(Enum):
    """Enum representing quadrant types in Hilbert curve"""
    A = 0
    B = 1
    C = 2
    D = 3


class HilbertCurveProcessor(QueryProcessorBase):
    """Query processor for Hilbert space-filling curve"""
    
    def __init__(self, grid_size: int, bits: int = None):
        """
        Initialize the Hilbert curve query processor
        
        Args:
            grid_size (int): Size of the grid (assuming square grid)
            bits (int, optional): Number of bits/order of the Hilbert curve.
                                 If None, calculated from grid_size.
        """
        super().__init__(grid_size)
        self.name = "HilbertCurveProcessor"
        
        # Calculate bits if not provided
        if bits is None:
            self.bits = int(np.ceil(np.log2(grid_size)))
        else:
            self.bits = bits
            
        # Actual grid size based on bits (must be a power of 2)
        self.actual_grid_size = 2 ** self.bits
    
    def determine_type(self, x: int, y: int, p: int, q: int, T: int) -> int:
        """
        Determine the type of the quadrant intersection with the query window
        
        Args:
            x (int): X-coordinate of the query window
            y (int): Y-coordinate of the query window
            p (int): Height of the query window
            q (int): Width of the query window
            T (int): Size of the current grid
            
        Returns:
            int: Type code (0-8)
        """
        half_T = T // 2
        
        cond0 = (x < half_T and y < half_T and (x + q) >= half_T and (y + p) >= half_T)
        cond1 = (x < half_T and y >= half_T and (x + q) >= half_T and (y + p) >= half_T)
        cond2 = (x < half_T and y < half_T and (x + q) >= half_T and (y + p) < half_T)
        cond3 = (x < half_T and y < half_T and (x + q) < half_T and (y + p) >= half_T)
        cond4 = (x >= half_T and y < half_T and (x + q) >= half_T and (y + p) >= half_T)
        cond5 = (x < half_T and y >= half_T and (x + q) < half_T and (y + p) >= half_T)
        cond6 = (x >= half_T and y >= half_T and (x + q) >= half_T and (y + p) >= half_T)
        cond7 = (x < half_T and y < half_T and (x + q) < half_T and (y + p) < half_T)
        cond8 = (x >= half_T and y < half_T and (x + q) >= half_T and (y + p) < half_T)
        
        if cond0:
            return 0
        elif cond1:
            return 1
        elif cond2:
            return 2
        elif cond3:
            return 3
        elif cond4:
            return 4
        elif cond5:
            return 5
        elif cond6:
            return 6
        elif cond7:
            return 7
        elif cond8:
            return 8
        else:
            return -1  # Undefined type
    
    def process_query(self, query_window: QueryWindow) -> List[Interval]:
        """
        Process a query window and return a list of 1D intervals on the Hilbert curve
        
        Args:
            query_window (QueryWindow): Query window in the 2D grid
            
        Returns:
            List[Interval]: List of intervals on the Hilbert curve
        """
        # Store segments as we identify them
        self.segments = []
        
        # Start recursive quadrant splitting with initial type A
        self.q_split(QuadrantType.A, self.actual_grid_size, 0, 
                    query_window.x, query_window.y, 
                    query_window.height, query_window.width)
        
        # Convert segments to Interval objects
        intervals = [Interval(seg.start, seg.end) for seg in self.segments]
        
        return intervals
    
    def add_sequence(self, mins: int, maxs: int) -> None:
        """
        Add a sequence to the segments list, merging with the previous segment if adjacent
        
        Args:
            mins (int): Start index of the sequence
            maxs (int): End index of the sequence
        """
        if not self.segments:
            self.segments.append(Interval(mins, maxs))
        else:
            if self.segments[-1].end + 1 == mins:
                self.segments[-1].end = max(self.segments[-1].end, maxs)
            else:
                self.segments.append(Interval(mins, maxs))
    
    def q_split(self, quadrant_type: QuadrantType, T: int, mino: int, x: int, y: int, p: int, q: int) -> None:
        """
        Recursively split quadrants to find segments on the Hilbert curve
        
        Args:
            quadrant_type (QuadrantType): Type of the current quadrant (A, B, C, D)
            T (int): Size of the current grid
            mino (int): Minimum index in the current quadrant
            x (int): X-coordinate of the query window
            y (int): Y-coordinate of the query window
            p (int): Height of the query window
            q (int): Width of the query window
        """
        # Stop recursion at leaf nodes
        if T < 1:
            return
            
        # Add the entire quadrant if it's completely covered by the query window
        if T == p and T == q:
            self.add_sequence(mino, mino + T * T - 1)
            return
        
        # Dispatch to the appropriate quadrant handling function
        if quadrant_type == QuadrantType.A:
            self.q_split_a(T, mino, x, y, p, q)
        elif quadrant_type == QuadrantType.B:
            self.q_split_b(T, mino, x, y, p, q)
        elif quadrant_type == QuadrantType.C:
            self.q_split_c(T, mino, x, y, p, q)
        elif quadrant_type == QuadrantType.D:
            self.q_split_d(T, mino, x, y, p, q)
    
    def q_split_a(self, T: int, mino: int, x: int, y: int, p: int, q: int) -> None:
        """
        Split type A quadrant
        
        Args:
            T (int): Size of the current grid
            mino (int): Minimum index in the current quadrant
            x (int): X-coordinate of the query window
            y (int): Y-coordinate of the query window
            p (int): Height of the query window
            q (int): Width of the query window
        """
        half_T = T // 2
        quarter_cells = half_T * half_T
        
        type_code = self.determine_type(x, y, p, q, T)
        
        if type_code == 0:
            self.q_split(QuadrantType.B, half_T, mino, x, y, half_T - y, half_T - x)
            self.q_split(QuadrantType.A, half_T, mino + quarter_cells, x, 0, y + p - half_T, half_T - x)
            self.q_split(QuadrantType.A, half_T, mino + quarter_cells * 2, 0, 0, y + p - half_T, x + q - half_T)
            self.q_split(QuadrantType.D, half_T, mino + quarter_cells * 3, 0, y, half_T - y, x + q - half_T)
        elif type_code == 1:
            self.q_split(QuadrantType.A, half_T, mino + quarter_cells, x, y - half_T, p, half_T - x)
            self.q_split(QuadrantType.A, half_T, mino + quarter_cells * 2, 0, y - half_T, p, x + q - half_T)
        elif type_code == 2:
            self.q_split(QuadrantType.B, half_T, mino, x, y, p, half_T - x)
            self.q_split(QuadrantType.D, half_T, mino + quarter_cells * 3, 0, y, p, x + q - half_T)
        elif type_code == 3:
            self.q_split(QuadrantType.B, half_T, mino, x, y, half_T - y, q)
            self.q_split(QuadrantType.A, half_T, mino + quarter_cells, x, 0, y + p - half_T, q)
        elif type_code == 4:
            self.q_split(QuadrantType.A, half_T, mino + quarter_cells * 2, x - half_T, 0, y + p - half_T, q)
            self.q_split(QuadrantType.D, half_T, mino + quarter_cells * 3, x - half_T, y, half_T - y, q)
        elif type_code == 5:
            self.q_split(QuadrantType.A, half_T, mino + quarter_cells, x, y - half_T, p, q)
        elif type_code == 6:
            self.q_split(QuadrantType.A, half_T, mino + quarter_cells * 2, x - half_T, y - half_T, p, q)
        elif type_code == 7:
            self.q_split(QuadrantType.B, half_T, mino, x, y, p, q)
        elif type_code == 8:
            self.q_split(QuadrantType.D, half_T, mino + quarter_cells * 3, x - half_T, y, p, q)
    
    def q_split_b(self, T: int, mino: int, x: int, y: int, p: int, q: int) -> None:
        """
        Split type B quadrant
        
        Args:
            T (int): Size of the current grid
            mino (int): Minimum index in the current quadrant
            x (int): X-coordinate of the query window
            y (int): Y-coordinate of the query window
            p (int): Height of the query window
            q (int): Width of the query window
        """
        half_T = T // 2
        quarter_cells = half_T * half_T
        
        type_code = self.determine_type(x, y, p, q, T)
        
        if type_code == 0:
            self.q_split(QuadrantType.A, half_T, mino, x, y, half_T - y, half_T - x)
            self.q_split(QuadrantType.B, half_T, mino + quarter_cells, 0, y, half_T - y, x + q - half_T)
            self.q_split(QuadrantType.B, half_T, mino + quarter_cells * 2, 0, 0, y + p - half_T, x + q - half_T)
            self.q_split(QuadrantType.C, half_T, mino + quarter_cells * 3, x, 0, y + p - half_T, half_T - x)
        elif type_code == 1:
            self.q_split(QuadrantType.B, half_T, mino + quarter_cells * 2, 0, y - half_T, p, x + q - half_T)
            self.q_split(QuadrantType.C, half_T, mino + quarter_cells * 3, x, y - half_T, p, half_T - x)
        elif type_code == 2:
            self.q_split(QuadrantType.A, half_T, mino, x, y, p, half_T - x)
            self.q_split(QuadrantType.B, half_T, mino + quarter_cells, 0, y, p, x + q - half_T)
        elif type_code == 3:
            self.q_split(QuadrantType.A, half_T, mino, x, y, half_T - y, q)
            self.q_split(QuadrantType.C, half_T, mino + quarter_cells * 3, x, 0, y + p - half_T, q)
        elif type_code == 4:
            self.q_split(QuadrantType.B, half_T, mino + quarter_cells, x - half_T, y, half_T - y, q)
            self.q_split(QuadrantType.B, half_T, mino + quarter_cells * 2, x - half_T, 0, y + p - half_T, q)
        elif type_code == 5:
            self.q_split(QuadrantType.C, half_T, mino + quarter_cells * 3, x, y - half_T, p, q)
        elif type_code == 6:
            self.q_split(QuadrantType.B, half_T, mino + quarter_cells * 2, x - half_T, y - half_T, p, q)
        elif type_code == 7:
            self.q_split(QuadrantType.A, half_T, mino, x, y, p, q)
        elif type_code == 8:
            self.q_split(QuadrantType.B, half_T, mino + quarter_cells, x - half_T, y, p, q)
    
    def q_split_c(self, T: int, mino: int, x: int, y: int, p: int, q: int) -> None:
        """
        Split type C quadrant
        
        Args:
            T (int): Size of the current grid
            mino (int): Minimum index in the current quadrant
            x (int): X-coordinate of the query window
            y (int): Y-coordinate of the query window
            p (int): Height of the query window
            q (int): Width of the query window
        """
        half_T = T // 2
        quarter_cells = half_T * half_T
        
        type_code = self.determine_type(x, y, p, q, T)
        
        if type_code == 0:
            self.q_split(QuadrantType.D, half_T, mino, 0, 0, y + p - half_T, x + q - half_T)
            self.q_split(QuadrantType.C, half_T, mino + quarter_cells, 0, y, half_T - y, x + q - half_T)
            self.q_split(QuadrantType.C, half_T, mino + quarter_cells * 2, x, y, half_T - y, half_T - x)
            self.q_split(QuadrantType.B, half_T, mino + quarter_cells * 3, x, 0, y + p - half_T, half_T - x)
        elif type_code == 1:
            self.q_split(QuadrantType.D, half_T, mino, 0, y - half_T, p, x + q - half_T)
            self.q_split(QuadrantType.B, half_T, mino + quarter_cells * 3, x, y - half_T, p, half_T - x)
        elif type_code == 2:
            self.q_split(QuadrantType.C, half_T, mino + quarter_cells, 0, y, p, x + q - half_T)
            self.q_split(QuadrantType.C, half_T, mino + quarter_cells * 2, x, y, p, half_T - x)
        elif type_code == 3:
            self.q_split(QuadrantType.C, half_T, mino + quarter_cells * 2, x, y, half_T - y, q)
            self.q_split(QuadrantType.B, half_T, mino + quarter_cells * 3, x, 0, y + p - half_T, q)
        elif type_code == 4:
            self.q_split(QuadrantType.D, half_T, mino, x - half_T, 0, y + p - half_T, q)
            self.q_split(QuadrantType.C, half_T, mino + quarter_cells, x - half_T, y, half_T - y, q)
        elif type_code == 5:
            self.q_split(QuadrantType.B, half_T, mino + quarter_cells * 3, x, y - half_T, p, q)
        elif type_code == 6:
            self.q_split(QuadrantType.D, half_T, mino, x - half_T, y - half_T, p, q)
        elif type_code == 7:
            self.q_split(QuadrantType.C, half_T, mino + quarter_cells * 2, x, y, p, q)
        elif type_code == 8:
            self.q_split(QuadrantType.C, half_T, mino + quarter_cells, x - half_T, y, p, q)
    
    def q_split_d(self, T: int, mino: int, x: int, y: int, p: int, q: int) -> None:
        """
        Split type D quadrant
        
        Args:
            T (int): Size of the current grid
            mino (int): Minimum index in the current quadrant
            x (int): X-coordinate of the query window
            y (int): Y-coordinate of the query window
            p (int): Height of the query window
            q (int): Width of the query window
        """
        half_T = T // 2
        quarter_cells = half_T * half_T
        
        type_code = self.determine_type(x, y, p, q, T)
        
        if type_code == 0:
            self.q_split(QuadrantType.C, half_T, mino, 0, 0, y + p - half_T, x + q - half_T)
            self.q_split(QuadrantType.D, half_T, mino + quarter_cells, x, 0, y + p - half_T, half_T - x)
            self.q_split(QuadrantType.D, half_T, mino + quarter_cells * 2, x, y, half_T - y, half_T - x)
            self.q_split(QuadrantType.A, half_T, mino + quarter_cells * 3, 0, y, half_T - y, x + q - half_T)
        elif type_code == 1:
            self.q_split(QuadrantType.C, half_T, mino, 0, y - half_T, p, x + q - half_T)
            self.q_split(QuadrantType.D, half_T, mino + quarter_cells, x, y - half_T, p, half_T - x)
        elif type_code == 2:
            self.q_split(QuadrantType.D, half_T, mino + quarter_cells * 2, x, y, p, half_T - x)
            self.q_split(QuadrantType.A, half_T, mino + quarter_cells * 3, 0, y, p, x + q - half_T)
        elif type_code == 3:
            self.q_split(QuadrantType.D, half_T, mino + quarter_cells, x, 0, y + p - half_T, q)
            self.q_split(QuadrantType.D, half_T, mino + quarter_cells * 2, x, y, half_T - y, q)
        elif type_code == 4:
            self.q_split(QuadrantType.C, half_T, mino, x - half_T, 0, y + p - half_T, q)
            self.q_split(QuadrantType.A, half_T, mino + quarter_cells * 3, x - half_T, y, half_T - y, q)
        elif type_code == 5:
            self.q_split(QuadrantType.D, half_T, mino + quarter_cells, x, y - half_T, p, q)
        elif type_code == 6:
            self.q_split(QuadrantType.C, half_T, mino, x - half_T, y - half_T, p, q)
        elif type_code == 7:
            self.q_split(QuadrantType.D, half_T, mino + quarter_cells * 2, x, y, p, q)
        elif type_code == 8:
            self.q_split(QuadrantType.A, half_T, mino + quarter_cells * 3, x - half_T, y, p, q)
