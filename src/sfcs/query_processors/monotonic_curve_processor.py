"""
Query processor for monotonic space-filling curves (row-major, column-major, Z-curve, learned Z-curve)
This implements the window query processing algorithm for monotonic curves using quadtree decomposition
"""
import numpy as np
from typing import List, Tuple, Optional, cast
import time
from .query_processor_base import QueryProcessorBase, QueryWindow, Interval


class MaxBlock:
    """Class representing a maximum block in the quadtree decomposition"""
    
    def __init__(self, x: int, y: int, width: int):
        """
        Initialize a maximum block
        
        Args:
            x (int): X-coordinate of the bottom-left corner
            y (int): Y-coordinate of the bottom-left corner
            width (int): Width of the block (assuming square block)
        """
        self.x = x
        self.y = y
        self.width = width
    
    def __repr__(self) -> str:
        return f"MaxBlock(x={self.x}, y={self.y}, width={self.width})"


class QuadTreeNode:
    """Node in a quadtree for spatial decomposition"""
    
    def __init__(self, x: int, y: int, width: int, height: int):
        """
        Initialize a quadtree node
        
        Args:
            x (int): X-coordinate of the bottom-left corner
            y (int): Y-coordinate of the bottom-left corner
            width (int): Width of the node
            height (int): Height of the node
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.children = [None, None, None, None]  # SW, SE, NW, NE


class QuadTree:
    """Quadtree for spatial decomposition of query windows"""
    
    def __init__(self, width: int, height: int):
        """
        Initialize a quadtree
        
        Args:
            width (int): Width of the grid
            height (int): Height of the grid
        """
        self.root = QuadTreeNode(0, 0, width, height)
    
    def decompose_window(self, node: QuadTreeNode, query_x: int, query_y: int, 
                         query_width: int, query_height: int, max_blocks: List[MaxBlock]) -> None:
        """
        Decompose a query window into maximum blocks using quadtree
        
        Args:
            node (QuadTreeNode): Current quadtree node
            query_x (int): X-coordinate of the query window
            query_y (int): Y-coordinate of the query window
            query_width (int): Width of the query window
            query_height (int): Height of the query window
            max_blocks (List[MaxBlock]): List to store maximum blocks
        """
        # Stop at leaf node or if node is a unit square
        if node.width == 1 and node.height == 1:
            return
        
        center_x = node.x + node.width // 2
        center_y = node.y + node.height // 2
        
        # Check each quadrant
        for i in range(4):
            # Determine quadrant coordinates
            quad_x = node.x if i % 2 == 0 else center_x
            quad_y = node.y if i < 2 else center_y
            quad_width = node.width // 2
            quad_height = node.height // 2
            
            # Condition 1: No intersection with query window
            if (quad_x + quad_width <= query_x or quad_y + quad_height <= query_y or
                quad_x >= query_x + query_width or quad_y >= query_y + query_height):
                continue
            # Condition 3: Quadrant is fully contained in query window
            elif (quad_x >= query_x and quad_y >= query_y and
                  quad_x + quad_width <= query_x + query_width and quad_y + quad_height <= query_y + query_height):
                # Add to maximum blocks list
                max_blocks.append(MaxBlock(quad_x, quad_y, quad_width))
            # Condition 2: Partial intersection, continue recursion
            else:
                node.children[i] = QuadTreeNode(quad_x, quad_y, quad_width, quad_height)
                self.decompose_window(node.children[i], query_x, query_y, 
                                     query_width, query_height, max_blocks)


class MonotonicCurveProcessor(QueryProcessorBase):
    """Query processor for monotonic space-filling curves"""
    
    def __init__(self, grid_size: int, curve_type: str):
        """
        Initialize the monotonic curve query processor
        
        Args:
            grid_size (int): Size of the grid (assuming square grid)
            curve_type (str): Type of monotonic curve ('row_major', 'column_major', 'z_curve', 'learned_z_curve')
        """
        # Ensure grid_size is a power of 2 for proper quadtree decomposition
        if not (grid_size & (grid_size - 1) == 0) and grid_size != 0:
            # Round up to the next power of 2
            grid_size = 1 << (grid_size - 1).bit_length()
            
        super().__init__(grid_size)
        self.curve_type = curve_type
        self.name = f"{curve_type.replace('_', ' ').title()}Processor"
        
        # Calculate the number of bits needed to represent grid coordinates
        self.bits = int(np.ceil(np.log2(grid_size))) if grid_size > 0 else 0
    
    def compute_z_code(self, x: int, y: int) -> int:
        """
        Compute Z-order code (Morton code) for a 2D point
        
        Args:
            x (int): X-coordinate
            y (int): Y-coordinate
            
        Returns:
            int: Z-order code
        """
        z_code = 0
        
        for i in range(self.bits):
            bit_x = ((x >> i) & 1)
            bit_y = ((y >> i) & 1)
            
            z_code |= (bit_x << (2 * i))
            z_code |= (bit_y << (2 * i + 1))
        
        return z_code
    
    def compute_row_major_code(self, x: int, y: int) -> int:
        """
        Compute row-major code for a 2D point
        
        Args:
            x (int): X-coordinate
            y (int): Y-coordinate
            
        Returns:
            int: Row-major code
        """
        return y * self.grid_size + x
    
    def compute_column_major_code(self, x: int, y: int) -> int:
        """
        Compute column-major code for a 2D point
        
        Args:
            x (int): X-coordinate
            y (int): Y-coordinate
            
        Returns:
            int: Column-major code
        """
        return x * self.grid_size + y
    
    def compute_curve_range(self, block: MaxBlock) -> Tuple[int, int]:
        """
        Compute the range of curve indices for a maximum block
        
        Args:
            block (MaxBlock): Maximum block
            
        Returns:
            Tuple[int, int]: Min and max indices in the space-filling curve
        """
        x, y, width = block.x, block.y, block.width
        
        if self.curve_type == 'row_major':
            z_min = self.compute_row_major_code(x, y)
            z_max = self.compute_row_major_code(x + width - 1, y + width - 1)
        elif self.curve_type == 'column_major':
            z_min = self.compute_column_major_code(x, y)
            z_max = self.compute_column_major_code(x + width - 1, y + width - 1)
        else:  # z_curve or learned_z_curve (both use the same z-order encoding)
            z_min = self.compute_z_code(x, y)
            z_max = self.compute_z_code(x + width - 1, y + width - 1)
        
        return z_min, z_max
    
    def _process_row_major(self, query_window: QueryWindow) -> List[Interval]:
        """
        Process a query window for row-major order
        
        Args:
            query_window (QueryWindow): Query window in the 2D grid
            
        Returns:
            List[Interval]: List of intervals on the row-major curve
        """
        intervals = []
        for y in range(query_window.y, query_window.y + query_window.height):
            # Calculate the start and end indices for this row
            start = self.compute_row_major_code(query_window.x, y)
            end = self.compute_row_major_code(query_window.x + query_window.width - 1, y)
            intervals.append(Interval(start, end))
        return intervals
    
    def _process_column_major(self, query_window: QueryWindow) -> List[Interval]:
        """
        Process a query window for column-major order
        
        Args:
            query_window (QueryWindow): Query window in the 2D grid
            
        Returns:
            List[Interval]: List of intervals on the column-major curve
        """
        intervals = []
        for x in range(query_window.x, query_window.x + query_window.width):
            # Calculate the start and end indices for this column
            start = self.compute_column_major_code(x, query_window.y)
            end = self.compute_column_major_code(x, query_window.y + query_window.height - 1)
            intervals.append(Interval(start, end))
        return intervals
    
    def _process_with_quadtree(self, query_window: QueryWindow) -> List[Interval]:
        """
        Process a query window using quadtree decomposition (for Z-curve and learned Z-curve)
        
        Args:
            query_window (QueryWindow): Query window in the 2D grid
            
        Returns:
            List[Interval]: List of intervals on the space-filling curve
        """
        # Create a quadtree for decomposition
        quad_tree = QuadTree(self.grid_size, self.grid_size)
        max_blocks = []
        
        # Step 1: Decompose the query window using quadtree
        quad_tree.decompose_window(quad_tree.root, query_window.x, query_window.y,
                                 query_window.width, query_window.height, max_blocks)
        
        # Step 2: Compute curve ranges for each maximum block
        z_ranges = []
        for block in max_blocks:
            z_min, z_max = self.compute_curve_range(block)
            z_ranges.append((z_min, z_max))
        
        # Step 3: Sort and merge ranges
        z_ranges.sort()  # Sort by start index
        
        # Merge overlapping or adjacent intervals
        merged_ranges = []
        
        for z_range in z_ranges:
            if not merged_ranges:
                merged_ranges.append(z_range)
            else:
                prev_start, prev_end = merged_ranges[-1]
                curr_start, curr_end = z_range
                
                # If current range overlaps or is adjacent to previous range
                if curr_start <= prev_end + 1:
                    # Merge the ranges
                    merged_ranges[-1] = (prev_start, max(prev_end, curr_end))
                else:
                    # Add as a new range
                    merged_ranges.append(z_range)
        
        # Convert to Interval objects
        intervals = [Interval(start, end) for start, end in merged_ranges]
        
        return intervals
    
    def process_query(self, query_window: QueryWindow) -> List[Interval]:
        """
        Process a query window and return a list of 1D intervals on the space-filling curve
        
        Args:
            query_window (QueryWindow): Query window in the 2D grid
            
        Returns:
            List[Interval]: List of intervals on the space-filling curve
        """
        # Use specialized processing for row-major and column-major orders
        if self.curve_type == 'row_major':
            return self._process_row_major(query_window)
        elif self.curve_type == 'column_major':
            return self._process_column_major(query_window)
        else:
            # Use quadtree decomposition for Z-curve and learned Z-curve
            return self._process_with_quadtree(query_window)
