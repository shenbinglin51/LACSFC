"""
Base interface for query processors for space-filling curves
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, List, Dict, Any, Optional


class QueryWindow:
    """Class representing a query window in a 2D grid"""
    
    def __init__(self, x: int, y: int, width: int, height: int):
        """
        Initialize a query window
        
        Args:
            x (int): X-coordinate of the bottom-left corner
            y (int): Y-coordinate of the bottom-left corner
            width (int): Width of the window
            height (int): Height of the window
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    
    def __repr__(self) -> str:
        return f"QueryWindow(x={self.x}, y={self.y}, width={self.width}, height={self.height})"


class Interval:
    """Class representing a 1D interval on the space-filling curve"""
    
    def __init__(self, start: int, end: int):
        """
        Initialize an interval
        
        Args:
            start (int): Start index of the interval
            end (int): End index of the interval (inclusive)
        """
        self.start = start
        self.end = end
    
    def __repr__(self) -> str:
        return f"Interval({self.start}, {self.end})"
    
    def __lt__(self, other):
        """For sorting intervals by start index"""
        return self.start < other.start


class QueryProcessorBase(ABC):
    """Base class for all query processors"""
    
    def __init__(self, grid_size: int):
        """
        Initialize the query processor
        
        Args:
            grid_size (int): Size of the grid (assuming square grid)
        """
        self.grid_size = grid_size
        self.name = self.__class__.__name__
    
    @abstractmethod
    def process_query(self, query_window: QueryWindow) -> List[Interval]:
        """
        Process a query window and return a list of 1D intervals on the space-filling curve
        
        Args:
            query_window (QueryWindow): Query window in the 2D grid
            
        Returns:
            List[Interval]: List of intervals on the space-filling curve
        """
        pass
    
    def batch_process_queries(self, query_windows: List[QueryWindow]) -> Tuple[List[List[Interval]], int, float]:
        """
        Process a batch of query windows and return statistics
        
        Args:
            query_windows (List[QueryWindow]): List of query windows
            
        Returns:
            Tuple[List[List[Interval]], int, float]: 
                - List of lists of intervals for each query
                - Total number of segments
                - Processing time in milliseconds
        """
        import time
        
        start_time = time.time()
        all_intervals = []
        total_segments = 0
        
        for query_window in query_windows:
            intervals = self.process_query(query_window)
            all_intervals.append(intervals)
            total_segments += len(intervals)
        
        end_time = time.time()
        processing_time_ms = (end_time - start_time) * 1000
        
        return all_intervals, total_segments, processing_time_ms
    
    def read_query_windows_from_csv(self, filename: str) -> List[QueryWindow]:
        """
        Read query windows from a CSV file
        
        Args:
            filename (str): Path to the CSV file
            
        Returns:
            List[QueryWindow]: List of query windows
        """
        import csv
        
        query_windows = []
        
        try:
            with open(filename, 'r') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    x = int(row[0])
                    y = int(row[1])
                    width = int(row[2])
                    height = int(row[3])
                    query_windows.append(QueryWindow(x, y, width, height))
        except Exception as e:
            print(f"Error reading CSV file: {e}")
        
        return query_windows
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata for the query processor
        
        Returns:
            Dict[str, Any]: Dictionary containing metadata
        """
        return {
            "name": self.name,
            "grid_size": self.grid_size,
        }
