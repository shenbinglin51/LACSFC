"""
Base interface for Space-Filling Curves (SFCs)
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, List, Dict, Any, Optional


class SFCBase(ABC):
    """Base class for all Space-Filling Curves (SFCs)"""
    
    def __init__(self, dimensions: int, grid_size: Tuple[int, ...]):
        """
        Initialize the SFC
        
        Args:
            dimensions (int): Number of dimensions
            grid_size (Tuple[int, ...]): Size of the grid in each dimension
        """
        self.dimensions = dimensions
        self.grid_size = grid_size
        self.name = self.__class__.__name__

    @abstractmethod
    def encode(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Convert multi-dimensional coordinates to 1D indices
        
        Args:
            coordinates (np.ndarray): Array of multi-dimensional coordinates
                                     shape: (n_points, dimensions)
        
        Returns:
            np.ndarray: 1D indices, shape: (n_points,)
        """
        pass
    
    @abstractmethod
    def decode(self, indices: np.ndarray) -> np.ndarray:
        """
        Convert 1D indices to multi-dimensional coordinates
        
        Args:
            indices (np.ndarray): Array of 1D indices, shape: (n_points,)
            
        Returns:
            np.ndarray: Multi-dimensional coordinates, shape: (n_points, dimensions)
        """
        pass
    
    def get_traversal_order(self) -> np.ndarray:
        """
        Get the traversal order of the SFC
        
        Returns:
            np.ndarray: Array of coordinates in traversal order
                       shape: (total_points, dimensions)
        """
        # Generate all grid points
        grid_points = np.array(np.meshgrid(*[range(size) for size in self.grid_size]))
        grid_points = grid_points.reshape(self.dimensions, -1).T
        
        # Encode all points
        indices = self.encode(grid_points)
        
        # Sort points by index
        sorted_indices = np.argsort(indices)
        traversal_order = grid_points[sorted_indices]
        
        return traversal_order
    
    def get_permutation(self) -> np.ndarray:
        """
        Get the permutation vector of the SFC
        
        Returns:
            np.ndarray: Permutation vector, shape: (total_points,)
        """
        # Generate all grid points in row-major order
        grid_points = np.array(np.meshgrid(*[range(size) for size in self.grid_size]))
        grid_points = grid_points.reshape(self.dimensions, -1).T
        
        # Encode all points
        indices = self.encode(grid_points)
        
        # Get permutation
        permutation = np.argsort(indices)
        
        return permutation
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata for the SFC
        
        Returns:
            Dict[str, Any]: Dictionary containing metadata
        """
        return {
            "name": self.name,
            "dimensions": self.dimensions,
            "grid_size": self.grid_size,
        }


class TraditionalSFC(SFCBase):
    """Base class for traditional Space-Filling Curves"""
    
    def __init__(self, dimensions: int, grid_size: Tuple[int, ...]):
        super().__init__(dimensions, grid_size)
        self.sfc_type = "traditional"


class LearnedSFC(SFCBase):
    """Base class for learned Space-Filling Curves"""
    
    def __init__(self, dimensions: int, grid_size: Tuple[int, ...], **kwargs):
        super().__init__(dimensions, grid_size)
        self.sfc_type = "learned"
        self.model = None
        self.trained = False
    
    @abstractmethod
    def train(self, query_workload: np.ndarray, **kwargs) -> None:
        """
        Train the learned SFC based on query workload
        
        Args:
            query_workload (np.ndarray): Query workload data
            **kwargs: Additional training parameters
        """
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata for the learned SFC
        
        Returns:
            Dict[str, Any]: Dictionary containing metadata
        """
        metadata = super().get_metadata()
        metadata.update({
            "sfc_type": self.sfc_type,
            "trained": self.trained,
            # Add more learned SFC specific metadata
        })
        return metadata
