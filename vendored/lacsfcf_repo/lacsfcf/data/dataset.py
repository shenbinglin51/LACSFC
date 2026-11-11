"""Point dataset support for continuous multi-dimensional data."""
from typing import List, Optional
import numpy as np

from lacsfcf.data.grid import Grid, Cell
from lacsfcf.data.query_workload import Query


class PointDataset:
    """Multi-dimensional point dataset D.
    
    Represents a collection of n points in d-dimensional space, where each
    point x = (x^(1), ..., x^(d)) ∈ [0,1]^d (unit hypercube).
    
    This class bridges continuous point data with the discrete grid model,
    enabling range queries on continuous data using grid-based SFC indexing.
    
    Attributes:
        points: (n, d) array of continuous coordinates ∈ [0,1]^d
        grid: Grid object defining the quantization
        cells: List of quantized cell coordinates for each point
        n: Number of points
        d: Dimensionality
    """
    
    def __init__(self, points: np.ndarray, grid: Grid):
        """Initialize point dataset.
        
        Args:
            points: (n, d) numpy array, each point ∈ [0,1]^d
            grid: Grid object for quantization
            
        Raises:
            ValueError: If point dimensions don't match grid dimensions
        """
        if points.ndim != 2:
            raise ValueError(f"Points must be 2D array, got shape {points.shape}")
        
        n, d = points.shape
        if d != grid.d:
            raise ValueError(f"Point dimension {d} != grid dimension {grid.d}")
        
        # Validate points are in [0,1]^d
        if np.any(points < 0) or np.any(points > 1):
            raise ValueError("All points must be in [0,1]^d")
        
        self.points = points
        self.grid = grid
        self.n = n
        self.d = d
        
        # Quantize all points to cells
        self.cells = [grid.quantize_point(p) for p in points]
    
    def range_query(self, query: Query) -> List[int]:
        """Execute range query and return indices of matching points.
        
        Returns all points that fall within the query window.
        
        Args:
            query: Query object with lo and hi bounds
            
        Returns:
            List of point indices that satisfy the query
            
        Example:
            >>> points = np.random.rand(1000, 2)
            >>> grid = Grid(d=2, K=4)
            >>> dataset = PointDataset(points, grid)
            >>> query = Query(lo=(3, 4), hi=(8, 11))
            >>> indices = dataset.range_query(query)
            >>> result_points = dataset.points[indices]
        """
        result = []
        for i, cell in enumerate(self.cells):
            if query.hits(cell):
                result.append(i)
        return result
    
    def query_result_points(self, query: Query) -> np.ndarray:
        """Return actual point coordinates that satisfy the query.
        
        Args:
            query: Query object
            
        Returns:
            (m, d) array of points that satisfy the query, where m ≤ n
        """
        indices = self.range_query(query)
        if len(indices) == 0:
            return np.empty((0, self.d), dtype=self.points.dtype)
        return self.points[indices]
    
    def query_count(self, query: Query) -> int:
        """Count number of points that satisfy the query.
        
        Args:
            query: Query object
            
        Returns:
            Number of points in the query window
        """
        return len(self.range_query(query))
    
    def build_sfc_index(self, code_map: np.ndarray) -> np.ndarray:
        """Build SFC-based index for the points.
        
        Maps each point to its position in the space-filling curve.
        
        Args:
            code_map: SFC code map from grid cells to curve positions
            
        Returns:
            (n,) array where result[i] is the SFC position of point i
            
        Example:
            >>> # After generating SFC
            >>> sfc_positions = dataset.build_sfc_index(code_map)
            >>> # Sort points by SFC order
            >>> sorted_indices = np.argsort(sfc_positions)
            >>> sorted_points = dataset.points[sorted_indices]
        """
        sfc_positions = np.zeros(self.n, dtype=np.int32)
        for i, cell in enumerate(self.cells):
            sfc_positions[i] = code_map[cell]
        return sfc_positions
    
    def get_cell_density(self) -> np.ndarray:
        """Compute point density per cell.
        
        Returns:
            Array of shape grid.grid_shape with point counts per cell
        """
        density = np.zeros(self.grid.grid_shape, dtype=np.int32)
        for cell in self.cells:
            density[cell] += 1
        return density
    
    @staticmethod
    def generate_random(n: int, grid: Grid, seed: Optional[int] = None) -> 'PointDataset':
        """Generate random point dataset uniformly in [0,1]^d.
        
        Args:
            n: Number of points
            grid: Grid object
            seed: Random seed
            
        Returns:
            PointDataset with n random points
        """
        rng = np.random.default_rng(seed)
        points = rng.uniform(0.0, 1.0, size=(n, grid.d))
        return PointDataset(points, grid)
    
    @staticmethod
    def generate_clustered(
        n: int,
        grid: Grid,
        num_clusters: int = 5,
        cluster_std: float = 0.1,
        seed: Optional[int] = None
    ) -> 'PointDataset':
        """Generate clustered point dataset.
        
        Creates points around random cluster centers with Gaussian distribution.
        
        Args:
            n: Number of points
            grid: Grid object
            num_clusters: Number of cluster centers
            cluster_std: Standard deviation of clusters
            seed: Random seed
            
        Returns:
            PointDataset with clustered points
        """
        rng = np.random.default_rng(seed)
        
        # Generate cluster centers
        centers = rng.uniform(0.2, 0.8, size=(num_clusters, grid.d))
        
        # Assign points to clusters
        points = []
        for _ in range(n):
            center = centers[rng.integers(0, num_clusters)]
            point = rng.normal(center, cluster_std)
            # Clip to [0,1]^d
            point = np.clip(point, 0.0, 1.0)
            points.append(point)
        
        return PointDataset(np.array(points), grid)
    
    def __len__(self) -> int:
        """Return number of points."""
        return self.n
    
    def __repr__(self) -> str:
        return f"PointDataset(n={self.n}, d={self.d}, grid={self.grid.grid_side}^{self.grid.d})"
