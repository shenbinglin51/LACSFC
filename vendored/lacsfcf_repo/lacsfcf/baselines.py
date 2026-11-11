"""Baseline SFC methods for comparison."""
import numpy as np
from lacsfcf.data.grid import Grid
from lacsfcf.metrics.range_query import mean_seg, mean_span


def row_major_sfc(grid: Grid) -> np.ndarray:
    """Generate row-major (Z-order) space-filling curve.
    
    For a 2D grid, this is simply scanning row by row:
      (0,0), (0,1), (0,2), ..., (0,N-1),
      (1,0), (1,1), (1,2), ..., (1,N-1),
      ...
      (N-1,0), (N-1,1), ..., (N-1,N-1)
    
    Returns:
        code_map: array of shape grid_shape mapping cell coords to position in curve
    """
    N = grid.grid_side
    d = grid.d
    
    code_map = np.zeros(grid.grid_shape, dtype=np.int32)
    
    if d == 2:
        # 2D row-major order
        pos = 0
        for i in range(N):
            for j in range(N):
                code_map[i, j] = pos
                pos += 1
    else:
        # General d-dimensional row-major order
        import itertools
        pos = 0
        for coords in itertools.product(range(N), repeat=d):
            code_map[coords] = pos
            pos += 1
    
    return code_map


def hilbert_sfc(grid: Grid) -> np.ndarray:
    """Generate Hilbert space-filling curve.
    
    Uses the hilbertcurve package to generate a standard Hilbert curve.
    For a 2D grid of size N×N, we use p iterations where 2^p = N.
    
    Returns:
        code_map: array of shape grid_shape mapping cell coords to position in curve
    """
    try:
        from hilbertcurve.hilbertcurve import HilbertCurve
    except ImportError:
        raise ImportError("Please install hilbertcurve: pip install hilbertcurve")
    
    N = grid.grid_side
    d = grid.d
    
    # Calculate p such that 2^p = N
    # For K=4: N=16, p=4
    # For K=5: N=32, p=5
    # For K=6: N=64, p=6
    p = grid.K
    
    # Verify that N = 2^p
    if 2**p != N:
        raise ValueError(f"Grid side {N} is not a power of 2 (2^{p} = {2**p})")
    
    # Create Hilbert curve object
    hilbert_curve = HilbertCurve(p, d)
    
    # Generate code_map
    code_map = np.zeros(grid.grid_shape, dtype=np.int32)
    
    if d == 2:
        # For 2D, iterate through all cells and get their Hilbert distance
        for i in range(N):
            for j in range(N):
                # Get Hilbert distance for this point
                distance = hilbert_curve.distance_from_coordinates([i, j])
                code_map[i, j] = distance
    else:
        # General d-dimensional case
        import itertools
        for coords in itertools.product(range(N), repeat=d):
            distance = hilbert_curve.distance_from_coordinates(list(coords))
            code_map[coords] = distance
    
    return code_map


def evaluate_baseline(grid: Grid, queries, method='row_major'):
    """Evaluate a baseline SFC method on given queries.
    
    Args:
        grid: Grid object
        queries: List of Query objects
        method: 'row_major' or 'hilbert'
    
    Returns:
        dict with mean_seg and mean_span
    """
    if method == 'row_major':
        code_map = row_major_sfc(grid)
    elif method == 'hilbert':
        code_map = hilbert_sfc(grid)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    seg = mean_seg(code_map, queries)
    span = mean_span(code_map, queries)
    
    return {
        'mean_seg': float(seg),
        'mean_span': float(span),
    }
