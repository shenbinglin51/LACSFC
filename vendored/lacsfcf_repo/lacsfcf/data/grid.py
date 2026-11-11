"""d-dimensional grid management for LACSFCF."""
import itertools
from typing import Dict, List, Optional, Tuple

import numpy as np

Cell  = Tuple[int, ...]
Block = Tuple[int, ...]


class Grid:
    """Represents a d-dimensional grid with 2^K cells per dimension.

    Grid cells: coordinates in {0,...,2^K - 1}^d
    Blocks    : 2×...×2 hypercubes; coordinates in {0,...,2^(K-1) - 1}^d
    """

    def __init__(self, d: int, K: int):
        self.d          = d
        self.K          = K
        self.grid_side  = 2 ** K          # cells per dimension
        self.block_side = 2 ** (K - 1)    # blocks per dimension
        self.num_cells  = self.grid_side  ** d
        self.num_blocks = self.block_side ** d

        self.grid_shape  = tuple(self.grid_side  for _ in range(d))
        self.block_shape = tuple(self.block_side for _ in range(d))

        # Number of dual-graph edges, computed analytically so that Grid
        # construction stays O(1) even for huge grids (e.g. d=2 K=14 has
        # 134M edges -- materializing the tuple list is prohibitive). The
        # actual (b1, b2, dim) list and the (b1,b2)->idx map are built lazily
        # on first access via the ``dual_edges`` property / ``edge_index``.
        #   #edges = sum over dims of (#blocks with coord[k] < B-1)
        #         = d * B^(d-1) * (B-1)
        B = self.block_side
        self.num_dual_edges = d * (B ** (d - 1)) * (B - 1) if B > 0 else 0

        self._dual_edges: Optional[List[Tuple[Block, Block, int]]] = None
        self._dual_edge_index: Optional[Dict[Tuple[Block, Block], int]] = None

    # ------------------------------------------------------------------
    def _build_dual_edges(self):
        B = self.block_side
        d = self.d
        self._dual_edges = []
        self._dual_edge_index = {}
        idx = 0
        for block in itertools.product(*[range(B)] * d):
            for k in range(d):
                if block[k] < B - 1:
                    b1 = block
                    b2 = tuple(block[j] + 1 if j == k else block[j] for j in range(d))
                    self._dual_edges.append((b1, b2, k))
                    key = (b1, b2)
                    self._dual_edge_index[key] = idx
                    self._dual_edge_index[(b2, b1)] = idx
                    idx += 1
        # Consistency check against the analytic count from __init__.
        assert len(self._dual_edges) == self.num_dual_edges, (
            f"dual edge count mismatch: built {len(self._dual_edges)} "
            f"!= analytic {self.num_dual_edges}"
        )

    # ------------------------------------------------------------------
    @property
    def dual_edges(self) -> List[Tuple[Block, Block, int]]:
        """List of (b1, b2, dim) for all adjacent block pairs (lazy)."""
        if self._dual_edges is None:
            self._build_dual_edges()
        return self._dual_edges

    def edge_index(self, b1: Block, b2: Block) -> int:
        """Return the integer index of a dual edge (builds index lazily)."""
        if self._dual_edge_index is None:
            self._build_dual_edges()
        return self._dual_edge_index[(b1, b2)]

    # ------------------------------------------------------------------
    def cell_to_block(self, cell: Cell) -> Tuple[Block, Tuple[int, ...]]:
        """Return (block_coord, local_offset) for a cell."""
        block  = tuple(c // 2 for c in cell)
        offset = tuple(c % 2  for c in cell)
        return block, offset

    def block_cells(self, block: Block) -> List[Cell]:
        """Return all 2^d cells in the block (unordered)."""
        return [
            tuple(2 * block[k] + s[k] for k in range(self.d))
            for s in itertools.product(range(2), repeat=self.d)
        ]

    # ------------------------------------------------------------------
    def code_map_from_path(self, path: List[Cell]) -> np.ndarray:
        """Build code_map[coords] = SFC code (position in path).

        Returns ndarray of shape grid_shape with int32 dtype.
        """
        code = np.empty(self.grid_shape, dtype=np.int32)
        for t, cell in enumerate(path):
            code[cell] = t
        return code

    def path_from_code_map(self, code_map: np.ndarray) -> List[Cell]:
        """Recover ordered path from a code_map array."""
        n = self.num_cells
        idx = np.argsort(code_map.ravel())
        # Convert flat indices back to d-tuples
        path = []
        shape = self.grid_shape
        for flat in idx:
            cell = []
            rem  = int(flat)
            for k in range(self.d - 1, -1, -1):
                cell.append(rem % shape[k])
                rem //= shape[k]
            path.append(tuple(reversed(cell)))
        return path

    # ------------------------------------------------------------------
    def quantize_point(self, x: np.ndarray) -> Cell:
        """Quantize a continuous point x ∈ [0,1]^d to a grid cell.
        
        Maps a point in the unit hypercube to its corresponding cell coordinate.
        
        Args:
            x: Continuous coordinates, array of shape (d,) with values ∈ [0,1]
            
        Returns:
            Cell coordinate tuple (c_0, ..., c_{d-1}) where c_i ∈ [0, 2^K - 1]
            
        Example:
            >>> grid = Grid(d=2, K=4)  # 16×16 grid
            >>> x = np.array([0.5, 0.75])
            >>> cell = grid.quantize_point(x)
            >>> cell
            (8, 12)
        """
        if len(x) != self.d:
            raise ValueError(f"Point dimension {len(x)} != grid dimension {self.d}")
        
        cell = tuple(
            min(int(np.floor(x[i] * self.grid_side)), self.grid_side - 1)
            for i in range(self.d)
        )
        return cell
    
    def quantize_query_window(self, w_L: np.ndarray, w_U: np.ndarray):
        """Quantize a continuous query window to a grid query.
        
        Maps a continuous window [w_L, w_U] ⊂ [0,1]^d to a discrete Query object.
        
        Args:
            w_L: Window lower bound, array of shape (d,) with values ∈ [0,1]
            w_U: Window upper bound, array of shape (d,) with values ∈ [0,1]
            
        Returns:
            Query object with quantized lo and hi bounds
            
        Example:
            >>> grid = Grid(d=2, K=4)
            >>> w_L = np.array([0.2, 0.3])
            >>> w_U = np.array([0.5, 0.7])
            >>> query = grid.quantize_query_window(w_L, w_U)
            >>> query.lo, query.hi
            ((3, 4), (8, 11))
        """
        from lacsfcf.data.query_workload import Query
        
        if len(w_L) != self.d or len(w_U) != self.d:
            raise ValueError(f"Window dimension mismatch: expected {self.d}")
        
        lo = self.quantize_point(w_L)
        hi = self.quantize_point(w_U)
        return Query(lo=lo, hi=hi)

    # ------------------------------------------------------------------
    def assert_ac(self, path: List[Cell]) -> None:
        """Assert that path is AC (every step ℓ₁ == 1) and covers all cells."""
        assert len(path) == self.num_cells, (
            f"Path length {len(path)} != {self.num_cells}"
        )
        for i in range(len(path) - 1):
            dist = sum(abs(path[i][k] - path[i + 1][k]) for k in range(self.d))
            assert dist == 1, (
                f"Non-AC step at position {i}: {path[i]} → {path[i+1]}, ℓ₁={dist}"
            )
