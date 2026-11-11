"""Binary Reflected Gray Code (BRGC) utilities for d-dimensional blocks."""
from functools import lru_cache
from typing import List, Tuple

Cell = Tuple[int, ...]


@lru_cache(maxsize=16)
def gray_code_sequence(d: int) -> List[Tuple[int, ...]]:
    """Return list of 2^d d-bit Gray code tuples in BRGC order.

    Example (d=2): [(0,0),(0,1),(1,1),(1,0)]
    """
    if d == 0:
        return [()]
    prev = gray_code_sequence(d - 1)
    return tuple((0,) + c for c in prev) + tuple((1,) + c for c in reversed(prev))


def brgc_cycle_edges(d: int) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """Return all edges of the BRGC Hamiltonian cycle for a d-bit code.

    Returns list of (u, v) where u and v are consecutive in the BRGC
    (including the wrap-around edge from last back to first).
    """
    seq = gray_code_sequence(d)
    n = len(seq)
    return [(seq[i], seq[(i + 1) % n]) for i in range(n)]


def face_portal_edge(d: int, dim: int, side: int) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Return the canonical BRGC portal edge on a given face of a unit block.

    A portal edge has both endpoints on the face (coord[dim] == side).
    This returns the geometrically minimal such edge that can be paired
    across adjacent blocks via φ(x) = x + e_k.

    Args:
        d:    number of dimensions
        dim:  the dimension of the face (0-indexed)
        side: 0 for low face (coord=0), 1 for high face (coord=1)

    Returns:
        (u, v) tuple of local cell coordinates (each in {0,1}^d)
    """
    seq = gray_code_sequence(d)
    n = len(seq)
    for i in range(n):
        u = seq[i]
        v = seq[(i + 1) % n]
        if u[dim] == side and v[dim] == side:
            return (u, v)
    raise RuntimeError(f"No portal edge found for d={d}, dim={dim}, side={side}")
