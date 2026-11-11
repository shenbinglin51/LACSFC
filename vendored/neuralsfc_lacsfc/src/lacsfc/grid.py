"""Grid helpers: shape validation, block shape, parity masks, coordinates."""

from __future__ import annotations

import numpy as np


def validate_shape(shape) -> tuple[int, ...]:
    shape = tuple(int(s) for s in shape)
    if len(shape) < 2:
        raise ValueError("dimension d must be >= 2")
    for s in shape:
        if s < 2 or s % 2 != 0:
            raise ValueError(f"every side must be even and >= 2, got {shape}")
    return shape


def block_shape(shape) -> tuple[int, ...]:
    return tuple(s // 2 for s in shape)


def num_cells(shape) -> int:
    return int(np.prod(shape, dtype=np.int64))


def index_dtype(shape):
    """Narrowest signed dtype able to hold every flat index of ``shape``."""
    return np.int32 if num_cells(shape) <= np.iinfo(np.int32).max else np.int64


def strides(shape) -> np.ndarray:
    st = np.ones(len(shape), dtype=np.int64)
    for k in range(len(shape) - 2, -1, -1):
        st[k] = st[k + 1] * shape[k + 1]
    return st


def all_coords(shape) -> np.ndarray:
    """(N, d) coordinates of all cells in C (row-major) flat order."""
    d = len(shape)
    return np.indices(shape, dtype=np.int64).reshape(d, -1).T


def unravel(flat, shape) -> np.ndarray:
    """Flat indices -> (..., d) coordinates."""
    return np.stack(np.unravel_index(np.asarray(flat, dtype=np.int64), shape), axis=-1)


def ravel(coords, shape) -> np.ndarray:
    """(..., d) coordinates -> flat indices."""
    coords = np.asarray(coords, dtype=np.int64)
    return np.ravel_multi_index(tuple(np.moveaxis(coords, -1, 0)), shape).astype(np.int64)


def parity_mask(block_coords) -> np.ndarray:
    """Bit k of the result is block_coords[..., k] mod 2."""
    block_coords = np.asarray(block_coords, dtype=np.int64)
    d = block_coords.shape[-1]
    return ((block_coords & 1) << np.arange(d, dtype=np.int64)).sum(axis=-1)


def bits_to_offsets(mask, d: int) -> np.ndarray:
    """Bitmask (...,) -> 0/1 offsets (..., d), bit k <-> dimension k."""
    mask = np.asarray(mask, dtype=np.int64)
    return (mask[..., None] >> np.arange(d, dtype=np.int64)) & 1
