"""
BitAllocZCurve: Z-order curve with learned per-dimension bit allocation.
Adapted from bitallocLZ/src/uc_curve.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


def _alloc_from_weights(weights: Sequence[float], total_bits: int, k: int) -> List[int]:
    """Convert continuous weights to integer bit allocation.
    Guarantees sum(alloc) == total_bits and 0 <= alloc[i] <= k.
    """
    w = np.asarray(weights, dtype=np.float64)
    w = np.maximum(w, 0.0)
    s = float(np.sum(w))
    if s <= 0.0:
        w = np.ones_like(w)
        s = float(np.sum(w))

    raw = float(total_bits) * w / s
    bits = np.floor(raw).astype(int)
    rem = int(total_bits - int(bits.sum()))

    if rem > 0:
        order = np.argsort(-(raw - bits))
        for i in order[:rem]:
            bits[int(i)] += 1

    bits = np.clip(bits, 0, int(k))

    while int(bits.sum()) > int(total_bits):
        candidates = np.where(bits > 0)[0]
        if candidates.size == 0:
            break
        j = int(candidates[np.argmax(bits[candidates])])
        bits[j] -= 1

    while int(bits.sum()) < int(total_bits):
        fractional = raw - bits
        candidates = np.where(bits < int(k))[0]
        if candidates.size == 0:
            break
        j = int(candidates[np.argmax(fractional[candidates])])
        bits[j] += 1

    current_sum = int(bits.sum())
    if current_sum != int(total_bits):
        diff = int(total_bits) - current_sum
        if diff > 0:
            for _ in range(abs(diff)):
                candidates = np.where(bits < int(k))[0]
                if candidates.size == 0:
                    break
                bits[int(candidates[0])] += 1
        else:
            for _ in range(abs(diff)):
                candidates = np.where(bits > 0)[0]
                if candidates.size == 0:
                    break
                bits[int(candidates[0])] -= 1

    final_sum = int(bits.sum())
    if final_sum != int(total_bits):
        raise RuntimeError(
            f"_alloc_from_weights failed: weights={list(weights)}, "
            f"total_bits={total_bits}, k={k}, alloc={bits.tolist()}, sum={final_sum}"
        )

    return [int(x) for x in bits.tolist()]


@dataclass(frozen=True)
class BitAllocZCurve:
    """Z-order curve with per-dimension bit allocation.

    Parameters
    ----------
    d : int
        Number of spatial dimensions.
    k : int
        Bits per dimension in the grid (grid_size = 2^k per axis).
    alloc : tuple[int, ...]
        Number of bits allocated to each dimension; must sum to total_bits.
    total_bits : int
        Total bits in the Z-code (≤ d*k, ≤ 64).
    """
    d: int
    k: int
    alloc: Tuple[int, ...]
    total_bits: int

    def __post_init__(self) -> None:
        if len(self.alloc) != int(self.d):
            raise ValueError("alloc length must equal d")
        if any(int(a) < 0 or int(a) > int(self.k) for a in self.alloc):
            raise ValueError("alloc entries must be in [0, k]")
        alloc_sum = sum(int(a) for a in self.alloc)
        if alloc_sum != int(self.total_bits):
            raise ValueError(
                f"sum(alloc)={alloc_sum} must equal total_bits={self.total_bits}"
            )
        if int(self.total_bits) > 64:
            raise ValueError("total_bits must be <= 64")

    @classmethod
    def from_weights(
        cls,
        *,
        d: int,
        k: int,
        weights: Sequence[float],
        total_bits: int,
    ) -> "BitAllocZCurve":
        alloc = _alloc_from_weights(
            weights=weights, total_bits=int(total_bits), k=int(k)
        )
        return cls(d=int(d), k=int(k), alloc=tuple(alloc), total_bits=int(total_bits))

    @classmethod
    def uniform(cls, *, d: int, k: int, total_bits: int | None = None) -> "BitAllocZCurve":
        """Create a uniform allocation (standard Z-curve)."""
        tb = int(total_bits) if total_bits is not None else int(d) * int(k)
        weights = [1.0] * int(d)
        return cls.from_weights(d=d, k=k, weights=weights, total_bits=tb)

    def encode_cell(self, cell: np.ndarray) -> int:
        """Encode a single d-dimensional grid cell to a Z-code integer."""
        c = np.asarray(cell, dtype=np.int64)
        if c.ndim != 1 or c.shape[0] != int(self.d):
            raise ValueError("cell dimension mismatch")

        out = 0
        remaining = [int(a) for a in self.alloc]
        appended = 0

        for bit in range(int(self.k) - 1, -1, -1):
            if not any(r > 0 for r in remaining):
                break
            for dim in range(int(self.d)):
                if remaining[dim] <= 0:
                    continue
                b = (int(c[dim]) >> bit) & 1
                out = (out << 1) | int(b)
                remaining[dim] -= 1
                appended += 1
                if appended >= int(self.total_bits):
                    return int(out)
        return int(out)

    def encode_cells(self, cells: np.ndarray) -> np.ndarray:
        """Encode a batch of cells. cells shape: (n, d). Returns uint64 array."""
        arr = np.asarray(cells, dtype=np.int64)
        if arr.ndim != 2 or arr.shape[1] != int(self.d):
            raise ValueError("cells shape mismatch: expected (n, d)")
        out = np.empty(arr.shape[0], dtype=np.int64)
        for i in range(arr.shape[0]):
            out[i] = self.encode_cell(arr[i])
        return out

    def alloc_summary(self) -> str:
        return f"d={self.d} k={self.k} total_bits={self.total_bits} alloc={list(self.alloc)}"
