"""Reflected Gray block cycle H_d and facet-edge assignment sigma (Lemma L-sigma').

Local edge s connects gray(s) and gray((s + 1) mod 2^d). Bit k <-> dimension k.
sigma[k, b] is a local edge whose both endpoints have bit k equal to b, and all
2d entries are distinct.
"""

from __future__ import annotations

import numpy as np

# Base case S(3) on the Gray path P_3 (bit 2 is the most significant bit).
_BASE3 = {(2, 0): 0, (2, 1): 4, (1, 0): 6, (1, 1): 2, (0, 0): 3, (0, 1): 1}
# d = 2: the 4-cycle, each edge lies in exactly one facet.
_BASE2 = {(0, 0): 3, (0, 1): 1, (1, 0): 0, (1, 1): 2}


def gray_sequence(d: int) -> np.ndarray:
    idx = np.arange(1 << d, dtype=np.int64)
    return idx ^ (idx >> 1)


def edge_endpoints(d: int, s) -> tuple[np.ndarray, np.ndarray]:
    n = 1 << d
    s = np.asarray(s, dtype=np.int64)
    a = s ^ (s >> 1)
    t = (s + 1) % n
    return a, t ^ (t >> 1)


def edge_flip_dim(d: int, s) -> np.ndarray:
    a, b = edge_endpoints(d, s)
    x = np.asarray(a ^ b, dtype=np.int64)
    return np.log2(x).astype(np.int64)


def facet_assignment(d: int) -> np.ndarray:
    """Return sigma as an int64 array of shape (d, 2)."""
    if d < 2:
        raise ValueError("d must be >= 2")
    sig = np.full((d, 2), -1, dtype=np.int64)
    if d == 2:
        for (k, b), s in _BASE2.items():
            sig[k, b] = s
        return sig
    for (k, b), s in _BASE3.items():
        sig[k, b] = s
    for t in range(3, d):
        half = 1 << t
        used = set(sig[:t].ravel().tolist())
        sig[t, 0] = next(s for s in range(half - 1) if s not in used)
        sig[t, 1] = half
    return sig


def check_facet_assignment(d: int, sig: np.ndarray) -> None:
    """Raise AssertionError if sig is not a valid facet-edge assignment."""
    n = 1 << d
    assert sig.shape == (d, 2)
    flat = sig.ravel()
    assert len(set(flat.tolist())) == 2 * d, "sigma is not injective"
    assert np.all((flat >= 0) & (flat < n)), "sigma out of range"
    a, b = edge_endpoints(d, sig)
    for k in range(d):
        for bit in range(2):
            assert ((int(a[k, bit]) >> k) & 1) == bit, f"edge for facet {(k, bit)} not in facet"
            assert ((int(b[k, bit]) >> k) & 1) == bit, f"edge for facet {(k, bit)} not in facet"
