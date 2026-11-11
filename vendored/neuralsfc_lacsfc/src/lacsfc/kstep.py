"""k-step orders with a jump set (technical_plan.md §13, Proposition K).

A sequence of cells sigma is a k-step order with jump set J if every consecutive pair outside J has
L_inf distance <= k. For a box Q = [lo, hi] (inclusive) every entry of Q along sigma (a cell of Q
whose predecessor is absent or outside Q) lies in the k-band B_k(Q), is the head of a jump, or is
the first cell of sigma; exits are symmetric. B_k(Q) only counts faces of Q that are not on the
grid boundary, because a predecessor outside Q must lie inside the grid.
"""

from __future__ import annotations

import numpy as np


def coords_of(flat, shape):
    return np.stack(np.unravel_index(np.asarray(flat, dtype=np.int64), shape), 1)


def steps_linf(seq_coords):
    return np.abs(np.diff(seq_coords, axis=0)).max(1)


def in_box(c, lo, hi):
    return ((c >= lo) & (c <= hi)).all(1)


def band_mask(c, lo, hi, k, shape):
    lo, hi, G = np.asarray(lo), np.asarray(hi), np.asarray(shape)
    near = ((c - lo < k) & (lo > 0)) | ((hi - c < k) & (hi < G - 1))
    return in_box(c, lo, hi) & near.any(1)


def band_size(lo, hi, k, shape):
    lo, hi, G = np.asarray(lo), np.asarray(hi), np.asarray(shape)
    full = int(np.prod(hi - lo + 1))
    ilo = np.where(lo > 0, lo + k, lo)
    ihi = np.where(hi < G - 1, hi - k, hi)
    return full - int(np.prod(np.maximum(ihi - ilo + 1, 0)))


def query_stats(seq_coords, jump, lo, hi, k, shape):
    """Per-box statistics of a k-step order; jump[i] marks the pair (i, i + 1).

    Returns a dict with the number of Proposition-K violations (must be 0), cells of sigma in Q,
    segments, jump endpoints in Q, and |B_k(Q)|.
    """
    q = in_box(seq_coords, lo, hi)
    f = np.zeros(1, dtype=bool)
    entry = q & ~np.concatenate([f, q[:-1]])
    exit_ = q & ~np.concatenate([q[1:], f])
    head = np.concatenate([f, jump])
    tail = np.concatenate([jump, f])
    first = np.zeros(q.size, dtype=bool)
    first[0] = True
    last = first[::-1]
    B = band_mask(seq_coords, lo, hi, k, shape)
    viol = int((entry & ~(B | head | first)).sum() + (exit_ & ~(B | tail | last)).sum())
    return dict(viol=viol, n_in=int(q.sum()), segs=int(entry.sum()),
                jq=int((q & (head | tail)).sum()), band=band_size(lo, hi, k, shape))
