"""M1: workload model for low-variance co-coverage weights.

A query workload is modelled as a mixture of K product distributions: queries are clustered
(k-means on normalised centre / side length), and within a component the per-axis intervals
[lo_k, hi_k] are assumed independent across axes, each with its empirical 1-D distribution.
Then, for component c with weight w_c,

    J~[k, u] = sum_c w_c * P_c(lo_k <= u_k <= hi_k - 1) * prod_{j != k} P_c(lo_j <= u_j <= hi_j).

K = 1 is a product model (estimates sum_k s_k numbers instead of dN), K = m recovers the
empirical J exactly (each query is a box, i.e. a product). The final weights are
J_lam = (1 - lam) J^ + lam J~, with (K, lam) chosen by held-out Phi (T5: variance x (1 - lam),
bias lam * D(J~)).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grid import num_cells, validate_shape
from .metrics import path_phi
from .weights import _normalize, pair_weights


def _interval_marginal(lo, hi, w, s):
    diff = np.zeros(s + 1)
    keep = hi >= lo
    np.add.at(diff, lo[keep], w[keep])
    np.add.at(diff, hi[keep] + 1, -w[keep])
    return np.cumsum(diff)[:s]


def mixture_weights(shape, lo, hi, labels=None, weights=None) -> np.ndarray:
    """J~ of a mixture of product models; same layout and scale as ``pair_weights``."""
    shape, lo, hi, weights = _normalize(shape, lo, hi, weights)
    d = len(shape)
    labels = np.zeros(lo.shape[0], dtype=np.int64) if labels is None else np.asarray(labels)
    if labels.shape != (lo.shape[0],):
        raise ValueError("labels must have shape (m,)")
    out = np.zeros((d,) + shape)
    for c in np.unique(labels):
        sel = labels == c
        wc = weights[sel].sum()
        if wc <= 0:
            continue
        w = weights[sel] / wc
        cover = [_interval_marginal(lo[sel, j], hi[sel, j], w, shape[j]) for j in range(d)]
        pair = [_interval_marginal(lo[sel, j], hi[sel, j] - 1, w, shape[j]) for j in range(d)]
        for k in range(d):
            t = np.full(shape, wc)
            for j in range(d):
                v = pair[j] if j == k else cover[j]
                t = t * v.reshape((1,) * j + (-1,) + (1,) * (d - j - 1))
            out[k] += t
    return out.reshape(d, num_cells(shape))


def cluster_queries(shape, lo, hi, K: int, seed: int = 0, iters: int = 50) -> np.ndarray:
    """k-means++ labels on [centre / s, length / s]; K >= m gives one query per component."""
    shape = validate_shape(shape)
    s = np.asarray(shape, dtype=np.float64)
    lo = np.atleast_2d(np.asarray(lo))
    hi = np.atleast_2d(np.asarray(hi))
    m = lo.shape[0]
    if K <= 1:
        return np.zeros(m, dtype=np.int64)
    if K >= m:
        return np.arange(m, dtype=np.int64)
    X = np.hstack([(lo + hi + 1) / (2 * s), (hi - lo + 1) / s])
    rng = np.random.default_rng(seed)
    cent = [X[rng.integers(m)]]
    for _ in range(1, K):
        d2 = np.min(((X[:, None, :] - np.asarray(cent)[None]) ** 2).sum(-1), axis=1)
        p = d2 / d2.sum() if d2.sum() > 0 else np.full(m, 1.0 / m)
        cent.append(X[rng.choice(m, p=p)])
    cent = np.asarray(cent)
    labels = np.full(m, -1, dtype=np.int64)
    for _ in range(iters):
        new = ((X[:, None, :] - cent[None]) ** 2).sum(-1).argmin(axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
        for c in range(K):
            if np.any(labels == c):
                cent[c] = X[labels == c].mean(axis=0)
    return np.unique(labels, return_inverse=True)[1].astype(np.int64)


def shrink(J_hat: np.ndarray, J_model: np.ndarray, lam: float) -> np.ndarray:
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lam must be in [0, 1]")
    return (1.0 - lam) * J_hat + lam * J_model


@dataclass(frozen=True)
class WeightSelection:
    J: np.ndarray
    K: int
    lam: float
    scores: dict


def select_weights(shape, lo, hi, build, Ks=(1, 2, 4, 8), lams=(0.0, 0.5, 0.8, 1.0),
                   val_frac: float = 0.25, seed: int = 0) -> WeightSelection:
    """Choose (K, lam) maximising held-out Phi of ``build(J)`` (a J -> order callable), refit on all."""
    shape = validate_shape(shape)
    lo = np.atleast_2d(np.asarray(lo, dtype=np.int64))
    hi = np.atleast_2d(np.asarray(hi, dtype=np.int64))
    m = lo.shape[0]
    perm = np.random.default_rng(seed).permutation(m)
    nv = max(1, int(round(val_frac * m)))
    val, fit = perm[:nv], perm[nv:]
    J_val = pair_weights(shape, lo[val], hi[val])
    J_fit = pair_weights(shape, lo[fit], hi[fit])
    scores = {}
    for K in Ks:
        J_mod = mixture_weights(shape, lo[fit], hi[fit],
                                cluster_queries(shape, lo[fit], hi[fit], K, seed))
        for lam in lams:
            scores[(K, lam)] = float(path_phi(build(shrink(J_fit, J_mod, lam)), J_val, shape))
    K, lam = max(scores, key=lambda key: (scores[key], -key[0], key[1]))
    J = shrink(pair_weights(shape, lo, hi),
               mixture_weights(shape, lo, hi, cluster_queries(shape, lo, hi, K, seed)), lam)
    return WeightSelection(J, K, lam, scores)
