"""M4: configuration selection by held-out Phi over (builder, K, lam).

Each builder maps weights J -> AC order (e.g. B3, Hilbert + windowed ILS, Snake). Weights are the
M1 estimate J_lam = (1 - lam) J^ + lam J~_K fitted on the fit split. The configuration with the
largest Phi on the validation split is returned. Since Phi_Q(pi) is in [0, |Q|], Hoeffding plus a
union bound over the M candidates gives, with probability >= 1 - delta,
    Phi(chosen) >= max_c Phi(c) - 2 * B * sqrt(ln(2M / delta) / (2 n_val)),  B = max |Q|,
for the curves built on the fit split (``refit=False``). ``refit=True`` rebuilds the chosen
configuration on all queries (usually better, but outside the guarantee).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grid import validate_shape
from .metrics import path_phi
from .weights import pair_weights
from .workload_model import cluster_queries, mixture_weights, shrink


@dataclass(frozen=True)
class ConfigSelection:
    order: np.ndarray
    J: np.ndarray
    builder: str
    K: int
    lam: float
    scores: dict


def _weights(shape, lo, hi, K, lam, seed, J_hat=None):
    J_hat = pair_weights(shape, lo, hi) if J_hat is None else J_hat
    if lam == 0.0:
        return J_hat
    return shrink(J_hat, mixture_weights(shape, lo, hi, cluster_queries(shape, lo, hi, K, seed)), lam)


def best_per_builder(scores: dict) -> dict:
    """builder -> (K, lam) with the largest validation score."""
    best = {}
    for (b, K, lam), v in scores.items():
        if b not in best or v > scores[(b,) + best[b]]:
            best[b] = (K, lam)
    return best


def select_config(shape, lo, hi, builders: dict, Ks=(1, 2, 4), lams=(0.0, 0.5, 1.0),
                  val_frac: float = 0.25, seed: int = 0, refit: bool = True,
                  score_fn=None) -> ConfigSelection:
    shape = validate_shape(shape)
    lo = np.atleast_2d(np.asarray(lo, dtype=np.int64))
    hi = np.atleast_2d(np.asarray(hi, dtype=np.int64))
    m = lo.shape[0]
    perm = np.random.default_rng(seed).permutation(m)
    nv = max(1, int(round(val_frac * m)))
    val, fit = perm[:nv], perm[nv:]
    if score_fn is None:
        J_val = pair_weights(shape, lo[val], hi[val])

        def score_fn(o, _lo, _hi):
            return float(path_phi(o, J_val, shape))
    J_fit_hat = pair_weights(shape, lo[fit], hi[fit])
    scores, orders = {}, {}
    for K in Ks:
        for lam in lams:
            J = _weights(shape, lo[fit], hi[fit], K, lam, seed, J_fit_hat)
            for name, build in builders.items():
                o = build(J)
                scores[(name, K, lam)] = float(score_fn(o, lo[val], hi[val]))
                orders[(name, K, lam)] = (o, J)
    key = max(scores, key=lambda k: (scores[k], -k[1], k[2]))
    name, K, lam = key
    if refit:
        J = _weights(shape, lo, hi, K, lam, seed)
        order = builders[name](J)
    else:
        order, J = orders[key]
    return ConfigSelection(order, J, name, K, lam, scores)
