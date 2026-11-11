"""Amortised (cross-instance) weight generation (paper Sec. 5.3 Tier 3, E5).

One generator FG_theta maps per-dual-edge features of a task (data distribution, workload
statistics from the task's own training queries, page size, hierarchy level) to a residual on
the analytic weights: W = W0n + FG_theta(feat). theta is trained across many tasks with
antithetic evolution strategies on the mean execution cost of each task's *held-out* queries,
so the training signal is generalisation, not in-sample fit. At inference a new task needs
one forward pass and no oracle call. numpy only (reuses ``learn.edge_features``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .grid import block_shape, unravel, validate_shape
from .learn import edge_features


def edge_levels(shape, sq) -> np.ndarray:
    """Number of aligned dyadic super-block boundaries crossed by each dual edge
    (trailing zeros of the upper block coordinate along the edge axis)."""
    ca = unravel(sq.block_a, block_shape(validate_shape(shape)))
    v = ca[np.arange(sq.size), sq.dim].astype(np.int64) + 1
    lvl = np.zeros(sq.size, dtype=np.int64)
    for _ in range(63):
        m = (v & 1) == 0
        if not m.any():
            break
        lvl += m
        v = np.where(m, v >> 1, v)
    return lvl


def task_features(shape, J, cell_of_point, sq, page_size: int, d_max: int = 3):
    """Fixed-width per-edge features for any d <= d_max, and normalised W0.

    Columns: W0n, 4 gate J values, 6 log-densities (blocks and gate cells), d_max block
    positions, d_max axis one-hot, level / log2(max side), log1p(block points / page) for
    both blocks."""
    shape = validate_shape(shape)
    d = len(shape)
    if d > d_max:
        raise ValueError(f"d={d} > d_max={d_max}")
    feat, w0n = edge_features(shape, J, cell_of_point, sq)
    base, pos, onehot = feat[:, :11], feat[:, 11:11 + d], feat[:, 11 + d:11 + 2 * d]
    pad = np.zeros((sq.size, d_max - d))
    lvl = edge_levels(shape, sq) / max(np.log2(max(shape)), 1.0)
    nb = block_shape(shape)
    cnt = np.bincount(np.asarray(cell_of_point, dtype=np.int64), minlength=int(np.prod(shape)))
    blk = np.ravel_multi_index(tuple((unravel(np.arange(cnt.size), shape) // 2).T), nb)
    bcnt = np.bincount(blk, weights=cnt, minlength=int(np.prod(nb)))
    pg = np.log1p(np.stack([bcnt[sq.block_a], bcnt[sq.block_b]], axis=1) / page_size)
    out = np.column_stack([base, pos, pad, onehot, pad, lvl, pg])
    return out, w0n


def workload_features(shape, lo, hi, J, sq) -> np.ndarray:
    """Axis-relative workload-shape features per dual edge (permutation-equivariant over axes).

    Columns: mean log relative side along the edge axis minus its mean over axes (elongation);
    d * (J mass on the edge axis / total J) - 1; mean over queries of the std over axes of the
    log relative side (task anisotropy); mean log relative query volume / d."""
    shape = validate_shape(shape)
    d = len(shape)
    side = (np.asarray(hi, dtype=np.float64) - np.asarray(lo, dtype=np.float64) + 1.0)
    ls = np.log(side / np.asarray(shape, dtype=np.float64))
    mean_ls = ls.mean(axis=0)
    elong = mean_ls - mean_ls.mean()
    mass = np.asarray(J, dtype=np.float64).reshape(d, -1).sum(axis=1)
    share = d * mass / max(float(mass.sum()), 1e-12) - 1.0
    aniso = float(ls.std(axis=1).mean()) if d > 1 else 0.0
    vol = float(ls.sum(axis=1).mean()) / d
    k = np.asarray(sq.dim)
    return np.column_stack([elong[k], share[k], np.full(sq.size, aniso), np.full(sq.size, vol)])


class MLP:
    """Per-edge MLP n_in -> hidden -> hidden -> 1 (tanh), parameters as one flat vector.
    ``init`` zeroes the output layer, so FG starts exactly at W0n."""

    def __init__(self, n_in: int, hidden: int = 32):
        self.n_in, self.hidden = n_in, hidden
        self.shapes = [(n_in, hidden), (hidden,), (hidden, hidden), (hidden,), (hidden, 1), (1,)]
        self.sizes = [int(np.prod(s)) for s in self.shapes]
        self.n_params = sum(self.sizes)

    def init(self, rng: np.random.Generator) -> np.ndarray:
        parts = []
        for k, s in enumerate(self.shapes):
            if k >= 4 or len(s) == 1:
                parts.append(np.zeros(s).ravel())
            else:
                parts.append((rng.standard_normal(s) / np.sqrt(s[0])).ravel())
        return np.concatenate(parts)

    def unpack(self, theta):
        out, i = [], 0
        for s, n in zip(self.shapes, self.sizes):
            out.append(theta[i:i + n].reshape(s))
            i += n
        return out

    def __call__(self, theta, feat) -> np.ndarray:
        W1, b1, W2, b2, W3, b3 = self.unpack(theta)
        h = np.tanh(feat @ W1 + b1)
        h = np.tanh(h @ W2 + b2)
        return (h @ W3 + b3)[:, 0]


@dataclass
class ESConfig:
    iters: int = 150
    pop: int = 16
    tasks_per_iter: int = 8
    sigma: float = 0.05
    lr: float = 0.02
    weight_decay: float = 0.0
    seed: int = 0


@dataclass
class ESResult:
    theta: np.ndarray
    history: list = field(default_factory=list)


def _centered_ranks(x: np.ndarray) -> np.ndarray:
    r = np.empty(x.size)
    r[np.argsort(x)] = np.arange(x.size)
    return r / max(x.size - 1, 1) - 0.5


def es_train(loss_fn, n_tasks: int, theta0: np.ndarray, cfg: ESConfig = ESConfig(),
             callback=None, map_fn=None) -> ESResult:
    """Antithetic ES with centred-rank shaping and Adam, minimising
    mean_{task in batch} loss_fn(theta, task). The same task batch is used for every
    population member in an iteration (common random numbers). ``map_fn(jobs)``, if given,
    returns [loss_fn(theta, task) for theta, task in jobs] (e.g. via a process pool); the
    result is identical to the serial path."""
    rng = np.random.default_rng(cfg.seed)
    theta = np.array(theta0, dtype=np.float64)
    m, v = np.zeros_like(theta), np.zeros_like(theta)
    b1, b2, eps = 0.9, 0.999, 1e-8
    hist = []
    for it in range(cfg.iters):
        batch = rng.choice(n_tasks, size=min(cfg.tasks_per_iter, n_tasks), replace=False)
        noise = rng.standard_normal((cfg.pop, theta.size))
        if map_fn is None:
            lp = np.array([np.mean([loss_fn(theta + cfg.sigma * e, t) for t in batch]) for e in noise])
            lm = np.array([np.mean([loss_fn(theta - cfg.sigma * e, t) for t in batch]) for e in noise])
        else:
            jobs = [(theta + s * cfg.sigma * e, t) for s in (1, -1) for e in noise for t in batch]
            vals = np.asarray(map_fn(jobs), dtype=np.float64).reshape(2, cfg.pop, len(batch))
            lp = np.array([np.mean(list(row)) for row in vals[0]])
            lm = np.array([np.mean(list(row)) for row in vals[1]])
        r = _centered_ranks(np.concatenate([lp, lm]))
        g = ((r[:cfg.pop] - r[cfg.pop:]) @ noise) / (2 * cfg.pop * cfg.sigma)
        g = g + cfg.weight_decay * theta
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mh, vh = m / (1 - b1 ** (it + 1)), v / (1 - b2 ** (it + 1))
        theta = theta - cfg.lr * mh / (np.sqrt(vh) + eps)
        rec = dict(iter=it, pop_mean=float(np.mean(np.concatenate([lp, lm]))),
                   pop_best=float(min(lp.min(), lm.min())))
        if callback is not None:
            rec.update(callback(it, theta) or {})
        hist.append(rec)
    return ESResult(theta=theta, history=hist)
