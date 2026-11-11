"""Learning backbone weights for the non-additive execution cost (paper Sec. 5.4).

The oracle builds P(W) = CoverMerge(MWST(G', W)) and measures the mean I/O cost
C = c_seek * seeks + c_page * pages over a query set on the materialized layout.
FG is a residual generator W = W0 + FG(features) with a zero-initialised head, so it
starts at the analytic weights W0 = DeltaPhi(J) (normalised). EG is a small message-
passing evaluator over the dual graph trained to regress the oracle score. Training
alternates Stage-E (fit EG on a replay buffer) and Stage-F (update FG through EG).
A validation safeguard keeps the cheaper of the learned curve and the W0 curve.
Requires torch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from .construct import build_curve, delta_phi, max_spanning_tree, merge_squares
from .grid import block_shape, num_cells, unravel, validate_shape
from .storage import io_stats


class CostOracle:
    def __init__(self, shape, J, cell_of_point, lo, hi, page_size: int = 64,
                 c_seek: float = 1.0, c_page: float = 1.0, max_cells: int = 1 << 16):
        shape = validate_shape(shape)
        if num_cells(shape) > max_cells:
            raise ValueError(f"grid has {num_cells(shape)} cells > max_cells={max_cells}")
        self.shape = shape
        self.J = np.asarray(J)
        self.cell_of_point = np.asarray(cell_of_point, dtype=np.int64)
        self.lo, self.hi = np.atleast_2d(lo), np.atleast_2d(hi)
        self.page_size, self.c_seek, self.c_page = int(page_size), float(c_seek), float(c_page)
        self.sq = merge_squares(shape)
        self.n_blocks = num_cells(block_shape(shape))
        self.n_edges = self.sq.size
        self.calls = 0

    def curve(self, W):
        tree = max_spanning_tree(self.n_blocks, self.sq.block_a, self.sq.block_b,
                                 np.asarray(W, dtype=np.float64))
        return build_curve(self.shape, self.J, tree)

    def costs(self, rank, lo=None, hi=None) -> np.ndarray:
        lo = self.lo if lo is None else lo
        hi = self.hi if hi is None else hi
        st = io_stats(rank, self.cell_of_point, lo, hi, self.shape, page_sizes=(self.page_size,))
        return self.c_seek * st.seeks + self.c_page * st.pages[0]

    def score(self, W, lo=None, hi=None) -> float:
        self.calls += 1
        return -float(self.costs(self.curve(W).rank, lo, hi).mean())

    def tree_cost(self, tree, lo=None, hi=None) -> float:
        self.calls += 1
        return float(self.costs(build_curve(self.shape, self.J, tree).rank, lo, hi).mean())


def cost_breakdown(oracle: CostOracle, rank, lo=None, hi=None) -> dict:
    """Per-query cost, an order-independent lower bound (one seek and ceil(points/B) pages
    for a non-empty result) and the excess above it."""
    lo = oracle.lo if lo is None else lo
    hi = oracle.hi if hi is None else hi
    st = io_stats(rank, oracle.cell_of_point, lo, hi, oracle.shape, page_sizes=(oracle.page_size,))
    cost = oracle.c_seek * st.seeks + oracle.c_page * st.pages[0]
    lb = (oracle.c_seek * (st.points > 0)
          + oracle.c_page * -(-st.points // oracle.page_size))
    return dict(cost=cost, lb=lb, excess=cost - lb, seeks=st.seeks, grid_segs=st.grid_segs,
                points=st.points)


def _tree_path(n_nodes, a, b, edge_ids, s, t):
    adj = [[] for _ in range(n_nodes)]
    for x, y, k in zip(a.tolist(), b.tolist(), edge_ids.tolist()):
        adj[x].append((y, k))
        adj[y].append((x, k))
    via = {s: -1}
    prev = {s: s}
    stack = [s]
    while stack:
        x = stack.pop()
        if x == t:
            break
        for y, k in adj[x]:
            if y not in via:
                via[y], prev[y] = k, x
                stack.append(y)
    path = []
    while t != s:
        path.append(via[t])
        t = prev[t]
    return path


def tree_local_search(oracle: CostOracle, tree0, budget: int, rng: np.random.Generator):
    """First-improvement edge swaps on the spanning tree (add a uniformly random non-tree
    dual edge, drop a uniformly random edge of the cycle it closes), scored by the oracle's
    mean cost on its own queries. Returns (tree, cost, cost after each step)."""
    sq = oracle.sq
    tree = np.asarray(tree0, dtype=bool).copy()
    cur = oracle.tree_cost(tree)
    hist = []
    for _ in range(budget):
        out = np.flatnonzero(~tree)
        if out.size == 0:
            hist.append(cur)
            continue
        e = int(rng.choice(out))
        ids = np.flatnonzero(tree)
        path = _tree_path(oracle.n_blocks, sq.block_a[ids], sq.block_b[ids], ids,
                          int(sq.block_a[e]), int(sq.block_b[e]))
        f = int(rng.choice(path))
        tree[f], tree[e] = False, True
        c = oracle.tree_cost(tree)
        if c < cur:
            cur = c
        else:
            tree[f], tree[e] = True, False
        hist.append(cur)
    return tree, cur, np.asarray(hist)


def tv_pairs(shape, sq) -> np.ndarray:
    """Pairs of same-direction dual edges whose A-blocks are face-adjacent."""
    nb = block_shape(validate_shape(shape))
    d = len(nb)
    idx = -np.ones((d, num_cells(nb)), dtype=np.int64)
    idx[sq.dim, sq.block_a] = np.arange(sq.size)
    ca = unravel(sq.block_a, nb)
    strides = [int(np.prod(nb[j + 1:], dtype=np.int64)) for j in range(d)]
    out = []
    for j in range(d):
        e = np.flatnonzero(ca[:, j] + 1 < nb[j])
        f = idx[sq.dim[e], sq.block_a[e] + strides[j]]
        keep = f >= 0
        out.append(np.stack([e[keep], f[keep]], axis=1))
    return np.concatenate(out) if out else np.zeros((0, 2), dtype=np.int64)


def _normalise(w):
    sd = float(w.std())
    return (w - float(w.mean())) / (sd if sd > 0 else 1.0)


def edge_features(shape, J, cell_of_point, sq):
    """Per-dual-edge features and normalised W0 = DeltaPhi(J)."""
    shape = validate_shape(shape)
    nb = block_shape(shape)
    d = len(shape)
    N = num_cells(shape)
    J = np.asarray(J, dtype=np.float64)
    cnt = np.bincount(np.asarray(cell_of_point, dtype=np.int64), minlength=N).astype(np.float64)
    blk = np.ravel_multi_index(tuple((unravel(np.arange(N), shape) // 2).T), nb)
    bcnt = np.bincount(blk, weights=cnt, minlength=num_cells(nb))
    w0n = _normalise(delta_phi(J, sq))
    jm = J.max() if J.max() > 0 else 1.0
    gate = np.stack([J[sq.dim, sq.a1], J[sq.dim, sq.a2],
                     J[sq.flip, np.minimum(sq.a1, sq.a2)],
                     J[sq.flip, np.minimum(sq.b1, sq.b2)]], axis=1) / jm
    dens = np.log1p(np.stack([bcnt[sq.block_a], bcnt[sq.block_b],
                              cnt[sq.a1], cnt[sq.a2], cnt[sq.b1], cnt[sq.b2]], axis=1))
    dens /= max(float(dens.max()), 1.0)
    pos = unravel(sq.block_a, nb) / np.maximum(np.asarray(nb) - 1, 1)
    onehot = np.eye(d)[sq.dim]
    feat = np.column_stack([w0n, gate, dens, pos, onehot])
    return feat, w0n


class WeightGenerator(nn.Module):
    def __init__(self, n_feat: int, hidden: int = 64):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(n_feat, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU())
        self.head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, feat, w0):
        return w0 + self.head(self.body(feat)).squeeze(-1)


class Evaluator(nn.Module):
    """Edge MLP -> node mean -> edge update -> mean readout. Inputs include, per edge,
    its margin to the largest weight at each endpoint block (what MWST reacts to)."""

    def __init__(self, n_feat: int, sq, n_blocks: int, hidden: int = 64):
        super().__init__()
        a = torch.as_tensor(sq.block_a, dtype=torch.long)
        b = torch.as_tensor(sq.block_b, dtype=torch.long)
        deg = torch.bincount(a, minlength=n_blocks) + torch.bincount(b, minlength=n_blocks)
        self.register_buffer("a", a)
        self.register_buffer("b", b)
        self.register_buffer("deg", deg.clamp_min(1).to(torch.float64))
        self.n_blocks = n_blocks
        self.edge = nn.Sequential(nn.Linear(n_feat + 3, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU())
        self.upd = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU())
        self.out = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, W, feat):
        B = W.shape[0]
        m = W.new_full((B, self.n_blocks), -torch.inf)
        m = m.scatter_reduce(1, self.a.expand(B, -1), W, "amax")
        m = m.scatter_reduce(1, self.b.expand(B, -1), W, "amax")
        x = torch.cat([feat.expand(B, -1, -1), W[..., None],
                       (W - m[:, self.a])[..., None], (W - m[:, self.b])[..., None]], dim=-1)
        h = self.edge(x)
        s = h.new_zeros(B, self.n_blocks, h.shape[-1])
        s = s.index_add(1, self.a, h).index_add(1, self.b, h) / self.deg[None, :, None]
        h = self.upd(torch.cat([h, s[:, self.a], s[:, self.b]], dim=-1))
        return self.out(h.mean(dim=1)).squeeze(-1)


@dataclass
class LearnConfig:
    rounds: int = 20
    init_samples: int = 32
    samples_per_round: int = 16
    sigma: float = 0.5
    mix: tuple = (0.5, 0.25, 0.25)
    e_steps: int = 100
    e_batch: int = 32
    f_steps: int = 50
    lr_e: float = 1e-3
    lr_f: float = 1e-3
    lam_tv: float = 1e-2
    mu: float = 1e-2
    hidden: int = 64
    seed: int = 0
    threads: int = 4
    device: str = "cpu"


@dataclass
class LearnResult:
    W0: np.ndarray
    W_learned: np.ndarray
    W_search: np.ndarray
    curve: object
    choice: str
    train_score_learned: float
    train_score_baseline: float
    train_score_search: float
    val_cost: float
    val_cost_learned: float
    val_cost_baseline: float
    oracle_calls: int
    history: list = field(default_factory=list)


def learn_weights(oracle: CostOracle, cell_of_point, cfg: LearnConfig = LearnConfig(),
                  val=None) -> LearnResult:
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(cfg.threads)
    rng = np.random.default_rng(cfg.seed)
    calls0 = oracle.calls

    feat_np, w0n = edge_features(oracle.shape, oracle.J, cell_of_point, oracle.sq)
    dev = torch.device(cfg.device)
    feat, w0t = torch.as_tensor(feat_np, device=dev), torch.as_tensor(w0n, device=dev)
    pairs = torch.as_tensor(tv_pairs(oracle.shape, oracle.sq), device=dev)
    E = oracle.n_edges
    fg = WeightGenerator(feat.shape[1], cfg.hidden).double().to(dev)
    eg = Evaluator(feat.shape[1], oracle.sq, oracle.n_blocks, cfg.hidden).double().to(dev)
    opt_f = torch.optim.Adam(fg.parameters(), lr=cfg.lr_f)
    opt_e = torch.optim.Adam(eg.parameters(), lr=cfg.lr_e)
    buf_W, buf_R = [], []

    def evaluate(W):
        r = oracle.score(W)
        buf_W.append(np.array(W, dtype=np.float64))
        buf_R.append(r)
        return r

    def fg_out():
        with torch.no_grad():
            return fg(feat, w0t).cpu().numpy().copy()

    def sample(k):
        n_fg = int(round(k * cfg.mix[0]))
        n_base = int(round(k * cfg.mix[1]))
        cur = fg_out()
        for _ in range(n_fg):
            evaluate(cur + cfg.sigma * rng.standard_normal(E))
        for _ in range(n_base):
            evaluate(w0n + cfg.sigma * rng.standard_normal(E))
        for _ in range(max(k - n_fg - n_base, 0)):
            evaluate(rng.standard_normal(E))

    w0_raw = delta_phi(oracle.J, oracle.sq)
    base_score = oracle.score(w0_raw)
    best = (base_score, w0_raw)
    cur = fg_out()
    s0 = evaluate(cur)
    if s0 > best[0]:
        best = (s0, cur)
    sample(cfg.init_samples)

    history = []
    for r in range(cfg.rounds):
        X = torch.as_tensor(np.stack(buf_W), device=dev)
        Y = torch.as_tensor(np.asarray(buf_R), device=dev)
        Yn = (Y - Y.mean()) / Y.std().clamp_min(1e-8)
        e_loss = 0.0
        for _ in range(cfg.e_steps):
            idx = torch.as_tensor(rng.integers(0, len(X), size=min(cfg.e_batch, len(X))),
                                  device=dev)
            loss = torch.mean((eg(X[idx], feat) - Yn[idx]) ** 2)
            opt_e.zero_grad()
            loss.backward()
            opt_e.step()
            e_loss = loss.item()

        for p in eg.parameters():
            p.requires_grad_(False)
        for _ in range(cfg.f_steps):
            W = fg(feat, w0t)
            loss = (-eg(W[None], feat).mean()
                    + cfg.lam_tv * (W[pairs[:, 0]] - W[pairs[:, 1]]).abs().mean()
                    + cfg.mu * torch.mean((W - w0t) ** 2))
            opt_f.zero_grad()
            loss.backward()
            opt_f.step()
        for p in eg.parameters():
            p.requires_grad_(True)

        cur = fg_out()
        s = evaluate(cur)
        if s > best[0]:
            best = (s, cur)
        sample(cfg.samples_per_round)
        history.append(dict(round=r, e_loss=e_loss, fg_score=s, best=best[0],
                            baseline=base_score, buffer=len(buf_R)))

    W_learned = best[1]
    i_s = int(np.argmax(buf_R))
    vlo, vhi = val if val is not None else (oracle.lo, oracle.hi)
    curve_l = oracle.curve(W_learned)
    curve_b = oracle.curve(w0_raw)
    vl = float(oracle.costs(curve_l.rank, vlo, vhi).mean())
    vb = float(oracle.costs(curve_b.rank, vlo, vhi).mean())
    learned = vl < vb
    return LearnResult(W0=w0_raw, W_learned=W_learned, W_search=buf_W[i_s],
                       curve=curve_l if learned else curve_b,
                       choice="learned" if learned else "baseline",
                       train_score_learned=best[0], train_score_baseline=base_score,
                       train_score_search=float(buf_R[i_s]),
                       val_cost=min(vl, vb), val_cost_learned=vl, val_cost_baseline=vb,
                       oracle_calls=oracle.calls - calls0, history=history)
