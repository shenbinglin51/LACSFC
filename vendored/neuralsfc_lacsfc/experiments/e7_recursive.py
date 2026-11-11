"""E7 (Plan B step 2): recursive dyadic family F_R with the exact DP decoder, out of sample.

Same tasks as E6-(2) (seeds 100-104, all families, 64x64 and 16x16x16, B in {256, 1024}).
Curves, all built from query sample A only:
  hilbert  fixed Hilbert curve
  w0       cover-and-merge, MWST(dPhi(J_A))
  fg       learned generator (theta from E5c sf+aug) + cover-and-merge
  dp0      F_R curve maximizing Phi(J_A) (ties broken toward Hilbert edges)
  dpv      F_R curve maximizing J_A + lam * mean(J_A) * 1[Hilbert edge], lam picked on B
  dps      F_R curve maximizing blur_r(J_A) (r passes of a [1,2,1]/4 kernel on every axis), r picked on B
  gbo      GBO of Yang-He-Chen (TKDE 2016) on queries A, following Alg. 1-2 / Proc. 3 (alpha = 3; not AC)
  gbo_s    earlier simplified GBO-style order on J_A (grid edges only, Hilbert ties; not AC)
Safeguards pick on B: safe_fg = {fg, w0, hilbert}, safe_dp = {fg, w0, hilbert, dpv},
safe_dps = {fg, w0, hilbert, dpv, dps}, safe_dpa = safe_dps + dpa.
  dpa      as dps with a per-axis number of blur passes r_k in {0,1,2,4,8,16}, chosen on B by two sweeps
           of coordinate descent starting from (r_s, ..., r_s) of dps
Headroom diagnostics (not selectable): dpAB = dp0 on J of A+B, dpT = dp0 on J of the test set T
(leaky, in-sample on T), dpBig = dp0 on J of 8000 fresh queries of the same family, independent of T.

    python experiments/e7_recursive.py --workers 8 --out-tag e7
"""

from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from e4_learn import synth
from e5_amortized import FAMILIES, ROOT, _geonames, make_task, queries, write_csv
from lacsfc import hilbert_rank, pair_weights
from lacsfc.amortize import MLP
from lacsfc.grid import num_cells
from lacsfc.recursive import recursive_rank
from lacsfc.storage import bin_points

LAMS = (0.0, 0.25, 1.0, 4.0)
BLURS = (1, 2, 4, 8)
BLURS_AX = (0, 1, 2, 4, 8, 16)


def blur(J, shape, r):
    """r passes of the separable binomial kernel [1, 2, 1] / 4 along every axis (edge-padded).

    r may be a per-axis sequence (r_1, ..., r_d) of passes along each spatial axis.
    """
    rs = (r,) * len(shape) if np.isscalar(r) else tuple(r)
    G = J.reshape((len(shape),) + tuple(shape)).astype(np.float64)
    for k in range(len(shape)):
        idx = [k] + [slice(None)] * len(shape)
        src = list(idx)
        idx[k + 1], src[k + 1] = -1, -2
        G[tuple(idx)] = G[tuple(src)]
    for ax in range(1, G.ndim):
        for _ in range(rs[ax - 1]):
            P = np.concatenate([G.take([0], axis=ax), G, G.take([-1], axis=ax)], axis=ax)
            n = G.shape[ax]
            G = 0.25 * P.take(range(0, n), axis=ax) + 0.5 * G + 0.25 * P.take(range(2, n + 2), axis=ax)
    return G.reshape(J.shape)


def _coaccess(shape, lo, hi, delta):
    """C[u] = number of boxes containing both u and u + delta (0 where u + delta is outside the grid)."""
    d = len(shape)
    a = np.maximum(lo, lo - delta)
    b = np.minimum(hi, hi - delta)
    ok = (a <= b).all(1)
    a, b = a[ok], b[ok]
    D = np.zeros(tuple(s + 1 for s in shape))
    for corner in range(1 << d):
        sel = [(corner >> k) & 1 for k in range(d)]
        idx = tuple(np.where(sel[k], b[:, k] + 1, a[:, k]) for k in range(d))
        np.add.at(D, idx, (-1) ** sum(sel))
    for k in range(d):
        D = np.cumsum(D, axis=k)
    return D[tuple(slice(0, s) for s in shape)]


def gbo_rank(shape, cells, lo, hi, alpha=3.0, seed=0, max_pairwise=6000):
    """GBO (Yang, He, Chen, TKDE 28(3), 2016, Alg. 1-2 and Proc. 3), following the paper.

    Vertices are the non-empty cells only. Edges join cell pairs with Euclidean distance < alpha, with
    weight = (#queries containing both cells) + 0.5 / distance (Eq. 3). Edges are taken in descending
    weight order (ties in random order) under degree <= 2 and no cycles. The paper's second pass over
    zero-workload edges between open vertices is the tail of the same order, because every workload
    weight is >= 1 and every pure distance weight is < 1. The remaining fragments and isolated cells
    are then joined ignoring the threshold, shortest Euclidean endpoint distance first, with the same
    degree and cycle rules; above max_pairwise endpoints this falls back to nearest-endpoint chaining.
    Empty cells hold no records, so they are appended at the end; their position does not change
    the cost C. The order is generally not AC.
    """
    d, N = len(shape), num_cells(shape)
    rng = np.random.default_rng(seed)
    co = np.indices(shape).reshape(d, -1).T
    nonempty = np.zeros(N, dtype=bool)
    nonempty[np.unique(cells)] = True
    st = np.cumprod((1,) + tuple(shape[::-1]))[:-1][::-1]
    r = int(np.ceil(alpha)) - 1
    grid = np.indices((2 * r + 1,) * d).reshape(d, -1).T - r
    deltas = [g for g in grid if (g ** 2).sum() < alpha ** 2 and g.any()
              and g[np.nonzero(g)[0][0]] > 0]
    U, V, Wt = [], [], []
    lo, hi = np.asarray(lo), np.asarray(hi)
    for g in deltas:
        C = _coaccess(shape, lo, hi, g).reshape(-1)
        inside = ((co + g >= 0) & (co + g < np.asarray(shape))).all(1)
        u = np.nonzero(inside & nonempty)[0]
        v = u + int((g * st).sum())
        keep = nonempty[v]
        u, v = u[keep], v[keep]
        U.append(u)
        V.append(v)
        Wt.append(C[u] + 0.5 / float(np.sqrt((g ** 2).sum())))
    U, V, Wt = np.concatenate(U), np.concatenate(V), np.concatenate(Wt)
    perm = rng.permutation(U.size)
    order = perm[np.argsort(-Wt[perm], kind="stable")]

    parent = list(range(N))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    deg = [0] * N
    adj = [[] for _ in range(N)]
    n_comp = int(nonempty.sum())

    def link(u, v):
        nonlocal n_comp
        if deg[u] < 2 and deg[v] < 2:
            ru, rv = find(u), find(v)
            if ru != rv:
                parent[ru] = rv
                adj[u].append(v)
                adj[v].append(u)
                deg[u] += 1
                deg[v] += 1
                n_comp -= 1

    for u, v in zip(U[order].tolist(), V[order].tolist()):
        if n_comp == 1:
            break
        link(u, v)

    ends = np.array([c for c in np.nonzero(nonempty)[0] if deg[c] < 2], dtype=np.int64)
    if n_comp > 1 and ends.size <= max_pairwise:
        P = co[ends].astype(np.float64)
        D2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
        iu, ju = np.triu_indices(ends.size, 1)
        dd = D2[iu, ju]
        p2 = rng.permutation(dd.size)
        for k in p2[np.argsort(dd[p2], kind="stable")].tolist():
            if n_comp == 1:
                break
            link(int(ends[iu[k]]), int(ends[ju[k]]))
        ends = np.array([c for c in np.nonzero(nonempty)[0] if deg[c] < 2], dtype=np.int64)

    seq, seen = [], np.zeros(N, dtype=bool)
    cur = np.zeros(d)
    pend = list(ends.tolist())
    while pend:
        P = co[pend]
        j = int(((P - cur) ** 2).sum(1).argmin())
        x, prev = pend[j], -1
        while True:
            seq.append(x)
            seen[x] = True
            nxt = [y for y in adj[x] if y != prev]
            if not nxt:
                break
            prev, x = x, nxt[0]
        cur = co[x]
        pend = [e for e in pend if not seen[e]]
    seq.extend(np.nonzero(~nonempty)[0].tolist())
    rank = np.empty(N, dtype=np.int64)
    rank[np.asarray(seq)] = np.arange(N)
    return rank


def gbo_simple_rank(shape, J, H):
    """Simplified GBO-style order used before the full paper was read (kept for comparison only).

    Grid edges are linked greedily by descending J (ties: Hilbert edges first) under degree <= 2 and
    no cycles, giving a path cover; fragments are chained by jumping from the current tail to the
    nearest (L1) unvisited fragment endpoint, starting from the endpoint nearest the origin.
    The result is generally not AC.
    """
    d, N = len(shape), num_cells(shape)
    co = np.indices(shape).reshape(d, -1)
    st = np.cumprod((1,) + tuple(shape[::-1]))[:-1][::-1]
    ks = np.concatenate([np.full(int((co[k] < shape[k] - 1).sum()), k) for k in range(d)])
    us = np.concatenate([np.nonzero(co[k] < shape[k] - 1)[0] for k in range(d)])
    vs = us + st[ks]
    order = np.lexsort((-H[ks, us], -J[ks, us]))
    parent = np.arange(N)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    adj = np.full((N, 2), -1)
    deg = np.zeros(N, dtype=np.int64)
    for e in order:
        u, v = int(us[e]), int(vs[e])
        if deg[u] < 2 and deg[v] < 2:
            ru, rv = find(u), find(v)
            if ru != rv:
                parent[ru] = rv
                adj[u, deg[u]], adj[v, deg[v]] = v, u
                deg[u] += 1
                deg[v] += 1
    ends = np.nonzero(deg < 2)[0]
    alive = np.ones(len(ends), dtype=bool)
    pos = {int(c): i for i, c in enumerate(ends)}
    seq = []
    cur = co[:, 0] * 0
    while alive.any():
        cand = np.nonzero(alive)[0]
        j = cand[np.abs(co[:, ends[cand]] - cur[:, None]).sum(0).argmin()]
        prev, x = -1, int(ends[j])
        while True:
            seq.append(x)
            nxt = [y for y in adj[x, :deg[x]] if y != prev]
            if not nxt:
                break
            prev, x = x, int(nxt[0])
        alive[j] = False
        alive[pos[x]] = False
        cur = co[:, x]
    rank = np.empty(N, dtype=np.int64)
    rank[np.asarray(seq)] = np.arange(N)
    return rank


def hilbert_edges(shape):
    order = np.argsort(hilbert_rank(shape))
    H = np.zeros((len(shape), num_cells(shape)))
    x, y = np.minimum(order[:-1], order[1:]), np.maximum(order[:-1], order[1:])
    st = np.cumprod((1,) + tuple(shape[::-1]))[:-1][::-1]
    k = np.searchsorted(-st, -(y - x))
    H[k, x] = 1.0
    return H


def big_sample(ds, shape, fam, seed, cfg, m=8000):
    rng = np.random.default_rng(seed)
    X = synth(cfg.n_synth, len(shape), rng) if ds == "synth" else _geonames(len(shape))
    _, coords = bin_points(X, shape, cfg.binning)
    return queries(fam, coords, shape, m, np.random.default_rng(seed + 10**6), cfg.max_frac)


def run(job):
    ds, shape, fam, page, seed, cfg = job
    t = make_task(ds, shape, fam, page, seed, cfg)
    z = np.load(cfg.theta)
    net = MLP(int(z["n_in"]), int(z["hidden"]))
    o, J = t.oB, t.oA.J
    H = hilbert_edges(shape)
    scale = float(J[H > 0].mean()) or 1.0
    ranks = {"hilbert": hilbert_rank(shape), "w0": o.curve(t.dphi).rank,
             "fg": o.curve(t.w0n + net(z["theta"], t.feat)).rank}
    dp = {lam: recursive_rank(shape, J + (lam * scale + 1e-9 * scale) * H) for lam in LAMS}
    cBdp = {lam: float(o.costs(r).mean()) for lam, r in dp.items()}
    lam_v = min(LAMS, key=lambda l: cBdp[l])
    ranks["dp0"], ranks["dpv"] = dp[0.0], dp[lam_v]
    ds_ = {r: recursive_rank(shape, blur(J, shape, r) + 1e-9 * scale * H) for r in BLURS}
    cBs = {r: float(o.costs(rk).mean()) for r, rk in ds_.items()}
    cBs[0] = cBdp[0.0]
    r_s = min(cBs, key=lambda r: cBs[r])
    ranks["dps"] = dp[0.0] if r_s == 0 else ds_[r_s]
    ra = (r_s,) * len(shape)
    cache = {ra: (cBs[r_s], ranks["dps"])}
    for _ in range(2):
        for k in range(len(shape)):
            for v in BLURS_AX:
                rt = ra[:k] + (v,) + ra[k + 1:]
                if rt not in cache:
                    rk = recursive_rank(shape, blur(J, shape, rt) + 1e-9 * scale * H)
                    cache[rt] = (float(o.costs(rk).mean()), rk)
                if cache[rt][0] < cache[ra][0]:
                    ra = rt
    ranks["dpa"] = cache[ra][1]
    ranks["gbo"] = gbo_rank(shape, t.oA.cell_of_point, t.oA.lo, t.oA.hi, alpha=3.0, seed=seed)
    ranks["gbo_s"] = gbo_simple_rank(shape, J, H)
    for name, (lo, hi) in (("dpAB", (np.concatenate([t.oA.lo, o.lo]), np.concatenate([t.oA.hi, o.hi]))),
                           ("dpT", t.te), ("dpBig", big_sample(ds, shape, fam, seed, cfg))):
        Jx = pair_weights(shape, lo, hi)
        ranks[name] = recursive_rank(shape, Jx + 1e-9 * float(Jx[H > 0].mean() or 1.0) * H)
    cB = {k: float(o.costs(r).mean()) for k, r in ranks.items()}
    cT = {k: float(o.costs(r, *t.te).mean()) for k, r in ranks.items()}
    side = np.asarray(t.te[1]) - np.asarray(t.te[0]) + 1
    vol = side.prod(1).astype(np.float64)
    row = dict(t.key, lam_v=lam_v, blur_r=r_s, blur_ax="-".join(map(str, ra)), cells_Q=float(vol.mean()),
               cells_bdQ=float((vol - np.maximum(side - 2, 0).prod(1)).mean()))
    for name, cands in (("safe_fg", ("fg", "w0", "hilbert")),
                        ("safe_dp", ("fg", "w0", "hilbert", "dpv")),
                        ("safe_dps", ("fg", "w0", "hilbert", "dpv", "dps")),
                        ("safe_dpa", ("fg", "w0", "hilbert", "dpv", "dps", "dpa"))):
        pick = min(cands, key=lambda k: cB[k])
        cT[name] = cT[pick]
        row[name + "_pick"] = pick
    row.update({f"T_{k}": v for k, v in cT.items()})
    row.update({f"B_{k}": v for k, v in cB.items()})
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["synth", "geonames"])
    ap.add_argument("--shapes", nargs="+", default=["64x64", "16x16x16"])
    ap.add_argument("--families", nargs="+", default=list(FAMILIES))
    ap.add_argument("--pages", type=int, nargs="+", default=[256, 1024])
    ap.add_argument("--seeds", type=int, nargs="+", default=[100, 101, 102, 103, 104])
    ap.add_argument("--theta", default=str(ROOT / "experiments" / "results" / "e5c_sfaug_all_theta.npz"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-tag", default="e7")
    a = ap.parse_args()
    cfg = SimpleNamespace(n_synth=100_000, binning="quantile", m_a=400, m_b=400, m_t=1000,
                          max_frac=0.25, max_cells=1 << 18, shape_feats=True, elong_max=8.0,
                          theta=a.theta)
    jobs = [(ds, tuple(int(v) for v in s.split("x")), fam, p, sd, cfg)
            for ds in a.datasets for s in a.shapes for fam in a.families for p in a.pages
            for sd in a.seeds]
    t0 = time.perf_counter()
    with Pool(a.workers) as pool:
        rows = pool.map(run, jobs, chunksize=1)
    write_csv(ROOT / "experiments" / "results" / f"{a.out_tag}_test.csv", rows)
    print(f"{len(rows)} tasks, {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
