"""E3: real data (GeoNames cities500, CC BY 4.0) + density-driven box queries.

Questions: (a) does the grid-segment objective that Phi optimises track the
I/O proxies of a point layout (seeks = runs in the sorted point array, pages
touched)? (b) does the M4 portfolio {Snake, L1, L1+SM} still beat data-agnostic
curves on real, axis-correlated data?

Data: lon, lat, log1p(population), elevation (dem, -9999 -> 0); the first d
columns are used for a d-dimensional grid. Queries: centre = cell of a random
data point, side lengths log-uniform (``indep`` per axis / ``iso`` shared).

    python experiments/e3_real.py --shapes 256x256 64x64x64 16x16x16x16
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd

from e2_m4 import make_builders
from e2_scale import is_pow2_cube
from lacsfc import hilbert_rank, snake_rank, zorder_rank
from lacsfc.select import _weights, best_per_builder, select_config
from lacsfc.storage import bin_points, data_boxes, io_stats

ROOT = Path(__file__).resolve().parents[1]


def load_geonames(path):
    df = pd.read_csv(path, sep="\t", header=None, usecols=[4, 5, 14, 16],
                     names=["lat", "lon", "pop", "dem"], quoting=3, keep_default_na=False,
                     na_values=[""], dtype={4: float, 5: float, 14: float, 16: float})
    return np.column_stack([df["lon"], df["lat"], np.log1p(df["pop"]), df["dem"].clip(lower=0)])


def rank_of(order):
    r = np.empty_like(order)
    r[order] = np.arange(order.size, dtype=np.int64)
    return r


def spearman(a, b):
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def run_one(X, shape, binning, qkind, seed, args):
    d = len(shape)
    cells, coords = bin_points(X[:, :d], shape, binning)
    rng = np.random.default_rng(seed)
    iso = qkind == "iso"
    lo, hi = data_boxes(coords, shape, args.m_train, rng, args.max_frac, iso)
    tlo, thi = data_boxes(coords, shape, args.m_test, rng, args.max_frac, iso)

    builders = make_builders(shape, args, seed)
    keys = [(n, K, lam) for K in args.Ks for lam in args.lams for n in builders]
    rec = []

    def score(o, vlo, vhi):
        st = io_stats(rank_of(np.asarray(o, dtype=np.int64)), cells, vlo, vhi, shape, args.pages)
        v = {"phi": -st.grid_segs.mean(), "seeks": -st.seeks.mean()}
        for j, B in enumerate(args.pages):
            v[f"xpages{B}"] = -(st.pages[j] - -(-st.points // B)).mean()
        rec.append(v)
        return v["phi"]

    t0 = time.perf_counter()
    sel = select_config(shape, lo, hi, builders, Ks=args.Ks, lams=args.lams, seed=seed, score_fn=score)
    sec = time.perf_counter() - t0
    val = dict(zip(keys, rec))

    ranks = {"snake": snake_rank(shape)}
    if is_pow2_cube(shape):
        ranks["zorder"] = zorder_rank(shape)
        ranks["hilbert"] = hilbert_rank(shape)
    for name, (K, lam) in best_per_builder(sel.scores).items():
        if name in ranks:
            continue
        o = sel.order if name == sel.builder else builders[name](_weights(shape, lo, hi, K, lam, seed))
        ranks[name] = rank_of(np.asarray(o, dtype=np.int64))
    picks = {}
    for crit in args.select_by:
        name, K, lam = max(keys, key=lambda k: (val[k][crit], -k[1], k[2]))
        if (name, K, lam) == (sel.builder, sel.K, sel.lam):
            o = sel.order
        else:
            o = builders[name](_weights(shape, lo, hi, K, lam, seed))
        ranks[f"portfolio_{crit}"] = rank_of(np.asarray(o, dtype=np.int64))
        picks[f"pick_{crit}"] = f"{name}/{K}/{lam}"

    base = {"shape": "x".join(map(str, shape)), "binning": binning, "queries": qkind, "seed": seed,
            "pick": f"{sel.builder}/{sel.K}/{sel.lam}", **picks, "sec_select": sec,
            "nonempty_frac": np.unique(cells).size / int(np.prod(shape))}
    rows, pooled, names = [], {"g": [], "s": [], **{f"x{B}": [] for B in args.pages}}, []
    for name, r in ranks.items():
        st = io_stats(r, cells, tlo, thi, shape, args.pages)
        row = dict(base, method=name, grid_segs=st.grid_segs.mean(), seeks=st.seeks.mean(),
                   points=st.points.mean())
        for j, B in enumerate(args.pages):
            xp = st.pages[j] - -(-st.points // B)
            row[f"pages{B}"] = st.pages[j].mean()
            row[f"xpages{B}"] = xp.mean()
            if not name.startswith("portfolio"):
                pooled[f"x{B}"].append(xp)
        rows.append(row)
        if not name.startswith("portfolio"):
            names.append(name)
            pooled["g"].append(st.grid_segs)
            pooled["s"].append(st.seeks)
    ac = np.array([n != "zorder" for n in names])
    G = np.stack(pooled["g"]).astype(np.float64)
    S = np.stack(pooled["s"]).astype(np.float64)
    corr = dict(base, rho_seg_seek=spearman(G.ravel(), S.ravel()),
                within_seg_seek=_within(G, S), within_ac_seg_seek=_within(G[ac], S[ac]))
    for B in args.pages:
        Xp = np.stack(pooled[f"x{B}"]).astype(np.float64)
        corr[f"rho_seg_xpages{B}"] = spearman(G.ravel(), Xp.ravel())
        corr[f"within_seg_xpages{B}"] = _within(G, Xp)
        corr[f"within_ac_seg_xpages{B}"] = _within(G[ac], Xp[ac])
    return rows, corr


def _within(A, B):
    """Pearson correlation of curve-induced variation: per query, subtract the mean over methods."""
    a = (A - A.mean(axis=0)).ravel()
    b = (B - B.mean(axis=0)).ravel()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else float("nan")


def write(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "geonames" / "cities500.txt"))
    ap.add_argument("--shapes", nargs="+", default=["256x256", "1024x1024", "64x64x64", "16x16x16x16"])
    ap.add_argument("--binnings", nargs="+", default=["minmax", "quantile"])
    ap.add_argument("--queries", nargs="+", default=["indep", "iso"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--m-train", type=int, default=2000)
    ap.add_argument("--m-test", type=int, default=1000)
    ap.add_argument("--max-frac", type=float, default=0.25)
    ap.add_argument("--pages", type=int, nargs="+", default=[16, 128])
    ap.add_argument("--Ks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--lams", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    ap.add_argument("--builders", nargs="+", default=["snake", "l1", "l1sm"])
    ap.add_argument("--select-by", nargs="+", default=["phi"],
                    help="validation criteria: phi, seeks, xpages<B> (one portfolio per criterion)")
    ap.add_argument("--max-loop", type=int, default=256)
    ap.add_argument("--reach", type=int, default=1024)
    ap.add_argument("--sm-window", type=int, default=4096)
    ap.add_argument("--min-tile", type=int, default=8)
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--passes", type=int, default=6)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--kick", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "experiments" / "results" / "e3_real.csv"))
    args = ap.parse_args()

    X = load_geonames(args.data)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cout = out.with_name(out.stem + "_corr.csv")
    rows, corrs = [], []
    for s in args.shapes:
        shape = tuple(int(x) for x in s.split("x"))
        for binning in args.binnings:
            for qk in args.queries:
                for seed in range(args.seeds):
                    r, c = run_one(X, shape, binning, qk, seed, args)
                    rows += r
                    corrs.append(c)
                    cells = " ".join(f"{x['method']}={x['grid_segs']:.2f}/{x['seeks']:.2f}/{x[f'pages{args.pages[-1]}']:.2f}"
                                     for x in r)
                    print(f"{s:>12} {binning:>8} {qk:>5} seed={seed} pick={c['pick']} sel={c['sec_select']:.0f}s "
                          f"nonempty={c['nonempty_frac']:.2f} rho(seg,seek)={c['rho_seg_seek']:.2f} | "
                          f"grid/seek/pages{args.pages[-1]}: {cells}", flush=True)
                write(out, rows)
                write(cout, corrs)

    df = pd.DataFrame(rows)
    mets = ["grid_segs", "seeks"] + [f"xpages{B}" for B in args.pages]
    agg = df.groupby(["shape", "binning", "queries", "method"], sort=False)[mets].mean()
    print("\nmean per query; (rel. to best data-agnostic curve among snake/zorder/hilbert, per metric)")
    for key, grp in agg.groupby(level=[0, 1, 2], sort=False):
        g = grp.droplevel([0, 1, 2])
        ref = g.loc[[m for m in ("snake", "zorder", "hilbert") if m in g.index]].min()
        print(f"\n{' '.join(key)}")
        print(f"{'method':>20} " + " ".join(f"{m:>18}" for m in mets))
        for m, v in g.iterrows():
            print(f"{m:>20} " + " ".join(f"{v[k]:>9.2f} ({v[k] / ref[k] - 1:+.3f})" for k in mets))
    cdf = pd.DataFrame(corrs)
    print("\nrho_* = Spearman pooled over queries x non-portfolio methods (query-size confounded); "
          "within_* = Pearson of per-query deviations from the method mean (curve-induced); mean over seeds")
    print(cdf.groupby(["shape", "binning", "queries"], sort=False)
          [[c for c in cdf.columns if c.startswith(("rho_", "within_"))] + ["nonempty_frac"]]
          .mean().round(3).to_string())
    print(f"\ncsv -> {out}, {cout}")


if __name__ == "__main__":
    main()
