"""E2-M4: validation-based selection over (builder, K, lam) vs each single builder and a test-set oracle.

    python experiments/e2_m4.py --shapes 32x32x32 16x16x16x16
    python experiments/e2_m4.py --builders snake l1 l1sm --Ks 1 2 4 8 16 32 --out .../e2_m4_sm.csv
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from e2_scale import WORKLOADS, is_pow2_cube, make_workload, order_of, phi_any
from lacsfc import build_curve, expected_box_size, hilbert_rank, pair_weights, snake_rank
from lacsfc.fastls import windowed_ils
from lacsfc.regions import region_curve_auto
from lacsfc.select import _weights, best_per_builder, select_config
from lacsfc.splitmerge import split_merge_search


def make_builders(shape, args, seed):
    sn = order_of(snake_rank(shape))
    b = {"snake": lambda J: sn,
         "l1": lambda J: build_curve(shape, J).order,
         "l1sm": lambda J: split_merge_search(build_curve(shape, J).order, J, shape,
                                             max_loop=args.max_loop, reach=args.reach,
                                             window=args.sm_window).order,
         "b3": lambda J: region_curve_auto(shape, J, min_tile=args.min_tile)[0].order}
    if is_pow2_cube(shape):
        h = order_of(hilbert_rank(shape))
        b["hilbert"] = lambda J: h
        b["hilbert_ils"] = lambda J: windowed_ils(h, J, shape, window=args.window, passes=args.passes,
                                                  iters=args.iters, kick=args.kick, seed=seed).order
    return {k: v for k, v in b.items() if k in args.builders}


def run_one(shape, kind, seed, args):
    rng = np.random.default_rng(seed)
    lo, hi = make_workload(shape, kind, args.m_train, rng, args.max_frac)
    tlo, thi = make_workload(shape, kind, args.m_test, rng, args.max_frac)
    Jt = pair_weights(shape, tlo, thi)
    EQt = expected_box_size(tlo, thi)
    builders = make_builders(shape, args, seed)

    t0 = time.perf_counter()
    sel = select_config(shape, lo, hi, builders, Ks=args.Ks, lams=args.lams, seed=seed)
    sec = time.perf_counter() - t0

    row = {"shape": "x".join(map(str, shape)), "workload": kind, "seed": seed, "pick": sel.builder,
           "K": sel.K, "lam": sel.lam, "sec_select": sec, "E_Q_test": EQt,
           "seg_portfolio": EQt - phi_any(sel.order, Jt, shape)}
    for name, (K, lam) in best_per_builder(sel.scores).items():
        if name == sel.builder:
            row[f"seg_{name}"] = row["seg_portfolio"]
        else:
            o = builders[name](_weights(shape, lo, hi, K, lam, seed))
            row[f"seg_{name}"] = EQt - phi_any(o, Jt, shape)
    row["seg_oracle_hindsight"] = min(row[f"seg_{n}"] for n in builders)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", nargs="+", default=["256x256", "32x32x32", "64x64x64", "16x16x16x16"])
    ap.add_argument("--workloads", nargs="+", default=list(WORKLOADS))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--m-train", type=int, default=2000)
    ap.add_argument("--m-test", type=int, default=1000)
    ap.add_argument("--max-frac", type=float, default=0.25)
    ap.add_argument("--Ks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--lams", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--passes", type=int, default=6)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--kick", type=int, default=8)
    ap.add_argument("--min-tile", type=int, default=8)
    ap.add_argument("--builders", nargs="+", default=["snake", "l1", "l1sm"])
    ap.add_argument("--max-loop", type=int, default=256)
    ap.add_argument("--reach", type=int, default=1024)
    ap.add_argument("--sm-window", type=int, default=4096)
    ap.add_argument("--out", default=str(Path(__file__).parent / "results" / "e2_m4.csv"))
    args = ap.parse_args()

    rows = []
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for s in args.shapes:
        shape = tuple(int(x) for x in s.split("x"))
        for kind in args.workloads:
            for seed in range(args.seeds):
                row = run_one(shape, kind, seed, args)
                rows.append(row)
                segs = " ".join(f"{k[4:]}={v:.3f}" for k, v in row.items() if k.startswith("seg_"))
                print(f"{row['shape']:>12} {kind:>11} seed={seed} pick={row['pick']}/{row['K']}/{row['lam']} "
                      f"sel={row['sec_select']:.0f}s | {segs}", flush=True)
            keys = list(dict.fromkeys(k for r in rows for k in r))
            with out.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(rows)

    methods = [b for b in ("snake", "l1", "l1sm", "b3", "hilbert_ils") if b in args.builders]
    methods += ["portfolio", "oracle_hindsight"]
    print("\nmean test segments per query (rel. to snake); regret = portfolio / oracle_hindsight - 1; "
          "oracle_hindsight = hindsight / in-sample reference (best builder chosen on test queries)")
    print(f"{'shape':>12} {'workload':>11} " + " ".join(f"{m:>17}" for m in methods) + "  regret  picks")
    for s in args.shapes:
        for kind in args.workloads:
            grp = [r for r in rows if r["shape"] == s and r["workload"] == kind]
            if not grp:
                continue
            base = np.mean([r["seg_snake"] for r in grp])
            cells = []
            for m in methods:
                if f"seg_{m}" in grp[0]:
                    v = np.mean([r[f"seg_{m}"] for r in grp])
                    cells.append(f"{v:>8.2f} ({v / base - 1:+.3f})")
                else:
                    cells.append(f"{'-':>17}")
            reg = np.mean([r["seg_portfolio"] for r in grp]) / np.mean([r["seg_oracle_hindsight"] for r in grp]) - 1
            picks = ",".join(r["pick"] for r in grp)
            print(f"{s:>12} {kind:>11} " + " ".join(f"{c:>17}" for c in cells) + f"  {reg:+.3f}  {picks}")
    print(f"\ncsv -> {out}")


if __name__ == "__main__":
    main()
