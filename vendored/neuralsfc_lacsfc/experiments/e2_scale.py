"""E2: scale. MWST vs Snake/Hilbert initialisation under the same windowed-ILS budget.

J is estimated from training queries; segments are evaluated on held-out test queries
of the same distribution via Theorem 1 (seg = |Q| - Phi_Q), spot-checked by direct counts.

    python experiments/e2_scale.py --shapes 256x256 1024x1024 32x32x32 16x16x16x16
"""

from __future__ import annotations

import argparse
import csv
import resource
import time
from pathlib import Path

import numpy as np

from lacsfc import (avg_segments, build_curve, build_curve_auto, correlated_boxes,
                    expected_box_size, hilbert_rank, is_ac_path, pair_weights, random_boxes,
                    segment_count, snake_rank, upper_bound_ub0, zorder_rank)
from lacsfc.fastls import windowed_ils
from lacsfc.grid import unravel
from lacsfc.regions import region_curve_auto

WORKLOADS = ("uniform", "hotspot", "mixed", "small", "aniso", "mixed_aniso",
             "diag", "iso", "iso_diag")


def _long_in(d, k, max_frac):
    f = np.full(d, max(max_frac / 10, 1e-9))
    f[k] = 2 * max_frac
    return f


def make_workload(shape, kind, m, rng, max_frac):
    d = len(shape)
    if kind == "uniform":
        return random_boxes(shape, m, rng, max_frac=max_frac)
    if kind == "hotspot":
        return random_boxes(shape, m, rng, max_frac=max_frac, center=np.full(d, 0.3), spread=0.1)
    if kind == "small":
        return random_boxes(shape, m, rng, max_frac=max_frac / 4)
    if kind == "aniso":
        return random_boxes(shape, m, rng, max_frac=_long_in(d, d - 1, max_frac))
    if kind == "mixed_aniso":
        lo1, hi1 = random_boxes(shape, m // 2, rng, max_frac=_long_in(d, 0, max_frac),
                                center=np.full(d, 0.25), spread=0.12)
        lo2, hi2 = random_boxes(shape, m - m // 2, rng, max_frac=_long_in(d, d - 1, max_frac),
                                center=np.full(d, 0.75), spread=0.12)
        return np.vstack([lo1, lo2]), np.vstack([hi1, hi2])
    if kind in ("diag", "iso", "iso_diag"):
        return correlated_boxes(shape, m, rng, max_frac=max_frac, kind=kind)
    lo1, hi1 = random_boxes(shape, m // 2, rng, max_frac=max_frac)
    lo2, hi2 = random_boxes(shape, m - m // 2, rng, max_frac=max_frac / 2,
                            center=np.linspace(0.7, 0.3, d), spread=0.08)
    return np.vstack([lo1, lo2]), np.vstack([hi1, hi2])


def order_of(rank):
    o = np.empty_like(rank)
    o[rank] = np.arange(rank.size, dtype=np.int64)
    return o


def phi_any(order, J, shape):
    """Sum of J over consecutive pairs that are grid-adjacent.

    phi_any is valid for adjacency-continuous orders only; general segment evaluation must use
    direct code-interval counting.
    """
    c = unravel(order, shape)
    diff = c[1:] - c[:-1]
    adj = np.abs(diff).sum(axis=1) == 1
    x, y = order[:-1][adj], order[1:][adj]
    k = np.argmax(np.abs(diff[adj]), axis=1)
    return float(J[k, np.minimum(x, y)].sum())


def mean_segments(order, lo, hi, J, shape):
    if is_ac_path(order, shape, start=None):
        return expected_box_size(lo, hi) - phi_any(order, J, shape)
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size, dtype=order.dtype)
    return avg_segments(rank, lo, hi, shape)


def is_pow2_cube(shape):
    return len(set(shape)) == 1 and shape[0] & (shape[0] - 1) == 0


def run_one(shape, kind, seed, args):
    rng = np.random.default_rng(seed)
    lo, hi = make_workload(shape, kind, args.m_train, rng, args.max_frac)
    tlo, thi = make_workload(shape, kind, args.m_test, rng, args.max_frac)
    t0 = time.perf_counter()
    J = pair_weights(shape, lo, hi)
    sec_J = time.perf_counter() - t0
    Jt = pair_weights(shape, tlo, thi)
    EQt = expected_box_size(tlo, thi)

    t0 = time.perf_counter()
    curve = build_curve(shape, J)
    sec_build = time.perf_counter() - t0
    t0 = time.perf_counter()
    auto_order, auto_perm, _ = build_curve_auto(shape, J)
    sec_auto = time.perf_counter() - t0
    t0 = time.perf_counter()
    b3, b3_tile = region_curve_auto(shape, J, min_tile=args.min_tile)
    sec_b3 = time.perf_counter() - t0

    inits = {"lacsfc": curve.order, "lacsfc_auto": auto_order, "b3": b3.order,
             "snake": order_of(snake_rank(shape))}
    plain = {}
    if is_pow2_cube(shape):
        inits["hilbert"] = order_of(hilbert_rank(shape))
        plain["zorder"] = order_of(zorder_rank(shape))

    row = {"shape": "x".join(map(str, shape)), "d": len(shape), "N": int(np.prod(shape)),
           "workload": kind, "seed": seed, "E_Q_test": EQt,
           "sec_J": sec_J, "sec_build": sec_build, "sec_auto": sec_auto,
           "auto_perm": "".join(map(str, auto_perm)), "sec_b3": sec_b3, "b3_tile": b3_tile,
           "b3_choice_hist": "/".join(f"{n}:{c}" for n, c in zip(
               b3.names, np.bincount(b3.choice, minlength=len(b3.names))))}
    orders = dict(plain)
    for name, o in inits.items():
        orders[name] = o
        res = windowed_ils(o, J, shape, window=args.window, passes=args.passes,
                           iters=args.iters, kick=args.kick, seed=seed)
        orders[f"{name}_ils"] = res.order
        row[f"sec_ils_{name}"] = res.seconds

    ub_train = upper_bound_ub0(J, shape)
    for name, o in orders.items():
        row[f"seg_{name}"] = mean_segments(o, tlo, thi, Jt, shape)
        if is_ac_path(o, shape, start=None):
            row[f"ub0gap_train_{name}"] = 1 - phi_any(o, J, shape) / ub_train if ub_train > 0 else float("nan")

    check = rng.choice(args.m_test, size=min(20, args.m_test), replace=False)
    for name in ("lacsfc_ils", "snake"):
        rank = np.empty_like(orders[name])
        rank[orders[name]] = np.arange(rank.size)
        direct = np.mean([segment_count(rank, tlo[q], thi[q], shape) for q in check])
        Jq = pair_weights(shape, tlo[check], thi[check])
        assert abs(direct - (expected_box_size(tlo[check], thi[check]) - phi_any(orders[name], Jq, shape))) < 1e-6

    ref = "hilbert" if "hilbert" in orders else "snake"
    row["ref"] = ref
    for name in orders:
        row[f"rel_{name}"] = row[f"seg_{name}"] / row[f"seg_{ref}"] - 1
    row["maxrss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", nargs="+", default=["256x256", "1024x1024", "32x32x32",
                                                    "64x64x64", "16x16x16x16"])
    ap.add_argument("--workloads", nargs="+", default=list(WORKLOADS))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--m-train", type=int, default=2000)
    ap.add_argument("--m-test", type=int, default=1000)
    ap.add_argument("--max-frac", type=float, default=0.25)
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--passes", type=int, default=6)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--kick", type=int, default=8)
    ap.add_argument("--min-tile", type=int, default=8)
    ap.add_argument("--out", default=str(Path(__file__).parent / "results" / "e2_scale.csv"))
    args = ap.parse_args()

    rows = []
    for s in args.shapes:
        shape = tuple(int(x) for x in s.split("x"))
        for kind in args.workloads:
            for seed in range(args.seeds):
                row = run_one(shape, kind, seed, args)
                rows.append(row)
                rel = " ".join(f"{k[4:]}={v:+.4f}" for k, v in row.items() if k.startswith("rel_"))
                print(f"{row['shape']:>12} {kind:>8} seed={seed} build={row['sec_build']:.1f}s "
                      f"ils={row['sec_ils_lacsfc']:.1f}s | vs {row['ref']}: {rel}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    methods = ["zorder", "snake", "snake_ils", "hilbert", "hilbert_ils", "lacsfc", "lacsfc_ils",
               "lacsfc_auto", "lacsfc_auto_ils", "b3", "b3_ils"]
    print("\nmean segments per test query (rel. to ref in brackets)")
    print(f"{'shape':>12} {'workload':>8} " + " ".join(f"{m:>18}" for m in methods)
          + f" {'build_s':>8} {'ils_s':>7} {'auto_s':>7} {'b3_s':>7}")
    for s in args.shapes:
        for kind in args.workloads:
            grp = [r for r in rows if r["shape"] == s and r["workload"] == kind]
            cells = []
            for m in methods:
                if f"seg_{m}" in grp[0]:
                    seg = np.mean([r[f"seg_{m}"] for r in grp])
                    rel = np.mean([r[f"rel_{m}"] for r in grp])
                    cells.append(f"{seg:>9.2f} ({rel:+.3f})")
                else:
                    cells.append(f"{'-':>18}")
            print(f"{s:>12} {kind:>8} " + " ".join(f"{c:>18}" for c in cells)
                  + f" {np.mean([r['sec_build'] for r in grp]):>8.1f}"
                  + f" {np.mean([r['sec_ils_lacsfc'] for r in grp]):>7.1f}"
                  + f" {np.mean([r['sec_auto'] for r in grp]):>7.1f}"
                  + f" {np.mean([r['sec_b3'] for r in grp]):>7.1f}")
    print(f"\ncsv -> {out}")


if __name__ == "__main__":
    main()
