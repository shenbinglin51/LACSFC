"""E5: amortised cross-instance weight generator (paper Sec. 5.3 Tier 3).

A task = (dataset, grid, workload family, page size, seed). Each task has three disjoint
query sets: A (features, J_A, W0, per-instance LS), B (FG training loss on train tasks;
validation for the safeguard on test tasks), T (reporting only). FG_theta is trained by
cross-task ES on cost(B) / cost_W0(B) - 1 over TRAIN tasks and applied to unseen TEST tasks
with one forward pass. Splits:
  E5a  test tasks = new seeds (new synthetic data / new queries), all families;
  E5b  --holdout-family F: train tasks exclude F, test tasks are F only.
  E5c  --shape-feats adds axis-relative workload-shape features (amortize.workload_features);
       --augment elong adds training tasks with random-axis, random-ratio elongated boxes.
  E6   --load-theta reuses a trained FG (no training) on --test-shapes / --test-seed-start;
       --ls-fg-budget runs per-instance LS warm-started from the FG tree (Tier 3 + Tier 2);
       family popw = data-centred boxes whose centres are drawn with probability ~ sqrt(population)
       (GeoNames) or a seeded log-normal point popularity (synth).

    python experiments/e5_amortized.py --out-tag e5a
    python experiments/e5_amortized.py --holdout-family iso_diag --out-tag e5b_iso_diag
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

import ch7_make_tasks as c7
from e2_scale import make_workload
from e4_learn import synth
from lacsfc import hilbert_rank, max_spanning_tree, pair_weights
from lacsfc import tasks as lt
from lacsfc.amortize import ESConfig, MLP, edge_levels, es_train, task_features, workload_features
from lacsfc.construct import build_curve, delta_phi, merge_squares
from lacsfc.grid import block_shape, num_cells
from lacsfc.learn import CostOracle, tree_local_search
from lacsfc.storage import bin_points, data_boxes
from lacsfc.workloads import random_boxes

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("data", "data_iso", "uniform", "hotspot", "aniso", "iso_diag")


@lru_cache(maxsize=None)
def _geonames_full():
    from e3_real import load_geonames
    return load_geonames(ROOT / "data" / "geonames" / "cities500.txt")


def _geonames(d):
    return _geonames_full()[:, :d]


def popw_boxes(coords, shape, m, rng, max_frac, p):
    """data_boxes with centre points drawn with probability p (independent log-uniform sides)."""
    sides = np.asarray(shape, dtype=np.int64)
    max_len = np.maximum(1, np.floor(max_frac * sides)).astype(np.float64)
    length = np.clip(np.rint(max_len ** rng.random((m, len(shape)))), 1, max_len).astype(np.int64)
    c = np.asarray(coords, dtype=np.int64)[rng.choice(len(coords), size=m, p=p)]
    lo = np.clip(c - length // 2, 0, sides - length)
    return lo, lo + length - 1


def queries(family, coords, shape, m, rng, max_frac, frac=None):
    if family == "elong":
        return random_boxes(shape, m, rng, max_frac=frac)
    if family == "popw":
        return popw_boxes(coords, shape, m, rng, max_frac, frac)
    if family == "data":
        return data_boxes(coords, shape, m, rng, max_frac)
    if family == "data_iso":
        return data_boxes(coords, shape, m, rng, max_frac, iso=True)
    return make_workload(shape, family, m, rng, max_frac)


def e5_queries(dataset, shape, family, X, rng, args):
    d = len(shape)
    cells, coords = bin_points(X, shape, args.binning)
    frac = None
    if family == "elong":
        k, r = int(rng.integers(d)), float(np.exp(rng.uniform(np.log(2.0), np.log(args.elong_max))))
        frac = np.full(d, args.max_frac / np.sqrt(r))
        frac[k] = min(args.max_frac * np.sqrt(r), 1.0)
    elif family == "popw":
        w = (np.sqrt(np.expm1(_geonames_full()[:, 2])) if dataset == "geonames"
             else rng.lognormal(0.0, 1.5, size=len(coords)))
        frac = w / w.sum()
    A = queries(family, coords, shape, args.m_a, rng, args.max_frac, frac)
    B = queries(family, coords, shape, args.m_b, rng, args.max_frac, frac)
    T = queries(family, coords, shape, args.m_t, rng, args.max_frac, frac)
    return cells, A, B, T


def tasks_gen_queries(dataset, shape, family, seed, X, args):
    """--task-gen tasks: cells and A / B / T cell boxes from lacsfc.tasks.make_task, as in
    ch7_make_tasks.build_task (equal-width 2^K grid, task seed from (dataset, d, family, seed)).
    Family data_skew_rand = data_skew with the long axis drawn from the task seed."""
    d = len(shape)
    K = int(shape[0]).bit_length() - 1
    if any(int(s) != 1 << K for s in shape):
        raise ValueError(f"--task-gen tasks needs a 2^K cube grid, got {shape}")
    tseed = c7.task_seed(dataset, d, family, seed)
    if family == "data_skew_rand":
        ax = int(np.random.default_rng(tseed).integers(d))
        spec_AB = spec_T = lt.WorkloadSpec(center="data", axis_skew=True, skew_axis=ax)
    else:
        spec_AB, spec_T = c7.specs(family)
    t = lt.make_task(X, K, spec_AB, spec_T, mA=args.m_a, mB=args.m_b, mT=args.m_t, seed=tseed,
                     binning="equal")
    cells = np.ravel_multi_index(tuple(t["cells"].T), shape).astype(np.int64)
    return cells, (t["A_low"], t["A_high"]), (t["B_low"], t["B_high"]), (t["T_low"], t["T_high"])


@dataclass
class Task:
    key: dict
    shape: tuple
    feat: np.ndarray
    w0n: np.ndarray
    dphi: np.ndarray
    lvl: np.ndarray
    oB: CostOracle
    oA: CostOracle
    te: tuple
    cB_w0: float


def make_task(dataset, shape, family, page, seed, args) -> Task:
    rng = np.random.default_rng(seed)
    d = len(shape)
    X = synth(args.n_synth, d, rng) if dataset == "synth" else _geonames(d)
    if args.task_gen == "tasks":
        cells, A, B, T = tasks_gen_queries(dataset, shape, family, seed, X, args)
    else:
        cells, A, B, T = e5_queries(dataset, shape, family, X, rng, args)
    J = pair_weights(shape, *A)
    oA = CostOracle(shape, J, cells, *A, page_size=page, max_cells=args.max_cells)
    oB = CostOracle(shape, J, cells, *B, page_size=page, max_cells=args.max_cells)
    sq = oA.sq
    feat, w0n = task_features(shape, J, cells, sq, page, d_max=args.d_max)
    if args.shape_feats:
        feat = np.column_stack([feat, workload_features(shape, *A, J, sq)])
    dphi = delta_phi(J, sq)
    cB_w0 = -oB.score(dphi)
    key = dict(dataset=dataset, shape="x".join(map(str, shape)), family=family, page=page,
               seed=seed)
    return Task(key, shape, feat, w0n, dphi, edge_levels(shape, sq), oB, oA, T, cB_w0)


def task_keys(seeds, families, args):
    return [(ds, tuple(int(v) for v in s.split("x")), fam, page, seed)
            for ds in args.datasets for s in args.shapes for fam in families
            for page in args.pages for seed in seeds]


_W = {}  # state inherited by forked workers: train, net, theta, args


def _build_key(key):
    return make_task(*key, _W["args"])


def build_tasks(seeds, families, args, workers=1):
    keys = task_keys(seeds, families, args)
    if workers > 1:
        _W["args"] = args
        with mp.get_context("fork").Pool(workers) as pool:
            return pool.map(_build_key, keys, chunksize=1)
    return [make_task(*k, args) for k in keys]


def _train_loss(job):
    theta, i = job
    t, net = _W["train"][i], _W["net"]
    return -t.oB.score(t.w0n + net(theta, t.feat)) / t.cB_w0 - 1.0


def _eval_train(i):
    return evaluate(_W["train"][i], _W["net"], _W["theta"], _W["args"])


def _eval_key(key):
    return evaluate(make_task(*key, _W["args"]), _W["net"], _W["theta"], _W["args"])


def eval_stream(keys, workers):
    """Build, evaluate and discard one task at a time (in forked workers if workers > 1)."""
    if workers > 1:
        with mp.get_context("fork").Pool(workers) as pool:
            return pool.map(_eval_key, keys, chunksize=1)
    return [_eval_key(k) for k in keys]


def evaluate(task: Task, net, theta, args):
    o, T = task.oB, task.te
    nb, sq = o.n_blocks, o.sq
    W_fg = task.w0n + net(theta, task.feat)
    W_hier = task.dphi / max(float(np.abs(task.dphi).max()), 1e-12) - args.hier_beta * task.lvl
    ranks = {"hilbert": hilbert_rank(task.shape),
             "w0": o.curve(task.dphi).rank,
             "hier": o.curve(W_hier).rank,
             "fg": o.curve(W_fg).rank}
    if args.ls_budget > 0:
        t0 = max_spanning_tree(nb, sq.block_a, sq.block_b, task.dphi)
        tree, _, _ = tree_local_search(task.oA, t0, args.ls_budget,
                                       np.random.default_rng(task.key["seed"]))
        ranks["ls"] = build_curve(task.shape, o.J, tree).rank
    if args.ls_fg_budget > 0:
        t0 = max_spanning_tree(nb, sq.block_a, sq.block_b, W_fg)
        tree, _, _ = tree_local_search(task.oA, t0, args.ls_fg_budget,
                                       np.random.default_rng(task.key["seed"]))
        ranks["lsfg"] = build_curve(task.shape, o.J, tree).rank
    cB = {k: float(o.costs(r).mean()) for k, r in ranks.items()}
    cT = {k: float(o.costs(r, *T).mean()) for k, r in ranks.items()}
    for name, cands in (("safe_base", ("w0", "hilbert")), ("safe_fg", ("fg", "w0", "hilbert"))):
        pick = min(cands, key=lambda k: cB[k])
        cT[name] = cT[pick]
        cT[name + "_pick"] = pick
    row = dict(task.key)
    for k, v in cT.items():
        row[f"T_{k}"] = v
    for k, v in cB.items():
        row[f"B_{k}"] = v
    return row


def summarise(rows, label):
    methods = [m for m in ("w0", "hier", "fg", "ls", "lsfg", "safe_base", "safe_fg")
               if f"T_{m}" in rows[0]]
    H = np.array([r["T_hilbert"] for r in rows])
    W = np.array([r["T_w0"] for r in rows])
    print(f"\n== {label}: {len(rows)} tasks; test C rel. to Hilbert / W0 (mean, 95% CI half-width; + = better)")
    for m in methods:
        x = np.array([r[f"T_{m}"] for r in rows])
        for ref, R in (("H", H), ("W0", W)):
            g = 100 * (R - x) / R
            ci = 1.96 * g.std(ddof=1) / np.sqrt(g.size) if g.size > 1 else float("nan")
            print(f"  {m:>10} vs {ref:<3} {g.mean():+7.2f}% ± {ci:5.2f}   wins {np.mean(g > 0):.2f}")
    fams = sorted({r["family"] for r in rows})
    print("  per family, page: fg / hier / w0 / safe_fg vs Hilbert (%)")
    for f in fams:
        for p in sorted({r["page"] for r in rows}):
            g = [r for r in rows if r["family"] == f and r["page"] == p]
            if not g:
                continue
            h = np.array([r["T_hilbert"] for r in g])
            vals = [np.mean(100 * (h - np.array([r[f"T_{m}"] for r in g])) / h)
                    for m in ("fg", "hier", "w0", "safe_fg")]
            print(f"    {f:>9} B={p:<5} " + " ".join(f"{v:+7.2f}" for v in vals))


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["synth", "geonames"])
    ap.add_argument("--shapes", nargs="+", default=["64x64", "16x16x16"])
    ap.add_argument("--families", nargs="+", default=list(FAMILIES))
    ap.add_argument("--holdout-family", default=None)
    ap.add_argument("--shape-feats", action="store_true")
    ap.add_argument("--augment", nargs="*", default=[], choices=["elong"])
    ap.add_argument("--load-theta", default=None)
    ap.add_argument("--test-shapes", nargs="+", default=None)
    ap.add_argument("--test-families", nargs="+", default=None)
    ap.add_argument("--test-seed-start", type=int, default=100)
    ap.add_argument("--ls-fg-budget", type=int, default=0)
    ap.add_argument("--elong-max", type=float, default=8.0)
    ap.add_argument("--pages", type=int, nargs="+", default=[256, 1024])
    ap.add_argument("--binning", default="quantile")
    ap.add_argument("--train-seeds", type=int, default=2)
    ap.add_argument("--test-seeds", type=int, default=1)
    ap.add_argument("--n-synth", type=int, default=100_000)
    ap.add_argument("--m-a", type=int, default=400)
    ap.add_argument("--m-b", type=int, default=400)
    ap.add_argument("--m-t", type=int, default=1000)
    ap.add_argument("--max-frac", type=float, default=0.25)
    ap.add_argument("--max-cells", type=int, default=1 << 16)
    ap.add_argument("--d-max", type=int, default=3)
    ap.add_argument("--task-gen", default="e5", choices=["e5", "tasks"])
    ap.add_argument("--train-seed-start", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--pop", type=int, default=16)
    ap.add_argument("--tasks-per-iter", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--hier-beta", type=float, default=1.0)
    ap.add_argument("--ls-budget", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-tag", default="e5a")
    args = ap.parse_args()

    res_dir = ROOT / "experiments" / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    if args.holdout_family:
        tr_fams = [f for f in args.families if f != args.holdout_family]
        te_fams = [args.holdout_family]
    else:
        tr_fams = te_fams = list(args.families)
    tr_fams = tr_fams + [f for f in args.augment if f not in tr_fams]

    if args.test_families:
        te_fams = list(args.test_families)

    t0 = time.perf_counter()
    if args.load_theta:
        z = np.load(args.load_theta)
        args.d_max = int(z["d_max"]) if "d_max" in z.files else 3
        net = MLP(int(z["n_in"]), int(z["hidden"]))
        args.shapes = args.test_shapes or args.shapes
        keys = task_keys(range(args.test_seed_start, args.test_seed_start + args.test_seeds),
                         te_fams, args)
        _W.update(net=net, theta=z["theta"], args=args)
        rows_te = eval_stream(keys, args.workers)
        write_csv(res_dir / f"{args.out_tag}_test.csv", rows_te)
        summarise(rows_te, f"TEST tasks (loaded {args.load_theta})")
        print(f"total {time.perf_counter() - t0:.0f}s")
        return

    train = build_tasks(range(args.train_seed_start, args.train_seed_start + args.train_seeds),
                        tr_fams, args, args.workers)
    print(f"built {len(train)} train tasks in {time.perf_counter() - t0:.1f}s", flush=True)

    net = MLP(train[0].feat.shape[1], args.hidden)
    theta0 = net.init(np.random.default_rng(args.seed))
    _W.update(train=train, net=net, args=args)
    pool = mp.get_context("fork").Pool(args.workers) if args.workers > 1 else None

    def loss(theta, i):
        return _train_loss((theta, i))

    def map_fn(jobs):
        return pool.map(_train_loss, [(th, int(i)) for th, i in jobs], chunksize=1)

    def all_losses(theta):
        jobs = [(theta, i) for i in range(len(train))]
        return map_fn(jobs) if pool else [loss(th, i) for th, i in jobs]

    def cb(it, theta):
        if it % 10 == 0 or it == args.iters - 1:
            full = float(np.mean(all_losses(theta)))
            print(f"iter {it:4d}  train-B rel. cost vs W0 {100 * full:+.3f}%  "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
            return dict(train_rel=full)
        return None

    cfg = ESConfig(iters=args.iters, pop=args.pop, tasks_per_iter=args.tasks_per_iter,
                   sigma=args.sigma, lr=args.lr, seed=args.seed)
    res = es_train(loss, len(train), theta0, cfg, callback=cb, map_fn=map_fn if pool else None)
    if pool:
        pool.close()
        pool.join()
    np.savez(res_dir / f"{args.out_tag}_theta.npz", theta=res.theta, n_in=net.n_in,
             hidden=net.hidden, d_max=args.d_max)
    write_csv(res_dir / f"{args.out_tag}_history.csv", res.history)

    _W["theta"] = res.theta
    args.shapes = args.test_shapes or args.shapes
    keys = task_keys(range(args.test_seed_start, args.test_seed_start + args.test_seeds), te_fams, args)
    rows_te = eval_stream(keys, args.workers)
    if rows_te:
        write_csv(res_dir / f"{args.out_tag}_test.csv", rows_te)
    ls_budget, args.ls_budget = args.ls_budget, 0
    if args.workers > 1:
        with mp.get_context("fork").Pool(args.workers) as p:
            rows_tr = p.map(_eval_train, range(len(train)), chunksize=1)
    else:
        rows_tr = [_eval_train(i) for i in range(len(train))]
    args.ls_budget = ls_budget
    write_csv(res_dir / f"{args.out_tag}_train.csv", rows_tr)
    summarise(rows_tr, "TRAIN tasks (T queries; FG saw these tasks' B queries)")
    if rows_te:
        summarise(rows_te, "TEST tasks (unseen)")
    print(f"total {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
