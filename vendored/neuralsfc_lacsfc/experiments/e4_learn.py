"""E4: learned backbone weights (FG/EG, paper Sec. 5.4) vs analytic W0 = DeltaPhi(J).

Objective: mean I/O cost C = c_seek * seeks + c_page * pages on the point layout.
Training queries drive J, W0 and the oracle; validation queries drive the safeguard;
all methods are reported on independent test queries. Small grids only (max 2^16 cells).

    python experiments/e4_learn.py --datasets synth geonames --shapes 64x64 16x16x16
"""

from __future__ import annotations

import argparse
import csv
import resource
import time
from pathlib import Path

import numpy as np

from lacsfc import hilbert_rank, pair_weights, snake_rank
from lacsfc.learn import CostOracle, LearnConfig, learn_weights
from lacsfc.storage import bin_points, data_boxes

ROOT = Path(__file__).resolve().parents[1]


def synth(n, d, rng):
    k = 6
    centers = rng.random((k, d))
    scales = rng.uniform(0.02, 0.1, size=(k, 1))
    lab = rng.integers(0, k, size=n)
    x = centers[lab] + scales[lab] * rng.standard_normal((n, d))
    x[: n // 10] = rng.random((n // 10, d))
    return np.clip(x, 0, 1)


def load(name, d, rng, n_synth):
    if name == "synth":
        return synth(n_synth, d, rng)
    from e3_real import load_geonames
    return load_geonames(ROOT / "data" / "geonames" / "cities500.txt")[:, :d]


def rank_of(order):
    r = np.empty_like(order)
    r[order] = np.arange(order.size, dtype=r.dtype)
    return r


def run_one(name, shape, seed, args):
    rng = np.random.default_rng(seed)
    X = load(name, len(shape), rng, args.n_synth)
    cells, coords = bin_points(X, shape, args.binning)
    tr = data_boxes(coords, shape, args.m_train, rng, args.max_frac)
    va = data_boxes(coords, shape, args.m_val, rng, args.max_frac)
    te = data_boxes(coords, shape, args.m_test, rng, args.max_frac)
    J = pair_weights(shape, *tr)
    oracle = CostOracle(shape, J, cells, *tr, page_size=args.page,
                        c_seek=args.c_seek, c_page=args.c_page, max_cells=args.max_cells)
    cfg = LearnConfig(rounds=args.rounds, init_samples=args.init_samples,
                      samples_per_round=args.samples_per_round, e_steps=args.e_steps,
                      f_steps=args.f_steps, hidden=args.hidden, seed=seed, threads=args.threads,
                      device=args.device)
    t0 = time.perf_counter()
    res = learn_weights(oracle, cells, cfg, val=va)
    dt = time.perf_counter() - t0

    def test_cost(rank):
        return float(oracle.costs(rank, *te).mean())

    ranks = {"hilbert": hilbert_rank(shape), "snake": snake_rank(shape),
             "W0": oracle.curve(res.W0).rank, "learned": oracle.curve(res.W_learned).rank,
             "search": oracle.curve(res.W_search).rank, "deployed": res.curve.rank}
    row = dict(dataset=name, shape="x".join(map(str, shape)), seed=seed, choice=res.choice,
               oracle_calls=res.oracle_calls, train_s=round(dt, 2),
               train_W0=-res.train_score_baseline, train_learned=-res.train_score_learned,
               train_search=-res.train_score_search,
               val_W0=res.val_cost_baseline, val_learned=res.val_cost_learned)
    for k, r in ranks.items():
        row[f"test_{k}"] = test_cost(r)
    row["gain_learned_vs_W0_%"] = 100 * (row["test_W0"] - row["test_learned"]) / row["test_W0"]
    row["gain_deployed_vs_W0_%"] = 100 * (row["test_W0"] - row["test_deployed"]) / row["test_W0"]
    row["maxrss_MB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["synth", "geonames"])
    ap.add_argument("--shapes", nargs="+", default=["64x64", "16x16x16"])
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--binning", default="quantile")
    ap.add_argument("--n-synth", type=int, default=100_000)
    ap.add_argument("--m-train", type=int, default=400)
    ap.add_argument("--m-val", type=int, default=400)
    ap.add_argument("--m-test", type=int, default=1000)
    ap.add_argument("--max-frac", type=float, default=0.25)
    ap.add_argument("--page", type=int, default=64)
    ap.add_argument("--c-seek", type=float, default=1.0)
    ap.add_argument("--c-page", type=float, default=1.0)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--init-samples", type=int, default=32)
    ap.add_argument("--samples-per-round", type=int, default=16)
    ap.add_argument("--e-steps", type=int, default=100)
    ap.add_argument("--f-steps", type=int, default=50)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-cells", type=int, default=1 << 16)
    ap.add_argument("--out", default=str(ROOT / "experiments" / "results" / "e4_learn.csv"))
    args = ap.parse_args()

    rows = []
    for name in args.datasets:
        for s in args.shapes:
            shape = tuple(int(v) for v in s.split("x"))
            for seed in range(args.seeds):
                row = run_one(name, shape, seed, args)
                rows.append(row)
                print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()},
                      flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
