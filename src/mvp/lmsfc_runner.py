from __future__ import annotations

import argparse
import time

import numpy as np

from .layout import build_artifacts
from .lmsfc.plugin import LMSFCPlugin
from .lmsfc.masks_io import load_masks_json, save_masks_json
from .lmsfc.training_sa import train_sa
from .quantizer import Quantizer
from .query_engine import QueryEngine
from .workloads import generate_points_uniform, generate_queries_loguniform_area


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--qn", type=int, default=1_000)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--b", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--area-min", type=float, default=1e-4)
    ap.add_argument("--area-max", type=float, default=0.2)
    ap.add_argument("--train-sa", action="store_true")
    ap.add_argument("--train-iters", type=int, default=200)
    ap.add_argument("--train-per", type=int, default=10)
    ap.add_argument("--train-restarts", type=int, default=5)
    ap.add_argument("--train-qn", type=int, default=0)
    ap.add_argument("--objective", type=str, default="query_s")
    ap.add_argument("--timing-repeats", type=int, default=3)
    ap.add_argument("--timing-no-warmup", action="store_true")
    ap.add_argument("--semantic", type=str, default="cell", choices=["cell"])
    ap.add_argument("--w-page-accesses", type=float, default=1.0)
    ap.add_argument("--w-segments", type=float, default=1.0)
    ap.add_argument("--masks-file", type=str, default="")
    ap.add_argument("--threshold", type=int, default=None)
    ap.add_argument("--save-masks", type=str, default="")
    args = ap.parse_args()

    rng = np.random.default_rng(int(args.seed))
    points = generate_points_uniform(rng=rng, n=int(args.n), d=2)
    queries = generate_queries_loguniform_area(
        rng=rng,
        qn=int(args.qn),
        d=2,
        area_min=float(args.area_min),
        area_max=float(args.area_max),
    )

    quantizer = Quantizer(d=2, k=int(args.k))
    masks = None
    threshold = None
    if str(args.masks_file):
        mf = load_masks_json(str(args.masks_file))
        if int(mf.d) != 2:
            raise ValueError(f"masks file d={mf.d} != runner d=2")
        if int(mf.k) != int(args.k):
            raise ValueError(f"masks file k={mf.k} != runner k={args.k}")
        masks = list(mf.masks)
        threshold = mf.threshold

    if args.threshold is not None:
        threshold = int(args.threshold)

    train_queries = queries
    test_queries = queries
    if bool(args.train_sa):
        qn_train = int(args.train_qn)
        if qn_train <= 0 or qn_train > len(queries):
            qn_train = len(queries)
        train_queries = queries[:qn_train]
        test_queries = queries[qn_train:] if qn_train < len(queries) else queries

    if bool(args.train_sa):
        r = train_sa(
            points=points,
            queries=train_queries,
            quantizer=quantizer,
            b=int(args.b),
            threshold=threshold,
            objective=str(args.objective),
            semantic=str(args.semantic),
            timing_repeats=int(args.timing_repeats),
            timing_warmup=(not bool(args.timing_no_warmup)),
            iters=int(args.train_iters),
            per=int(args.train_per),
            restarts=int(args.train_restarts),
            w_page_accesses=float(args.w_page_accesses),
            w_segments=float(args.w_segments),
            initial_masks=masks,
            seed=int(args.seed),
        )
        masks = r.masks
        print(f"sa_cost={r.cost:.6f}")
        print(f"sa_masks={masks}")

    if masks is None:
        sfc = LMSFCPlugin.make_standard(d=2, k=int(args.k))
    else:
        sfc = LMSFCPlugin(name="LMSFC-SA", grid_size=tuple([quantizer.m] * 2), masks=masks)
    if threshold is not None:
        sfc.threshold = int(threshold)

    if str(args.save_masks):
        save_masks_json(
            str(args.save_masks),
            d=2,
            k=int(args.k),
            masks=getattr(sfc, "masks", []),
            threshold=getattr(sfc, "threshold", None),
        )

    artifacts = build_artifacts(points=points, sfc=sfc, quantizer=quantizer, b=int(args.b))
    sfc.set_page_start_codes(artifacts.page_start_codes)

    engine = QueryEngine(artifacts=artifacts, sfc=sfc, quantizer=quantizer, semantic=str(args.semantic))

    t0 = time.perf_counter()
    page_accesses = 0
    segments = 0
    util = 0.0
    for q in test_queries:
        r = engine.query(q)
        page_accesses += r.metrics.physical_pages
        segments += r.metrics.segments
        util += r.metrics.mean_page_amp
    t1 = time.perf_counter()

    qn = len(test_queries)
    if bool(args.train_sa):
        print(f"train_queries={len(train_queries)} test_queries={len(test_queries)} n={len(points)} k={args.k} b={args.b}")
    else:
        print(f"queries={qn} n={len(points)} k={args.k} b={args.b}")
    print(f"avg_page_accesses={page_accesses / qn:.4f}")
    print(f"avg_segments={segments / qn:.4f}")
    print(f"avg_page_amplification={util / qn:.6f}")
    print(f"total_time_s={t1 - t0:.4f}")


if __name__ == "__main__":
    main()
