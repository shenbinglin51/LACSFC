from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .layout import build_artifacts
from .lmsfc.masks_io import load_masks_json
from .lmsfc.plugin import LMSFCPlugin
from .quantizer import Quantizer
from .sfc_plugins import make_hilbert, make_z_curve
from .types import Interval, QuantizedQuery, SFCPlugin
from .workloads import generate_queries_loguniform_area


def _merge_sorted_codes(codes: Sequence[int]) -> List[Interval]:
    if not codes:
        return []
    out: List[Interval] = []
    start = int(codes[0])
    end = int(codes[0])
    for v in codes[1:]:
        iv = int(v)
        if iv <= end + 1:
            end = iv
        else:
            out.append(Interval(start=start, end=end))
            start = iv
            end = iv
    out.append(Interval(start=start, end=end))
    return out


def _intervals_to_set(intervals: Sequence[Interval], *, max_expand: int) -> set[int]:
    total = 0
    for itv in intervals:
        total += int(itv.end) - int(itv.start) + 1
        if total > int(max_expand):
            raise ValueError(f"interval expansion too large: {total} > {max_expand}")

    out: set[int] = set()
    for itv in intervals:
        out.update(range(int(itv.start), int(itv.end) + 1))
    return out


def _enumerate_cells(low_cell: np.ndarray, high_cell: np.ndarray) -> np.ndarray:
    low = np.asarray(low_cell, dtype=np.int64)
    high = np.asarray(high_cell, dtype=np.int64)
    d = int(low.shape[0])
    grids = [np.arange(int(low[i]), int(high[i]) + 1, dtype=np.int64) for i in range(d)]
    mesh = np.meshgrid(*grids, indexing="ij")
    return np.stack([m.reshape(-1) for m in mesh], axis=1)


def _cells_to_points(cells: np.ndarray, m: int) -> np.ndarray:
    cells = np.asarray(cells, dtype=np.float64)
    return (cells + 0.5) / float(m)


def _make_sfc(*, curve: str, k: int, masks_file: str, threshold: Optional[int]) -> SFCPlugin:
    curve = str(curve)
    if curve == "z":
        return make_z_curve(k=int(k))
    if curve == "hilbert":
        return make_hilbert(k=int(k))
    if curve == "lmsfc_std":
        sfc = LMSFCPlugin.make_standard(d=2, k=int(k))
        if threshold is not None:
            sfc.threshold = int(threshold)
        return sfc
    if curve == "lmsfc":
        if not str(masks_file):
            raise ValueError("--masks-file is required for curve=lmsfc")
        mf = load_masks_json(str(masks_file))
        if int(mf.d) != 2:
            raise ValueError(f"masks file d={mf.d} != 2")
        if int(mf.k) != int(k):
            raise ValueError(f"masks file k={mf.k} != k={k}")
        m = 1 << int(k)
        sfc = LMSFCPlugin(name="LMSFC", grid_size=(m, m), masks=list(mf.masks))
        if mf.threshold is not None:
            sfc.threshold = int(mf.threshold)
        if threshold is not None:
            sfc.threshold = int(threshold)
        return sfc

    raise ValueError("unsupported curve")


@dataclass(frozen=True)
class CheckResult:
    ok: bool
    exact_intervals: List[Interval]
    plugin_intervals: List[Interval]
    missing: List[int]
    extra: List[int]


def _compute_exact_intervals(*, sfc: SFCPlugin, low_cell: np.ndarray, high_cell: np.ndarray) -> List[Interval]:
    cells = _enumerate_cells(low_cell=low_cell, high_cell=high_cell)
    codes = [int(sfc.encode_cell(cells[i])) for i in range(cells.shape[0])]
    codes.sort()
    return _merge_sorted_codes(codes)


def _get_plugin_intervals(*, sfc: SFCPlugin, low_cell: np.ndarray, high_cell: np.ndarray) -> List[Interval]:
    intervals_fn = getattr(sfc, "intervals", None)
    if not callable(intervals_fn):
        raise ValueError("sfc has no intervals()")
    q = QuantizedQuery(low_cell=np.asarray(low_cell, dtype=np.int64), high_cell=np.asarray(high_cell, dtype=np.int64))
    out = intervals_fn(q)
    if out is None:
        raise ValueError("intervals() returned None (missing page_start_codes?)")
    out2 = [Interval(int(it.start), int(it.end)) for it in out]
    out2.sort()
    return out2


def check_one_window(
    *,
    sfc: SFCPlugin,
    low_cell: np.ndarray,
    high_cell: np.ndarray,
    max_expand: int = 1_000_000,
) -> CheckResult:
    exact = _compute_exact_intervals(sfc=sfc, low_cell=low_cell, high_cell=high_cell)
    plugin = _get_plugin_intervals(sfc=sfc, low_cell=low_cell, high_cell=high_cell)

    exact_set = _intervals_to_set(exact, max_expand=max_expand)
    plugin_set = _intervals_to_set(plugin, max_expand=max_expand)

    missing = sorted(exact_set - plugin_set)
    extra = sorted(plugin_set - exact_set)

    return CheckResult(
        ok=(not missing and not extra),
        exact_intervals=exact,
        plugin_intervals=plugin,
        missing=missing,
        extra=extra,
    )


def _attach_page_start_codes(*, sfc: SFCPlugin, k: int, b: int) -> None:
    setter = getattr(sfc, "set_page_start_codes", None)
    if not callable(setter):
        return

    m = 1 << int(k)
    all_cells = _enumerate_cells(low_cell=np.zeros(2, dtype=np.int64), high_cell=np.asarray([m - 1, m - 1], dtype=np.int64))
    points = _cells_to_points(all_cells, m=m)
    quantizer = Quantizer(d=2, k=int(k))
    artifacts = build_artifacts(points=points, sfc=sfc, quantizer=quantizer, b=int(b))
    setter(artifacts.page_start_codes)


def cmd_intervals_check(args: argparse.Namespace) -> None:
    k = int(args.k)
    m = 1 << k

    sfc = _make_sfc(curve=str(args.curve), k=k, masks_file=str(args.masks_file), threshold=args.threshold)
    _attach_page_start_codes(sfc=sfc, k=k, b=int(args.b))

    rng = np.random.default_rng(int(args.seed))

    failures = 0
    for _ in range(int(args.trials)):
        x0 = int(rng.integers(0, m))
        y0 = int(rng.integers(0, m))
        x1 = int(rng.integers(0, m))
        y1 = int(rng.integers(0, m))
        low_cell = np.asarray([min(x0, x1), min(y0, y1)], dtype=np.int64)
        high_cell = np.asarray([max(x0, x1), max(y0, y1)], dtype=np.int64)

        r = check_one_window(sfc=sfc, low_cell=low_cell, high_cell=high_cell, max_expand=int(args.max_expand))
        if not r.ok:
            failures += 1
            print(f"FAIL low={low_cell.tolist()} high={high_cell.tolist()} missing={len(r.missing)} extra={len(r.extra)}")
            if int(args.print_intervals):
                print(f"exact={[(it.start, it.end) for it in r.exact_intervals]}")
                print(f"plugin={[(it.start, it.end) for it in r.plugin_intervals]}")
            if int(args.print_codes):
                if r.missing:
                    print(f"missing_codes={r.missing[:200]}")
                if r.extra:
                    print(f"extra_codes={r.extra[:200]}")

    ok = failures == 0
    print(f"curve={args.curve} k={k} trials={int(args.trials)} ok={ok} failures={failures}")


def cmd_viz(args: argparse.Namespace) -> None:
    k = int(args.k)
    viz_k_obj = getattr(args, "viz_k", None)
    viz_k = int(viz_k_obj) if viz_k_obj is not None else int(k)
    if viz_k <= 0:
        raise ValueError("--viz-k must be > 0")
    if viz_k > k:
        raise ValueError("--viz-k must be <= --k")
    m = 1 << int(viz_k)
    viz_high_bits = bool(getattr(args, "viz_high_bits", False))
    shift = int(k) - int(viz_k)

    sfc = _make_sfc(curve=str(args.curve), k=k, masks_file=str(args.masks_file), threshold=args.threshold)

    cells = _enumerate_cells(low_cell=np.zeros(2, dtype=np.int64), high_cell=np.asarray([m - 1, m - 1], dtype=np.int64))
    codes = np.empty(cells.shape[0], dtype=np.int64)
    for i in range(cells.shape[0]):
        cell = cells[i].astype(np.uint64, copy=False)
        if viz_high_bits and shift > 0:
            cell = (cell << np.uint64(shift)).astype(np.uint64, copy=False)
        codes[i] = int(sfc.encode_cell(cell))

    order = np.argsort(codes, kind="mergesort")

    out_csv = str(args.out_csv)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "code", "x", "y"])
        for rank, idx in enumerate(order.tolist()):
            x, y = cells[int(idx)].tolist()
            w.writerow([rank, int(codes[int(idx)]), int(x), int(y)])

    print(f"wrote_csv={out_csv}")

    if not bool(args.png):
        return

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        print("matplotlib_not_available")
        return

    prefix_bits = int(getattr(args, "prefix_bits", 0) or 0)
    prefix_alpha = float(getattr(args, "prefix_alpha", 0.25) or 0.25)
    prefix_cmap = str(getattr(args, "prefix_cmap", "hsv") or "hsv")

    masks_obj = getattr(sfc, "masks", None)
    masks: Optional[List[int]] = None
    if masks_obj is not None:
        try:
            masks = [int(v) for v in masks_obj]
        except Exception:
            masks = None

    bg = None
    if prefix_bits > 0 and masks:
        varying: List[int] = []

        for mm in masks:
            mm = int(mm)
            pos_asc: List[int] = []
            tmp = int(mm)
            while tmp:
                lsb = tmp & -tmp
                pos_asc.append(int(lsb.bit_length() - 1))
                tmp ^= lsb
            # pdep maps value bit j -> pos_asc[j]
            start_j = int(shift) if viz_high_bits else 0
            end_j = start_j + int(viz_k)
            if start_j < 0:
                start_j = 0
            if end_j > len(pos_asc):
                end_j = len(pos_asc)
            varying.extend(pos_asc[start_j:end_j])

        varying = sorted(set(int(v) for v in varying), reverse=True)
        if varying:
            pb = min(int(prefix_bits), len(varying))
            sel = varying[:pb]
            bg = np.empty((m, m), dtype=np.int64)
            for idx in range(cells.shape[0]):
                x, y = cells[idx].tolist()
                code = int(codes[idx])
                gid = 0
                for b in sel:
                    gid = (gid << 1) | ((code >> int(b)) & 1)
                bg[int(y), int(x)] = int(gid)

    pts = cells[order].astype(np.float64)
    xs = pts[:, 0] + 0.5
    ys = pts[:, 1] + 0.5

    fig = plt.figure(figsize=(6, 6))
    if bg is not None:
        plt.imshow(
            bg,
            origin="lower",
            extent=(0, m, 0, m),
            interpolation="nearest",
            cmap=prefix_cmap,
            alpha=float(prefix_alpha),
        )
    plt.plot(xs, ys, linewidth=0.7)

    label = str(getattr(args, "label", "none"))
    if label not in {"none", "code", "rank"}:
        raise ValueError("--label must be one of: none, code, rank")
    if label != "none":
        label_max_m = int(getattr(args, "label_max_m", 32))
        label_force = bool(getattr(args, "label_force", False))
        if m > label_max_m and not label_force:
            print(f"skip_labeling_m_too_large m={m} label_max_m={label_max_m} (use --label-force to override)")
        else:
            fontsize = float(getattr(args, "label_fontsize", 6.0))
            if label == "code":
                for i in range(cells.shape[0]):
                    x, y = cells[i].tolist()
                    plt.text(float(x) + 0.5, float(y) + 0.5, str(int(codes[i])), ha="center", va="center", fontsize=fontsize)
            else:
                for rank, idx in enumerate(order.tolist()):
                    x, y = cells[int(idx)].tolist()
                    plt.text(float(x) + 0.5, float(y) + 0.5, str(int(rank)), ha="center", va="center", fontsize=fontsize)
    plt.xlim(0, m)
    plt.ylim(0, m)
    plt.gca().set_aspect("equal", adjustable="box")
    title = f"{args.curve} k={k}"
    if int(viz_k) != int(k):
        title += f" viz_k={viz_k}"
        if viz_high_bits:
            title += "(high)"
        else:
            title += "(low)"
    title += f" m={m}"
    if bg is not None:
        title += f" prefix_bits={prefix_bits}"
    plt.title(title)

    out_png = str(args.out_png)
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"wrote_png={out_png}")


def _exact_segments_span(*, sfc: SFCPlugin, low_cell: np.ndarray, high_cell: np.ndarray) -> Tuple[int, int, int]:
    cells = _enumerate_cells(low_cell=low_cell, high_cell=high_cell)
    codes = [int(sfc.encode_cell(cells[i])) for i in range(cells.shape[0])]
    codes.sort()
    n_codes = int(len(codes))
    if n_codes <= 0:
        return 0, 0, 0
    span = int(int(codes[-1]) - int(codes[0]) + 1)
    seg = 1
    prev = int(codes[0])
    for c in codes[1:]:
        ci = int(c)
        if ci != prev + 1:
            seg += 1
        prev = ci
    return int(seg), int(span), int(n_codes)


def cmd_exact_locality(args: argparse.Namespace) -> None:
    curves = [c.strip() for c in str(args.curves).split(",") if c.strip()]
    if not curves:
        raise ValueError("--curves must be non-empty")

    report_quantiles = bool(getattr(args, "quantiles", False))

    k = int(args.k)
    qn = int(args.qn)
    area_min = float(args.area_min)
    area_max = float(args.area_max)
    max_cells = int(args.max_cells)
    seed = int(args.seed)

    rng = np.random.default_rng(seed)
    quantizer = Quantizer(d=2, k=k)
    queries = generate_queries_loguniform_area(rng=rng, qn=qn, d=2, area_min=area_min, area_max=area_max)

    stats = []
    for curve in curves:
        sfc = _make_sfc(curve=str(curve), k=k, masks_file=str(args.masks_file), threshold=args.threshold)

        segs: List[int] = []
        spans: List[int] = []
        ncells: List[int] = []
        span_per_cell: List[float] = []
        skipped = 0
        for q in queries:
            qq = quantizer.quantize_window(q)
            low = np.asarray(qq.low_cell, dtype=np.int64)
            high = np.asarray(qq.high_cell, dtype=np.int64)
            dx = int(abs(int(high[0]) - int(low[0])) + 1)
            dy = int(abs(int(high[1]) - int(low[1])) + 1)
            n = int(dx * dy)
            if n <= 0:
                skipped += 1
                continue
            if max_cells > 0 and n > max_cells:
                skipped += 1
                continue

            seg, span, n_codes = _exact_segments_span(sfc=sfc, low_cell=low, high_cell=high)
            if seg <= 0 or span <= 0 or n_codes <= 0:
                skipped += 1
                continue
            segs.append(int(seg))
            spans.append(int(span))
            ncells.append(int(n_codes))
            span_per_cell.append(float(span) / float(n_codes))

        n_eval = len(segs)
        if n_eval <= 0:
            row = {
                "curve": curve,
                "n_eval": 0,
                "n_skip": int(skipped),
                "mean_cells": 0.0,
                "mean_seg": 0.0,
                "mean_span": 0.0,
                "mean_span_per_cell": 0.0,
                "med_cells": 0.0,
                "med_seg": 0.0,
                "med_span": 0.0,
            }
            if report_quantiles:
                row.update(
                    {
                        "p90_span": 0.0,
                        "p95_span": 0.0,
                        "p99_span": 0.0,
                        "max_span": 0.0,
                        "p90_spc": 0.0,
                        "p95_spc": 0.0,
                        "p99_spc": 0.0,
                        "max_spc": 0.0,
                    }
                )
            stats.append(row)
            continue

        segs_sorted = sorted(segs)
        spans_sorted = sorted(spans)
        ncells_sorted = sorted(ncells)
        spc_sorted = sorted(span_per_cell)

        mean_seg = float(sum(segs) / n_eval)
        mean_span = float(sum(spans) / n_eval)
        mean_cells = float(sum(ncells) / n_eval)
        mean_span_per_cell = float(sum(span_per_cell) / n_eval)

        def median(v: List[int]) -> float:
            if not v:
                return 0.0
            m = len(v)
            if m % 2 == 1:
                return float(v[m // 2])
            return 0.5 * float(v[m // 2 - 1] + v[m // 2])

        def quantile_sorted(v: List[float], q: float) -> float:
            if not v:
                return 0.0
            q = float(q)
            if q <= 0.0:
                return float(v[0])
            if q >= 1.0:
                return float(v[-1])
            n = len(v)
            pos = q * float(n - 1)
            lo = int(pos)
            hi = min(n - 1, lo + 1)
            frac = pos - float(lo)
            return float(v[lo] * (1.0 - frac) + v[hi] * frac)

        def quantile_sorted_int(v: List[int], q: float) -> float:
            return quantile_sorted([float(x) for x in v], q)

        row = {
            "curve": curve,
            "n_eval": int(n_eval),
            "n_skip": int(skipped),
            "mean_cells": float(mean_cells),
            "mean_seg": float(mean_seg),
            "mean_span": float(mean_span),
            "mean_span_per_cell": float(mean_span_per_cell),
            "med_cells": float(median(ncells_sorted)),
            "med_seg": float(median(segs_sorted)),
            "med_span": float(median(spans_sorted)),
        }

        if report_quantiles:
            row.update(
                {
                    "p90_span": float(quantile_sorted_int(spans_sorted, 0.90)),
                    "p95_span": float(quantile_sorted_int(spans_sorted, 0.95)),
                    "p99_span": float(quantile_sorted_int(spans_sorted, 0.99)),
                    "max_span": float(spans_sorted[-1]),
                    "p90_spc": float(quantile_sorted(spc_sorted, 0.90)),
                    "p95_spc": float(quantile_sorted(spc_sorted, 0.95)),
                    "p99_spc": float(quantile_sorted(spc_sorted, 0.99)),
                    "max_spc": float(spc_sorted[-1]),
                }
            )

        stats.append(row)

    cols = [
        "curve",
        "n_eval",
        "n_skip",
        "mean_cells",
        "mean_seg",
        "mean_span",
        "mean_span_per_cell",
        "med_cells",
        "med_seg",
        "med_span",
    ]
    if report_quantiles:
        cols.extend(
            [
                "p90_span",
                "p95_span",
                "p99_span",
                "max_span",
                "p90_spc",
                "p95_spc",
                "p99_spc",
                "max_spc",
            ]
        )

    print("\t".join(cols))
    for r in stats:
        parts: List[str] = []
        for c in cols:
            v = r.get(c)
            if c in {"curve"}:
                parts.append(str(v))
            elif c in {"n_eval", "n_skip"}:
                parts.append(str(int(v or 0)))
            elif c.endswith("_spc") or c in {"mean_span_per_cell"}:
                parts.append(f"{float(v or 0.0):.4f}")
            else:
                parts.append(f"{float(v or 0.0):.2f}")
        print("\t".join(parts))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_chk = sub.add_parser("intervals-check")
    ap_chk.add_argument(
        "--curve",
        type=str,
        default="lmsfc_std",
        choices=["z", "hilbert", "lmsfc_std", "lmsfc"],
    )
    ap_chk.add_argument("--k", type=int, default=4)
    ap_chk.add_argument("--b", type=int, default=1)
    ap_chk.add_argument("--seed", type=int, default=0)
    ap_chk.add_argument("--trials", type=int, default=50)
    ap_chk.add_argument("--masks-file", type=str, default="")
    ap_chk.add_argument("--threshold", type=int, default=None)
    ap_chk.add_argument("--max-expand", type=int, default=1_000_000)
    ap_chk.add_argument("--print-intervals", type=int, default=1)
    ap_chk.add_argument("--print-codes", type=int, default=0)
    ap_chk.set_defaults(func=cmd_intervals_check)

    ap_viz = sub.add_parser("viz")
    ap_viz.add_argument(
        "--curve",
        type=str,
        default="z",
        choices=["z", "hilbert", "lmsfc_std", "lmsfc"],
    )
    ap_viz.add_argument("--k", type=int, default=4)
    ap_viz.add_argument("--viz-k", type=int, default=None)
    ap_viz.add_argument("--viz-high-bits", action="store_true")
    ap_viz.add_argument("--masks-file", type=str, default="")
    ap_viz.add_argument("--threshold", type=int, default=None)
    ap_viz.add_argument("--out-csv", type=str, default="./curve_order.csv")
    ap_viz.add_argument("--png", action="store_true")
    ap_viz.add_argument("--out-png", type=str, default="./curve_path.png")
    ap_viz.add_argument("--prefix-bits", type=int, default=0)
    ap_viz.add_argument("--prefix-alpha", type=float, default=0.25)
    ap_viz.add_argument("--prefix-cmap", type=str, default="hsv")
    ap_viz.add_argument("--label", type=str, default="none", choices=["none", "code", "rank"])
    ap_viz.add_argument("--label-fontsize", type=float, default=6.0)
    ap_viz.add_argument("--label-max-m", type=int, default=32)
    ap_viz.add_argument("--label-force", action="store_true")
    ap_viz.set_defaults(func=cmd_viz)

    ap_loc = sub.add_parser("exact-locality")
    ap_loc.add_argument("--curves", type=str, default="z,lmsfc_std,lmsfc")
    ap_loc.add_argument("--k", type=int, default=8)
    ap_loc.add_argument("--qn", type=int, default=200)
    ap_loc.add_argument("--seed", type=int, default=0)
    ap_loc.add_argument("--area-min", type=float, default=1e-4)
    ap_loc.add_argument("--area-max", type=float, default=0.2)
    ap_loc.add_argument("--max-cells", type=int, default=20000)
    ap_loc.add_argument("--quantiles", action="store_true")
    ap_loc.add_argument("--masks-file", type=str, default="")
    ap_loc.add_argument("--threshold", type=int, default=None)
    ap_loc.set_defaults(func=cmd_exact_locality)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
