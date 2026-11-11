"""Aggregate EXP-1110 sensitivity sweeps and the scale sweep into CSV / Markdown / LaTeX tables.

Inputs
  results/exp_1110_sweeps/cells/<sweep>__<value>__<workload>__s<seed>.jsonl   (run_exp_1110_sweeps.sh)
  results/exp_0530_layerC_scale_stage2_v1/jsonl/C__<wl>__n<N>__mixed__<curve>.jsonl  (EXP-0530 Layer C)
  results/exp_1110_scale_supplement/jsonl/C__<wl>__n<N>__mixed__<curve>.jsonl         (run_exp_1110_scale_supplement.sh)
For the scale sweep, Hilbert rows from the supplement are only used as an anchor check against Layer C.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METRICS = [("query_us", "latency ($\\mu$s/query, Python engine)", 1.0), ("avg_pa", "avg pages accessed", 1.0),
           ("avg_seg", "avg\\_seg", 1.0), ("avg_amp", "avg\\_amp", 1.0),
           ("build_s", "index construction time (s)", 1.0)]
NAME = {"ZCurve": "Z-curve", "AdaCurve": "AdaCurve$^\\dagger$", "GBO": "GBO$^\\ddagger$"}
# Display row order: the three LACSFC-* variants are collapsed into one "LACSFC" row
# (per-seed best across the variants present in each cell). Per-variant rows are kept
# in the CSV output only.
ORDER = ["RowMajor", "ColumnMajor", "ZCurve", "Hilbert", "LMSFC-Std", "LMSFC-BO", "BitAllocZ-BO", "PiecewiseSFC",
         "AdaCurve", "GBO", "QUILTS", "LACSFC"]
LACSFC_VARIANTS = {"LACSFC-H", "LACSFC-Opt", "LACSFC-Meta"}
LACSFC_AGG = "LACSFC"
EXCLUDE_FROM_BEST = {"AdaCurve"}
SWEEP_TITLE = {
    "grid": "grid resolution $k$", "sel": "query window area", "aspect": "query aspect ratio",
    "page": "page size $B$", "train": "number of training queries", "shift": "train$\\to$test distribution",
    "scale": "data size $N$ (v2: all methods, 120/120 queries)",
    "trainper": "LMSFC-BO training sample ($N/p$ points, $N=1$M)",
}
CELL_RE = re.compile(r"^(?P<sweep>[a-z]+)__(?P<val>.+)__(?P<wl>[a-z0-9_]+?)__s(?P<seed>\d+)$")
SCALE_RE = re.compile(r":C:(?P<wl>[a-z0-9_]+):n(?P<n>\d+):mixed:(?P<curve>[a-z_]+):s(?P<seed>\d+)$")


def _val_key(v: str):
    m = re.search(r"[-+]?\d*\.?\d+(e[-+]?\d+)?", v)
    return (0, float(m.group(0)), v) if m and not v.startswith(("hot", "uni")) else (1, 0.0, v)


def _metric(res, met, n_test):
    if met == "query_us":
        return None if res.get("query_s") is None else float(res["query_s"]) * 1e6 / n_test
    return None if res.get(met) is None else float(res[met])


def load_sweeps(cell_dirs, n_test):
    """-> data[(sweep, wl)][val][curve][metric] = {seed: value}"""
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    seen = set()
    for f in [p for d in cell_dirs for p in sorted(Path(d).glob("*.jsonl"))]:
        m = CELL_RE.match(f.stem)
        if not m:
            continue
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            for curve, res in r["results"].items():
                if (f.stem, curve) in seen:
                    continue
                seen.add((f.stem, curve))
                seed = int(m["seed"])
                for met, _, _ in METRICS:
                    v = _metric(res, met, n_test)
                    if v is not None:
                        data[(m["sweep"], m["wl"])][m["val"]][curve][met][seed] = v
    return data


def load_scale(dirs, n_test):
    """-> data[wl][N][curve][metric] = {seed: value}; first dir wins (Layer C before supplement)."""
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    anchor = []
    for src_i, d in enumerate(dirs):
        for f in sorted(Path(d).glob("C__*.jsonl")):
            for line in f.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                m = SCALE_RE.search(r.get("exp", ""))
                if not m:
                    continue
                (curve, res), = r["results"].items()
                wl, n, seed = m["wl"], int(m["n"]), int(m["seed"])
                for met, _, _ in METRICS:
                    v = _metric(res, met, n_test)
                    if v is None:
                        continue
                    slot = data[wl][n][curve][met]
                    if seed in slot:
                        if met not in ("query_us", "build_s") and src_i > 0:
                            anchor.append((wl, n, curve, seed, met, slot[seed], v))
                        continue
                    slot[seed] = v
    return data, anchor


def _fmt(v: np.ndarray, scale: float) -> str:
    v = v * scale
    return f"{v.mean():.3f}" if len(v) < 2 else f"{v.mean():.3f}±{v.std(ddof=1):.3f}"


def _rows(curves):
    return [c for c in ORDER if c in curves] + sorted(c for c in curves if c not in ORDER)


def _lacsfc_agg(curve_map):
    """curve_map: {curve: {metric: {seed: value}}} -> {metric: {seed: min over variants present}}.

    The LACSFC aggregate reports, for every seed, the best value among whichever
    LACSFC-* variants were run in that cell (best-of-2 in sweeps, best-of-3 where
    Meta also ran)."""
    out = defaultdict(dict)
    for c in LACSFC_VARIANTS & set(curve_map):
        for met, slot in curve_map[c].items():
            for s, v in slot.items():
                if s in out[met]:
                    out[met][s] = min(out[met][s], v)
                else:
                    out[met][s] = v
    return out


def _display_curves(curve_map):
    """Curve names for the table: variants hidden, single LACSFC_AGG row added."""
    cs = [c for c in curve_map if c not in LACSFC_VARIANTS]
    if LACSFC_VARIANTS & set(curve_map):
        cs.append(LACSFC_AGG)
    return cs


def emit(title, cols, get, out_md, out_tex, out_csv, tag):
    """cols: list of (key, header); get(curve, key, metric) -> np.ndarray or None"""
    curves = set()
    for key, _ in cols:
        curves |= set(get(None, key, None) or [])
    rows = _rows(curves)
    for met, met_tex, scale in METRICS:
        best = {}
        for key, _ in cols:
            means = [get(c, key, met).mean() for c in rows if c not in EXCLUDE_FROM_BEST and get(c, key, met) is not None]
            best[key] = min(means) if means else None
        out_md.append(f"\n### {title} — {met}\n")
        out_md.append("| Method | " + " | ".join(h for _, h in cols) + " |")
        out_md.append("|---|" + "---|" * len(cols))
        tex = ["\\begin{table}[t]", "\\centering\\scriptsize",
               f"\\caption{{{title}: {met_tex} (mean$\\pm$s.d. over seeds; lower is better).}}",
               f"\\label{{tab:exp1110_{tag}_{met}}}", "\\begin{tabular}{l" + "r" * len(cols) + "}", "\\toprule",
               "Method & " + " & ".join(h for _, h in cols) + " \\\\", "\\midrule"]
        for c in rows:
            md_cells, tex_cells = [], []
            for key, _ in cols:
                v = get(c, key, met)
                if v is None:
                    md_cells.append("N/A")
                    tex_cells.append("--")
                    continue
                s = _fmt(v, scale)
                out_csv.append([tag, met, c, key, len(v), float(v.mean() * scale),
                                float(v.std(ddof=1) * scale) if len(v) > 1 else ""])
                is_best = c not in EXCLUDE_FROM_BEST and best[key] is not None and np.isclose(v.mean(), best[key])
                md_cells.append(f"**{s}**" if is_best else s)
                t = s.replace("±", "\\,$\\pm$\\,")
                tex_cells.append("\\textbf{" + t + "}" if is_best else t)
            out_md.append(f"| {c} | " + " | ".join(md_cells) + " |")
            tex.append(NAME.get(c, c) + " & " + " & ".join(tex_cells) + " \\")
        # Per-variant LACSFC detail: CSV only (display tables show the aggregate row).
        for c in sorted(LACSFC_VARIANTS):
            for key, _ in cols:
                v = get(c, key, met)
                if v is not None:
                    out_csv.append([tag, met, c, key, len(v), float(v.mean() * scale),
                                    float(v.std(ddof=1) * scale) if len(v) > 1 else ""])
        tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
        out_tex.append("\n".join(tex))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", default=[str(ROOT / "results/exp_1110_sweeps/cells"),
                                                  str(ROOT / "results/exp_1110_scale_v2/cells"),
                                                  str(ROOT / "results/exp_1110_extras/cells")])
    ap.add_argument("--scale-dirs", nargs="+", default=[
        str(ROOT / "results/exp_0530_layerC_scale_stage2_v1/jsonl"),
        str(ROOT / "results/exp_1110_scale_supplement/jsonl")])
    ap.add_argument("--out-dir", default=str(ROOT / "results/exp_1110_paper_tables"))
    ap.add_argument("--sweep-test-qn", type=int, default=120, help="test queries per sweep/scale-v2 cell (TEST_QN)")
    ap.add_argument("--scale-test-qn", type=int, default=20, help="test queries per EXP-0530 Layer C cell (QN-TRAIN_QN)")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md, tex, rows_csv = ["# EXP-1110 sweeps (auto-generated; partial cells are reported with their seed count n)\n\n"
                         "LACSFC = per-seed best of the available LACSFC-* variants (per-variant rows are kept in the CSV)."], [
        "% auto-generated by examples/make_exp_1110_sweep_tables.py; AdaCurve excluded from best marking\n"
        "% LACSFC row = per-seed best of LACSFC-H/-Opt/-Meta; per-variant rows appear only in the CSV"], []

    sweeps = load_sweeps(args.cells, args.sweep_test_qn)
    for (sweep, wl) in sorted(sweeps):
        vals = sorted(sweeps[(sweep, wl)], key=_val_key)
        block = sweeps[(sweep, wl)]

        agg = {v: _lacsfc_agg(block[v]) for v in vals}

        def get(c, key, met, block=block, agg=agg):
            if c is None:
                return _display_curves(block[key])
            if c == LACSFC_AGG:
                slot = agg[key].get(met)
            else:
                slot = block[key].get(c, {}).get(met)
            return np.asarray(list(slot.values())) if slot else None

        cols = [(v, v + f" (n={len(next(iter(block[v].values()))['avg_seg'])})") for v in vals]
        emit(f"{wl}, {SWEEP_TITLE.get(sweep, sweep)}", cols, get, md, tex, rows_csv, f"{sweep}_{wl}")

    scale, anchor = load_scale(args.scale_dirs, args.scale_test_qn)
    for wl in sorted(scale):
        ns = sorted(scale[wl])

        agg = {n: _lacsfc_agg(scale[wl][n]) for n in ns}

        def get(c, key, met, wl=wl, agg=agg):
            if c is None:
                return _display_curves(scale[wl][key])
            if c == LACSFC_AGG:
                slot = agg[key].get(met)
            else:
                slot = scale[wl][key].get(c, {}).get(met)
            return np.asarray(list(slot.values())) if slot else None

        cols = [(n, f"{n // 1_000_000}M") for n in ns]
        emit(f"{wl}, data scale (EXP-0530 Layer C settings: k=4, 20/20 queries)", cols, get, md, tex, rows_csv,
             f"scaleC_{wl}")

    bad = [a for a in anchor if not np.isclose(a[5], a[6], rtol=1e-6)]
    md.append(f"\n## Scale anchor check (supplement vs Layer C, structural metrics)\n\n"
              f"{len(anchor)} overlapping values, {len(bad)} mismatches.")
    for a in bad[:20]:
        md.append(f"- MISMATCH {a}")

    (out_dir / "exp_1110_sweeps.md").write_text("\n".join(md) + "\n")
    (out_dir / "exp_1110_sweeps.tex").write_text("\n".join(tex))
    with open(out_dir / "exp_1110_sweeps.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["table", "metric", "curve", "setting", "n_seeds", "mean", "std"])
        w.writerows(rows_csv)
    print(f"wrote {out_dir}/exp_1110_sweeps.(md|tex|csv); anchor overlaps={len(anchor)} mismatches={len(bad)}")


if __name__ == "__main__":
    main()
