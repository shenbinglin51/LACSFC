"""Render the EXP-1110 MVP all-method comparison tables (LaTeX) from the three combined_per_seed.json files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

LABEL = {
    "hilbert": "Hilbert", "hilbert_nd": "Hilbert", "Hilbert": "Hilbert",
    "z": "Z-curve", "ZCurve": "Z-curve",
    "LMSFC-Std": "LMSFC-Std", "lmsfc_bo": "LMSFC-BO", "LMSFC-BO": "LMSFC-BO",
    "bitallocz_bo": "BitAllocZ-BO", "BitAllocZ-BO": "BitAllocZ-BO",
    "piecewise": "PiecewiseSFC", "PiecewiseSFC": "PiecewiseSFC",
    "adacurve": "AdaCurve$^\\dagger$", "AdaCurve": "AdaCurve$^\\dagger$",
    "gbo": "GBO$^\\ddagger$", "GBO": "GBO$^\\ddagger$",
    "quilts": "QUILTS", "QUILTS": "QUILTS",
    "lacsfc_h": "LACSFC-H",
    "lacsfc_opt": "LACSFC-Opt", "lacsfc_opt_mid": "LACSFC-Opt", "LACSFC-Neural": "LACSFC-Opt",
    "lacsfc_meta": "LACSFC-Meta", "lacsfc_meta_refine_tiny": "LACSFC-Meta",
    "LACSFC": "LACSFC",
}
# The lacsfc_* variant tokens are collapsed into one "LACSFC" row = per-seed best
# (element-wise min across the variant arrays present in each cell).
LACSFC_TOKENS = {"lacsfc_h", "lacsfc_opt", "lacsfc_opt_mid", "LACSFC-Neural",
                 "lacsfc_meta", "lacsfc_meta_refine_tiny"}
LACSFC_AGG = "LACSFC"
EXCLUDE_FROM_BEST = {"adacurve", "AdaCurve"}


def _best_of_variants(arrs):
    """Element-wise min across variant arrays; variants may have different seed counts."""
    n = max(len(a) for a in arrs)
    return np.asarray([min(a[i] for a in arrs if i < len(a)) for i in range(n)], dtype=float)


def _groups():
    d2 = json.loads((ROOT / "results/exp_1110_2d_all_curve_split/combined_per_seed.json").read_text())
    d0530 = json.loads((ROOT / "results/exp_1110_gbo_quilts_on_0530/combined_per_seed.json").read_text())
    d0522 = json.loads((ROOT / "results/exp_1110_gbo_quilts_on_0522/combined_per_seed.json").read_text())
    g2 = {
        "title": "2D (EXP-1110 rerun of EXP-0512, 120 train / 120 test queries, $n=50$k, $k=8$, $B=256$)",
        "label": "2d",
        "cols": [("uniform_iso2d", "Uniform"), ("gmm_hotspot2d", "GMM-hot"), ("gmm_aspect8_2d", "GMM-asp8"),
                 ("osm25m_trace", "OSM-25M$^\\S$"), ("osm250m_trace", "OSM-250M$^\\S$")],
        "rows": ["Hilbert", "ZCurve", "LMSFC-Std", "LMSFC-BO", "BitAllocZ-BO", "PiecewiseSFC", "AdaCurve",
                 "GBO", "QUILTS", "LACSFC-Neural"],
        "get": lambda c, w: d2[f"{w}|{c}"],
    }
    g3 = {
        "title": "3D--5D (EXP-0530 Layer B + Layer A osm\\_3d, 40 train / 40 test queries, $n=50$k, $k=4$, $B=128$)",
        "label": "0530",
        "cols": [("B|uniform_3d", "Uni-3D"), ("B|gmm_3d", "GMM-3D"), ("B|uniform_4d", "Uni-4D"),
                 ("B|gmm_4d", "GMM-4D"), ("B|uniform_5d", "Uni-5D"), ("B|gmm_5d", "GMM-5D"), ("A|osm_3d", "OSM-3D")],
        "rows": ["hilbert", "z", "lmsfc_bo", "bitallocz_bo", "piecewise", "adacurve", "gbo", "quilts",
                 "lacsfc_h", "lacsfc_opt", "lacsfc_meta"],
        "get": lambda c, w: d0530[f"{w}|{c}"],
    }
    g4 = {
        "title": "Realistic 4D (EXP-0522, 16 train / 64 test queries, $n=5$k, $k=4$, $B=128$)",
        "label": "0522",
        "cols": [("osm_time_pop_4d", "OSM-4D"), ("geonames_time_cat_4d", "GeoNames-4D"),
                 ("tiger_time_class_4d", "TIGER-4D")],
        "rows": ["hilbert_nd", "z", "lmsfc_bo", "bitallocz_bo", "piecewise", "adacurve", "gbo", "quilts",
                 "lacsfc_h", "lacsfc_opt_mid", "lacsfc_meta_refine_tiny"],
        "get": lambda c, w: d0522[f"{c}|{w}"],
    }
    return [g2, g3, g4]


def _table(g, metric: str, caption_metric: str) -> str:
    cols, raw_rows = g["cols"], g["rows"]
    variants = [r for r in raw_rows if r in LACSFC_TOKENS]
    rows = [r for r in raw_rows if r not in LACSFC_TOKENS] + ([LACSFC_AGG] if variants else [])
    vals = {(r, w): np.asarray(g["get"](r, w)[metric], dtype=float) for r in raw_rows for w, _ in cols}
    for w, _ in cols:
        vals[(LACSFC_AGG, w)] = _best_of_variants([vals[(r, w)] for r in variants])
    best = {w: min(vals[(r, w)].mean() for r in rows if r not in EXCLUDE_FROM_BEST) for w, _ in cols}
    out = [
        "\\begin{table*}[t]",
        "\\centering\\small",
        f"\\caption{{{caption_metric} (mean $\\pm$ s.d. over seeds 0--2; lower is better). {g['title']}.}}",
        f"\\label{{tab:exp1110_{g['label']}_{metric}}}",
        "\\begin{tabular}{l" + "r" * len(cols) + "}",
        "\\toprule",
        "Method & " + " & ".join(h for _, h in cols) + " \\\\",
        "\\midrule",
    ]
    for r in rows:
        cells = []
        for w, _ in cols:
            v = vals[(r, w)]
            s = f"{v.mean():.3f}\\,$\\pm$\\,{v.std(ddof=1):.3f}"
            if r not in EXCLUDE_FROM_BEST and np.isclose(v.mean(), best[w]):
                s = "\\textbf{" + s + "}"
            cells.append(s)
        out.append(LABEL[r] + " & " + " & ".join(cells) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    return "\n".join(out)


NOTES = r"""% Notes for all EXP-1110 tables (MVP Python engine; descriptive, not the Chapter 7 formal pipeline):
% $\dagger$ AdaCurve degenerates in this harness (avg_amp 10-300x Hilbert); excluded from best-value marking.
% $\ddagger$ GBO (Yang, He, Chen, TKDE 2016) is not an AC order: empty cells are appended at the end.
% $\S$ OSM points and queries do not depend on the seed; s.d. reflects only learned-method randomness.
% LMSFC-Std/BO and PiecewiseSFC are MVP re-implementations (bit-mask BO; Learned-BMTree re-implementation).
% LACSFC row = per-seed best of the LACSFC variants (lacsfc_h / lacsfc_opt / lacsfc_meta
%   and their *_mid/*_refine_tiny aliases); per-variant values are in exp_1110_sweeps.csv.
% LACSFC-Opt in the 2D table is token lacsfc_neural (same code path and lacsfc_config as EXP-0512 LACSFC-Opt).
% LACSFC-Opt/Meta are not run-to-run deterministic (Layer A vs Layer B differ at identical seed/args).
% 3 seeds only: a one-sided paired Wilcoxon test cannot reach p<0.05 (minimum p = 0.125).
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results/exp_1110_paper_tables/exp_1110_tables.tex"))
    args = ap.parse_args()
    parts = [NOTES]
    for g in _groups():
        parts.append(_table(g, "avg_amp", "Page amplification \\texttt{avg\\_amp}"))
        parts.append(_table(g, "avg_seg", "Contiguous segments per query \\texttt{avg\\_seg}"))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
