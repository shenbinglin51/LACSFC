# LACSFC — EXP-1110 Reproduction Artifact

Anonymous artifact for the paper experiments (EXP-1110): all compared methods,
the benchmark engine, raw per-seed results, and the exact scripts that produce
the paper tables.

## Quick start

```bash
./reproduce.sh setup     # venv + pip deps + optional C++ extensions
./reproduce.sh data      # verify datasets (see data/*/README.md for the big files)
./reproduce.sh tables    # JSONL -> combined_per_seed.json -> CSV/MD/TeX tables
./reproduce.sh verify    # diff regenerated tables against results/exp_1110_paper_tables/
./reproduce.sh run --quick   # smoke pipeline (~30 min, 1 seed, small N)
./reproduce.sh run --paper   # full reproduction (days of compute)
```

`verify` is the zero-compute check: it regenerates every table from the shipped
raw JSONL and diffs against `results/exp_1110_paper_tables/`.

## Expected output

`./reproduce.sh verify` exits 0 and reports no diffs. The regenerated files
must match `results/exp_1110_paper_tables/` (`SUMMARY.md`, `exp_1110_sweeps.md`,
`*.csv`, `*.tex`) byte-for-byte; `avg_seg`/`avg_pa`/`avg_amp` are structural
metrics and reproduce exactly, while `query_us`/`build_s` are wall-clock and
will drift if you re-run the benchmark.

## Methods (table row -> implementation)

| Row | Code |
|---|---|
| RowMajor / ColumnMajor | `src/sfcs/traditional/{row_major,column_major}.py` |
| Z-curve | `src/sfcs/traditional/z_curve.py` |
| Hilbert | `src/sfcs/traditional/hilbert_curve.py` (+ `hilbertcurve` pkg for d>=3) |
| LMSFC-Std / LMSFC-BO | `src/mvp/lmsfc/` (bit-mask re-implementation) |
| BitAllocZ-BO | `src/mvp/bitallocz/` |
| PiecewiseSFC | `src/mvp/piecewise_sfc/` (Learned-BMTree re-implementation) |
| AdaCurve | `src/mvp/adacurve/paper_repro.py` |
| GBO | `src/mvp/rank_table.py` + `vendored/neuralsfc_lacsfc/experiments/e7_recursive.py` |
| QUILTS | `src/mvp/rank_table.py` + `vendored/neuralsfc_lacsfc/experiments/ch7_export.py` |
| LACSFC | `src/mvp/lacsfc/` + `vendored/lacsfcf_repo/lacsfcf/`; three variants (H / Opt / Meta) — the table row is the per-seed best across the variants present in each cell; per-variant values are in `exp_1110_sweeps.csv` |

## Layout

- `src/mvp/compare_runner.py` — benchmark entry (`python -m src.mvp.compare_runner`)
- `examples/` — one runner script per experiment family + table generators
- `results/` — raw per-cell JSONL + generated paper tables
- `vendored/` — external research code this benchmark depends on
- `data/` — datasets (large files: regenerate via `data/prepare.sh`; see data/*/README.md)

## Notes

- `query_us`/`build_s` are Python-engine timings and will drift on re-run;
  `avg_seg`/`avg_pa`/`avg_amp` are structural and reproduce exactly.
- The `lacsfc_brq` C++ extension is optional (`impl=boundary_py` is the default);
  building it needs `SDSL_BRQ_ROOT` or `vendored/sdsl-lite/BRQ`.
- Env vars: `LACSFCF_ROOT`, `NEURALSFC_ROOT`, `ADACURVE_ROOT`, `SDSL_BRQ_ROOT`,
  `LMSFC_ROOT`/`LMSFC_LIB` override the vendored copies.
