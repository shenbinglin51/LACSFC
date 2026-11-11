# EXP-1110: GBO and QUILTS added to the EXP-0522 4D matrix (2026-11-10)

- Inputs: the EXP-0522 files in `data/exp_0522_lacsfc_4d_all_baseline_v1/` (seeds 0-2). SHA-256 of every points/queries
  file equals the copy in `<earlier-checkout>/data/...` that EXP-0522 actually read.
- Settings as EXP-0522: n=5000, qn=80, `--train-qn 16` (first 16 train, 64 test), d=4, k=4, b=128, `--locality-qn 0`,
  `--profile`, repeat 1.
- New curves: `gbo`, `quilts` via `src/mvp/rank_table.py` (lacsfc `e7_recursive.gbo_rank`, alpha=3, seed=run seed;
  lacsfc `ch7_export.build_quilts`, rule (a)). Both fit on the 16 training queries only. GBO is not AC.
- Anchor: `hilbert_nd` rerun on the same 9 cells; avg_pa, avg_seg, avg_amp, avg_interval_count, avg_covered_code_len,
  avg_records_scanned, avg_pass_semantic equal the archived EXP-0522 values in 9/9 cells. Hilbert query_s reruns
  within about 8% of the May values.
- Tests: `python3 -m unittest tests.test_rank_table` (6 tests).
- `combined_per_seed.json`: per-seed avg_amp, avg_seg, avg_pa, query_s, build_s of the 9 archived curves plus GBO/QUILTS.
- Timing caveat: GBO/QUILTS timed 2026-11-10, the others in May 2026 on the same machine; Python engine, 64 test
  queries per cell, about 0.2 ms per query.
