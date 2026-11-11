# EXP-1110 2D: EXP-0512 2D all-curve matrix rerun with a disjoint train/test split (2026-11-10)

- Driver: `examples/run_exp_1110_2d_all_curve_split.sh`; 5 workloads x seeds 0-2, 15/15 cells completed.
- Why: EXP-0512 (`run_latest_all_curve_r2_v1.sh`) passed `--train-qn = --qn` (synthetic) and `--train-qn 0` (OSM);
  `compare_runner` then uses the same queries for training and testing, so its learned-curve numbers are in-sample.
  Confirmed by reproducing the archived gmm_hotspot2d seed-0 Hilbert avg_pa/avg_seg/avg_amp exactly with qn=120,
  train-qn=120.
- Here: n=50000, d=2, k=8, b=256, qn=240, train-qn=120. Every log starts with `qn_total=240 qn_train=120 qn_test=120`.
  OSM uses the first 240 rows of `data/OSM/queries_osm250m.csv` (120 train / 120 test).
- Not anchored: the test set differs from EXP-0512, so no cell can be compared value-for-value with the archive.
- LACSFC row: token `lacsfc_neural` (output label `LACSFC-Neural`). It runs the same compare_runner branch as the
  EXP-0512 `LACSFC-Opt` row and its `lacsfc_config` equals the archived one (k=8, max_evals=150, sigma0=0.25,
  block_space, weighted_mvp, init 0.60,0.15,0.25, learn_obj false). Reported as LACSFC-Opt.
- OSM workloads: points and queries do not depend on the seed, so non-learned curves have zero seed variance; the
  three seeds are replicates only for the learned curves.
- `combined_per_seed.json`: per-seed avg_amp, avg_seg, avg_pa, query_s, build_s.
- Timing caveat: an unrelated EXP-0530 Layer C job ran on the same machine during this run.
