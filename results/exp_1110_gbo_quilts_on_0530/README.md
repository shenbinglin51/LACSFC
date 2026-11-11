# EXP-1110: GBO and QUILTS added to EXP-0530 Layer A and Layer B (2026-11-10)

- Driver: `examples/run_exp_1110_gbo_quilts_on_0530.sh` (curves hilbert, gbo, quilts; seeds 0-2).
- Settings are the ones printed in the first line of every archived EXP-0530 log:
  `n=50000 qn_total=80 qn_train=40 qn_test=40 k=4 b=128`; hotspot_mixture queries (ratio 0.9), area 1e-5..5e-3.
  (The script defaults QN=40/TRAIN_QN=20 are not what was run.) osm_3d reads the archived
  `data/exp_0530_layerA_main_v1/osm_time_3d_s{0,1,2}_*` files; the synthetic workloads are regenerated from the seed.
- Anchor: hilbert rerun, 9 workloads x 3 seeds; avg_pa, avg_seg, avg_amp, avg_interval_count, avg_covered_code_len,
  avg_records_scanned, avg_pass_semantic equal the archived values in 27/27 cells.
- GBO / QUILTS fit on the 40 training queries only (`src/mvp/rank_table.py`).
- `_bad_qn40/`: a first attempt with qn=40/train-qn=20; anchor failed 0/8; not used.
- `combined_per_seed.json`: per-seed avg_amp, avg_seg, avg_pa, query_s for the 9 archived curves plus GBO/QUILTS,
  Layer B (uniform/gmm x 3d/4d/5d) and Layer A osm_3d. Layer A and Layer B uniform_4d / gmm_4d share seed and args:
  avg_pa/avg_seg/avg_amp are equal for every curve except LACSFC-Opt and LACSFC-Meta, which differ between the two
  layers (the archived LACSFC-Opt/Meta builds are not run-to-run deterministic).
- Timing caveat: an unrelated EXP-0530 Layer C job (n=15M) was running on the same machine during this run, and the
  archived curves were timed in May; query_s is not comparable across the two sources.
