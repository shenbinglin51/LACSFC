# EXP-0530 Layer-C scale data (not shipped — ~4 GB)

Files `osm_time_3d_s{0,1,2}_n<N>_{points.npy,queries.csv}` used by the
data-scale sweep (`run_exp_1110_scale_v2.sh`). Regenerate deterministically with
`./data/prepare.sh layerc` once `data/OSM/osm250mbinary32` is present —
it runs `examples/generate_realistic_highdim_workloads_v1.py` with
seed {0,1,2}, N in {1M..30M}, qn=40, k=4, spatial area 1e-5..5e-3.
