#!/usr/bin/env bash
# EXP-1110 reproduction entry point (black-box).
#
#   ./reproduce.sh setup                 python deps + C++ extensions
#   ./reproduce.sh data                  verify required datasets are present
#   ./reproduce.sh run [--quick|--paper] run the EXP-1110 experiment cells (raw JSONL)
#   ./reproduce.sh tables                JSONL -> combined_per_seed.json -> CSV/MD/TeX tables
#   ./reproduce.sh verify                regenerate artifacts and diff against the shipped results
#   ./reproduce.sh all  [--quick|--paper]
#
# Modes:
#   --paper        full paper configuration (default): seeds 0-2, N=50k, 120 test queries,
#                  LACSFC_MAX_EVALS=150, LMSFC-BO 30+6*20 evaluations. Days of compute.
#   --quick        smoke pipeline: 1 seed, N=5k, 20 test queries, reduced training budgets.
#   --tables-only  (with `tables`/`verify`) skip run entirely; regenerate tables from the
#                  raw JSONL shipped under results/.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

MODE="paper"
for arg in "$@"; do
  case "${arg}" in
    --quick) MODE="quick" ;;
    --paper) MODE="paper" ;;
    --tables-only) MODE="tables-only" ;;
  esac
done
cmd="${1:-help}"

# ---------------------------------------------------------------- environment knobs
# quick mode shrinks every budget the runner scripts expose via env vars.
if [[ "${MODE}" == "quick" ]]; then
  export SEEDS="0" N=5000 TEST_QN=20 TRAIN_QN=20
  export LACSFC_MAX_EVALS=20 BO_INIT=5 BO_ROUNDS=2 BO_PER_ROUND=5
  export SCALES="1000000" TRAINPERS="4" CELL_TIMEOUT=1200
fi
export PYTHONDONTWRITEBYTECODE=1

PY=python3
[[ -x .venv/bin/python3 ]] && PY=.venv/bin/python3

# ---------------------------------------------------------------- steps
step_setup() {
  ${PY} -m venv .venv
  PY=.venv/bin/python3
  ${PY} -m pip install --upgrade pip
  ${PY} -m pip install -r requirements.txt
  # C++ extensions (query interval extraction + LACSFC BRQ backend)
  (cd src/sfcs/query_processors && make)
  (cd src/mvp/lacsfc/cpp && "${PY}" setup_brq.py build_ext --inplace)
  echo "[setup] done. Interpeter: $(${PY} -V)"
}

step_data() {
  local missing=0 f
  local required=(
    data/OSM/osm250mbinary32
    data/OSM/queries_osm250m.csv
  )
  for s in 0 1 2; do
    required+=("data/exp_0530_layerA_main_v1/osm_time_3d_s${s}_points.npy"
               "data/exp_0530_layerA_main_v1/osm_time_3d_s${s}_queries.csv")
    for w in osm_time_pop_4d geonames_time_cat_4d tiger_time_class_4d; do
      required+=("data/exp_0522_lacsfc_4d_all_baseline_v1/${w}_s${s}_points.npy"
                 "data/exp_0522_lacsfc_4d_all_baseline_v1/${w}_s${s}_queries.csv")
    done
  done
  # LACSFC-Meta tier-3 checkpoint (extras 'meta' sweep)
  required+=("results/exp_0520_lacsfc_tier_ablation_budget_v1/ckpt/lacsfc_exp0520_tier3_4d_generator.pt"
             "results/exp_0520_lacsfc_tier_ablation_budget_v1/ckpt/lacsfc_exp0520_tier3_4d_evaluator.pt")
  for f in "${required[@]}"; do
    if [[ ! -f "${f}" ]]; then
      echo "[missing] ${f}"
      missing=1
    fi
  done
  if [[ ${missing} -eq 0 ]]; then
    echo "[data] all required files present"
  else
    echo "[data] MISSING files: see data/OSM/README.md for download/preparation instructions."
    return 1
  fi
}

step_run() {
  if [[ "${MODE}" == "tables-only" ]]; then
    echo "[run] skipped (--tables-only)"
    return 0
  fi
  local stages=(
    run_exp_1110_sweeps.sh            # grid/sel/aspect/page/train/shift sweeps
    run_exp_1110_extras.sh            # trainper + LACSFC-Meta cells
    run_exp_1110_scale_v2.sh          # data-scale sweep (all methods)
    run_exp_1110_scale_supplement.sh  # scale supplement vs EXP-0530 Layer C anchor
    run_exp_1110_2d_all_curve_split.sh
    run_exp_1110_gbo_quilts_on_0530.sh
    run_exp_1110_gbo_quilts_on_0522.sh
  )
  for s in "${stages[@]}"; do
    echo "[run] === ${s} (mode=${MODE}) ==="
    bash "examples/${s}"
  done
}

step_tables() {
  ${PY} examples/combine_exp_1110_per_seed.py
  ${PY} examples/make_exp_1110_sweep_tables.py
  ${PY} examples/make_exp_1110_tables.py
}

step_verify() {
  local tmp
  tmp="$(mktemp -d)"
  echo "[verify] regenerating tables into ${tmp}"
  ${PY} examples/combine_exp_1110_per_seed.py --check || return 1
  ${PY} examples/make_exp_1110_sweep_tables.py --out-dir "${tmp}"
  ${PY} examples/make_exp_1110_tables.py --out "${tmp}/exp_1110_tables.tex"
  local ok=1
  for f in exp_1110_sweeps.csv exp_1110_sweeps.md exp_1110_sweeps.tex; do
    if ! diff -q "${tmp}/${f}" "results/exp_1110_paper_tables/${f}" > /dev/null; then
      echo "[verify] DIFFERS: ${f}"
      ok=0
    fi
  done
  if ! diff -q "${tmp}/exp_1110_tables.tex" "results/exp_1110_paper_tables/exp_1110_tables.tex" > /dev/null; then
    echo "[verify] DIFFERS: exp_1110_tables.tex"
    ok=0
  fi
  if [[ ${ok} -eq 1 ]]; then
    echo "[verify] all regenerated tables identical to results/exp_1110_paper_tables/"
  else
    echo "[verify] differences found (expected if cells were re-run: timing metrics drift;"
    echo "         avg_seg/avg_pa/avg_amp should still match cell-for-cell)"
    return 1
  fi
}

case "${cmd}" in
  setup)  step_setup ;;
  data)   step_data ;;
  run)    step_run ;;
  tables) step_tables ;;
  verify) step_verify ;;
  all)
    [[ "${MODE}" != "tables-only" ]] && step_setup
    step_data || echo "[all] continuing despite missing data (non-OSM cells still run)"
    step_run
    step_tables
    step_verify
    ;;
  *) sed -n '2,18p' "$0"; exit 1 ;;
esac
