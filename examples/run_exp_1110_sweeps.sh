#!/usr/bin/env bash
# EXP-1110 sensitivity sweeps (all methods, disjoint train/test queries), modelled on the LMSFC (PVLDB'23) and
# LACSFC-0601 experiment sections:
#   grid   : grid resolution k per dimensionality (LACSFC Table 6, extended to all methods and d=2..5)
#   sel    : query selectivity / window area (LMSFC Fig. 7)
#   aspect : query aspect ratio (LMSFC Fig. 9)
#   page   : page size B (LMSFC Sec. 7.8 / Table 5 analogue)
#   train  : training-workload size (LMSFC Fig. 12)
#   shift  : train->test query-distribution shift (LACSFC Table 7)
# Every cell is one compare_runner process writing its own JSONL; qn = train_qn + TEST_QN, so the first train_qn
# generated queries train and the remaining TEST_QN queries test.
# Budget: LACSFC-Opt uses LACSFC_MAX_EVALS=150 objective evaluations; LMSFC-BO gets the same 150 evaluations
# (BO_INIT + BO_ROUNDS*BO_PER_ROUND = 30 + 6*20), unlike the 2D main table which used the runner default (440).
# Usage: SWEEPS="grid sel" JOBS=3 CPUS=4-9 bash examples/run_exp_1110_sweeps.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

OUT_DIR="${OUT_DIR:-results/exp_1110_sweeps}"
SWEEPS="${SWEEPS:-grid sel shift aspect page train}"
SEEDS="${SEEDS:-0 1 2}"
JOBS="${JOBS:-3}"
CPUS="${CPUS:-4-9}"
MEM_GB="${MEM_GB:-6}"
CELL_TIMEOUT="${CELL_TIMEOUT:-3600}"
N="${N:-50000}"
TEST_QN="${TEST_QN:-120}"
TRAIN_QN="${TRAIN_QN:-120}"
LACSFC_MAX_EVALS="${LACSFC_MAX_EVALS:-150}"
BO_INIT="${BO_INIT:-30}"
BO_ROUNDS="${BO_ROUNDS:-6}"
BO_PER_ROUND="${BO_PER_ROUND:-20}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
OSM_POINTS="${OSM_POINTS:-data/OSM/osm250mbinary32}"
CURVES_2D="${CURVES_2D:-row,column,z,hilbert,lmsfc_std,lmsfc_bo,bitallocz_bo,piecewise,adacurve,gbo,quilts,lacsfc_h,lacsfc_opt}"
CURVES_ND="${CURVES_ND:-z,hilbert,lmsfc_std,lmsfc_bo,bitallocz_bo,piecewise,adacurve,gbo,quilts,lacsfc_h,lacsfc_opt}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "${OUT_DIR}/cells" "${OUT_DIR}/logs"

# workload -> "d points-args default_k default_b area_min area_max"
wl() {
  case "$1" in
    osm_2d) echo "2|--points real --points-file ${OSM_POINTS}|8|256|1e-4|0.05" ;;
    gmm_2d) echo "2|--points gmm|8|256|1e-4|0.05" ;;
    gmm_3d) echo "3|--points gmm|4|128|1e-5|5e-3" ;;
    gmm_4d) echo "4|--points gmm|4|128|1e-5|5e-3" ;;
    gmm_5d) echo "5|--points gmm|3|128|1e-5|5e-3" ;;
    *) echo "unknown workload $1" >&2; return 1 ;;
  esac
}

# emit: sweep|value|workload|seed|k|b|train_qn|area_min|area_max|test_dist|train_dist|extra
emit_cells() {
  local s w k b a ar td tr t
  for s in ${SEEDS}; do
    for sw in ${SWEEPS}; do
      case "${sw}" in
        grid)
          for k in 6 8 10; do for w in osm_2d gmm_2d; do echo "grid|k${k}|${w}|${s}|${k}|||||hotspot_mixture|hotspot_mixture|"; done; done
          for k in 3 4 5 6; do echo "grid|k${k}|gmm_3d|${s}|${k}|||||hotspot_mixture|hotspot_mixture|"; done
          for k in 3 4 5; do echo "grid|k${k}|gmm_4d|${s}|${k}|||||hotspot_mixture|hotspot_mixture|"; done
          for k in 3 4; do echo "grid|k${k}|gmm_5d|${s}|${k}|||||hotspot_mixture|hotspot_mixture|"; done ;;
        sel)
          for a in 1e-5 1e-4 1e-3 1e-2; do for w in osm_2d gmm_4d; do echo "sel|a${a}|${w}|${s}||||${a}|${a}|hotspot_mixture|hotspot_mixture| "; done; done ;;
        aspect)
          for ar in 0.125 0.25 1 4 8; do for w in osm_2d gmm_4d; do echo "aspect|r${ar}|${w}|${s}||||||aspect_ratio|aspect_ratio|--aspect-ratio ${ar}"; done; done ;;
        page)
          for b in 64 128 256 512 1024; do for w in osm_2d gmm_4d; do echo "page|b${b}|${w}|${s}||${b}||||hotspot_mixture|hotspot_mixture|"; done; done ;;
        train)
          for t in 20 60 120 240; do echo "train|t${t}|osm_2d|${s}|||${t}|||hotspot_mixture|hotspot_mixture|"; done ;;
        shift)
          for tr in hotspot_mixture uniform; do for td in hotspot_mixture uniform; do for w in osm_2d gmm_4d; do
            echo "shift|${tr}-${td}|${w}|${s}||||||${td}|${tr}|"; done; done; done ;;
        *) echo "unknown sweep ${sw}" >&2; return 1 ;;
      esac
    done
  done
}

run_cell() {
  IFS='|' read -r sw val w s k b tq amin amax td tr extra <<< "$1"
  IFS='|' read -r d pargs dk db damin damax <<< "$(wl "${w}")"
  k="${k:-${dk}}"; b="${b:-${db}}"; tq="${tq:-${TRAIN_QN}}"; amin="${amin:-${damin}}"; amax="${amax:-${damax}}"
  local curves="${CURVES_ND}"; [[ "${d}" == "2" ]] && curves="${CURVES_2D}"
  local id="${sw}__${val}__${w}__s${s}"
  local out="${OUT_DIR}/cells/${id}.jsonl" log="${OUT_DIR}/logs/${id}.log"
  if [[ -s "${out}" ]]; then echo "[skip] ${id}"; return 0; fi
  local cmd=(python3 -m src.mvp.compare_runner
    --exp-tag "exp_1110_sweep:${sw}:${val}:${w}:s${s}" --output "${out}.part" --seed "${s}" --repeat 1
    --n "${N}" --qn "$(( tq + TEST_QN ))" --train-qn "${tq}" --d "${d}" --k "${k}" --b "${b}"
    --area-min "${amin}" --area-max "${amax}" --curves "${curves}"
    --query-dist "${td}" --train-query-dist "${tr}" --hotspot-ratio 0.9
    --train-bo-init-size "${BO_INIT}" --train-bo-rounds "${BO_ROUNDS}" --train-bo-evals-per-round "${BO_PER_ROUND}"
    --lacsfc-k "${k}" --lacsfc-max-evals "${LACSFC_MAX_EVALS}" --lacsfc-sigma0 0.25 --lacsfc-mode block_space
    --lacsfc-objective weighted_mvp --lacsfc-no-learn-obj --lacsfc-init-obj 0.60,0.15,0.25 --profile)
  # shellcheck disable=SC2206
  cmd+=(${pargs} ${extra})
  if [[ "${DRY_RUN}" == "1" ]]; then echo "${cmd[*]}"; return 0; fi
  local t0; t0=$(date +%s)
  if systemd-run --user --scope -q -p MemoryMax="${MEM_GB}G" -p MemorySwapMax=0 \
       nice -n 10 timeout --kill-after=30 "${CELL_TIMEOUT}" "${cmd[@]}" > "${log}" 2>&1; then
    mv "${out}.part" "${out}"
    echo "[done] ${id} $(( $(date +%s) - t0 ))s"
  else
    echo "[FAIL rc=$?] ${id} $(( $(date +%s) - t0 ))s (see ${log})"
  fi
}
export -f run_cell wl
export OUT_DIR N TEST_QN TRAIN_QN LACSFC_MAX_EVALS BO_INIT BO_ROUNDS BO_PER_ROUND OSM_POINTS CURVES_2D CURVES_ND DRY_RUN MEM_GB CELL_TIMEOUT

emit_cells > "${OUT_DIR}/cells.txt"
echo "[plan] $(wc -l < "${OUT_DIR}/cells.txt") cells, JOBS=${JOBS}, CPUS=${CPUS}"
taskset -c "${CPUS}" xargs -d '\n' -P "${JOBS}" -I{} bash -c 'run_cell "$1"' _ {} < "${OUT_DIR}/cells.txt"
echo "[all done] ${OUT_DIR}"
