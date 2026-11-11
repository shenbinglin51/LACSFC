#!/usr/bin/env bash
# EXP-1110 2D: rerun of the EXP-0512 2D all-curve matrix (run_latest_all_curve_r2_v1.sh settings) with a disjoint
# train/test split, plus GBO and QUILTS. EXP-0512 used --train-qn = --qn (synthetic) and --train-qn 0 (OSM), so the
# learned curves were fit on the test queries. Here QN=240 / TRAIN_QN=120: the first 120 generated queries train,
# the next 120 test; OSM takes the first 240 rows of the query file, 120 train / 120 test.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

OUT_DIR="${OUT_DIR:-results/exp_1110_2d_all_curve_split}"
SEEDS="${SEEDS:-0 1 2}"
WORKLOADS="${WORKLOADS:-uniform_iso2d gmm_hotspot2d gmm_aspect8_2d osm25m_trace osm250m_trace}"
CURVES="${CURVES:-z,hilbert,piecewise,lmsfc_std,lmsfc_bo,bitallocz_bo,adacurve,lacsfc_neural,gbo,quilts}"
N="${N:-50000}"
QN="${QN:-240}"
TRAIN_QN="${TRAIN_QN:-120}"
D=2
K="${K:-8}"
B="${B:-256}"
LACSFC_K="${LACSFC_K:-${K}}"
LACSFC_MAX_EVALS="${LACSFC_MAX_EVALS:-150}"
LACSFC_SIGMA0="${LACSFC_SIGMA0:-0.25}"
LACSFC_MODE="${LACSFC_MODE:-block_space}"
LACSFC_OBJECTIVE="${LACSFC_OBJECTIVE:-weighted_mvp}"
LACSFC_INIT_OBJ="${LACSFC_INIT_OBJ:-0.60,0.15,0.25}"
AREA_MIN="${AREA_MIN:-1e-4}"
AREA_MAX="${AREA_MAX:-0.05}"
ASPECT_RATIO="${ASPECT_RATIO:-8.0}"
HOTSPOT_RATIO="${HOTSPOT_RATIO:-1.0}"
QUERY_FILE="${QUERY_FILE:-data/OSM/queries_osm250m.csv}"
RESUME="${RESUME:-1}"

mkdir -p "${OUT_DIR}/logs"

workload_args() {
  case "$1" in
    uniform_iso2d) printf '%s\n' "--points uniform --query-dist uniform --train-query-dist uniform" ;;
    gmm_hotspot2d) printf '%s\n' "--points gmm --query-dist hotspot_mixture --train-query-dist hotspot_mixture --hotspot-ratio ${HOTSPOT_RATIO}" ;;
    gmm_aspect8_2d) printf '%s\n' "--points gmm --query-dist aspect_ratio --train-query-dist aspect_ratio --aspect-ratio ${ASPECT_RATIO}" ;;
    osm25m_trace) printf '%s\n' "--points real --points-file data/OSM/osm25mbinary --queries-file ${QUERY_FILE}" ;;
    osm250m_trace) printf '%s\n' "--points real --points-file data/OSM/osm250mbinary32 --queries-file ${QUERY_FILE}" ;;
    *) echo "Unknown workload: $1" >&2; return 1 ;;
  esac
}

for workload in ${WORKLOADS}; do
  read -r -a wargs <<< "$(workload_args "${workload}")"
  output_file="${OUT_DIR}/${workload}_all_curves.jsonl"
  for seed in ${SEEDS}; do
    tag="exp_1110_2d_split:${workload}:s${seed}"
    if [[ "${RESUME}" == "1" && -f "${output_file}" ]] && grep -q "\"exp\": \"${tag}\"" "${output_file}"; then
      echo "[skip] ${tag}"
      continue
    fi
    t0=$(date +%s)
    python3 -m src.mvp.compare_runner \
      --exp-tag "${tag}" --output "${output_file}" --seed "${seed}" --repeat 1 \
      --n "${N}" --qn "${QN}" --train-qn "${TRAIN_QN}" --d "${D}" --k "${K}" --b "${B}" \
      --area-min "${AREA_MIN}" --area-max "${AREA_MAX}" --curves "${CURVES}" \
      --lacsfc-k "${LACSFC_K}" --lacsfc-max-evals "${LACSFC_MAX_EVALS}" --lacsfc-sigma0 "${LACSFC_SIGMA0}" \
      --lacsfc-mode "${LACSFC_MODE}" --lacsfc-objective "${LACSFC_OBJECTIVE}" --lacsfc-no-learn-obj \
      --lacsfc-init-obj "${LACSFC_INIT_OBJ}" "${wargs[@]}" --profile \
      > "${OUT_DIR}/logs/${workload}__s${seed}.log" 2>&1
    echo "[done] ${tag} $(( $(date +%s) - t0 ))s $(head -1 "${OUT_DIR}/logs/${workload}__s${seed}.log")"
  done
done
echo "[done] wrote ${OUT_DIR}"
