#!/usr/bin/env bash
# EXP-1110: add GBO and QUILTS (plus a Hilbert anchor) to EXP-0530 Layer A and Layer B.
# Settings are those printed in the archived EXP-0530 logs: n=50000 qn_total=80 qn_train=40 qn_test=40 k=4 b=128,
# hotspot_mixture queries (ratio 0.9), area 1e-5..5e-3. osm_3d reads the archived data/exp_0530_layerA_main_v1 files.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

OUT="${OUT:-results/exp_1110_gbo_quilts_on_0530}"
SEEDS="${SEEDS:-0 1 2}"
CURVES="${CURVES:-hilbert gbo quilts}"
mkdir -p "${OUT}/jsonl" "${OUT}/logs"

COMMON=(--repeat 1 --n 50000 --qn 80 --train-qn 40 --k 4 --lacsfc-k 4 --b 128
        --area-min 1e-5 --area-max 5e-3 --profile)
QA=(--query-dist hotspot_mixture --train-query-dist hotspot_mixture --hotspot-ratio 0.9)

run() {
  local layer="$1" w="$2" d="$3" c="$4" s="$5"
  shift 5
  local tok="${c}"
  if [[ "${c}" == "hilbert" && "${d}" != "2" ]]; then tok="hilbert_nd"; fi
  local out="${OUT}/jsonl/${layer}__${w}__n50000__mixed__${c}.jsonl"
  local tag="exp_1110_on_0530:${layer}:${w}:n50000:mixed:${c}:s${s}"
  if [[ -f "${out}" ]] && grep -q "\"exp\": \"${tag}\"" "${out}"; then
    echo "[skip] ${tag}"
    return 0
  fi
  local t0 ec
  t0=$(date +%s)
  timeout 1800 python3 -m src.mvp.compare_runner --exp-tag "${tag}" --output "${out}" --seed "${s}" --d "${d}" \
    --curves "${tok}" "${COMMON[@]}" "$@" > "${OUT}/logs/${layer}__${w}__${c}__s${s}.log" 2>&1
  ec=$?
  echo "${layer} ${w} ${c} s${s} exit=${ec} $(( $(date +%s) - t0 ))s"
}

for s in ${SEEDS}; do
  for c in ${CURVES}; do
    for wd in uniform_3d:3:uniform gmm_3d:3:gmm uniform_4d:4:uniform gmm_4d:4:gmm uniform_5d:5:uniform gmm_5d:5:gmm; do
      IFS=: read -r w d p <<< "${wd}"
      run B "${w}" "${d}" "${c}" "${s}" --points "${p}" "${QA[@]}"
    done
    for wd in uniform_4d:4:uniform gmm_4d:4:gmm; do
      IFS=: read -r w d p <<< "${wd}"
      run A "${w}" "${d}" "${c}" "${s}" --points "${p}" "${QA[@]}"
    done
    run A osm_3d 3 "${c}" "${s}" --points real \
      --points-file "data/exp_0530_layerA_main_v1/osm_time_3d_s${s}_points.npy" \
      --queries-file "data/exp_0530_layerA_main_v1/osm_time_3d_s${s}_queries.csv" --require-n
  done
done
echo "[done] ${OUT}"
