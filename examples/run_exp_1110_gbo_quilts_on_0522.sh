#!/usr/bin/env bash
# EXP-1110: add GBO and QUILTS (plus a hilbert_nd anchor) to the EXP-0522 4D real-data matrix.
# Settings as EXP-0522: n=5000, qn_total=80 (--train-qn 16 -> 16 train / 64 test), d=4, k=4, b=128,
# --locality-qn 0, --profile, repeat 1. Points/queries come from the archived
# data/exp_0522_lacsfc_4d_all_baseline_v1/<wl>_s<seed>_{points.npy,queries.csv} files.
# Output layout matches exp_1110_gbo_quilts_on_0522/{jsonl,logs}: <wl>__<curve>.jsonl.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

OUT="${OUT:-results/exp_1110_gbo_quilts_on_0522}"
DATA_DIR="${DATA_DIR:-data/exp_0522_lacsfc_4d_all_baseline_v1}"
SEEDS="${SEEDS:-0 1 2}"
CURVES="${CURVES:-hilbert_nd gbo quilts}"
WORKLOADS="${WORKLOADS:-osm_time_pop_4d geonames_time_cat_4d tiger_time_class_4d}"
mkdir -p "${OUT}/jsonl" "${OUT}/logs"

run() {
  local w="$1" c="$2" s="$3"
  local out="${OUT}/jsonl/${w}__${c}.jsonl"
  local tag="exp_1110_on_0522:${w}:${c}:s${s}"
  if [[ -f "${out}" ]] && grep -q "\"exp\": \"${tag}\"" "${out}"; then
    echo "[skip] ${tag}"
    return 0
  fi
  local t0 ec
  t0=$(date +%s)
  timeout 1800 python3 -m src.mvp.compare_runner --exp-tag "${tag}" --output "${out}" --seed "${s}" \
    --repeat 1 --n 5000 --d 4 --k 4 --b 128 --qn 80 --train-qn 16 --locality-qn 0 --profile \
    --curves "${c}" --points real \
    --points-file "${DATA_DIR}/${w}_s${s}_points.npy" \
    --queries-file "${DATA_DIR}/${w}_s${s}_queries.csv" --require-n \
    > "${OUT}/logs/${w}__${c}__s${s}.log" 2>&1
  ec=$?
  echo "${w} ${c} s${s} exit=${ec} $(( $(date +%s) - t0 ))s"
}

for s in ${SEEDS}; do
  for c in ${CURVES}; do
    for w in ${WORKLOADS}; do
      run "${w}" "${c}" "${s}"
    done
  done
done
echo "[done] ${OUT}"
