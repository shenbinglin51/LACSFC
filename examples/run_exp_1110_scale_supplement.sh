#!/usr/bin/env bash
# EXP-1110 scale supplement: copy of run_exp_0530_full_curve_matrix_v1.sh (left untouched because EXP-0530 Layer C
# is still running from it). Re-runs Layer C cells on the same cached data (DATA_DIR, same seeds/N/QN/K/B) for the
# methods Layer C did not include, plus Hilbert as a structural anchor against the Layer C Hilbert rows.
# Usage: LAYER=C SCALES="1000000 2000000 5000000 10000000 15000000 20000000" SEEDS="0 1 2" XL_SCALE= \
#        bash examples/run_exp_1110_scale_supplement.sh
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

OUT_DIR="${OUT_DIR:-results/exp_1110_scale_supplement}"
DATA_DIR="${DATA_DIR:-data/exp_0530_layerC_scale_stage2_v1}"
export REALISTIC_SOURCE_OVERRIDE="${REALISTIC_SOURCE_OVERRIDE:-data/OSM/osm250mbinary32}"
LAYER="${LAYER:-smoke}"
SEEDS="${SEEDS:-0}"
RESUME="${RESUME:-1}"
RUN_ANALYSIS="${RUN_ANALYSIS:-1}"

N="${N:-1000}"
QN="${QN:-40}"
TRAIN_QN="${TRAIN_QN:-20}"
K="${K:-4}"
LACSFC_K="${LACSFC_K:-${K}}"
B="${B:-128}"
AREA_MIN="${AREA_MIN:-1e-5}"
AREA_MAX="${AREA_MAX:-5e-3}"
HOTSPOT_RATIO="${HOTSPOT_RATIO:-0.9}"
ASPECT_RATIO="${ASPECT_RATIO:-8.0}"
QUERY_KIND="${QUERY_KIND:-mixed}"

CURVES_ALL="${CURVES_ALL:-row column z hilbert lmsfc_bo piecewise bitallocz_bo adacurve lacsfc_h lacsfc_opt lacsfc_meta}"
CURVES_SCALE="${CURVES_SCALE:-hilbert lmsfc_std gbo quilts adacurve}"
CURVES_VARIANT="${CURVES_VARIANT:-hilbert lacsfc_h lacsfc_opt lacsfc_meta}"
CURVES_QUERY="${CURVES_QUERY:-z hilbert lmsfc_bo piecewise lacsfc_opt lacsfc_meta}"
SMOKE_CURVES="${SMOKE_CURVES:-z hilbert}"

LACSFC_MAX_EVALS="${LACSFC_MAX_EVALS:-3}"
META_REFINE_EVALS="${META_REFINE_EVALS:-1}"
POPSIZE="${POPSIZE:-20}"
SIGMA0="${SIGMA0:-0.25}"
LACSFC_MODE="${LACSFC_MODE:-block_space}"
LACSFC_OBJECTIVE="${LACSFC_OBJECTIVE:-weighted_mvp}"
LACSFC_INIT_OBJ="${LACSFC_INIT_OBJ:-0.60,0.15,0.25}"

LMSFC_BO_INIT_SIZE="${LMSFC_BO_INIT_SIZE:-8}"
LMSFC_BO_ROUNDS="${LMSFC_BO_ROUNDS:-2}"
LMSFC_BO_EVALS_PER_ROUND="${LMSFC_BO_EVALS_PER_ROUND:-8}"
BITALLOCZ_BO_INIT="${BITALLOCZ_BO_INIT:-5}"
BITALLOCZ_BO_ITERS="${BITALLOCZ_BO_ITERS:-16}"
ADACURVE_EPOCHS="${ADACURVE_EPOCHS:-10}"
ADACURVE_TRAIN_SAMPLE_SIZE="${ADACURVE_TRAIN_SAMPLE_SIZE:-2048}"

TIER3_CKPT_DIR="${TIER3_CKPT_DIR:-results/exp_0520_lacsfc_tier_ablation_budget_v1/ckpt}"
TIER3_MODEL_NAME="${TIER3_MODEL_NAME:-lacsfc_exp0520_tier3_4d}"
WALL_CLOCK_CAP_S="${WALL_CLOCK_CAP_S:-2400}"

mkdir -p "${OUT_DIR}/jsonl" "${OUT_DIR}/logs" "${OUT_DIR}/status" "${OUT_DIR}/summary" "${DATA_DIR}"
LOG_TSV="${OUT_DIR}/walltime.tsv"
if [[ ! -f "${LOG_TSV}" ]]; then
  printf 'layer\tworkload\td\tn\tquery_kind\tcurve\tseed\twall_s\texit_code\tstatus\tjsonl\tlog\n' > "${LOG_TSV}"
fi

has_completed_tag() {
  local output_file="$1"
  local tag="$2"
  [[ -f "${output_file}" ]] && grep -q '"exp": "'"${tag}"'"' "${output_file}"
}

workload_dim() {
  case "$1" in
    *_2d|tiger_2d|geonames_2d|osm_2d) printf '2\n' ;;
    *_3d|osm_3d|uniform_3d|gmm_3d) printf '3\n' ;;
    *_4d|uniform_4d|gmm_4d) printf '4\n' ;;
    *_5d|uniform_5d|gmm_5d) printf '5\n' ;;
    *) echo "Cannot infer dimension from workload: $1" >&2; return 1 ;;
  esac
}

workload_points_kind() {
  case "$1" in
    uniform_*) printf 'synthetic:uniform\n' ;;
    gmm_*) printf 'synthetic:gmm\n' ;;
    *) printf 'real\n' ;;
  esac
}

query_args() {
  case "${QUERY_KIND}" in
    uniform) printf '%s\n' "--query-dist uniform --train-query-dist uniform" ;;
    skewed) printf '%s\n' "--query-dist hotspot --train-query-dist hotspot" ;;
    mixed) printf '%s\n' "--query-dist hotspot_mixture --train-query-dist hotspot_mixture --hotspot-ratio ${HOTSPOT_RATIO}" ;;
    aspect) printf '%s\n' "--query-dist aspect_ratio --train-query-dist aspect_ratio --aspect-ratio ${ASPECT_RATIO}" ;;
    *) echo "Unknown QUERY_KIND=${QUERY_KIND}" >&2; return 1 ;;
  esac
}

real_source_file() {
  case "$1" in
    osm_2d) printf '%s\n' "${OSM2D_SOURCE_OVERRIDE:-data/OSM/osm250mbinary32}" ;;
    geonames_2d) printf '%s\n' "data/tiger_tweets/geonames_2d.csv" ;;
    tiger_2d) printf '%s\n' "data/tiger_tweets/tiger_us_2d.csv" ;;
    *) printf '\n' ;;
  esac
}

realistic_workload_name() {
  case "$1" in
    osm_3d) printf '%s\n' "osm_time_3d" ;;
    geonames_3d) printf '%s\n' "geonames_time_3d" ;;
    tiger_3d) printf '%s\n' "tiger_time_3d" ;;
    osm_4d) printf '%s\n' "osm_time_pop_4d" ;;
    geonames_4d) printf '%s\n' "geonames_time_cat_4d" ;;
    tiger_4d) printf '%s\n' "tiger_time_class_4d" ;;
    *) printf '\n' ;;
  esac
}

ensure_realistic_data() {
  local workload="$1" seed="$2"
  local gen_workload
  gen_workload="$(realistic_workload_name "${workload}")"
  if [[ -z "${gen_workload}" ]]; then
    return 1
  fi
  # IMPORTANT: include the scale N in the cache stem. Without it, scale
  # experiments at different N silently reuse the first generated file
  # (the EXP-0530 Layer C osm_3d 1M/5M/10M/30M data-reuse bug).
  local stem="${gen_workload}_s${seed}_n${N}"
  local points_file="${DATA_DIR}/${stem}_points.npy"
  local queries_file="${DATA_DIR}/${stem}_queries.csv"
  if [[ -f "${points_file}" && -f "${queries_file}" && "${RESUME}" == "1" ]]; then
    printf '%s\t%s\t%s\n' "${gen_workload}" "${points_file}" "${queries_file}"
    return 0
  fi
  # Optional real-source override (e.g. data/OSM/osm250mbinary32 for >25M scales).
  local source_arg=()
  if [[ -n "${REALISTIC_SOURCE_OVERRIDE:-}" ]]; then
    source_arg=(--source-points-file "${REALISTIC_SOURCE_OVERRIDE}")
  fi
  python "${ROOT_DIR}/examples/generate_realistic_highdim_workloads_v1.py" \
    --workload "${gen_workload}" \
    --out-dir "${DATA_DIR}" \
    --out-prefix "${stem}" \
    "${source_arg[@]}" \
    --seed "${seed}" \
    --n "${N}" \
    --qn "${QN}" \
    --k "${K}" \
    --spatial-area-min "${AREA_MIN}" \
    --spatial-area-max "${AREA_MAX}" >/dev/null
  printf '%s\t%s\t%s\n' "${gen_workload}" "${points_file}" "${queries_file}"
}

curve_token() {
  local curve="$1" dim="$2"
  case "${curve}" in
    row|column)
      if [[ "${dim}" != "2" ]]; then
        return 2
      fi
      printf '%s\n' "${curve}"
      ;;
    hilbert)
      if [[ "${dim}" == "2" ]]; then printf 'hilbert\n'; else printf 'hilbert_nd\n'; fi
      ;;
    z|lmsfc_bo|lmsfc_std|piecewise|bitallocz_bo|adacurve|gbo|quilts|lacsfc_h|lacsfc_opt|lacsfc_meta)
      printf '%s\n' "${curve}"
      ;;
    *) return 1 ;;
  esac
}

curve_extra_args() {
  local curve="$1"
  case "${curve}" in
    lacsfc_opt)
      printf '%s\n' "--lacsfc-max-evals ${LACSFC_MAX_EVALS}" ;;
    lacsfc_meta)
      if [[ -f "${TIER3_CKPT_DIR}/${TIER3_MODEL_NAME}_metadata.json" ]]; then
        printf '%s\n' "--lacsfc-max-evals ${META_REFINE_EVALS} --lacsfc-meta-model-dir ${TIER3_CKPT_DIR} --lacsfc-meta-model-name ${TIER3_MODEL_NAME} --lacsfc-meta-no-evaluator --lacsfc-meta-fallback-cmaes"
      else
        printf '%s\n' "--lacsfc-max-evals ${META_REFINE_EVALS}"
      fi
      ;;
    *)
      printf '\n' ;;
  esac
}

append_status() {
  local layer="$1" workload="$2" dim="$3" n="$4" qkind="$5" curve="$6" seed="$7" wall="$8" ec="$9" status="${10}" output_file="${11}" log_file="${12}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${layer}" "${workload}" "${dim}" "${n}" "${qkind}" "${curve}" "${seed}" "${wall}" "${ec}" "${status}" "${output_file}" "${log_file}" >> "${LOG_TSV}"
}

run_case() {
  local layer="$1" workload="$2" curve="$3" seed="$4" n_value="$5" qkind="$6"
  local dim token kind output_file log_file tag qargs curve_extra base_args extra_args t0 t1 wall ec status
  dim="$(workload_dim "${workload}")" || return 0
  token="$(curve_token "${curve}" "${dim}")"
  local token_status=$?
  output_file="${OUT_DIR}/jsonl/${layer}__${workload}__n${n_value}__${qkind}__${curve}.jsonl"
  log_file="${OUT_DIR}/logs/${layer}__${workload}__n${n_value}__${qkind}__${curve}__s${seed}.log"
  tag="exp_0530:${layer}:${workload}:n${n_value}:${qkind}:${curve}:s${seed}"
  if [[ "${token_status}" == "2" ]]; then
    append_status "${layer}" "${workload}" "${dim}" "${n_value}" "${qkind}" "${curve}" "${seed}" 0 0 "N/A-unsupported" "${output_file}" "${log_file}"
    echo "[skip] ${tag} unsupported"
    return 0
  fi
  if [[ "${token_status}" != "0" ]]; then
    append_status "${layer}" "${workload}" "${dim}" "${n_value}" "${qkind}" "${curve}" "${seed}" 0 1 "N/A-unknown_curve" "${output_file}" "${log_file}"
    return 0
  fi
  if [[ "${RESUME}" == "1" ]] && has_completed_tag "${output_file}" "${tag}"; then
    echo "[skip] ${tag} completed"
    return 0
  fi
  qargs="$(QUERY_KIND="${qkind}" query_args)" || return 0
  read -r -a qargs_arr <<< "${qargs}"
  curve_extra="$(curve_extra_args "${curve}")"
  read -r -a curve_extra_arr <<< "${curve_extra}"
  kind="$(workload_points_kind "${workload}")"
  base_args=(--exp-tag "${tag}" --output "${output_file}" --seed "${seed}" --repeat 1 --n "${n_value}" --qn "${QN}" --train-qn "${TRAIN_QN}" --d "${dim}" --k "${K}" --lacsfc-k "${LACSFC_K}" --b "${B}" --area-min "${AREA_MIN}" --area-max "${AREA_MAX}" --curves "${token}" --profile --train-bo-init-size "${LMSFC_BO_INIT_SIZE}" --train-bo-rounds "${LMSFC_BO_ROUNDS}" --train-bo-evals-per-round "${LMSFC_BO_EVALS_PER_ROUND}" --bitallocz-bo-init "${BITALLOCZ_BO_INIT}" --bitallocz-bo-iters "${BITALLOCZ_BO_ITERS}" --adacurve-epochs "${ADACURVE_EPOCHS}" --adacurve-train-sample-size "${ADACURVE_TRAIN_SAMPLE_SIZE}" --lacsfc-sigma0 "${SIGMA0}" --lacsfc-popsize "${POPSIZE}" --lacsfc-mode "${LACSFC_MODE}" --lacsfc-objective "${LACSFC_OBJECTIVE}" --lacsfc-no-learn-obj --lacsfc-init-obj "${LACSFC_INIT_OBJ}")
  if [[ "${kind}" == synthetic:* ]]; then
    base_args+=(--points "${kind#synthetic:}" "${qargs_arr[@]}")
  else
    local source_file gen_info gen_name points_file queries_file
    source_file="$(real_source_file "${workload}")"
    if [[ -n "${source_file}" ]]; then
      if [[ ! -f "${source_file}" ]]; then
        append_status "${layer}" "${workload}" "${dim}" "${n_value}" "${qkind}" "${curve}" "${seed}" 0 1 "N/A-missing_data" "${output_file}" "${log_file}"
        echo "[skip] ${tag} missing data ${source_file}"
        return 0
      fi
      base_args+=(--points real --points-file "${source_file}" --require-n "${qargs_arr[@]}")
    else
      gen_info="$(N="${n_value}" ensure_realistic_data "${workload}" "${seed}")" || {
        append_status "${layer}" "${workload}" "${dim}" "${n_value}" "${qkind}" "${curve}" "${seed}" 0 1 "N/A-no_generator" "${output_file}" "${log_file}"
        return 0
      }
      IFS=$'\t' read -r gen_name points_file queries_file <<< "${gen_info}"
      base_args+=(--points real --points-file "${points_file}" --queries-file "${queries_file}" --require-n)
    fi
  fi
  echo "[run] ${tag} token=${token} cap=${WALL_CLOCK_CAP_S}s"
  t0=$(date +%s)
  set +e
  timeout --kill-after=30 "${WALL_CLOCK_CAP_S}" python -m src.mvp.compare_runner "${base_args[@]}" "${curve_extra_arr[@]}" >"${log_file}" 2>&1
  ec=$?
  set -e
  t1=$(date +%s)
  wall=$(( t1 - t0 ))
  case "${ec}" in
    0) status="ok" ;;
    124) status="N/A-timeout" ;;
    137) status="N/A-OOM-or-killed" ;;
    *) status="N/A-exit_${ec}" ;;
  esac
  append_status "${layer}" "${workload}" "${dim}" "${n_value}" "${qkind}" "${curve}" "${seed}" "${wall}" "${ec}" "${status}" "${output_file}" "${log_file}"
  echo "[done] ${tag} ${status} wall=${wall}s"
}

layer_plan() {
  case "${LAYER}" in
    smoke)
      printf '%s\n' "smoke uniform_3d 200 0 mixed ${SMOKE_CURVES}"
      printf '%s\n' "smoke gmm_3d 200 0 mixed ${SMOKE_CURVES}"
      ;;
    A)
      local workloads="${WORKLOADS:-tiger_2d geonames_2d osm_2d osm_3d uniform_4d gmm_4d}"
      for wl in ${workloads}; do for seed in ${SEEDS}; do printf '%s\n' "A ${wl} ${N} ${seed} mixed ${CURVES_ALL}"; done; done
      ;;
    B)
      local workloads="${WORKLOADS:-uniform_3d uniform_4d uniform_5d gmm_3d gmm_4d gmm_5d}"
      for wl in ${workloads}; do for seed in ${SEEDS}; do printf '%s\n' "B ${wl} ${N} ${seed} mixed ${CURVES_ALL}"; done; done
      ;;
    C)
      local workloads="${WORKLOADS:-osm_2d osm_3d gmm_4d}"
      local scales="${SCALES:-1000000 5000000 10000000}"
      for wl in ${workloads}; do for scale in ${scales}; do for seed in ${SEEDS}; do printf '%s\n' "C ${wl} ${scale} ${seed} mixed ${CURVES_SCALE}"; done; done; done
      if [[ -z "${XL_SCALE+x}" ]]; then
        for wl in ${workloads}; do printf '%s\n' "C ${wl} 30000000 0 mixed ${CURVES_SCALE}"; done
      elif [[ -n "${XL_SCALE}" ]]; then
        for wl in ${workloads}; do printf '%s\n' "C ${wl} ${XL_SCALE} 0 mixed ${CURVES_SCALE}"; done
      fi
      ;;
    E)
      local workloads="${WORKLOADS:-osm_2d osm_3d gmm_4d gmm_5d}"
      local seeds="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
      for wl in ${workloads}; do for seed in ${seeds}; do printf '%s\n' "E ${wl} ${N} ${seed} mixed ${CURVES_VARIANT}"; done; done
      ;;
    F)
      local workloads="${WORKLOADS:-osm_2d gmm_4d}"
      local qkinds="${QUERY_KINDS:-uniform skewed mixed}"
      for wl in ${workloads}; do for qk in ${qkinds}; do for seed in ${SEEDS}; do printf '%s\n' "F ${wl} ${N} ${seed} ${qk} ${CURVES_QUERY}"; done; done; done
      ;;
    *) echo "Unknown LAYER=${LAYER}" >&2; return 1 ;;
  esac
}

while read -r layer workload n_value seed qkind curves; do
  [[ -z "${layer}" ]] && continue
  for curve in ${curves}; do
    run_case "${layer}" "${workload}" "${curve}" "${seed}" "${n_value}" "${qkind}"
  done
done < <(layer_plan)

if [[ "${RUN_ANALYSIS}" == "1" ]]; then
  python "${ROOT_DIR}/examples/analyze_exp_0530_full_curve_matrix_v1.py" \
    --input-dir "${OUT_DIR}/jsonl" \
    --walltime "${LOG_TSV}" \
    --out-dir "${OUT_DIR}/summary"
fi

echo "[done] EXP-0530 layer=${LAYER} results under ${OUT_DIR}"
