#!/usr/bin/env bash
# Prepares large datasets omitted from the artifact.
# Usage:
#   ./data/prepare.sh osm CSV        # build data/OSM/osm250mbinary32 from a 2-col CSV of uint32 x,y
#   ./data/prepare.sh layerc         # regenerate Layer-C npy files (requires osm250mbinary32)
#   ./data/prepare.sh all            # layerc only (osm file must already exist)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

step_osm() {
  local csv="$1"
  python - "$csv" <<'PY'
import sys, numpy as np
csv_path = sys.argv[1]
out = "data/OSM/osm250mbinary32"
# Format (src/mvp/workloads.py load_points_real):
#   <u4 count header> + count pairs of <u4 (x, y)
pts = np.loadtxt(csv_path, delimiter=",", dtype="<u4")
with open(out, "wb") as f:
    np.array([len(pts)], dtype="<u4").tofile(f)
    pts.tofile(f)
print(f"[prepare] wrote {out}: {len(pts)} points")
PY
}

step_layerc() {
  local src="data/OSM/osm250mbinary32"
  [[ -f "$src" ]] || { echo "[prepare] missing $src (run 'osm' step first)" >&2; return 1; }
  local out_dir="data/exp_0530_layerC_scale_stage2_v1"
  mkdir -p "$out_dir"
  for seed in 0 1 2; do
    for n in 1000000 2000000 5000000 10000000 15000000 20000000 30000000; do
      local stem="osm_time_3d_s${seed}_n${n}"
      [[ -f "${out_dir}/${stem}_points.npy" ]] && continue
      python examples/generate_realistic_highdim_workloads_v1.py \
        --workload osm_time_3d \
        --source-points-file "$src" \
        --out-dir "$out_dir" \
        --out-prefix "$stem" \
        --seed "$seed" --n "$n" \
        --qn 40 --k 4 \
        --spatial-area-min 1e-5 --spatial-area-max 5e-3
      echo "[prepare] generated ${stem}"
    done
  done
}

case "${1:-all}" in
  osm)    [[ -n "${2:-}" ]] || { echo "usage: $0 osm <csv>" >&2; exit 1; }; step_osm "$2" ;;
  layerc) step_layerc ;;
  all)    step_layerc ;;
  *) echo "usage: $0 {osm <csv>|layerc|all}" >&2; exit 1 ;;
esac
