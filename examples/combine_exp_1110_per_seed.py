"""Rebuild the three combined_per_seed.json files consumed by make_exp_1110_tables.py.

Each combined file merges per-seed metric values from archived experiment JSONL
plus the EXP-1110 rerun JSONL (GBO/QUILTS/Hilbert anchors). First directory wins
for a duplicate (key, seed): the archives are listed before the rerun dirs so
the archived values are kept for anchored curves and the rerun dirs only add
the new GBO/QUILTS rows.

Outputs (paths relative to repo root):
  results/exp_1110_gbo_quilts_on_0530/combined_per_seed.json   key = "<layer>|<wl>|<curve>"
  results/exp_1110_2d_all_curve_split/combined_per_seed.json   key = "<wl>|<curve>"
  results/exp_1110_gbo_quilts_on_0522/combined_per_seed.json   key = "<curve>|<wl>"
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

METRICS = ("avg_amp", "avg_seg", "avg_pa", "query_s", "build_s")

LAYERED_RE = re.compile(r"^(?P<layer>[A-Z])__(?P<wl>.+?)__n(?P<n>\d+)__mixed__(?P<curve>.+)$")
WL_CURVE_RE = re.compile(r"^(?P<wl>.+?)__(?P<curve>[a-z0-9_]+)$")
ALL_CURVES_RE = re.compile(r"^(?P<wl>.+?)_all_curves$")


def _collect(files, key_fn, out, keep_fn=None, metrics=METRICS):
    """files: iterable of jsonl paths. key_fn(stem) -> (key_prefix, curve_from_name).
    out[key][metric][seed] = value; first record wins per (key, seed, metric)."""
    for f in files:
        parsed = key_fn(f.stem)
        if parsed is None:
            continue
        prefix, curve_name = parsed
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            seed = int(r["seed"])
            for curve, res in r["results"].items():
                key = prefix.format(curve=curve_name or curve)
                if keep_fn is not None and not keep_fn(key):
                    continue
                slot = out[key]
                for met in metrics:
                    v = res.get(met)
                    if v is not None and seed not in slot[met]:
                        slot[met][seed] = float(v)


def _emit(out):
    """{key: {metric: {seed: val}}} -> {key: {metric: [val by seed]}}"""
    return {k: {m: [slot[m][s] for s in sorted(slot[m])] for m in slot if slot[m]}
            for k, slot in sorted(out.items())}


def combine_0530():
    out = defaultdict(lambda: defaultdict(dict))
    dirs = [RES / "exp_0530_layerA_main_v1/jsonl",
            RES / "exp_0530_layerB_dimdist_v1/jsonl",
            RES / "exp_1110_gbo_quilts_on_0530/jsonl"]

    def key_fn(stem):
        m = LAYERED_RE.match(stem)
        return (f"{m['layer']}|{m['wl']}|{{curve}}", m["curve"]) if m else None

    # archived Layer A contributes only osm_3d (the other A workloads duplicate Layer B
    # cells modulo the non-deterministic LACSFC-Opt/Meta rows). The shipped 0530 file
    # carries no build_s column.
    keep = lambda key: not key.startswith("A|") or key.startswith("A|osm_3d|")
    _collect([f for d in dirs for f in sorted(d.glob("*.jsonl"))], key_fn, out, keep,
             metrics=("avg_amp", "avg_seg", "avg_pa", "query_s"))
    return _emit(out)


def combine_2d_split():
    out = defaultdict(lambda: defaultdict(dict))
    d = RES / "exp_1110_2d_all_curve_split"

    def key_fn(stem):
        m = ALL_CURVES_RE.match(stem)
        return (f"{m['wl']}|{{curve}}", None) if m else None

    _collect(sorted(d.glob("*.jsonl")), key_fn, out)
    return _emit(out)


def combine_0522():
    out = defaultdict(lambda: defaultdict(dict))
    dirs = [RES / "exp_0522_lacsfc_4d_all_baseline_v1/jsonl",
            RES / "exp_1110_gbo_quilts_on_0522/jsonl"]

    def key_fn(stem):
        m = WL_CURVE_RE.match(stem)
        return (f"{m['curve']}|{m['wl']}", m["curve"]) if m else None

    _collect([f for d in dirs for f in sorted(d.glob("*.jsonl"))], key_fn, out)
    return _emit(out)


TARGETS = {
    "exp_1110_gbo_quilts_on_0530": combine_0530,
    "exp_1110_2d_all_curve_split": combine_2d_split,
    "exp_1110_gbo_quilts_on_0522": combine_0522,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="compare regenerated content against the existing files; exit 1 on mismatch")
    ap.add_argument("--out-dir", default=None,
                    help="write to <out-dir>/<name>/combined_per_seed.json instead of results/ "
                         "(useful for --check-style dry runs)")
    args = ap.parse_args()

    bad = 0
    for name, fn in TARGETS.items():
        data = fn()
        dst_dir = Path(args.out_dir) / name if args.out_dir else RES / name
        dst = dst_dir / "combined_per_seed.json"
        if args.check:
            ref_path = RES / name / "combined_per_seed.json"
            ref = json.loads(ref_path.read_text()) if ref_path.exists() else None
            if ref == data:
                print(f"[ok] {name}: {len(data)} keys identical")
            else:
                bad += 1
                ref_keys, new_keys = set(ref or {}), set(data)
                print(f"[MISMATCH] {name}: missing={sorted(ref_keys - new_keys)[:5]} "
                      f"extra={sorted(new_keys - ref_keys)[:5]}")
                for k in sorted(ref_keys & new_keys):
                    if ref[k] != data[k]:
                        print(f"  value diff at {k}: ref={ref[k]} new={data[k]}")
                        break
        else:
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst.write_text(json.dumps(data, indent=1) + "\n")
            print(f"wrote {dst} ({len(data)} keys)")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
