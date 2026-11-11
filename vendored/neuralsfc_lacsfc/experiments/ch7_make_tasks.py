"""Chapter 7 shared tasks (exp_chapter7_spec section 5, C4).

Same code path as experiments/make_tasks.py (lacsfc.tasks.make_task, grid K = 20 // d, task seed
from (dataset, d, family, seed index), equal-width binning), extended with
  - family data_iso = WorkloadSpec(center="data", aspect_max=1.0);
  - datasets osm, tiger (4D semi-synthetic: 2 real coordinates + 2 synthetic attributes), points
    from <repo>/examples/generate_realistic_highdim_workloads_v1.make_points with
    n = 250,000 and point seed 0 (one point set per dataset, shared by all query seeds, as for
    GeoNames).
Writes <out>/d{d}/<name>.npz and <out>/manifest.json (meta, file SHA-256, SHA-256 of the generator
script and of the raw source files).

    python experiments/ch7_make_tasks.py --datasets geonames osm tiger --d 4
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from e3_real import load_geonames  # noqa: E402
from lacsfc.tasks import WorkloadSpec, make_task, save_task  # noqa: E402
from make_tasks import FAMILIES as G6_FAMILIES, grid_K, task_name, task_seed  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BENCH = Path(__file__).resolve().parents[3]
GENERATOR = BENCH / "examples" / "generate_realistic_highdim_workloads_v1.py"
SEMI = {"osm": ("osm_time_pop_4d", BENCH / "data" / "OSM" / "osm25mbinary"),
        "tiger": ("tiger_time_class_4d", BENCH / "data" / "tiger_tweets" / "tiger_us_2d.csv")}
FAMILIES = dict(G6_FAMILIES, data_iso=WorkloadSpec(center="data", aspect_max=1.0))
DEFAULT_FAMILIES = ["data", "uniform", "hotspot", "data_skew", "data>hotspot", "data_iso"]
N_SEMI = 250_000
POINT_SEED = 0


def specs(family: str):
    a, _, t = family.partition(">")
    return FAMILIES[a], FAMILIES[t or a]


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _generator():
    spec = importlib.util.spec_from_file_location("ch7_semisynth_generator", GENERATOR)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_points(dataset: str, d: int, geonames_path) -> np.ndarray:
    if dataset == "geonames":
        return load_geonames(geonames_path)[:, :d]
    if dataset not in SEMI:
        raise ValueError(f"unknown dataset {dataset!r}")
    if d != 4:
        raise ValueError(f"{dataset} is 4D only")
    workload, source = SEMI[dataset]
    return _generator().make_points(workload, str(source), N_SEMI, POINT_SEED, 8)


def build_task(X: np.ndarray, dataset: str, d: int, family: str, s: int, mA=1000, mB=500, mT=1000) -> dict:
    spec_AB, spec_T = specs(family)
    task = make_task(X, grid_K(d), spec_AB, spec_T, mA=mA, mB=mB, mT=mT,
                     seed=task_seed(dataset, d, family, s), binning="equal")
    task["meta"].update(dataset=dataset, family=family, seed_index=int(s))
    return task


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(Path.home() / "lacsfc_cache" / "ch7" / "tasks"))
    ap.add_argument("--geonames", default=str(ROOT / "data" / "geonames" / "cities500.txt"))
    ap.add_argument("--datasets", nargs="+", default=["geonames", "osm", "tiger"])
    ap.add_argument("--d", type=int, nargs="+", default=[4])
    ap.add_argument("--families", nargs="+", default=DEFAULT_FAMILIES)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 11)))
    ap.add_argument("--mA", type=int, default=1000)
    ap.add_argument("--mB", type=int, default=500)
    ap.add_argument("--mT", type=int, default=1000)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    man_path = out / "manifest.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {}
    sources = manifest.setdefault("_sources", {})
    for ds in args.datasets:
        if ds == "geonames":
            sources[ds] = {"path": args.geonames, "sha256": sha256_file(args.geonames)}
        else:
            sources[ds] = {"path": str(SEMI[ds][1]), "sha256": sha256_file(SEMI[ds][1]),
                           "generator": str(GENERATOR), "generator_sha256": sha256_file(GENERATOR),
                           "workload": SEMI[ds][0], "n": N_SEMI, "point_seed": POINT_SEED}
    for d in args.d:
        (out / f"d{d}").mkdir(exist_ok=True)
        for ds in args.datasets:
            if ds != "geonames" and d != 4:
                continue
            X = load_points(ds, d, args.geonames)
            sources[ds][f"points_d{d}_sha256"] = hashlib.sha256(np.ascontiguousarray(X).tobytes()).hexdigest()
            for fam in args.families:
                for s in args.seeds:
                    name = task_name(ds, d, fam, s)
                    path = out / f"d{d}" / f"{name}.npz"
                    task = build_task(X, ds, d, fam, s, args.mA, args.mB, args.mT)
                    meta = save_task(path, task)
                    meta["file_sha256"] = sha256_file(path)
                    manifest[name] = meta
                    print(name, task["K"], meta["n"], meta["file_sha256"][:12], flush=True)
                    man_path.write_text(json.dumps(manifest, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
