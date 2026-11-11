"""Formal shared tasks on GeoNames (verification_plan A.1, A.2, A.6).

One file per (d, family, seed): points = first d columns of e3_real.load_geonames,
grid K = floor(20 / d), windows A / B / T from lacsfc.tasks.make_task. A family
"F>G" draws A and B from F and T from G (drift, A.2.4). The task seed is derived
from (dataset, d, family, seed index), so different families do not share streams.
Writes <out>/manifest.json (meta + sha256 of every file). Seed index 0 was used by the
smoke run whose T results were seen (verification_plan D8), so formal seeds start at 1.

    python experiments/make_tasks.py            # d 2 3 4, seeds 1-5, equal + quantile (C.1)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from e3_real import load_geonames  # noqa: E402
from lacsfc.tasks import WorkloadSpec, save_task, make_task  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = {
    "data": WorkloadSpec(center="data"),
    "uniform": WorkloadSpec(center="uniform"),
    "hotspot": WorkloadSpec(center="hotspot"),
    "data_skew": WorkloadSpec(center="data", axis_skew=True, skew_axis=0),
}
DEFAULT_FAMILIES = ["data", "uniform", "hotspot", "data_skew", "data>hotspot"]


def grid_K(d: int) -> int:
    return 20 // int(d)


def task_seed(dataset: str, d: int, family: str, s: int) -> int:
    h = hashlib.sha256(f"{dataset}|d{d}|{family}|{s}".encode()).hexdigest()
    return int(h[:8], 16)


def specs(family: str):
    a, _, t = family.partition(">")
    return FAMILIES[a], FAMILIES[t or a]


def task_name(dataset: str, d: int, family: str, s: int, binning: str = "equal") -> str:
    suffix = "" if binning == "equal" else f"_{binning}"
    return f"{dataset}_d{d}_{family.replace('>', '-to-')}_s{s}{suffix}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(ROOT / "experiments" / "cache" / "tasks"))
    ap.add_argument("--geonames", default=str(ROOT / "data" / "geonames" / "cities500.txt"))
    ap.add_argument("--d", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--families", nargs="+", default=DEFAULT_FAMILIES)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--binning", nargs="+", default=["equal", "quantile"], choices=["equal", "quantile"])
    ap.add_argument("--mA", type=int, default=1000)
    ap.add_argument("--mB", type=int, default=500)
    ap.add_argument("--mT", type=int, default=1000)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    X = load_geonames(args.geonames)
    man_path = out / "manifest.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {}
    for d in args.d:
        for fam in args.families:
            spec_AB, spec_T = specs(fam)
            for s in args.seeds:
                for binning in args.binning:
                    name = task_name("geonames", d, fam, s, binning)
                    path = out / f"{name}.npz"
                    task = make_task(X[:, :d], grid_K(d), spec_AB, spec_T, mA=args.mA, mB=args.mB, mT=args.mT,
                                     seed=task_seed("geonames", d, fam, s), binning=binning)
                    task["meta"].update(dataset="geonames", family=fam, seed_index=int(s))
                    meta = save_task(path, task)
                    meta["file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                    manifest[name] = meta
                    print(name, task["K"], meta["n"], meta["file_sha256"][:12], flush=True)
                    man_path.write_text(json.dumps(manifest, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
