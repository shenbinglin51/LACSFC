"""Chapter 7 external baselines (exp_chapter7_spec C3; decisions of 2026-11-09): fit on A only, export a
full-grid rank per task for ch7_methods.load_external.

Methods and sources
    lmsfc      archived LMSFC C++ (<repo>/LMSFC/lmsfc.rar, 2023) with the ch7 patches P1-P8 of
               <repo>/LMSFC/lmsfc_ch7/PATCHES.md (d dims, fixed seed); TrainingBayesian with its wall-clock
               objective; trainingQueries = A, testingQueries (its top-3 re-ranking) = B
    bitallocz  bitallocLZ reproduction (<repo>/bitallocLZ/src): BO over per-column weights and the Z-value
               length L (learn_l), paper Def. 2 interleave, paper zone-map cost model on a 10% sample
    adacurve   Adacurve reproduction (<repo>/Adacurve/Adacurve/hd), run_repro.py driver config, cell-projected
    bmtree     official Learned-BMTree (= PiecewiseSFC) via experiments/baselines/bmtree_native.fit_bmtree
    quilts     <repo>/QUILTS design_multi; patterns = width_to_bits of each A cell box per dimension, distinct,
               admitted in the paper's order (smallest total first, then largest, then the rest; ties lexicographic)
Builders get fit_view(task) only: points, cells, K and the A windows; T is never passed, and B only to the methods
in SELECT_ON_B (LMSFC: B replaces the second half of the query file test.cpp gave TrainingBayesian for its final
top-3 choice). LMSFC times queries while training: build it with --workers 1 on an otherwise idle machine.
AdaCurve cell projection: occupied cell key = median AdaCurve position of its points; an empty cell takes the key
of the nearest preceding occupied cell in Z-order (-inf before the first); cells sorted by (key, Z-order rank).
BitAllocZ with L < d*K maps several cells to one Z-value; cells sorted by (Z-value, Z-order rank).

Outputs under <ext_root>/<method>/: <task>.code.bin (u64 rank per cell, C order), <task>.meta.json, and
manifest.json (written once all tasks are built; never overwritten with different content).
Run with a Python that has torch, scikit-learn and bayes_opt (a missing package stops the build instead of
silently changing the trainer); LMSFC needs <repo>/LMSFC/lmsfc_ch7/build.sh run first:

    python experiments/ch7_export.py --ext-root ~/lacsfc_cache/ch7/ext build --method lmsfc --workers 1
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import importlib.util
import json
import multiprocessing as mp
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LAC = HERE.parent
sys.path.insert(0, str(LAC / "src"))
sys.path.insert(0, str(HERE))

from lacsfc.baselines import zorder_rank  # noqa: E402
from lacsfc.tasks import load_task  # noqa: E402

MVP_ROOT = Path(os.environ.get("MVP_ROOT", LAC.parents[1]))
BITALLOC_ROOT = Path(os.environ.get("BITALLOC_ROOT", LAC.parents[1] / "bitallocLZ"))
ADACURVE_ROOT = Path(os.environ.get("ADACURVE_ROOT", LAC.parents[1] / "Adacurve"))
LMSFC_ROOT = Path(os.environ.get("LMSFC_ROOT", LAC.parents[1] / "LMSFC" / "lmsfc_ch7"))
QUILTS_ROOT = Path(os.environ.get("QUILTS_ROOT", LAC.parents[1] / "QUILTS"))
PAGE = 256
METHODS = ("lmsfc", "bitallocz", "adacurve", "bmtree", "quilts")
ADAPTERS = {
    "lmsfc": "archived LMSFC C++ + ch7 patches P1-P8 (lmsfc_ch7/PATCHES.md), TrainingBayesian wall-clock objective; "
             "fit on A, top-3 choice on B",
    "bitallocz": "bitallocLZ reproduction (BO over weights and L, Def. 2, paper cost model), A only; "
                 "cells sharing a Z-value ordered by Z-order rank",
    "adacurve": "Adacurve/hd reproduction (run_repro.py config, Z-order warmstart, checkpoint by A io_cost), "
                "cell-projected (median position; empty cells follow Z-order predecessor)",
    "bmtree": "official Learned-BMTree MCTS (experiments/baselines/bmtree_native.fit_bmtree), A only",
    "quilts": "QUILTS design_multi, A only; patterns = per-dim width_to_bits of the A cell boxes, distinct, admitted "
              "in the paper's order (smallest total, largest, rest; ties lexicographic)",
}
DEFAULTS = {
    "lmsfc": {"seed": 0, "page_size": PAGE, "per": 20, "iters": 1000},
    "bitallocz": {"seed": 0, "b": PAGE, "bo_init": 10, "bo_iters": 100, "scheme": "paper", "learn_l": True,
                  "l_min": 8, "cost_model": "paper", "sample_frac": 0.1, "max_intervals": 8192},
    "adacurve": {"seed": 0, "block_size": PAGE, "compress_dim_cap": 16, "n_iters": 1000,
                 "n_train_queries_per_iter": 30, "lr": 3e-4, "sample_size": 5000, "train_block_size": 50,
                 "soft_rank_eps": 0.1, "eps_anneal": False, "local_weight": 0.05, "warmstart": "zorder",
                 "warmstart_steps": 1500, "ckpt_every": 40},
    "bmtree": {"seed": 0, "data_sample_points": 10**5, "smallest_split_card": 0, "max_depth": 8, "split_depth": 5,
               "action_depth": 10, "rollouts": 10},
    "quilts": {"order": "paper"},
}
REQUIRES = {"lmsfc": ("sklearn", "bayes_opt"), "bitallocz": ("bayes_opt",), "adacurve": ("torch",),
            "bmtree": ("torch",), "quilts": ()}
FIT_KEYS = ("A_low", "A_high", "A_ql", "A_qh")
SELECT_KEYS = ("B_low", "B_high")
SELECT_ON_B = ("lmsfc",)


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def array_sha(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def view_keys(method=None) -> tuple:
    return FIT_KEYS + (SELECT_KEYS if method in SELECT_ON_B else ())


def fit_view(t: dict, method=None) -> dict:
    """What a builder may see: points, cells, K, the A split and (SELECT_ON_B only) the B cell boxes."""
    v = {"u": np.asarray(t["u"]), "cells": np.asarray(t["cells"]), "K": int(t["K"])}
    for k in view_keys(method):
        v[k] = np.asarray(t[k])
    return v


def grid_coords(shape) -> np.ndarray:
    return np.stack(np.unravel_index(np.arange(int(np.prod(shape)), dtype=np.int64), shape), axis=1)


def rank_from_codes(codes) -> np.ndarray:
    codes = np.asarray(codes)
    order = np.argsort(codes, kind="stable")
    if np.any(np.diff(codes[order]) == 0):
        raise RuntimeError("codes are not unique: not a full-grid order")
    rank = np.empty(order.size, dtype=np.int64)
    rank[order] = np.arange(order.size, dtype=np.int64)
    return rank


def check_perm(rank, N: int) -> None:
    rank = np.asarray(rank)
    if rank.shape != (N,) or not np.array_equal(np.sort(rank), np.arange(N)):
        raise RuntimeError(f"rank is not a permutation of the {N} grid cells")


def deposit_codes(coords: np.ndarray, masks) -> np.ndarray:
    """Vectorized lmsfc.zaddress.cal_z_address: bit j of coordinate i goes to the j-th lowest set bit of masks[i]."""
    coords = np.asarray(coords, dtype=np.uint64)
    out = np.zeros(coords.shape[0], dtype=np.uint64)
    for i, m in enumerate(masks):
        m, j = int(m), 0
        while m:
            lsb = m & -m
            out |= ((coords[:, i] >> np.uint64(j)) & np.uint64(1)) << np.uint64(lsb.bit_length() - 1)
            m ^= lsb
            j += 1
    return out


def rank_with_zorder_ties(codes, zrank) -> np.ndarray:
    """Full-grid rank sorted by (code, Z-order rank)."""
    seq = np.lexsort((np.asarray(zrank), np.asarray(codes)))
    rank = np.empty(seq.size, dtype=np.int64)
    rank[seq] = np.arange(seq.size, dtype=np.int64)
    return rank


def schedule_codes(coords: np.ndarray, curve) -> np.ndarray:
    """Vectorized bitallocLZ uc_curve.BitAllocZCurve.encode_cell (either interleave scheme)."""
    c = np.asarray(coords, dtype=np.uint64)
    out = np.zeros(c.shape[0], dtype=np.uint64)
    one = np.uint64(1)
    for dim, pos in curve._schedule:
        sh = np.uint64(int(curve.k) - int(curve.alloc[dim]) + int(pos))
        out = (out << one) | ((c[:, dim] >> sh) & one)
    return out


def project_point_order(cells_flat, pos, zrank) -> np.ndarray:
    """AdaCurve cell projection (module docstring): full-grid rank from a point-level order ``pos``."""
    cells_flat, pos, zrank = np.asarray(cells_flat), np.asarray(pos, dtype=np.float64), np.asarray(zrank)
    N = zrank.size
    o = np.argsort(cells_flat, kind="stable")
    cf, ps = cells_flat[o], pos[o]
    uniq, start, cnt = np.unique(cf, return_index=True, return_counts=True)
    key = np.full(N, np.nan)
    key[uniq] = [np.median(ps[s:s + c]) for s, c in zip(start, cnt)]
    zorder = np.argsort(zrank, kind="stable")
    kz = key[zorder]
    last = np.maximum.accumulate(np.where(np.isnan(kz), -1, np.arange(N)))
    key[zorder] = np.where(last >= 0, kz[np.maximum(last, 0)], -np.inf)
    seq = np.lexsort((zrank, key))
    rank = np.empty(N, dtype=np.int64)
    rank[seq] = np.arange(N, dtype=np.int64)
    return rank


# ---------------------------------------------------------------- builders (run in a worker process)

def _mvp():
    if str(MVP_ROOT) not in sys.path:
        sys.path.insert(0, str(MVP_ROOT))
    from src.mvp.quantizer import Quantizer
    from src.mvp.types import QueryWindow
    return Quantizer, QueryWindow


def morton_masks(d: int, K: int) -> list:
    """Interleaved start masks over d*K bits, dimension 0 on the highest bit of each group (test.cpp's
    2-D example masks 183251937962 / 91625968981 are this pattern for d = 2, K = 19)."""
    return [sum(1 << (j * d + d - 1 - i) for j in range(K)) for i in range(d)]


def lmsfc_exe() -> Path:
    exe = LMSFC_ROOT / "ch7_lmsfc"
    info = (LMSFC_ROOT / "ch7_lmsfc.buildinfo").read_text().split()
    if not exe.exists() or info[:2] != [sha256(exe), "ch7_lmsfc"]:
        raise RuntimeError(f"{exe} missing or differs from its buildinfo; run {LMSFC_ROOT / 'build.sh'}")
    return exe


def write_lmsfc_inputs(work: Path, cells, boxes: dict) -> None:
    """test.cpp formats: points = u32 count then u32 coordinates row-major; queries = csv lb..., ub... (inclusive)."""
    cells = np.ascontiguousarray(cells, dtype="<u4")
    with open(work / "points.bin", "wb") as f:
        f.write(np.uint32(len(cells)).astype("<u4").tobytes())
        f.write(cells.tobytes())
    for name, (lo, hi) in boxes.items():
        np.savetxt(work / f"{name}.csv", np.hstack([lo, hi]).astype(np.int64), fmt="%d", delimiter=",")


def build_lmsfc(v, cfg):
    import shutil
    import subprocess
    import tempfile
    exe = lmsfc_exe()
    d, K = v["cells"].shape[1], v["K"]
    if d * K > 63:
        raise RuntimeError("LMSFC masks are 64-bit")
    start = morton_masks(d, K)
    with tempfile.TemporaryDirectory(prefix="ch7_lmsfc_") as w:
        work = Path(w)
        write_lmsfc_inputs(work, v["cells"], {"A": (v["A_low"], v["A_high"]), "B": (v["B_low"], v["B_high"])})
        shutil.copy(LMSFC_ROOT / "rffile.init", work / "rffile")
        env = {**os.environ, "LMSFC_SEED": str(cfg["seed"]), "LMSFC_PYTHON": sys.executable,
               "LMSFC_RF": str(LMSFC_ROOT / "randomForestnew.py"), "OMP_NUM_THREADS": "1"}
        cmd = [str(exe), str(d), "points.bin", "A.csv", "B.csv", str(cfg["page_size"]), str(cfg["per"]),
               str(cfg["iters"]), "masks.txt"] + [str(m) for m in start]
        p = subprocess.run(cmd, cwd=work, env=env, capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(f"ch7_lmsfc failed ({p.returncode}): {p.stderr[-2000:]}")
        masks = [int(x) for x in (work / "masks.txt").read_text().split()]
        y = np.atleast_1d(np.loadtxt(work / "cacheY.csv", dtype=np.int64))
    if len(masks) != d or sum(masks) != (1 << (d * K)) - 1 or any(bin(m).count("1") != K for m in masks) \
            or any(a & b for i, a in enumerate(masks) for b in masks[i + 1:]):
        raise RuntimeError(f"LMSFC returned invalid masks {masks}")
    top3 = [ln.split()[1:] for ln in p.stdout.splitlines() if ln.startswith("testnum ")]
    codes = deposit_codes(grid_coords((1 << K,) * d), masks)
    return rank_from_codes(codes), {
        "masks": [hex(m) for m in masks], "start_masks": [hex(m) for m in start], "n_evaluated": int(y.size),
        "train_cost_us": {"min": int(y.min()), "median": float(np.median(y)), "max": int(y.max()),
                          "n_zero": int((y == 0).sum()), "n_distinct": int(np.unique(y).size)},
        "top3_B_cost_us_and_count": [[int(a), int(b)] for a, b in top3],
        "stdout_sha256": hashlib.sha256(p.stdout.encode()).hexdigest()}


BITALLOC_PKG = "_bitallocLZ_src"


def _bitalloc(mods=("uc_optimize", "uc_curve", "uc_quantizer", "uc_types")) -> dict:
    """bitallocLZ/src loaded under a private package name (its own name, src, clashes with the MVP's)."""
    if BITALLOC_PKG not in sys.modules:
        spec = importlib.util.spec_from_file_location(BITALLOC_PKG, BITALLOC_ROOT / "src" / "__init__.py",
                                                      submodule_search_locations=[str(BITALLOC_ROOT / "src")])
        mod = importlib.util.module_from_spec(spec)
        sys.modules[BITALLOC_PKG] = mod
        spec.loader.exec_module(mod)
    return {m: importlib.import_module(f"{BITALLOC_PKG}.{m}") for m in mods}


def build_bitallocz(v, cfg):
    m = _bitalloc()
    d, K = v["cells"].shape[1], v["K"]
    u = np.asarray(v["u"], dtype=np.float64)
    qz = m["uc_quantizer"].Quantizer(d=d, k=K)
    if not np.array_equal(qz.quantize_points(u), v["cells"]):
        raise RuntimeError("bitallocLZ Quantizer cells differ from the task cells")
    QW = m["uc_types"].QueryWindow
    queries = [QW(low=np.asarray(lo, dtype=np.float64), high=np.asarray(hi, dtype=np.float64))
               for lo, hi in zip(v["A_ql"], v["A_qh"])]
    qq = [qz.quantize_window(q) for q in queries]
    if not (np.array_equal(np.stack([q.low_cell for q in qq]), v["A_low"])
            and np.array_equal(np.stack([q.high_cell for q in qq]), v["A_high"])):
        raise RuntimeError("bitallocLZ quantized A windows differ from the task cell boxes")
    res = m["uc_optimize"].learn_alloc_bo(
        points=u, train_queries=queries, d=d, k=K, b=cfg["b"], total_bits=d * K, bo_init=cfg["bo_init"],
        bo_iters=cfg["bo_iters"], w_segments=0.0, w_page_accesses=0.0, w_candidates=0.0, seed=cfg["seed"],
        scheme=cfg["scheme"], learn_l=cfg["learn_l"], l_min=cfg["l_min"], l_max=None, cost_model=cfg["cost_model"],
        sample_frac=cfg["sample_frac"], max_intervals=cfg["max_intervals"])
    curve = m["uc_curve"].BitAllocZCurve(d=d, k=K, alloc=tuple(int(a) for a in res.alloc),
                                         total_bits=int(res.l_bits), scheme=cfg["scheme"])
    shape = (1 << K,) * d
    coords = grid_coords(shape)
    codes = schedule_codes(coords, curve)
    sample = np.random.default_rng(0).choice(len(coords), size=min(1000, len(coords)), replace=False)
    if any(int(codes[i]) != curve.encode_cell(coords[i]) for i in sample):
        raise RuntimeError("vectorized BitAllocZ codes differ from encode_cell")
    rank = rank_with_zorder_ties(codes, np.asarray(zorder_rank(shape)))
    return rank, {"alloc": [int(a) for a in curve.alloc], "L": int(curve.total_bits), "train_cost": float(res.cost),
                  "n_evals": len(res.trace), "distinct_codes": int(np.unique(codes).size)}


def _adacurve():
    if str(ADACURVE_ROOT) not in sys.path:
        sys.path.insert(0, str(ADACURVE_ROOT))
    from Adacurve.hd.adacurve import AdaCurveHD
    from Adacurve.hd.evaluate import io_cost
    return AdaCurveHD, io_cost


def build_adacurve(v, cfg):
    """AdaCurveHD with the run_repro.py driver config; checkpoints chosen by io_cost on A (as run_repro does on
    its training queries)."""
    AdaCurveHD, io_cost = _adacurve()
    pts = np.asarray(v["u"], dtype=np.float64)
    q_lo, q_hi = np.asarray(v["A_ql"], dtype=np.float64), np.asarray(v["A_qh"], dtype=np.float64)
    n, d = pts.shape
    K = v["K"]
    model = AdaCurveHD(compress_dim=min(cfg["compress_dim_cap"], max(4, d)), n_iters=cfg["n_iters"],
                       n_train_queries_per_iter=min(cfg["n_train_queries_per_iter"], len(q_lo)), lr=cfg["lr"],
                       sample_size=min(cfg["sample_size"], n), train_block_size=cfg["train_block_size"],
                       soft_rank_eps=cfg["soft_rank_eps"], eps_anneal=cfg["eps_anneal"],
                       local_weight=cfg["local_weight"], warmstart=cfg["warmstart"],
                       warmstart_steps=cfg["warmstart_steps"], seed=cfg["seed"], device="cpu", verbose=False)
    trace = []

    def hook(it, mdl):
        if it >= 0 and it % cfg["ckpt_every"] != 0 and it != mdl.n_iters - 1:
            return None
        val = io_cost(mdl.order(pts), pts, q_lo, q_hi, cfg["block_size"])["io_cost_pct"]
        trace.append([int(it), float(val)])
        return val

    model.fit(pts, q_lo, q_hi, eval_hook=hook)
    order = model.order(pts)
    pos = np.empty(n, dtype=np.int64)
    pos[order] = np.arange(n, dtype=np.int64)
    shape = (1 << K,) * d
    cells_flat = np.ravel_multi_index(tuple(v["cells"].T), shape)
    rank = project_point_order(cells_flat, pos, np.asarray(zorder_rank(shape)))
    return rank, {"occupied_cells": int(np.unique(cells_flat).size), "ckpt_trace": trace,
                  "best_A_io_cost_pct": min(t[1] for t in trace) if trace else None}


def build_bmtree(v, cfg):
    sys.path.insert(0, str(HERE / "baselines"))
    import bmtree_native
    tree, info = bmtree_native.fit_bmtree(v["cells"], v["A_low"], v["A_high"], v["K"], **cfg)
    d, K = v["cells"].shape[1], v["K"]
    return bmtree_native.tree_rank(tree, (1 << K,) * d), info


def _quilts():
    if str(QUILTS_ROOT) not in sys.path:
        sys.path.insert(0, str(QUILTS_ROOT))
    from quilts.design import design_multi, width_to_bits
    return design_multi, width_to_bits


def quilts_patterns(A_low, A_high, width_to_bits, order: str = "paper") -> list:
    """Distinct per-dim width_to_bits patterns of the A cell boxes (inclusive bounds), in admission order.
    paper = QUILTS Fig. 16 / plot_curves.py: smallest total first, then largest, then the rest (ascending)."""
    w = np.asarray(A_high, dtype=np.int64) - np.asarray(A_low, dtype=np.int64) + 1
    if np.any(w < 1):
        raise RuntimeError("A cell box with high < low")
    pats = sorted({tuple(width_to_bits(int(x)) for x in row) for row in w})
    if order != "paper":
        raise ValueError(f"unknown QUILTS order {order!r}")
    ps = sorted(pats, key=sum)
    return ps if len(ps) < 3 else [ps[0], ps[-1]] + ps[1:-1]


def build_quilts(v, cfg):
    design_multi, width_to_bits = _quilts()
    d, K = v["cells"].shape[1], v["K"]
    if d * K > 64:
        raise RuntimeError("QUILTS keys above 64 bits are not exported")
    pats = quilts_patterns(v["A_low"], v["A_high"], width_to_bits, cfg["order"])
    curve = design_multi([K] * d, pats, name="quilts")
    coords = grid_coords((1 << K,) * d)
    hi, lo = curve.encode(coords)
    if np.any(hi):
        raise RuntimeError("QUILTS key exceeds 64 bits")
    sample = np.random.default_rng(0).choice(len(coords), size=min(1000, len(coords)), replace=False)
    if any(int(lo[i]) != curve.encode_scalar(coords[i]) for i in sample):
        raise RuntimeError("vectorized QUILTS keys differ from encode_scalar")
    return rank_from_codes(lo), {
        "order_msb_to_lsb": list(curve.order), "notation_lsb_first": curve.notation(lsb_first=True),
        "n_patterns": len(pats), "n_admitted": len(curve.admitted), "admitted": [list(p) for p in curve.admitted],
        "n_dropped": len(curve.dropped)}


BUILDERS = {"lmsfc": build_lmsfc, "bitallocz": build_bitallocz, "adacurve": build_adacurve,
            "bmtree": build_bmtree, "quilts": build_quilts}


def method_sources(method: str) -> dict:
    files = [HERE / "ch7_export.py", LAC / "src" / "lacsfc" / "baselines.py", LAC / "src" / "lacsfc" / "tasks.py"]
    if method == "bmtree":
        bm = Path(os.environ.get("BMTREE_DIR", LAC.parents[2] / "Learned-BMTree"))
        files += [HERE / "baselines" / "bmtree_native.py", bm / "mcts.py", bm / "int2binary.py"]
        files += sorted((bm / "bmtree").glob("*.py")) + sorted((bm / "utils").glob("*.py"))
    elif method == "bitallocz":
        files += sorted((BITALLOC_ROOT / "src").glob("*.py"))
    elif method == "adacurve":
        a = ADACURVE_ROOT / "Adacurve"
        files += [ADACURVE_ROOT / "run_repro.py", a / "__init__.py", a / "hd" / "__init__.py",
                  a / "hd" / "adacurve.py", a / "hd" / "evaluate.py", a / "hd" / "sfc.py"]
    elif method == "lmsfc":
        r = LMSFC_ROOT
        files += [r / "ch7_main.cpp", r / "LearnedZorderPaging.h", r / "randomForestnew.py", r / "rffile.init",
                  r / "build.sh", r / "ch7_lmsfc", r / "ch7_lmsfc.buildinfo", r / "PATCHES.md", r.parent / "lmsfc.rar"]
        files += sorted((r / "entities").glob("*.h")) + sorted((r / "pgm").glob("*.hpp"))
        files += sorted(p for p in (r / "orig_2023").rglob("*") if p.is_file())
    elif method == "quilts":
        files += sorted((QUILTS_ROOT / "quilts").glob("*.py")) + [QUILTS_ROOT / "experiments" / "plot_curves.py"]
    else:
        raise ValueError(method)
    return {str(f): sha256(f) for f in files}


def env_info() -> dict:
    out = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
           "numpy": np.__version__}
    for m in ("torch", "sklearn", "bayes_opt"):
        try:
            out[m] = getattr(importlib.import_module(m), "__version__", "present")
        except ImportError:
            out[m] = None
    return out


def _init_worker():
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"


def build_one(job: dict) -> dict:
    """Fit one (method, task) on A, write <task>.code.bin then <task>.meta.json (each via rename)."""
    method, name, task_path, out_dir, cfg = job["method"], job["name"], job["task"], Path(job["out"]), job["cfg"]
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass
    t = load_task(task_path)
    v = fit_view(t, method)
    d, K = v["cells"].shape[1], v["K"]
    N = 1 << (K * d)
    started = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    t0 = time.perf_counter()
    rank, info = BUILDERS[method](v, dict(cfg))
    build_s = time.perf_counter() - t0
    check_perm(rank, N)
    code = out_dir / f"{name}.code.bin"
    tmp = out_dir / f".{name}.code.bin.tmp"
    np.asarray(rank, dtype="<u8").tofile(tmp)
    os.replace(tmp, code)
    meta = {"method": method, "task": name, "task_path": str(task_path), "task_sha256": sha256(task_path),
            "fit_split": "A+B_select" if method in SELECT_ON_B else "A",
            "train_query_sha256": {k: array_sha(t[k]) for k in view_keys(method)},
            "points_sha256": array_sha(t["u"]), "d": int(d), "K": int(K), "N": int(N), "config": cfg,
            "info": info, "build_seconds": build_s, "started": started, "file": code.name, "sha256": sha256(code),
            "env": env_info()}
    tmp = out_dir / f".{name}.meta.json.tmp"
    tmp.write_text(json.dumps(meta, indent=1, sort_keys=True))
    os.replace(tmp, out_dir / f"{name}.meta.json")
    return {"name": name, "build_seconds": build_s}


# ---------------------------------------------------------------- driver

def select_tasks(task_list, task_root) -> dict:
    if task_list is not None:
        return {n: {"task": str(e["task"])} for n, e in sorted(json.loads(Path(task_list).read_text()).items())}
    import ch7_run
    return ch7_run.default_tasks(task_root)


def check_done(out_dir: Path, name: str, task_path, cfg) -> bool:
    meta_p, code = out_dir / f"{name}.meta.json", out_dir / f"{name}.code.bin"
    if not meta_p.exists():
        if code.exists():
            raise RuntimeError(f"{code} exists without its meta file; inspect and remove it by hand")
        return False
    meta = json.loads(meta_p.read_text())
    if meta["task_sha256"] != sha256(task_path):
        raise RuntimeError(f"{name}: built on a different task file")
    if meta["config"] != cfg:
        raise RuntimeError(f"{name}: built with a different config")
    if not code.exists() or sha256(code) != meta["sha256"]:
        raise RuntimeError(f"{name}: rank file missing or differs from its meta file")
    return True


def write_manifest(method: str, out_dir: Path, tasks: dict, cfg: dict) -> Path:
    entries = {}
    for name, e in tasks.items():
        if not check_done(out_dir, name, e["task"], cfg):
            raise RuntimeError(f"{name} is not built")
        meta = json.loads((out_dir / f"{name}.meta.json").read_text())
        entries[name] = {k: meta[k] for k in ("file", "sha256", "task_sha256", "fit_split", "train_query_sha256",
                                              "points_sha256", "build_seconds", "info")}
        entries[name]["meta_sha256"] = sha256(out_dir / f"{name}.meta.json")
    man = {"adapter": ADAPTERS[method], "method": method, "builder": str(HERE / "ch7_export.py"),
           "config": cfg, "sources": method_sources(method), "env": env_info(), "tasks": entries}
    man_p = out_dir / "manifest.json"
    text = json.dumps(man, indent=1, sort_keys=True)
    if man_p.exists():
        old = json.loads(man_p.read_text())
        if {k: v for k, v in old.items() if k != "env"} != {k: v for k, v in man.items() if k != "env"}:
            raise RuntimeError(f"{man_p} exists with different content; not overwritten")
        return man_p
    man_p.write_text(text)
    return man_p


def cmd_build(method, ext_root: Path, task_list, task_root, workers: int, overrides: dict) -> None:
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}")
    bad = set(overrides) - set(DEFAULTS[method])
    if bad:
        raise ValueError(f"unknown config keys for {method}: {sorted(bad)}")
    missing = [m for m in REQUIRES[method] if importlib.util.find_spec(m) is None]
    if missing:
        raise RuntimeError(f"{method} needs {missing} in {sys.executable}")
    if method == "lmsfc" and workers > 1:
        raise ValueError("lmsfc trains on measured query times: use --workers 1 (or 0)")
    cfg = {**DEFAULTS[method], **overrides}
    tasks = select_tasks(task_list, task_root)
    out_dir = Path(ext_root) / method
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [{"method": method, "name": n, "task": e["task"], "out": str(out_dir), "cfg": cfg}
            for n, e in tasks.items() if not check_done(out_dir, n, e["task"], cfg)]
    print(f"{method}: {len(tasks) - len(jobs)} of {len(tasks)} tasks already built, {len(jobs)} to build", flush=True)
    if workers <= 0:
        for j in jobs:
            _init_worker()
            r = build_one(j)
            print(f"{method} {r['name']} ok ({r['build_seconds']:.0f} s)", flush=True)
    elif jobs:
        with mp.get_context("spawn").Pool(workers, initializer=_init_worker, maxtasksperchild=1) as pool:
            for r in pool.imap_unordered(build_one, jobs):
                print(f"{method} {r['name']} ok ({r['build_seconds']:.0f} s)", flush=True)
    print(f"manifest: {write_manifest(method, out_dir, tasks, cfg)}", flush=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ext-root", type=Path, required=True)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build")
    p.add_argument("--method", required=True, choices=METHODS)
    p.add_argument("--task-list", type=Path, help="JSON {name: {task}}; default: the X1 task set")
    p.add_argument("--task-root", type=Path, default=Path.home() / "lacsfc_cache" / "ch7" / "tasks")
    p.add_argument("--workers", type=int, default=1, help="0 = in this process (tests)")
    p.add_argument("--config", type=json.loads, default={}, help="JSON overrides of the method defaults (smoke)")
    a = ap.parse_args(argv)
    cmd_build(a.method, a.ext_root, a.task_list, a.task_root, a.workers, a.config)


if __name__ == "__main__":
    main()
