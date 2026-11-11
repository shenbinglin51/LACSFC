"""Shared experiment tasks (verification_plan A.1, A.2, A.5.1, Q1, Q2).

One task = normalized points + windows A / B / T, generated in [0, 1]^d and then
quantized to an equal-width 2^K grid. Every method (ours and baselines) reads the
same task file.

- Normalization: per-axis min-max to [0, 1]^d.
- Cells: left-closed / right-open, ``floor(u * 2^K)``, coordinate 1 -> last cell.
- Windows are closed; Q = cells [cell(lo), cell(hi)] per axis, so every point
  inside the window lies in Q (Q2 invariant).
- A, B, T come from independent child streams of one ``SeedSequence``; T may use
  a different workload spec (drift, A.2.4).
- Selectivity (A.6): main experiments use the fraction of points inside the
  window (``sel_mode="points"``, default); window volume is the sensitivity run.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class WorkloadSpec:
    center: str = "data"
    sel_min: float = 1e-4
    sel_max: float = 1e-1
    sel_mode: str = "points"
    aspect_max: float = 20.0
    axis_skew: bool = False
    skew_axis: int = 0
    n_hot: int = 3
    hot_std: float = 0.05


def normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    a, b = x.min(axis=0), x.max(axis=0)
    span = np.where(b > a, b - a, 1.0)
    return np.clip((x - a) / span, 0.0, 1.0)


def quantize(u: np.ndarray, K: int) -> np.ndarray:
    s = 1 << int(K)
    return np.clip(np.floor(np.asarray(u, dtype=np.float64) * s), 0, s - 1).astype(np.int64)


def quantize_windows(ql: np.ndarray, qh: np.ndarray, K: int):
    return quantize(ql, K), quantize(qh, K)


def quantile_edges(u: np.ndarray, K: int) -> np.ndarray:
    """Per-axis inner cell boundaries at the j / 2^K quantiles, shape (d, 2^K - 1)."""
    s = 1 << int(K)
    return np.quantile(np.asarray(u, dtype=np.float64), np.arange(1, s) / s, axis=0).T.copy()


def quantize_edges(v: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Cell coordinate = number of inner boundaries <= value (left-closed cells; monotone per axis)."""
    v = np.atleast_2d(np.asarray(v, dtype=np.float64))
    return np.column_stack([np.searchsorted(e, v[:, k], side="right")
                            for k, e in enumerate(edges)]).astype(np.int64)


def _shapes(m: int, d: int, rng: np.random.Generator, spec: WorkloadSpec) -> np.ndarray:
    """Unit-volume side vectors with max/min ratio r, r log-uniform in [1, aspect_max]."""
    r = np.exp(rng.random(m) * np.log(spec.aspect_max))
    t = rng.random((m, d))
    if d > 1:
        lo_ax = rng.integers(0, d, size=m)
        hi_ax = (lo_ax + rng.integers(1, d, size=m)) % d
        if spec.axis_skew:
            hi_ax = np.full(m, spec.skew_axis % d)
            lo_ax = (hi_ax + rng.integers(1, d, size=m)) % d
        t[np.arange(m), lo_ax] = 0.0
        t[np.arange(m), hi_ax] = 1.0
    s = np.exp(t * np.log(r)[:, None])
    return s / np.exp(np.log(s).mean(axis=1, keepdims=True))


def _sides_for_volume(shape: np.ndarray, vol: np.ndarray) -> np.ndarray:
    """Sides proportional to ``shape`` with product ``vol``; axes capped at 1, the
    remaining axes rescaled to keep the volume (the realized ratio may shrink)."""
    m, d = shape.shape
    sides = shape * (vol ** (1.0 / d))[:, None]
    for _ in range(d):
        cap = sides >= 1.0
        if not cap.any():
            break
        sides = np.where(cap, 1.0, sides)
        free = (~cap).sum(axis=1)
        prod_free = np.prod(np.where(cap, 1.0, sides), axis=1)
        f = np.where(free > 0, (vol / prod_free) ** (1.0 / np.maximum(free, 1)), 1.0)
        sides = np.where(cap, 1.0, sides * f[:, None])
    return np.minimum(sides, 1.0)


def _place(c: np.ndarray, sides: np.ndarray):
    lo = np.clip(c - sides / 2, 0.0, 1.0 - sides)
    return lo, lo + sides


def _centers(u: np.ndarray, m: int, rng: np.random.Generator, spec: WorkloadSpec, hot: np.ndarray):
    d = u.shape[1]
    if spec.center == "data":
        return u[rng.integers(0, len(u), size=m)]
    if spec.center == "uniform":
        return rng.random((m, d))
    if spec.center == "hotspot":
        k = rng.integers(0, len(hot), size=m)
        return np.clip(rng.normal(hot[k], spec.hot_std), 0.0, 1.0)
    raise ValueError(f"unknown center {spec.center!r}")


def _fraction_in(xs: np.ndarray, us: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Fraction of ``us`` (sorted by axis 0, ``xs = us[:, 0]``) inside the closed box."""
    i0 = np.searchsorted(xs, lo[0], side="left")
    i1 = np.searchsorted(xs, hi[0], side="right")
    s = us[i0:i1, 1:]
    return float(np.count_nonzero(np.all((s >= lo[1:]) & (s <= hi[1:]), axis=1))) / len(xs)


def gen_windows(u: np.ndarray, m: int, rng: np.random.Generator, spec: WorkloadSpec,
                hot: np.ndarray, min_hits: int = 400, iters: int = 30):
    """Return closed windows (ql, qh) in [0, 1]^d and the target selectivity per window.

    ``sel_mode="volume"``: target = window volume. ``sel_mode="points"``: target =
    fraction of points inside, matched by bisection on the volume. Each window is
    counted on a uniform subsample (a prefix of one random permutation, size the
    next power of two >= ``min_hits / target``, capped at all points), so about
    ``min_hits`` points are expected inside; with all points the match is exact up
    to bisection and point granularity.
    """
    m = int(m)
    d = u.shape[1]
    target = np.exp(np.log(spec.sel_min) + rng.random(m) * (np.log(spec.sel_max) - np.log(spec.sel_min)))
    shape = _shapes(m, d, rng, spec)
    c = _centers(u, m, rng, spec, hot)
    if spec.sel_mode == "volume":
        lo, hi = _place(c, _sides_for_volume(shape, target))
        return lo, hi, target
    if spec.sel_mode != "points":
        raise ValueError(f"unknown sel_mode {spec.sel_mode!r}")
    n = len(u)
    perm = rng.permutation(n)
    need = np.ceil(min_hits / target)
    size = np.minimum(n, 2 ** np.ceil(np.log2(np.maximum(need, 1)))).astype(np.int64)
    levels = {}
    for sz in np.unique(size):
        us = u[perm[:sz]]
        us = us[np.argsort(us[:, 0], kind="stable")]
        levels[int(sz)] = (np.ascontiguousarray(us[:, 0]), us)
    a = np.full(m, np.log(1e-12))
    b = np.zeros(m)
    for _ in range(iters):
        mid = 0.5 * (a + b)
        lo, hi = _place(c, _sides_for_volume(shape, np.exp(mid)))
        too_big = np.array([_fraction_in(*levels[int(size[i])], lo[i], hi[i]) > target[i]
                            for i in range(m)], dtype=bool)
        b = np.where(too_big, mid, b)
        a = np.where(too_big, a, mid)
    lo, hi = _place(c, _sides_for_volume(shape, np.exp(b)))
    return lo, hi, target


def make_task(x: np.ndarray, K: int, spec_AB: WorkloadSpec, spec_T: WorkloadSpec | None = None,
              mA: int = 1000, mB: int = 500, mT: int = 1000, seed: int = 0,
              binning: str = "equal") -> dict:
    """``binning`` = "equal" (2^K equal-width cells on [0, 1]) or "quantile" (per-axis quantile
    boundaries of the points). Points, hot centres and continuous windows do not depend on
    ``binning``; only the cell mapping does, so the two versions of a task are paired."""
    spec_T = spec_AB if spec_T is None else spec_T
    if binning not in ("equal", "quantile"):
        raise ValueError(f"unknown binning {binning!r}")
    u = normalize(x)
    if binning == "quantile":
        edges = quantile_edges(u, K)
        cell_of = lambda v: quantize_edges(v, edges)  # noqa: E731
    else:
        cell_of = lambda v: quantize(v, K)  # noqa: E731
    ss = np.random.SeedSequence(int(seed))
    s_hot, s_A, s_B, s_T = ss.spawn(4)
    n_hot = max(spec_AB.n_hot, spec_T.n_hot, 1)
    hot = u[np.random.default_rng(s_hot).integers(0, len(u), size=n_hot)]
    task = {"u": u, "cells": cell_of(u), "K": int(K), "hot": hot}
    if binning == "quantile":
        task["edges"] = edges
    for name, sst, spec, m in (("A", s_A, spec_AB, mA), ("B", s_B, spec_AB, mB), ("T", s_T, spec_T, mT)):
        ql, qh, tgt = gen_windows(u, m, np.random.default_rng(sst), spec, hot[:spec.n_hot] if spec.n_hot else hot)
        lo, hi = cell_of(ql), cell_of(qh)
        task.update({f"{name}_ql": ql, f"{name}_qh": qh, f"{name}_target": tgt,
                     f"{name}_low": lo, f"{name}_high": hi})
    task["meta"] = {"seed": int(seed), "K": int(K), "n": int(len(u)), "d": int(u.shape[1]),
                    "spec_AB": asdict(spec_AB), "spec_T": asdict(spec_T),
                    "mA": int(mA), "mB": int(mB), "mT": int(mT), "binning": binning}
    return task


def array_hashes(task: dict) -> dict:
    return {k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
            for k, v in sorted(task.items()) if isinstance(v, np.ndarray)}


def save_task(path, task: dict) -> dict:
    arrays = {k: v for k, v in task.items() if isinstance(v, np.ndarray)}
    arrays["K"] = np.asarray(task["K"])
    meta = dict(task["meta"], hashes=array_hashes(task))
    np.savez_compressed(path, meta=np.asarray(json.dumps(meta)), **arrays)
    return meta


def load_task(path) -> dict:
    z = np.load(path, allow_pickle=False)
    task = {k: z[k] for k in z.files if k not in ("meta", "K")}
    task["K"] = int(z["K"])
    task["meta"] = json.loads(str(z["meta"]))
    got = array_hashes({k: v for k, v in task.items() if isinstance(v, np.ndarray)})
    if got != task["meta"]["hashes"]:
        raise ValueError(f"task file {path} does not match its stored hashes")
    return task
