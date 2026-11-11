"""Exact maximum-Phi AC orders via OR-Tools CP-SAT (E1 reference; optional dependency).

Modes:
  "path_origin": Hamiltonian path starting at cell 0, free end (comparable to build_curve)
  "path_free"  : Hamiltonian path, free start and end
  "cycle"      : Hamiltonian cycle

Paths use a dummy node N closing the circuit. CP-SAT needs integer weights:
``w = round(J * scale)``; pass integer J (e.g. raw query counts) with scale=1 for exact optima.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from ortools.sat.python import cp_model

from .construct import adjacent_values
from .grid import num_cells, strides, unravel, validate_shape

MODES = ("path_origin", "path_free", "cycle")


@dataclass(frozen=True)
class ExactResult:
    mode: str
    order: np.ndarray
    phi: float
    bound: float
    optimal: bool
    seconds: float


def grid_edges(shape):
    """All grid-adjacent pairs (u, u + e_k) with u < v, and their dimension k."""
    shape = validate_shape(shape)
    N = num_cells(shape)
    c = unravel(np.arange(N, dtype=np.int64), shape)
    stride = strides(shape)
    us, vs, ks = [], [], []
    for k in range(len(shape)):
        u = np.flatnonzero(c[:, k] < shape[k] - 1)
        us.append(u)
        vs.append(u + stride[k])
        ks.append(np.full(u.size, k, dtype=np.int64))
    return np.concatenate(us), np.concatenate(vs), np.concatenate(ks)


def solve_exact(shape, J: np.ndarray, mode: str = "path_origin", scale: float = 1.0,
                time_limit: float = 60.0, workers: int = 8, hint=None) -> ExactResult:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    shape = validate_shape(shape)
    N = num_cells(shape)
    J = np.asarray(J, dtype=np.float64)
    u, v, k = grid_edges(shape)
    w = np.rint(J[k, u] * scale).astype(np.int64)

    model = cp_model.CpModel()
    arcs, terms, lit = [], [], {}
    for e in range(u.size):
        a, b, we = int(u[e]), int(v[e]), int(w[e])
        for x, y in ((a, b), (b, a)):
            z = model.NewBoolVar(f"x{x}_{y}")
            lit[(x, y)] = z
            arcs.append((x, y, z))
            if we:
                terms.append(we * z)
    if mode != "cycle":
        D = N
        for i in range(N):
            if mode == "path_free" or i == 0:
                z = model.NewBoolVar(f"s{i}")
                lit[(D, i)] = z
                arcs.append((D, i, z))
            if mode == "path_free" or i != 0:
                z = model.NewBoolVar(f"t{i}")
                lit[(i, D)] = z
                arcs.append((i, D, z))
    model.AddCircuit(arcs)
    model.Maximize(sum(terms))

    if hint is not None:
        seq = [int(x) for x in hint]
        on = set(zip(seq[:-1], seq[1:]))
        if mode == "cycle":
            on.add((seq[-1], seq[0]))
        else:
            on |= {(N, seq[0]), (seq[-1], N)}
        for key, z in lit.items():
            model.AddHint(z, key in on)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_workers = int(workers)
    t0 = time.perf_counter()
    status = solver.Solve(model)
    sec = time.perf_counter() - t0
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"CP-SAT returned {solver.StatusName(status)}")

    succ = {x: y for (x, y), z in lit.items() if solver.BooleanValue(z)}
    start = 0 if mode == "cycle" else succ[N]
    order = [start]
    while len(order) < N:
        order.append(succ[order[-1]])
    order = np.asarray(order, dtype=np.int64)
    phi = float(adjacent_values(J, order[:-1], order[1:], shape).sum())
    if mode == "cycle":
        phi += float(adjacent_values(J, order[-1:], order[:1], shape).sum())
    return ExactResult(mode, order, phi, solver.BestObjectiveBound() / scale,
                       status == cp_model.OPTIMAL, sec)
