"""Exact minimum weighted segment counts via OR-Tools CP-SAT (verification_plan C.2, G5-0 / G5).

Objectives (minimise S = sum_q w_q * segments_q over an order of all N grid cells):
  O1 "grid_segs" : maximal runs of consecutive positions inside Q, empty cells included
                   (= io_stats.grid_segs);
  O2 "seeks"     : maximal runs of Q's occupied cells after deleting the empty cells from the
                   order (= io_stats.seeks when every occupied cell holds >= 1 point).
Path constraints: ac=None (any permutation), "origin" (AC path starting at cell 0) or "free"
(AC path, free start). AC = consecutive cells are grid neighbours (lacsfc.is_ac_path).

Models:
  solve_o1                 edge-weight Hamiltonian path: S = sum_q w_q|Q| - sum_consecutive J(a,b),
                           J(a,b) = sum_q w_q [a in Q][b in Q]; complete digraph (ac=None) or grid.
  solve_o2_unconstrained   O2 depends only on the relative order of occupied cells, so it is
                           solve_o1's edge model on the occupied cells only (subsequence model);
                           empty cells are appended afterwards. Valid ONLY without AC constraint.
  solve_o2_state           path through ALL cells (empty cells kept); per query, a state bit
                           "last occupied cell so far is in Q" is carried across empty cells and a
                           seek starts at an occupied in-Q cell whose state predecessor is 0.
                           Used for AC-O2 (must not use the subsequence model) and, with
                           graph="complete", as a cross-check of the unconstrained O2 model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from ortools.sat.python import cp_model

from .exact import grid_edges
from .grid import num_cells, validate_shape

AC_MODES = (None, "origin", "free")


@dataclass(frozen=True)
class SegResult:
    model: str
    order: np.ndarray
    objective: int
    bound: float
    optimal: bool
    status: str
    seconds: float
    stats: dict = field(default_factory=dict)


def membership(shape, lo, hi) -> np.ndarray:
    """(m, N) bool: cell in box q (inclusive integer corners, C-order flat index)."""
    shape = validate_shape(shape)
    lo, hi = np.atleast_2d(lo), np.atleast_2d(hi)
    c = np.stack(np.unravel_index(np.arange(num_cells(shape)), shape), axis=1)
    return np.all((c[None] >= lo[:, None]) & (c[None] <= hi[:, None]), axis=2)


def _pairs(shape, nodes, graph):
    nodes = list(nodes)
    if graph == "complete":
        return [(a, b) for a in nodes for b in nodes if a != b]
    u, v, _ = grid_edges(shape)
    s = set(nodes)
    out = []
    for a, b in zip(u.tolist(), v.tolist()):
        if a in s and b in s:
            out += [(a, b), (b, a)]
    return out


def _solve(model, time_limit, workers):
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_workers = int(workers)
    t0 = time.perf_counter()
    status = solver.Solve(model)
    sec = time.perf_counter() - t0
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"CP-SAT returned {solver.StatusName(status)}")
    return solver, status, sec


def _walk(succ, D, n):
    order = [succ[D]]
    while len(order) < n:
        order.append(succ[order[-1]])
    return order


def _edge_path(shape, nodes, Jab, graph, start, time_limit, workers):
    """Max sum of Jab over consecutive pairs of a Hamiltonian path on ``nodes``."""
    nodes = list(nodes)
    D = -1
    model = cp_model.CpModel()
    idx = {c: i for i, c in enumerate(nodes)}
    idx[D] = len(nodes)
    lit, arcs, terms = {}, [], []
    for a, b in _pairs(shape, nodes, graph):
        z = model.NewBoolVar(f"x{a}_{b}")
        lit[(a, b)] = z
        arcs.append((idx[a], idx[b], z))
        w = int(Jab(a, b))
        if w:
            terms.append(w * z)
    for c in nodes:
        if start is None or c == start:
            z = model.NewBoolVar(f"s{c}")
            lit[(D, c)] = z
            arcs.append((idx[D], idx[c], z))
        z = model.NewBoolVar(f"t{c}")
        lit[(c, D)] = z
        arcs.append((idx[c], idx[D], z))
    model.AddCircuit(arcs)
    model.Maximize(sum(terms) if terms else 0)
    solver, status, sec = _solve(model, time_limit, workers)
    succ = {a: b for (a, b), z in lit.items() if solver.BooleanValue(z)}
    order = _walk(succ, D, len(nodes))
    return order, solver, status, sec


def _result(name, order, obj, bound, solver, status, sec):
    st = dict(status=solver.StatusName(status), objective=solver.ObjectiveValue(),
              best_bound=solver.BestObjectiveBound(), conflicts=solver.NumConflicts(),
              branches=solver.NumBranches(), wall_time=solver.WallTime())
    return SegResult(name, np.asarray(order, dtype=np.int64), int(round(obj)), float(bound),
                     status == cp_model.OPTIMAL, solver.StatusName(status), sec, st)


def solve_o1(shape, lo, hi, w, ac=None, time_limit=60.0, workers=8) -> SegResult:
    if ac not in AC_MODES:
        raise ValueError(f"ac must be one of {AC_MODES}")
    shape = validate_shape(shape)
    M = membership(shape, lo, hi)
    w = np.asarray(w, dtype=np.int64)
    const = int((w * M.sum(axis=1)).sum())
    J = (M.T.astype(np.int64) * w) @ M.astype(np.int64)
    order, solver, status, sec = _edge_path(shape, range(num_cells(shape)), lambda a, b: J[a, b],
                                            "complete" if ac is None else "grid",
                                            0 if ac == "origin" else None, time_limit, workers)
    return _result(f"o1_edge_{ac or 'unconstrained'}", order, const - solver.ObjectiveValue(),
                   const - solver.BestObjectiveBound(), solver, status, sec)


def solve_o2_unconstrained(shape, lo, hi, w, occupied, time_limit=60.0, workers=8) -> SegResult:
    """Subsequence model (unconstrained only): edge model on occupied cells, empties appended."""
    shape = validate_shape(shape)
    occ = np.asarray(occupied, dtype=bool)
    M = membership(shape, lo, hi) & occ[None]
    w = np.asarray(w, dtype=np.int64)
    const = int((w * M.sum(axis=1)).sum())
    J = (M.T.astype(np.int64) * w) @ M.astype(np.int64)
    sub, solver, status, sec = _edge_path(shape, np.flatnonzero(occ).tolist(), lambda a, b: J[a, b],
                                          "complete", None, time_limit, workers)
    order = list(sub) + np.flatnonzero(~occ).tolist()
    return _result("o2_subsequence_unconstrained", order, const - solver.ObjectiveValue(),
                   const - solver.BestObjectiveBound(), solver, status, sec)


def solve_o2_state(shape, lo, hi, w, occupied, ac="origin", graph=None, time_limit=60.0,
                   workers=8) -> SegResult:
    """Full-path state model: empty cells stay in the path and carry the last-occupied state."""
    if ac not in AC_MODES:
        raise ValueError(f"ac must be one of {AC_MODES}")
    shape = validate_shape(shape)
    N = num_cells(shape)
    graph = graph or ("complete" if ac is None else "grid")
    if ac is not None and graph != "grid":
        raise ValueError("AC modes require graph='grid'")
    occ = np.asarray(occupied, dtype=bool)
    M = membership(shape, lo, hi)
    w = np.asarray(w, dtype=np.int64)
    D = N
    start = 0 if ac == "origin" else None
    model = cp_model.CpModel()
    lit, arcs = {}, []
    for a, b in _pairs(shape, range(N), graph):
        z = model.NewBoolVar(f"x{a}_{b}")
        lit[(a, b)] = z
        arcs.append((a, b, z))
    for c in range(N):
        if start is None or c == start:
            z = model.NewBoolVar(f"s{c}")
            lit[(D, c)] = z
            arcs.append((D, c, z))
        z = model.NewBoolVar(f"t{c}")
        lit[(c, D)] = z
        arcs.append((c, D, z))
    model.AddCircuit(arcs)
    terms = []
    for q in np.flatnonzero(w > 0):
        inq = M[q]
        L = {D: 0}
        for c in range(N):
            L[c] = int(inq[c]) if occ[c] else model.NewBoolVar(f"L{q}_{c}")
        S = {}
        for c in np.flatnonzero(occ & inq).tolist():
            S[c] = model.NewBoolVar(f"S{q}_{c}")
            terms.append(int(w[q]) * S[c])
        for (a, b), z in lit.items():
            if b == D:
                continue
            la = L[a]
            if not occ[b]:
                model.Add(L[b] == la).OnlyEnforceIf(z)
            elif inq[b]:
                if isinstance(la, int):
                    if la == 0:
                        model.AddImplication(z, S[b])
                else:
                    model.AddBoolOr([z.Not(), la, S[b]])
    model.Minimize(sum(terms) if terms else 0)
    solver, status, sec = _solve(model, time_limit, workers)
    succ = {a: b for (a, b), z in lit.items() if solver.BooleanValue(z)}
    order = _walk(succ, D, N)
    return _result(f"o2_state_{ac or 'unconstrained'}_{graph}", order, solver.ObjectiveValue(),
                   solver.BestObjectiveBound(), solver, status, sec)
