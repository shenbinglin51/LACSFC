from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..layout import build_artifacts
from ..quantizer import Quantizer
from ..query_engine import QueryEngine
from ..types import QueryWindow
from .masks import validate_masks
from .plugin import LMSFCPlugin


@dataclass(frozen=True)
class ReinforceTrainResult:
    masks: List[int]
    cost: float
    logits: np.ndarray


def _logsumexp(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0:
        return -float("inf")
    m = float(np.max(a))
    return float(m + np.log(np.sum(np.exp(a - m))))


def _sample_gumbel(rng: np.random.Generator, shape: Tuple[int, ...], eps: float = 1e-8) -> np.ndarray:
    u = rng.random(shape, dtype=np.float64)
    return -np.log(-np.log(u + eps) + eps)


def _pl_logprob_and_grad(logits: np.ndarray, perm: np.ndarray) -> Tuple[float, np.ndarray]:
    logits = np.asarray(logits, dtype=np.float64)
    perm = np.asarray(perm, dtype=np.int64)
    n = int(logits.shape[0])
    if perm.shape != (n,):
        raise ValueError("perm shape mismatch")

    grad = np.zeros_like(logits, dtype=np.float64)
    logp = 0.0

    for t in range(n):
        remaining = perm[t:]
        rem_logits = logits[remaining]
        lse = _logsumexp(rem_logits)
        chosen = int(perm[t])
        logp += float(logits[chosen] - lse)

        grad[chosen] += 1.0
        probs = np.exp(rem_logits - lse)
        for idx, p in zip(remaining, probs):
            grad[int(idx)] -= float(p)

    return float(logp), grad


def _perm_to_masks(perm: np.ndarray, widths: Sequence[int]) -> List[int]:
    perm = np.asarray(perm, dtype=np.int64)
    out: List[int] = []
    start = 0
    for w in widths:
        end = start + int(w)
        bits = perm[start:end]
        m = 0
        for b in bits:
            m |= 1 << int(b)
        out.append(int(m))
        start = end
    return out


def train_reinforce(
    *,
    points: np.ndarray,
    queries: Sequence[QueryWindow],
    quantizer: Quantizer,
    b: int,
    objective: str = "metrics",
    semantic: str = "cell",
    timing_repeats: int = 3,
    timing_warmup: bool = True,
    iters: int = 200,
    per: int = 10,
    w_page_accesses: float = 1.0,
    w_segments: float = 1.0,
    lr: float = 0.05,
    tau_start: float = 1.0,
    tau_end: float = 0.2,
    baseline_momentum: float = 0.9,
    force_fallback_intervals: bool = True,
    seed: int = 0,
) -> ReinforceTrainResult:
    if per <= 0:
        raise ValueError("per must be > 0")
    if iters <= 0:
        raise ValueError("iters must be > 0")
    objective = str(objective)
    if objective not in {"metrics", "query_s", "page_accesses", "segments"}:
        raise ValueError("objective must be one of: metrics, query_s, page_accesses, segments")
    if timing_repeats <= 0:
        raise ValueError("timing_repeats must be > 0")

    semantic = str(semantic)
    if semantic != "cell":
        raise ValueError("semantic must be 'cell'")

    rng = np.random.default_rng(int(seed))

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != quantizer.d:
        raise ValueError(f"points must have shape (n,{quantizer.d})")

    d = int(quantizer.d)
    k = int(quantizer.k)

    widths = [int(k)] * int(d)
    total_width = int(sum(widths))
    if total_width <= 0:
        raise ValueError("invalid widths")
    if total_width > 64:
        raise ValueError("total_width must be <= 64")

    logits = np.linspace(0.0, -1.0, total_width, dtype=np.float64)

    n = points.shape[0]
    n_train = max(1, n // int(per))
    perm_points = rng.permutation(n)
    training_points = points[perm_points[:n_train]]

    if objective == "query_s":
        b_train = int(b)
    else:
        b_train = max(1, int(b) // int(per))

    qn = len(queries)
    qn_train = max(1, qn // int(per))

    def eval_cost(masks: Sequence[int], q_batch: Sequence[QueryWindow]) -> float:
        sfc = LMSFCPlugin(name="LMSFC-REINFORCE", grid_size=tuple([quantizer.m] * d), masks=list(masks))
        if force_fallback_intervals:
            setattr(sfc, "intervals", (lambda _q: None))
        artifacts = build_artifacts(points=training_points, sfc=sfc, quantizer=quantizer, b=b_train)
        setter = getattr(sfc, "set_page_start_codes", None)
        if callable(setter):
            setter(artifacts.page_start_codes)
        engine = QueryEngine(artifacts=artifacts, sfc=sfc, quantizer=quantizer, semantic=semantic)

        if objective == "query_s":
            if timing_warmup and q_batch:
                engine.query(q_batch[0])
            best = float("inf")
            for _ in range(int(timing_repeats)):
                t0 = time.perf_counter()
                for q in q_batch:
                    engine.query(q)
                t1 = time.perf_counter()
                best = min(best, float(t1 - t0))
            return float(best)

        page_accesses = 0
        segments = 0
        for q in q_batch:
            r = engine.query(q)
            page_accesses += int(r.metrics.physical_pages)
            segments += int(r.metrics.segments)

        qn2 = max(1, len(q_batch))
        avg_pa = page_accesses / qn2
        avg_seg = segments / qn2

        if objective == "page_accesses":
            return float(avg_pa)
        if objective == "segments":
            return float(avg_seg)
        return float(w_page_accesses * avg_pa + w_segments * avg_seg)

    baseline_reward = 0.0
    best_cost = float("inf")
    best_masks = _perm_to_masks(np.arange(total_width, dtype=np.int64), widths=widths)

    for step in range(int(iters)):
        if qn <= qn_train:
            q_batch = list(queries)
        else:
            idx = rng.choice(qn, size=qn_train, replace=False)
            q_batch = [queries[int(i)] for i in idx]

        tau = float(tau_start + (tau_end - tau_start) * (step / max(1, int(iters) - 1)))
        g = _sample_gumbel(rng=rng, shape=(total_width,))
        z = logits + float(tau) * g
        perm = np.argsort(-z, kind="mergesort")

        masks = _perm_to_masks(perm, widths=widths)
        validate_masks(masks, d)

        cost = eval_cost(masks, q_batch=q_batch)
        reward = -float(cost)

        if step == 0:
            baseline_reward = float(reward)
        advantage = float(reward - baseline_reward)
        baseline_reward = float(baseline_momentum * baseline_reward + (1.0 - baseline_momentum) * reward)

        _, grad = _pl_logprob_and_grad(logits=logits, perm=perm)
        logits = logits + float(lr) * float(advantage) * grad

        if float(cost) < float(best_cost):
            best_cost = float(cost)
            best_masks = list(masks)

    perm_det = np.argsort(-logits, kind="mergesort")
    masks_det = _perm_to_masks(perm_det, widths=widths)
    validate_masks(masks_det, d)

    final_cost = eval_cost(masks_det, q_batch=list(queries) if queries else [])
    if float(final_cost) <= float(best_cost):
        return ReinforceTrainResult(masks=list(masks_det), cost=float(final_cost), logits=np.asarray(logits, dtype=np.float64))

    return ReinforceTrainResult(masks=list(best_masks), cost=float(best_cost), logits=np.asarray(logits, dtype=np.float64))
