from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..types import Interval, QueryWindow, QuantizedQuery, SFCPlugin


@dataclass(frozen=True)
class AdaCurveBlockQueryStats:
    candidates: int
    true_hits: int
    blocks_checked: int
    blocks_scanned: int


class AdaCurvePaperReproPlugin(SFCPlugin):
    def __init__(
        self,
        k: int = 8,
        block_size_points: int = 256,
        compress_dim: int = 8,
        epochs: int = 50,
        learning_rate: float = 1e-3,
        query_batch_size: int = 16,
        train_sample_size: int = 4096,
        fp_weight: float = 1.0,
        tp_weight: float = 1.0,
        recon_weight: float = 0.05,
        smooth_weight: float = 0.01,
        margin: float = 1.0,
        seed: int = 7,
        name: str = "AdaCurve",
    ) -> None:
        self.name = name
        self._k = int(k)
        self._block_size_points = int(block_size_points)
        self._compress_dim = int(compress_dim)
        self._epochs = int(epochs)
        self._learning_rate = float(learning_rate)
        self._query_batch_size = int(query_batch_size)
        self._train_sample_size = int(train_sample_size)
        self._fp_weight = float(fp_weight)
        self._tp_weight = float(tp_weight)
        self._recon_weight = float(recon_weight)
        self._smooth_weight = float(smooth_weight)
        self._margin = float(margin)
        self._seed = int(seed)
        self._warm_anchor_weight = 1.0
        self._warm_loss_weight = 1.0
        self._points: Optional[np.ndarray] = None
        self._cells: Optional[np.ndarray] = None
        self._scores: Optional[np.ndarray] = None
        self._sort_order: Optional[np.ndarray] = None
        self._position_of_point: Optional[np.ndarray] = None
        self._block_start: Optional[np.ndarray] = None
        self._block_end: Optional[np.ndarray] = None
        self._block_min: Optional[np.ndarray] = None
        self._block_max: Optional[np.ndarray] = None
        self._cell_to_positions: Dict[tuple, np.ndarray] = {}
        self._is_built = False
        self._model = None

    def build(self, points: np.ndarray, training_queries: Optional[Sequence[QueryWindow]] = None) -> None:
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2:
            raise ValueError("points must have shape (n,d)")
        self._points = np.clip(points, 0.0, 1.0)
        grid = 1 << self._k
        self._cells = np.floor(self._points * grid).astype(np.int64)
        np.clip(self._cells, 0, grid - 1, out=self._cells)
        queries = list(training_queries or [])
        self._train(queries)
        self._finalize_layout()
        self._is_built = True

    def _make_model(self, d: int):
        import torch
        import torch.nn as nn

        hidden = max(32, 4 * int(self._compress_dim), 4 * d)

        class _Model(nn.Module):
            def __init__(self, in_dim: int, comp_dim: int, hidden_dim: int) -> None:
                super().__init__()
                self.attn = nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, in_dim),
                )
                self.encoder = nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, comp_dim),
                    nn.ReLU(),
                )
                self.decoder = nn.Sequential(
                    nn.Linear(comp_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, in_dim),
                )
                self.ranker = nn.Sequential(
                    nn.Linear(comp_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1),
                )

            def forward(self, x):
                w = torch.softmax(self.attn(x), dim=1)
                z = self.encoder(x * w)
                recon = self.decoder(z)
                score = self.ranker(z).squeeze(-1)
                return score, recon

        return _Model(d, int(self._compress_dim), hidden)

    def _morton_warmstart_target(self, cells: np.ndarray) -> np.ndarray:
        n, d = cells.shape
        total_bits = d * self._k
        if total_bits <= 62:
            codes = np.zeros(n, dtype=np.uint64)
            cu = cells.astype(np.uint64, copy=False)
            for bit in range(self._k):
                shift = self._k - 1 - bit
                for dim in range(d):
                    codes = (codes << np.uint64(1)) | ((cu[:, dim] >> np.uint64(shift)) & np.uint64(1))
            order = np.argsort(codes, kind="stable")
        else:
            bases = (np.maximum(1, cells.max(axis=0) + 1).astype(np.float64) ** np.arange(d - 1, -1, -1)).astype(np.float64)
            codes_f = cells.astype(np.float64) @ bases
            order = np.argsort(codes_f, kind="stable")
        pos = np.empty(n, dtype=np.float32)
        pos[order] = np.arange(n, dtype=np.float32)
        pos = (pos - float(pos.mean())) / (float(pos.std()) + 1e-6)
        return pos

    def _train(self, queries: Sequence[QueryWindow]) -> None:
        import torch
        import torch.nn.functional as F

        assert self._points is not None
        assert self._cells is not None
        torch.manual_seed(self._seed)
        rng = np.random.default_rng(self._seed)
        n, d = self._points.shape
        self._model = self._make_model(d)
        opt = torch.optim.Adam(self._model.parameters(), lr=self._learning_rate)
        x_all = torch.from_numpy(self._points.astype(np.float32))
        warm_target = torch.from_numpy(self._morton_warmstart_target(self._cells))
        sample_size = min(n, max(1, self._train_sample_size))
        query_lows = [np.asarray(q.low, dtype=np.float32) for q in queries]
        query_highs = [np.asarray(q.high, dtype=np.float32) for q in queries]
        if len(queries) == 0:
            queries = []
        for epoch in range(max(0, self._epochs)):
            opt.zero_grad()
            score, recon = self._model(x_all)
            score_n = (score - score.mean()) / (score.std() + 1e-6)
            loss = self._warm_loss_weight * F.mse_loss(score_n, warm_target) + self._recon_weight * F.mse_loss(recon, x_all)
            if n > sample_size:
                sample_idx_np = rng.choice(n, size=sample_size, replace=False)
            else:
                sample_idx_np = np.arange(n, dtype=np.int64)
            sample_idx = torch.from_numpy(sample_idx_np.astype(np.int64))
            sample_points = self._points[sample_idx_np]
            sample_scores = score_n[sample_idx]
            order_np = np.argsort(sample_scores.detach().cpu().numpy(), kind="stable")
            sorted_sample_idx_np = sample_idx_np[order_np]
            sorted_sample_points = sample_points[order_np]
            if len(queries) > 0:
                q_count = min(self._query_batch_size, len(queries))
                q_ids = rng.choice(len(queries), size=q_count, replace=len(queries) < q_count)
                wl_loss = self._workload_loss(score_n, sorted_sample_idx_np, sorted_sample_points, q_ids, query_lows, query_highs)
                loss = loss + wl_loss
            if self._smooth_weight > 0.0 and len(order_np) > 2:
                ordered_scores = sample_scores[torch.from_numpy(order_np.astype(np.int64))]
                ordered_points = torch.from_numpy(sorted_sample_points.astype(np.float32))
                spatial = torch.linalg.norm(ordered_points[1:] - ordered_points[:-1], dim=1)
                score_gap = torch.abs(ordered_scores[1:] - ordered_scores[:-1])
                loss = loss + self._smooth_weight * torch.mean(spatial / (score_gap + 0.1))
            loss.backward()
            opt.step()

    def _workload_loss(self, score_n, sorted_idx: np.ndarray, sorted_points: np.ndarray, q_ids, lows, highs):
        import torch
        import torch.nn.functional as F

        n_sorted = len(sorted_idx)
        bs = max(1, min(self._block_size_points, n_sorted))
        n_blocks = (n_sorted + bs - 1) // bs
        losses = []
        for qi in q_ids.tolist():
            lo = lows[int(qi)]
            hi = highs[int(qi)]
            in_query_sorted = np.all((sorted_points >= lo) & (sorted_points <= hi), axis=1)
            true_pos = np.nonzero(in_query_sorted)[0]
            if true_pos.size >= 2:
                ids = torch.from_numpy(sorted_idx[true_pos].astype(np.int64))
                s = score_n[ids]
                losses.append(self._tp_weight * self._soft_span(s) / float(max(1, true_pos.size)))
            for bi in range(n_blocks):
                s0 = bi * bs
                s1 = min(s0 + bs, n_sorted)
                pts = sorted_points[s0:s1]
                if pts.size == 0:
                    continue
                bmin = pts.min(axis=0)
                bmax = pts.max(axis=0)
                if not bool(np.all((bmax >= lo) & (bmin <= hi))):
                    continue
                mask = in_query_sorted[s0:s1]
                ids = torch.from_numpy(sorted_idx[s0:s1].astype(np.int64))
                block_scores = score_n[ids]
                if not bool(mask.any()):
                    spread = torch.max(block_scores) - torch.min(block_scores)
                    losses.append(self._fp_weight * F.relu(torch.as_tensor(self._margin, dtype=spread.dtype) - spread))
                elif int(mask.sum()) >= 2:
                    hit_ids = torch.from_numpy(sorted_idx[s0:s1][mask].astype(np.int64))
                    losses.append(self._tp_weight * self._soft_span(score_n[hit_ids]) / float(max(1, int(mask.sum()))))
        if len(losses) == 0:
            return score_n.sum() * 0.0
        return torch.stack(losses).mean()

    @staticmethod
    def _soft_span(values):
        import torch

        if values.numel() <= 1:
            return values.sum() * 0.0
        beta = 8.0
        hi = torch.sum(torch.softmax(values * beta, dim=0) * values)
        lo = torch.sum(torch.softmax(-values * beta, dim=0) * values)
        return hi - lo

    def _finalize_layout(self) -> None:
        import torch

        assert self._points is not None
        assert self._cells is not None
        assert self._model is not None
        with torch.no_grad():
            score, _ = self._model(torch.from_numpy(self._points.astype(np.float32)))
        raw_scores = score.detach().cpu().numpy().astype(np.float64)
        raw_scores = (raw_scores - float(raw_scores.mean())) / (float(raw_scores.std()) + 1e-9)
        warm_scores = self._morton_warmstart_target(self._cells).astype(np.float64)
        scores = raw_scores + self._warm_anchor_weight * warm_scores
        noise = np.random.default_rng(self._seed + 101).random(len(scores)) * 1e-9
        order = np.argsort(scores + noise, kind="stable")
        pos = np.empty(len(order), dtype=np.int64)
        pos[order] = np.arange(len(order), dtype=np.int64)
        self._scores = scores
        self._sort_order = order.astype(np.int64)
        self._position_of_point = pos
        n = len(order)
        bs = max(1, self._block_size_points)
        n_blocks = (n + bs - 1) // bs
        self._block_start = (np.arange(n_blocks, dtype=np.int64) * bs).astype(np.int64)
        self._block_end = np.minimum(self._block_start + bs, n).astype(np.int64)
        sorted_points = self._points[self._sort_order]
        self._block_min = np.empty((n_blocks, self._points.shape[1]), dtype=np.float32)
        self._block_max = np.empty((n_blocks, self._points.shape[1]), dtype=np.float32)
        for bi in range(n_blocks):
            chunk = sorted_points[int(self._block_start[bi]):int(self._block_end[bi])]
            self._block_min[bi] = chunk.min(axis=0)
            self._block_max[bi] = chunk.max(axis=0)
        grouped: Dict[tuple, List[int]] = {}
        for i, cell in enumerate(self._cells):
            key = tuple(int(v) for v in cell.tolist())
            grouped.setdefault(key, []).append(int(self._position_of_point[i]))
        self._cell_to_positions = {k: np.asarray(v, dtype=np.int64) for k, v in grouped.items()}

    def encode_points(self, points: np.ndarray, cells: Optional[np.ndarray] = None) -> np.ndarray:
        if self._position_of_point is None:
            raise RuntimeError("AdaCurvePaperReproPlugin is not built")
        points = np.asarray(points)
        if self._points is not None and points.shape == self._points.shape and np.allclose(points, self._points):
            return self._position_of_point.astype(np.int64, copy=True)
        if self._model is None:
            raise RuntimeError("AdaCurvePaperReproPlugin is not built")
        import torch

        with torch.no_grad():
            score, _ = self._model(torch.from_numpy(np.asarray(points, dtype=np.float32)))
        order = np.argsort(score.detach().cpu().numpy(), kind="stable")
        pos = np.empty(len(order), dtype=np.int64)
        pos[order] = np.arange(len(order), dtype=np.int64)
        return pos

    def encode_cell(self, cell: np.ndarray) -> int:
        if not self._is_built:
            raise RuntimeError("AdaCurvePaperReproPlugin is not built")
        key = tuple(int(v) for v in np.asarray(cell, dtype=np.int64).tolist())
        positions = self._cell_to_positions.get(key)
        if positions is None or len(positions) == 0:
            return 0
        return int(np.median(positions))

    def intervals(self, q: QuantizedQuery) -> Optional[List[Interval]]:
        if self._block_min is None or self._block_max is None or self._block_start is None or self._block_end is None:
            return None
        grid = float(1 << self._k)
        q_lower = np.asarray(q.low_cell, dtype=np.float32) / grid
        q_upper = (np.asarray(q.high_cell, dtype=np.float32) + 1.0) / grid
        inter = np.all((self._block_max >= q_lower) & (self._block_min <= q_upper), axis=1)
        hit_blocks = np.nonzero(inter)[0]
        intervals: List[Interval] = []
        for bi in hit_blocks.tolist():
            intervals.append(Interval(start=int(self._block_start[bi]), end=int(self._block_end[bi]) - 1))
        if len(intervals) == 0:
            return intervals
        merged: List[Interval] = []
        cur = intervals[0]
        for nxt in intervals[1:]:
            if int(nxt.start) <= int(cur.end) + 1:
                cur = Interval(start=int(cur.start), end=max(int(cur.end), int(nxt.end)))
            else:
                merged.append(cur)
                cur = nxt
        merged.append(cur)
        return merged

    def query_native(self, q: QueryWindow) -> AdaCurveBlockQueryStats:
        if self._points is None or self._sort_order is None or self._block_min is None or self._block_max is None or self._block_start is None or self._block_end is None:
            raise RuntimeError("AdaCurvePaperReproPlugin is not built")
        lo = np.asarray(q.low, dtype=np.float32)
        hi = np.asarray(q.high, dtype=np.float32)
        inter = np.all((self._block_max >= lo) & (self._block_min <= hi), axis=1)
        hit_blocks = np.nonzero(inter)[0]
        sorted_points = self._points[self._sort_order]
        candidates = 0
        true_hits = 0
        for bi in hit_blocks.tolist():
            s0 = int(self._block_start[bi])
            s1 = int(self._block_end[bi])
            pts = sorted_points[s0:s1]
            candidates += len(pts)
            true_hits += int(np.all((pts >= lo) & (pts <= hi), axis=1).sum())
        return AdaCurveBlockQueryStats(candidates=candidates, true_hits=true_hits, blocks_checked=len(self._block_start), blocks_scanned=len(hit_blocks))


def make_adacurve_paper_repro(
    k: int = 8,
    block_size_points: int = 256,
    compress_dim: int = 8,
    epochs: int = 50,
    learning_rate: float = 1e-3,
    query_batch_size: int = 16,
    train_sample_size: int = 4096,
    fp_weight: float = 1.0,
    tp_weight: float = 1.0,
    recon_weight: float = 0.05,
    smooth_weight: float = 0.01,
    margin: float = 1.0,
    seed: int = 7,
    name: str = "AdaCurve",
) -> AdaCurvePaperReproPlugin:
    return AdaCurvePaperReproPlugin(
        k=k,
        block_size_points=block_size_points,
        compress_dim=compress_dim,
        epochs=epochs,
        learning_rate=learning_rate,
        query_batch_size=query_batch_size,
        train_sample_size=train_sample_size,
        fp_weight=fp_weight,
        tp_weight=tp_weight,
        recon_weight=recon_weight,
        smooth_weight=smooth_weight,
        margin=margin,
        seed=seed,
        name=name,
    )
