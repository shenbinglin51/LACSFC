from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .types import Interval, QuantizedQuery, SFCPlugin
from .lmsfc.masks import standard_morton_masks
from .lmsfc.zaddress import _pdep_u64, cal_z_address
from ._paths import lacsfcf_root as _lacsfcf_root


@dataclass
class NumpyEncodeSFC(SFCPlugin):
    name: str
    _encoder: object
    grid_size: Tuple[int, ...]

    def encode_cell(self, cell: np.ndarray) -> int:
        cell = np.asarray(cell, dtype=np.int64)
        if cell.shape != (len(self.grid_size),):
            raise ValueError(f"cell must have shape ({len(self.grid_size)},)")
        coords = cell.reshape(1, -1)
        code = self._encoder.encode(coords)[0]
        return int(code)

    def intervals(self, q: QuantizedQuery) -> Optional[List[Interval]]:
        return None


class _MonotonicIntervals:
    curve_type: str

    def intervals(self, q: QuantizedQuery) -> Optional[List[Interval]]:
        if len(q.low_cell) != 2:
            return None

        from src.sfcs.query_processors.monotonic_curve_processor import MonotonicCurveProcessor
        from src.sfcs.query_processors.query_processor_base import QueryWindow as QW

        grid_size = int(self.grid_size[0])
        if grid_size != int(self.grid_size[1]):
            return None

        x = int(q.low_cell[0])
        y = int(q.low_cell[1])
        w = int(q.high_cell[0] - q.low_cell[0] + 1)
        h = int(q.high_cell[1] - q.low_cell[1] + 1)
        qw = QW(x=x, y=y, width=w, height=h)

        proc = MonotonicCurveProcessor(grid_size=grid_size, curve_type=self.curve_type)
        intervals = proc.process_query(qw)
        out = [Interval(int(it.start), int(it.end)) for it in intervals]
        out.sort()
        return out


def _merge_intervals_sorted(intervals: List[Interval]) -> List[Interval]:
    if not intervals:
        return []
    intervals.sort(key=lambda it: (int(it.start), int(it.end)))
    out: List[Interval] = [intervals[0]]
    for it in intervals[1:]:
        last = out[-1]
        if int(it.start) <= int(last.end) + 1:
            out[-1] = Interval(int(last.start), int(max(int(last.end), int(it.end))))
        else:
            out.append(it)
    return out


@dataclass
class LMSFCQuadtreeExactSFC(SFCPlugin):
    name: str
    grid_size: Tuple[int, int]
    masks: List[int]
    _contiguous_ok: List[bool]

    def __post_init__(self) -> None:
        m = int(self.grid_size[0])
        if m != int(self.grid_size[1]):
            raise ValueError("grid_size must be square")
        if m <= 0 or (m & (m - 1)):
            raise ValueError("grid_size must be power-of-two")

        k = m.bit_length() - 1
        ok: List[bool] = [True] * (k + 1)
        for level in range(0, k + 1):
            if level == 0:
                ok[level] = True
                continue
            v = (1 << int(level)) - 1
            var_mask = int(_pdep_u64(v, int(self.masks[0]))) | int(_pdep_u64(v, int(self.masks[1])))
            target = (1 << (2 * int(level))) - 1
            ok[level] = int(var_mask) == int(target)
        object.__setattr__(self, "_contiguous_ok", ok)

    def encode_cell(self, cell: np.ndarray) -> int:
        cell = np.asarray(cell, dtype=np.uint64)
        if cell.shape != (2,):
            raise ValueError("cell must have shape (2,)")
        return int(cal_z_address(cell, self.masks))

    def intervals(self, q: QuantizedQuery) -> Optional[List[Interval]]:
        if len(q.low_cell) != 2:
            return None

        m = int(self.grid_size[0])
        if m != int(self.grid_size[1]):
            return None

        lx = int(min(int(q.low_cell[0]), int(q.high_cell[0])))
        hx = int(max(int(q.low_cell[0]), int(q.high_cell[0])))
        ly = int(min(int(q.low_cell[1]), int(q.high_cell[1])))
        hy = int(max(int(q.low_cell[1]), int(q.high_cell[1])))

        if lx < 0:
            lx = 0
        if ly < 0:
            ly = 0
        if hx >= m:
            hx = m - 1
        if hy >= m:
            hy = m - 1

        def intersects(x0: int, y0: int, size: int) -> bool:
            x1 = x0 + size - 1
            y1 = y0 + size - 1
            return not (x1 < lx or x0 > hx or y1 < ly or y0 > hy)

        def inside(x0: int, y0: int, size: int) -> bool:
            x1 = x0 + size - 1
            y1 = y0 + size - 1
            return x0 >= lx and x1 <= hx and y0 >= ly and y1 <= hy

        out: List[Interval] = []
        stack: List[Tuple[int, int, int]] = [(0, 0, m)]
        while stack:
            x0, y0, size = stack.pop()
            if not intersects(x0, y0, size):
                continue

            if size == 1:
                code = self.encode_cell(np.asarray([x0, y0], dtype=np.uint64))
                out.append(Interval(int(code), int(code)))
                continue

            if inside(x0, y0, size):
                level = int(size).bit_length() - 1
                if level < len(self._contiguous_ok) and bool(self._contiguous_ok[level]):
                    start = self.encode_cell(np.asarray([x0, y0], dtype=np.uint64))
                    span = 1 << (2 * int(level))
                    out.append(Interval(int(start), int(start) + int(span) - 1))
                    continue

            half = size >> 1
            stack.append((x0 + half, y0 + half, half))
            stack.append((x0, y0 + half, half))
            stack.append((x0 + half, y0, half))
            stack.append((x0, y0, half))

        return _merge_intervals_sorted(out)


class RowMajorSFC(_MonotonicIntervals, NumpyEncodeSFC):
    curve_type = "row_major"


class ColumnMajorSFC(_MonotonicIntervals, NumpyEncodeSFC):
    curve_type = "column_major"


class ZCurveSFC(_MonotonicIntervals, NumpyEncodeSFC):
    curve_type = "z_curve"


@dataclass
class ZCurveQuadtreeExactSFC(SFCPlugin):
    name: str
    grid_size: Tuple[int, int]
    _masks: List[int]

    def encode_cell(self, cell: np.ndarray) -> int:
        cell = np.asarray(cell, dtype=np.uint64)
        if cell.shape != (2,):
            raise ValueError("cell must have shape (2,)")
        return int(cal_z_address(cell, self._masks))

    def intervals(self, q: QuantizedQuery) -> Optional[List[Interval]]:
        if len(q.low_cell) != 2:
            return None

        m = int(self.grid_size[0])
        if m != int(self.grid_size[1]):
            return None

        lx = int(min(int(q.low_cell[0]), int(q.high_cell[0])))
        hx = int(max(int(q.low_cell[0]), int(q.high_cell[0])))
        ly = int(min(int(q.low_cell[1]), int(q.high_cell[1])))
        hy = int(max(int(q.low_cell[1]), int(q.high_cell[1])))

        if lx < 0:
            lx = 0
        if ly < 0:
            ly = 0
        if hx >= m:
            hx = m - 1
        if hy >= m:
            hy = m - 1

        out: List[Interval] = []

        def emit(start: int, end: int) -> None:
            if not out:
                out.append(Interval(int(start), int(end)))
                return
            last = out[-1]
            if int(start) <= int(last.end) + 1:
                out[-1] = Interval(int(last.start), int(max(int(last.end), int(end))))
            else:
                out.append(Interval(int(start), int(end)))

        def intersects(x0: int, y0: int, size: int) -> bool:
            x1 = x0 + size - 1
            y1 = y0 + size - 1
            return not (x1 < lx or x0 > hx or y1 < ly or y0 > hy)

        def inside(x0: int, y0: int, size: int) -> bool:
            x1 = x0 + size - 1
            y1 = y0 + size - 1
            return x0 >= lx and x1 <= hx and y0 >= ly and y1 <= hy

        stack: List[Tuple[int, int, int]] = [(0, 0, m)]
        while stack:
            x0, y0, size = stack.pop()
            if not intersects(x0, y0, size):
                continue
            if inside(x0, y0, size):
                start = self.encode_cell(np.asarray([x0, y0], dtype=np.uint64))
                span = 1 << (2 * int(size).bit_length() - 2) if (size & (size - 1)) else 1 << (2 * int(np.log2(size)))
                # size is always a power of two in this traversal
                span = 1 << (2 * (size.bit_length() - 1))
                emit(start, int(start) + int(span) - 1)
                continue
            if size == 1:
                start = self.encode_cell(np.asarray([x0, y0], dtype=np.uint64))
                emit(start, start)
                continue

            half = size >> 1
            # push in reverse Morton quadrant order so pop() yields Morton order
            stack.append((x0 + half, y0 + half, half))
            stack.append((x0, y0 + half, half))
            stack.append((x0 + half, y0, half))
            stack.append((x0, y0, half))

        return out


class _HilbertIntervals:
    def intervals(self, q: QuantizedQuery) -> Optional[List[Interval]]:
        if len(q.low_cell) != 2:
            return None

        from src.sfcs.query_processors.hilbert_curve_processor import HilbertCurveProcessor
        from src.sfcs.query_processors.query_processor_base import QueryWindow as QW

        grid_size = int(self.grid_size[0])
        if grid_size != int(self.grid_size[1]):
            return None

        x = int(q.low_cell[0])
        y = int(q.low_cell[1])
        w = int(q.high_cell[0] - q.low_cell[0] + 1)
        h = int(q.high_cell[1] - q.low_cell[1] + 1)
        qw = QW(x=x, y=y, width=w, height=h)

        proc = HilbertCurveProcessor(grid_size=grid_size)
        intervals = proc.process_query(qw)
        out = [Interval(int(it.start), int(it.end)) for it in intervals]
        out.sort()
        return out


class HilbertSFC(_HilbertIntervals, NumpyEncodeSFC):
    pass


def make_row_major(k: int = 8) -> RowMajorSFC:
    from src.sfcs.traditional.row_major import RowMajor

    m = 1 << k
    encoder = RowMajor(dimensions=2, grid_size=(m, m))
    return RowMajorSFC(name="RowMajor", _encoder=encoder, grid_size=(m, m))


def make_column_major(k: int = 8) -> ColumnMajorSFC:
    from src.sfcs.traditional.column_major import ColumnMajor

    m = 1 << k
    encoder = ColumnMajor(dimensions=2, grid_size=(m, m))
    return ColumnMajorSFC(name="ColumnMajor", _encoder=encoder, grid_size=(m, m))


def make_z_curve(k: int = 8, d: int = 2) -> NumpyEncodeSFC:
    from src.sfcs.traditional.z_curve import ZCurve

    m = 1 << k
    encoder = ZCurve(dimensions=int(d), grid_size=tuple([m] * int(d)))
    if int(d) == 2:
        return ZCurveSFC(name="ZCurve", _encoder=encoder, grid_size=(m, m))
    return NumpyEncodeSFC(name="ZCurve", _encoder=encoder, grid_size=tuple([m] * int(d)))


def make_z_curve_quadtree_exact(k: int = 8) -> ZCurveQuadtreeExactSFC:
    m = 1 << int(k)
    masks = standard_morton_masks(2)
    return ZCurveQuadtreeExactSFC(name="ZCurve-Quadtree", grid_size=(m, m), _masks=masks)


def make_lmsfc_quadtree_exact(*, k: int = 8, masks: List[int], name: str) -> LMSFCQuadtreeExactSFC:
    m = 1 << int(k)
    return LMSFCQuadtreeExactSFC(name=str(name), grid_size=(m, m), masks=list(masks), _contiguous_ok=[])


def make_hilbert(k: int = 8) -> HilbertSFC:
    from src.sfcs.traditional.hilbert_curve import HilbertCurve

    m = 1 << k
    encoder = HilbertCurve(dimensions=2, grid_size=(m, m))
    return HilbertSFC(name="Hilbert", _encoder=encoder, grid_size=(m, m))


class _HilbertNdEncoder:
    """Batch encoder for N-dimensional Hilbert curve using hilbertcurve package."""

    def __init__(self, k: int, d: int) -> None:
        from hilbertcurve.hilbertcurve import HilbertCurve
        self._hc = HilbertCurve(p=k, n=d)

    def encode(self, cells: np.ndarray) -> np.ndarray:
        pts = np.asarray(cells, dtype=np.int64).tolist()
        dists = np.empty(len(pts), dtype=np.int64)
        for i, pt in enumerate(pts):
            dists[i] = int(self._hc.distance_from_coordinates(pt))
        return dists


def make_hilbert_nd(k: int = 5, d: int = 3) -> NumpyEncodeSFC:
    m = 1 << k
    encoder = _HilbertNdEncoder(k=k, d=d)
    return NumpyEncodeSFC(name="Hilbert", _encoder=encoder, grid_size=tuple([m] * d))


def make_lacsfc_h(
    k: int = 5,
    d: int = 2,
    points: Optional[np.ndarray] = None,
    seed: int = 0,
    name: str = "LACSFC-H",
):
    from .lacsfc.plugin import LACSFCPlugin

    sfc = LACSFCPlugin(k=k, d=d, seed=seed)
    sfc.name = name
    sfc._use_neural = False
    sfc._use_data_hotspots = True
    if points is not None:
        sfc.build(points)
    return sfc


def make_lacsfc_neural(
    d: int = 2,
    k: int = 5,
    points: Optional[np.ndarray] = None,
    query_windows=None,
    quantizer=None,
    max_evals: int = 1500,
    seed: int = 0,
    name: str = "LACSFC-Neural",
    cmaes_mode: str = "block_space",
    func_space_M: int = 200,
    sigma0: float = 0.5,
    popsize: Optional[int] = None,
    b: int = 256,
    es_epochs: int = 30,
    es_n_perturbations: int = 16,
    es_sigma: float = 0.1,
    es_dataset_size: int = 64,
    es_learn_obj_weights: bool = True,
    es_init_obj_weights: tuple = (0.60, 0.15, 0.25),
    cover_weight: float = 0.0,
    runtime_surrogate: str = "none",
    runtime_weight: float = 0.0,
    runtime_normalize: str = "raw",
    runtime_interval_weight: float = 0.0,
    cmaes_objective: str = "page_amp",
    cmaes_learn_obj_weights: bool = True,
    cmaes_init_obj_weights: tuple = (0.60, 0.15, 0.25),
    cmaes_init_strategy: str = "heuristic",
    meta_enable: bool = False,
    meta_model_dir: str = "",
    meta_model_name: str = "lacsfc_pretrained",
    meta_inner_steps: int = 0,
    meta_inner_lr: float = 1e-4,
    meta_use_evaluator: bool = True,
    meta_fallback_cmaes: bool = True,
    meta_hidden: int = 128,
    meta_layers: int = 3,
    meta_evaluator_hidden: int = 128,
    meta_evaluator_layers: int = 3,
):
    from .lacsfc.plugin import LACSFCPlugin

    sfc = LACSFCPlugin(k=k, d=d, seed=seed)
    sfc.name = name
    sfc._use_neural = True
    sfc._use_data_hotspots = True
    sfc._cmaes_max_evals = max_evals
    sfc._cmaes_mode = cmaes_mode
    sfc._cmaes_func_space_M = func_space_M
    sfc._cmaes_sigma0 = sigma0
    sfc._cmaes_cover_weight = float(cover_weight)
    sfc._cmaes_runtime_surrogate = str(runtime_surrogate)
    sfc._cmaes_runtime_weight = float(runtime_weight)
    sfc._cmaes_runtime_normalize = str(runtime_normalize)
    sfc._cmaes_runtime_interval_weight = float(runtime_interval_weight)
    sfc._cmaes_objective = str(cmaes_objective)
    sfc._cmaes_learn_obj_weights = bool(cmaes_learn_obj_weights)
    sfc._cmaes_init_obj_weights = tuple(cmaes_init_obj_weights)
    sfc._cmaes_init_strategy = str(cmaes_init_strategy)
    sfc._meta_enable = bool(meta_enable)
    sfc._meta_model_dir = str(meta_model_dir)
    sfc._meta_model_name = str(meta_model_name)
    sfc._meta_inner_steps = int(meta_inner_steps)
    sfc._meta_inner_lr = float(meta_inner_lr)
    sfc._meta_use_evaluator = bool(meta_use_evaluator)
    sfc._meta_fallback_cmaes = bool(meta_fallback_cmaes)
    sfc._meta_hidden = int(meta_hidden)
    sfc._meta_layers = int(meta_layers)
    sfc._meta_evaluator_hidden = int(meta_evaluator_hidden)
    sfc._meta_evaluator_layers = int(meta_evaluator_layers)
    sfc._b = b
    if popsize is not None:
        sfc._cmaes_popsize = popsize
    sfc._es_epochs = es_epochs
    sfc._es_n_perturbations = es_n_perturbations
    sfc._es_sigma = es_sigma
    sfc._train_dataset_size_v2 = es_dataset_size
    sfc._es_learn_obj_weights = es_learn_obj_weights
    sfc._es_init_obj_weights = es_init_obj_weights
    if points is not None:
        sfc.build(points, query_windows=query_windows, quantizer=quantizer)
    return sfc


def make_lacsfc_neural_v2(
    d: int = 2,
    k: int = 5,
    points: Optional[np.ndarray] = None,
    v2_ckpt_path: str = "",
    seed: int = 0,
    name: str = "LACSFC-Neural-v2",
):
    """Factory: LACSFC-Neural-v2. Loads a pre-trained FG checkpoint
    (produced by `examples/train_lacsfc_v2_offline.py`) and builds the SFC
    via the `_build_neural_v2` branch — no CMA-ES / generator_es training
    at build time. See docs/D4_v2_PR3.md for pipeline details.
    """
    from .lacsfc.plugin import LACSFCPlugin

    if not v2_ckpt_path:
        raise ValueError("make_lacsfc_neural_v2 requires v2_ckpt_path")

    sfc = LACSFCPlugin(k=k, d=d, seed=seed)
    sfc.name = name
    sfc._use_neural = False
    sfc._use_neural_v2 = True
    sfc._v2_ckpt_path = str(v2_ckpt_path)
    sfc._use_data_hotspots = True
    if points is not None:
        sfc.build(points)
    return sfc


def make_lacsfc_from_weights(
    k: int,
    d: int,
    weights: np.ndarray,
    points: Optional[np.ndarray] = None,
    name: str = "LACSFC-FromWeights",
    seed: int = 0,
    use_fast_construct: bool = False,
):
    """Factory: build an LACSFCPlugin directly from a pre-computed edge-weight
    vector (bypassing CMA-ES / Generator training).

    Used by the D-4 meta-learning / online-learning closure: a pretrained
    generator produces `weights`, which we inject straight into the plugin
    so the QueryEngine sees a real learned SFC.
    """
    import sys
    from pathlib import Path
    LACSFC_PATH = _lacsfcf_root()
    if str(LACSFC_PATH) not in sys.path:
        sys.path.insert(0, str(LACSFC_PATH))
    from lacsfcf.data.grid import Grid
    from .lacsfc.plugin import LACSFCPlugin

    sfc = LACSFCPlugin(k=k, d=d, seed=seed, use_fast_construct=use_fast_construct)
    sfc.name = name
    sfc._grid = Grid(d=d, K=k)

    w = np.asarray(weights, dtype=np.float32)
    sfc._assign_curve(w)
    sfc._init_query_processor()
    sfc._is_built = True
    return sfc


def make_lacsfc_metalearned(
    k: int,
    d: int,
    generator,
    target_queries,
    points: Optional[np.ndarray] = None,
    name: str = "LACSFC-Meta",
    seed: int = 0,
):
    """Factory: encode `target_queries` via SceneEncoder, run `generator` to
    obtain per-edge weights, and return an LACSFCPlugin built from those
    weights. Closes the meta-learning loop into QueryEngine.
    """
    import sys
    from pathlib import Path
    LACSFC_PATH = _lacsfcf_root()
    if str(LACSFC_PATH) not in sys.path:
        sys.path.insert(0, str(LACSFC_PATH))
    from lacsfcf.data.grid import Grid
    from lacsfcf.data.query_workload import QueryWorkload
    from lacsfcf.data.scene import SceneEncoder
    import torch

    grid = Grid(d=d, K=k)
    encoder = SceneEncoder(grid)
    wl = QueryWorkload(list(target_queries), grid)
    feats = encoder.encode(wl)

    generator.eval()
    with torch.no_grad():
        t_feats = torch.tensor(feats, dtype=torch.float32)
        weights = generator(t_feats).cpu().numpy()

    return make_lacsfc_from_weights(
        k=k, d=d, weights=weights, points=points, name=name, seed=seed,
    )


def make_piecewise_sfc(
    k: int = 8,
    d: int = 2,
    smallest_split_card: int = 100,
    max_depth: int = 20,
    points: Optional[np.ndarray] = None,
):
    from .piecewise_sfc.plugin import PiecewiseSFCPlugin
    sfc = PiecewiseSFCPlugin(k=k, d=d, smallest_split_card=smallest_split_card, max_depth=max_depth)
    if points is not None:
        sfc.build(points)
    return sfc


def make_piecewise_mcts(
    k: int = 8,
    d: int = 2,
    smallest_split_card: int = 100,
    max_depth: int = 20,
    points: Optional[np.ndarray] = None,
    queries: Optional[List] = None,
):
    """Factory for the query-guided (MCTS-style greedy) PiecewiseSFC.

    Build strategy uses per-node query-straddle minimization — the greedy
    rollout baseline of MCTS (see docs/MCTS_TRAINING_GUIDE.md). Requires
    both ``points`` and ``queries`` for eager build; otherwise returns an
    unbuilt plugin awaiting ``sfc.build(points, queries=...)``.
    """
    from .piecewise_sfc.plugin import PiecewiseSFCPlugin
    sfc = PiecewiseSFCPlugin(
        k=k,
        d=d,
        smallest_split_card=smallest_split_card,
        max_depth=max_depth,
        build_mode="query_guided",
    )
    sfc.name = "PiecewiseSFC-MCTS"
    if points is not None and queries is not None:
        sfc.build(points, queries=queries)
    return sfc


def make_adacurve(
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
):
    from .adacurve.paper_repro import make_adacurve_paper_repro
    
    return make_adacurve_paper_repro(
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


def make_bitallocz_bo(
    d: int = 2,
    k: int = 8,
    b: int = 256,
    total_bits: Optional[int] = None,
    bo_init: int = 5,
    bo_iters: int = 80,
    w_seg: float = 1.0,
    w_pa: float = 1.0,
    w_cand: float = 0.0,
    max_cells: int = 2_000_000,
    seed: int = 0,
    verbose: bool = False,
    name: str = "BitAllocZ-BO",
    points: Optional[np.ndarray] = None,
    quantized_queries=None,
    quantizer=None,
):
    """创建 BitAllocLZ（贝叶斯优化）插件。

    Args:
        d: 维度数（默认 2）
        k: 每维比特数，网格大小 = 2^k（默认 8）
        b: 页面大小（默认 256）
        total_bits: 总比特预算，默认 d*k（标准 Z-curve）
        bo_init: BO 随机初始探索次数（默认 5）
        bo_iters: BO 采集迭代次数（默认 80）
        w_seg: 区间数权重（默认 1.0）
        w_pa: 页访问权重（默认 1.0）
        w_cand: 候选数权重（默认 0.0）
        max_cells: 每查询最大枚举 cell 数（默认 2_000_000）
        seed: 随机种子（默认 0）
        verbose: 是否打印优化进度（默认 False）
        name: 插件名称
        points: 若提供则立即调用 build()
        quantized_queries: 与 points 配合使用
        quantizer: Quantizer 实例（可选）
    """
    from .bitallocz.plugin import BitAllocZPlugin

    plugin = BitAllocZPlugin(
        d=int(d), k=int(k), b=int(b),
        total_bits=total_bits,
        optimizer="bo",
        bo_init=int(bo_init),
        bo_iters=int(bo_iters),
        w_seg=float(w_seg), w_pa=float(w_pa), w_cand=float(w_cand),
        max_cells=int(max_cells),
        seed=int(seed),
        verbose=bool(verbose),
        name=str(name),
    )
    if points is not None and quantized_queries is not None:
        plugin.build(points, quantized_queries, quantizer=quantizer)
    return plugin


def make_bitallocz_gd(
    d: int = 2,
    k: int = 8,
    b: int = 256,
    total_bits: Optional[int] = None,
    gd_iters: int = 100,
    gd_lr: float = 0.1,
    w_seg: float = 1.0,
    w_pa: float = 1.0,
    w_cand: float = 0.0,
    max_cells: int = 2_000_000,
    seed: int = 0,
    verbose: bool = False,
    name: str = "BitAllocZ-GD",
    points: Optional[np.ndarray] = None,
    quantized_queries=None,
    quantizer=None,
):
    """创建 BitAllocLZ（梯度下降）插件。

    Args:
        d: 维度数（默认 2）
        k: 每维比特数，网格大小 = 2^k（默认 8）
        b: 页面大小（默认 256）
        total_bits: 总比特预算，默认 d*k
        gd_iters: 梯度下降迭代次数（默认 100）
        gd_lr: 学习率（默认 0.1）
        w_seg / w_pa / w_cand: 代价权重
        max_cells: 每查询最大枚举 cell 数
        seed: 随机种子
        verbose: 是否打印优化进度
        name: 插件名称
        points / quantized_queries / quantizer: 若提供则立即 build()
    """
    from .bitallocz.plugin import BitAllocZPlugin

    plugin = BitAllocZPlugin(
        d=int(d), k=int(k), b=int(b),
        total_bits=total_bits,
        optimizer="gd",
        gd_iters=int(gd_iters),
        gd_lr=float(gd_lr),
        w_seg=float(w_seg), w_pa=float(w_pa), w_cand=float(w_cand),
        max_cells=int(max_cells),
        seed=int(seed),
        verbose=bool(verbose),
        name=str(name),
    )
    if points is not None and quantized_queries is not None:
        plugin.build(points, quantized_queries, quantizer=quantizer)
    return plugin


def make_bitallocz_uniform(
    d: int = 2,
    k: int = 8,
    b: int = 256,
    total_bits: Optional[int] = None,
    name: str = "BitAllocZ-Uniform",
):
    """创建 BitAllocLZ 均匀分配 baseline（同 total_bits，公平对比用）。

    与 bitallocz_bo / bitallocz_gd 在相同 total_bits 下对比时使用。
    不需要训练数据，立即可用。
    """
    from .bitallocz.curve import BitAllocZCurve
    from .bitallocz.plugin import BitAllocZPlugin
    from .bitallocz.optimizer import LearnResult

    _d = int(d)
    _k = int(k)
    _tb = int(total_bits) if total_bits is not None else _d * _k

    curve = BitAllocZCurve.uniform(d=_d, k=_k, total_bits=_tb)

    plugin = BitAllocZPlugin(
        d=_d, k=_k, b=int(b),
        total_bits=_tb,
        optimizer="bo",
        name=str(name),
    )
    plugin._curve = curve
    plugin._learn_result = LearnResult(
        alloc=curve.alloc,
        cost=0.0,
        n_evals=0,
    )
    return plugin
