from __future__ import annotations

import bisect
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .index import BoundaryIndex
from .metrics import QueryMetrics, compute_metrics
from .quantizer import Quantizer
from .types import Interval, LayoutArtifacts, QueryWindow, Record, SFCPlugin


VECTORIZE_CANDIDATE_THRESHOLD = 16
BATCH_CANDIDATE_MIN_RECORDS = VECTORIZE_CANDIDATE_THRESHOLD


@dataclass(frozen=True)
class QueryResult:
    records: Tuple[Record, ...]
    intervals: Tuple[Interval, ...]
    metrics: QueryMetrics


@dataclass(frozen=True)
class QueryProfile:
    intervals_s: float
    lookup_s: float
    filter_s: float
    total_s: float
    interval_count: int
    covered_code_len: int
    records_scanned: int
    pass_code: int
    pass_semantic: int
    hit_segments: int
    page_count: int
    touched_page_record_count: int
    candidate_page_count: int
    candidate_page_record_count: int
    empty_candidate_page_count: int
    candidate_page_local_span_sum: int
    candidate_page_local_span_mean: float
    candidate_page_local_span_max: int
    page_interval_visits: int
    page_interval_overlaps: int
    candidate_slice_count: int
    candidate_slice_len_mean: float
    empty_slice_count: int
    nonempty_candidate_slice_count: int
    candidate_slice_len_nonempty_mean: float
    candidate_slice_len_max: int
    vectorized_candidate_slice_count: int
    scalar_candidate_slice_count: int
    vectorized_candidate_record_count: int
    scalar_candidate_record_count: int
    batched_candidate_slice_count: int
    batched_candidate_record_count: int
    candidate_record_count: int
    semantic_check_count: int
    accepted_record_count: int
    filter_bisect_s: float
    filter_candidate_scan_s: float
    filter_candidate_extract_s: float
    filter_candidate_array_s: float
    filter_semantic_check_s: float
    filter_materialize_s: float


@dataclass(frozen=True)
class PageArrayCacheEntry:
    cells: np.ndarray
    xs: np.ndarray
    has_code: np.ndarray


@dataclass(frozen=True)
class QueryProfiledResult:
    result: QueryResult
    profile: QueryProfile


def _point_in_window(x: np.ndarray, q: QueryWindow) -> bool:
    return bool(np.all(x >= q.low) and np.all(x <= q.high))


def _cell_in_range(cell: np.ndarray, low_cell: np.ndarray, high_cell: np.ndarray) -> bool:
    return bool(np.all(cell >= low_cell) and np.all(cell <= high_cell))



def _merge_sorted_codes(codes: np.ndarray) -> List[Interval]:
    if codes.size == 0:
        return []
    start = int(codes[0])
    end = int(codes[0])
    out: List[Interval] = []
    for v in codes[1:]:
        iv = int(v)
        if iv <= end + 1:
            end = iv
        else:
            out.append(Interval(start=start, end=end))
            start = iv
            end = iv
    out.append(Interval(start=start, end=end))
    return out


def _enumerate_cells(low_cell: np.ndarray, high_cell: np.ndarray, max_cells: int) -> np.ndarray:
    d = int(low_cell.shape[0])
    sizes = (high_cell - low_cell + 1).astype(np.int64)
    total = int(np.prod(sizes))
    if total > int(max_cells):
        raise ValueError(f"enumeration too large: {total} cells")

    grids = [np.arange(int(low_cell[i]), int(high_cell[i]) + 1, dtype=np.int64) for i in range(d)]
    mesh = np.meshgrid(*grids, indexing="ij")
    stacked = np.stack([m.reshape(-1) for m in mesh], axis=1)
    return stacked


def _intervals_from_plugin(
    sfc: SFCPlugin,
    low_cell: np.ndarray,
    high_cell: np.ndarray,
    max_cells: int,
) -> List[Interval]:
    q_like = type("_Q", (), {"low_cell": low_cell, "high_cell": high_cell})
    intervals_fn = getattr(sfc, "intervals", None)
    if callable(intervals_fn):
        out = intervals_fn(q_like)
        if out is not None:
            out2 = [Interval(int(it.start), int(it.end)) for it in out]
            out2.sort()
            return out2

    cells = _enumerate_cells(low_cell=low_cell, high_cell=high_cell, max_cells=max_cells)
    encoder = getattr(sfc, "_encoder", None)
    if encoder is not None and hasattr(encoder, "encode"):
        codes = np.asarray(encoder.encode(cells), dtype=np.int64)
    else:
        codes = np.empty(cells.shape[0], dtype=np.int64)
        for i in range(cells.shape[0]):
            codes[i] = int(sfc.encode_cell(cells[i]))

    codes.sort(kind="mergesort")
    return _merge_sorted_codes(codes)


def _code_in_intervals(code: int, intervals: Sequence[Interval]) -> bool:
    if not intervals:
        return False

    lo = 0
    hi = len(intervals) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        itv = intervals[mid]
        if code < itv.start:
            hi = mid - 1
        elif code > itv.end:
            lo = mid + 1
        else:
            return True
    return False


def _page_code_range(records: Sequence[Record]) -> Optional[Tuple[int, int]]:
    if not records:
        return None
    lo: Optional[int] = None
    hi: Optional[int] = None
    first = records[0].code
    last = records[-1].code
    if first is not None and last is not None:
        return int(first), int(last)
    for r in records:
        if r.code is None:
            continue
        v = int(r.code)
        if lo is None or v < lo:
            lo = v
        if hi is None or v > hi:
            hi = v
    if lo is None or hi is None:
        return None
    return int(lo), int(hi)


def _overlapping_intervals_for_page(
    intervals: Sequence[Interval],
    interval_ends: Sequence[int],
    page_start: int,
    page_end: int,
) -> Iterable[Interval]:
    i = bisect.bisect_left(interval_ends, int(page_start))
    n = len(intervals)
    while i < n:
        interval = intervals[i]
        if int(interval.start) > int(page_end):
            break
        yield interval
        i += 1


def _records_cells_points(records: Sequence[Record], quantizer: Quantizer) -> Tuple[np.ndarray, np.ndarray]:
    if not records:
        return np.empty((0, quantizer.d), dtype=np.int64), np.empty((0, quantizer.d), dtype=np.float64)
    xs = np.asarray([r.x for r in records], dtype=np.float64)
    if all(r.cell is not None for r in records):
        cells = np.asarray([r.cell for r in records], dtype=np.int64)
    else:
        cells = np.empty((len(records), quantizer.d), dtype=np.int64)
        missing: List[int] = []
        missing_xs: List[np.ndarray] = []
        for i, r in enumerate(records):
            if r.cell is None:
                missing.append(i)
                missing_xs.append(np.asarray(r.x, dtype=np.float64))
            else:
                cells[i] = np.asarray(r.cell, dtype=np.int64)
        if missing:
            quantized = quantizer.quantize_points(np.asarray(missing_xs, dtype=np.float64))
            for j, i in enumerate(missing):
                cells[i] = quantized[j]
    return cells, xs


def _page_records_cells_points(records: Sequence[Record], quantizer: Quantizer) -> PageArrayCacheEntry:
    cells, xs = _records_cells_points(records, quantizer)
    has_code = np.asarray([r.code is not None for r in records], dtype=bool)
    return PageArrayCacheEntry(cells=cells, xs=xs, has_code=has_code)


def _page_slices_records_cells_points(
    records: Sequence[Record],
    entry: PageArrayCacheEntry,
    slices: Sequence[Tuple[int, int]],
) -> Tuple[Tuple[Record, ...], np.ndarray, np.ndarray]:
    if not slices:
        return tuple(), entry.cells[:0], entry.xs[:0]
    if len(slices) == 1:
        start, end = slices[0]
        candidate_records = records[start:end]
        cells = entry.cells[start:end]
        xs = entry.xs[start:end]
        has_code = entry.has_code[start:end]
        if bool(np.all(has_code)):
            return tuple(candidate_records), cells, xs
        keep = np.flatnonzero(has_code)
        return tuple(candidate_records[int(i)] for i in keep), cells[keep], xs[keep]

    candidate_records_list: List[Record] = []
    cells_parts: List[np.ndarray] = []
    xs_parts: List[np.ndarray] = []
    has_code_parts: List[np.ndarray] = []
    for start, end in slices:
        candidate_records_list.extend(records[start:end])
        cells_parts.append(entry.cells[start:end])
        xs_parts.append(entry.xs[start:end])
        has_code_parts.append(entry.has_code[start:end])
    cells = np.concatenate(cells_parts, axis=0)
    xs = np.concatenate(xs_parts, axis=0)
    has_code = np.concatenate(has_code_parts, axis=0)
    if bool(np.all(has_code)):
        return tuple(candidate_records_list), cells, xs
    keep = np.flatnonzero(has_code)
    return tuple(candidate_records_list[int(i)] for i in keep), cells[keep], xs[keep]


def _semantic_accept_mask(
    cells: np.ndarray,
    xs: np.ndarray,
    low_cell: np.ndarray,
    high_cell: np.ndarray,
    q: QueryWindow,
) -> np.ndarray:
    if cells.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    cell_mask = np.all((cells >= low_cell) & (cells <= high_cell), axis=1)
    if not bool(np.any(cell_mask)):
        return cell_mask
    point_mask = np.zeros(cells.shape[0], dtype=bool)
    point_mask[cell_mask] = np.all((xs[cell_mask] >= q.low) & (xs[cell_mask] <= q.high), axis=1)
    return cell_mask & point_mask


class QueryEngine:
    def __init__(
        self,
        artifacts: LayoutArtifacts,
        sfc: SFCPlugin,
        quantizer: Quantizer,
        max_cells_fallback: int = 2_000_000,
        semantic: str = "cell",
    ) -> None:
        self._artifacts = artifacts
        self._sfc = sfc
        self._quantizer = quantizer
        self._index = BoundaryIndex(page_start_codes=artifacts.page_start_codes)
        self._max_cells_fallback = int(max_cells_fallback)
        semantic = str(semantic)
        if semantic != "cell":
            raise ValueError("semantic must be 'cell'")
        self._semantic = semantic
        self._page_array_cache: Dict[int, PageArrayCacheEntry] = {}

    @property
    def artifacts(self) -> LayoutArtifacts:
        return self._artifacts

    def _page_array_entry(self, pid: int) -> PageArrayCacheEntry:
        entry = self._page_array_cache.get(int(pid))
        if entry is None:
            entry = _page_records_cells_points(self._artifacts.pages[int(pid)].records, self._quantizer)
            self._page_array_cache[int(pid)] = entry
        return entry

    def query(self, q: QueryWindow) -> QueryResult:
        t_start = time.perf_counter()
        
        q_low = np.asarray(q.low, dtype=np.float64)
        q_high = np.asarray(q.high, dtype=np.float64)
        q2 = QueryWindow(low=np.minimum(q_low, q_high), high=np.maximum(q_low, q_high))

        qq = self._quantizer.quantize_window(q2)
        low_cell = np.asarray(qq.low_cell, dtype=np.int64)
        high_cell = np.asarray(qq.high_cell, dtype=np.int64)
        intervals = _intervals_from_plugin(
            sfc=self._sfc,
            low_cell=qq.low_cell,
            high_cell=qq.high_cell,
            max_cells=self._max_cells_fallback,
        )

        pages = self._index.lookup_intervals(intervals)
        interval_ends = [int(it.end) for it in intervals]
        out: List[Record] = []
        codes: List[int] = []  # 收集结果记录的 SFC codes
        
        for pid in pages:
            if pid < 0 or pid >= len(self._artifacts.pages):
                continue
            page = self._artifacts.pages[pid]
            
            # Check if page has Sort Dimension
            if page.sort_dim >= 0:
                # Sort Dimension 二分查找
                dim = page.sort_dim
                start_idx = bisect.bisect_left(page.records, low_cell[dim],
                                               key=lambda r: r.cell[dim] if r.cell is not None else -1)
                end_idx = bisect.bisect_right(page.records, high_cell[dim],
                                              key=lambda r: r.cell[dim] if r.cell is not None else -1)
                
                candidate_records = page.records[start_idx:end_idx]
                if len(candidate_records) >= VECTORIZE_CANDIDATE_THRESHOLD:
                    entry = self._page_array_entry(pid)
                    cells = entry.cells[start_idx:end_idx]
                    xs = entry.xs[start_idx:end_idx]
                    accept_mask = _semantic_accept_mask(cells, xs, low_cell, high_cell, q2)
                    for r, accepted in zip(candidate_records, accept_mask):
                        if not bool(accepted):
                            continue
                        out.append(r)
                        if r.code is not None:
                            codes.append(int(r.code))
                else:
                    for r in candidate_records:
                        if r.cell is None:
                            rc = self._quantizer.quantize_points(np.asarray([r.x], dtype=np.float64))[0]
                        else:
                            rc = np.asarray(r.cell, dtype=np.int64)
                        if not _cell_in_range(rc, low_cell=low_cell, high_cell=high_cell):
                            continue
                        if not _point_in_window(r.x, q2):
                            continue
                        out.append(r)
                        if r.code is not None:
                            codes.append(int(r.code))
            else:
                # Code 二分查找：对每个 code 区间分别处理
                page_range = _page_code_range(page.records)
                if page_range is None:
                    continue
                page_start, page_end = page_range
                candidate_slices: List[Tuple[int, int]] = []
                candidate_total = 0
                for interval in _overlapping_intervals_for_page(intervals, interval_ends, page_start, page_end):
                    start_idx = bisect.bisect_left(page.records, interval.start, 
                                                   key=lambda r: r.code if r.code is not None else -1)
                    end_idx = bisect.bisect_right(page.records, interval.end, 
                                                  key=lambda r: r.code if r.code is not None else -1)
                    if end_idx > start_idx:
                        candidate_slices.append((int(start_idx), int(end_idx)))
                        candidate_total += int(end_idx - start_idx)
                if candidate_total >= BATCH_CANDIDATE_MIN_RECORDS:
                    entry = self._page_array_entry(pid)
                    candidate_records, cells, xs = _page_slices_records_cells_points(page.records, entry, candidate_slices)
                    accept_mask = _semantic_accept_mask(cells, xs, low_cell, high_cell, q2)
                    for r, accepted in zip(candidate_records, accept_mask):
                        if not bool(accepted):
                            continue
                        out.append(r)
                        codes.append(int(r.code))
                else:
                    for start_idx, end_idx in candidate_slices:
                        for r in page.records[start_idx:end_idx]:
                            if r.code is None:
                                continue
                            if r.cell is None:
                                rc = self._quantizer.quantize_points(np.asarray([r.x], dtype=np.float64))[0]
                            else:
                                rc = np.asarray(r.cell, dtype=np.int64)
                            if not _cell_in_range(rc, low_cell=low_cell, high_cell=high_cell):
                                continue
                            if not _point_in_window(r.x, q2):
                                continue
                            out.append(r)
                            codes.append(int(r.code))

        t_end = time.perf_counter()
        query_time_ms = (t_end - t_start) * 1000.0  # 转换为毫秒
        
        metrics = compute_metrics(
            result_count=len(out),
            page_accesses=len(pages),
            b=self._artifacts.b,
            intervals=intervals,
            codes=codes,
            query_time_ms=query_time_ms,
        )
        return QueryResult(records=tuple(out), intervals=tuple(intervals), metrics=metrics)

    def query_profiled(self, q: QueryWindow) -> QueryProfiledResult:
        t0 = time.perf_counter()

        q_low = np.asarray(q.low, dtype=np.float64)
        q_high = np.asarray(q.high, dtype=np.float64)
        q2 = QueryWindow(low=np.minimum(q_low, q_high), high=np.maximum(q_low, q_high))

        qq = self._quantizer.quantize_window(q2)
        low_cell = np.asarray(qq.low_cell, dtype=np.int64)
        high_cell = np.asarray(qq.high_cell, dtype=np.int64)
        intervals = _intervals_from_plugin(
            sfc=self._sfc,
            low_cell=qq.low_cell,
            high_cell=qq.high_cell,
            max_cells=self._max_cells_fallback,
        )
        interval_count = int(len(intervals))
        covered_code_len = 0
        for it in intervals:
            covered_code_len += int(it.end) - int(it.start) + 1
        t1 = time.perf_counter()

        pages = self._index.lookup_intervals(intervals)
        interval_ends = [int(it.end) for it in intervals]
        t2 = time.perf_counter()

        out: List[Record] = []
        codes: List[int] = []  # 收集结果记录的 SFC codes
        records_scanned = 0
        pass_code = 0
        pass_semantic = 0
        hit_segments = 0
        page_count = 0
        touched_page_record_count = 0
        candidate_page_count = 0
        candidate_page_record_count = 0
        empty_candidate_page_count = 0
        candidate_page_local_span_sum = 0
        candidate_page_local_span_max = 0
        page_interval_visits = 0
        page_interval_overlaps = 0
        candidate_slice_count = 0
        empty_slice_count = 0
        nonempty_candidate_slice_count = 0
        candidate_slice_len_sum_nonempty = 0
        candidate_slice_len_max = 0
        vectorized_candidate_slice_count = 0
        scalar_candidate_slice_count = 0
        vectorized_candidate_record_count = 0
        scalar_candidate_record_count = 0
        batched_candidate_slice_count = 0
        batched_candidate_record_count = 0
        candidate_record_count = 0
        semantic_check_count = 0
        accepted_record_count = 0
        filter_bisect_s = 0.0
        filter_candidate_scan_s = 0.0
        filter_candidate_extract_s = 0.0
        filter_candidate_array_s = 0.0
        filter_semantic_check_s = 0.0
        filter_materialize_s = 0.0
        last_hit_code: Optional[int] = None
        for pid in pages:
            if pid < 0 or pid >= len(self._artifacts.pages):
                continue
            page_count += 1
            page = self._artifacts.pages[pid]
            page_record_count = int(len(page.records))
            touched_page_record_count += page_record_count
            
            # Check if page has Sort Dimension
            if page.sort_dim >= 0:
                # Sort Dimension 二分查找
                dim = page.sort_dim
                page_interval_visits += 1
                candidate_slice_count += 1
                t_bisect0 = time.perf_counter()
                start_idx = bisect.bisect_left(page.records, low_cell[dim],
                                               key=lambda r: r.cell[dim] if r.cell is not None else -1)
                end_idx = bisect.bisect_right(page.records, high_cell[dim],
                                              key=lambda r: r.cell[dim] if r.cell is not None else -1)
                filter_bisect_s += time.perf_counter() - t_bisect0
                slice_len = int(end_idx - start_idx)
                if start_idx == end_idx:
                    empty_slice_count += 1
                else:
                    nonempty_candidate_slice_count += 1
                    candidate_slice_len_sum_nonempty += slice_len
                    candidate_slice_len_max = max(candidate_slice_len_max, slice_len)
                    page_interval_overlaps += 1
                
                candidate_records = page.records[start_idx:end_idx]
                if slice_len > 0:
                    candidate_page_count += 1
                    candidate_page_record_count += page_record_count
                    candidate_page_local_span_sum += slice_len
                    candidate_page_local_span_max = max(candidate_page_local_span_max, slice_len)
                else:
                    empty_candidate_page_count += 1
                if len(candidate_records) >= VECTORIZE_CANDIDATE_THRESHOLD:
                    vectorized_candidate_slice_count += 1
                    t_extract0 = time.perf_counter()
                    n_candidates = len(candidate_records)
                    filter_candidate_extract_s += time.perf_counter() - t_extract0
                    t_array0 = time.perf_counter()
                    entry = self._page_array_entry(pid)
                    cells = entry.cells[start_idx:end_idx]
                    xs = entry.xs[start_idx:end_idx]
                    filter_candidate_array_s += time.perf_counter() - t_array0
                    n_candidates = len(candidate_records)
                    records_scanned += n_candidates
                    candidate_record_count += n_candidates
                    vectorized_candidate_record_count += n_candidates
                    pass_code += n_candidates
                    t_sem0 = time.perf_counter()
                    accept_mask = _semantic_accept_mask(cells, xs, low_cell, high_cell, q2)
                    semantic_check_count += n_candidates
                    filter_semantic_check_s += time.perf_counter() - t_sem0
                    t_mat0 = time.perf_counter()
                    for r, accepted in zip(candidate_records, accept_mask):
                        if not bool(accepted):
                            continue
                        pass_semantic += 1
                        accepted_record_count += 1
                        cur_code = int(r.code) if r.code is not None else 0
                        if last_hit_code is None:
                            hit_segments = 1
                            last_hit_code = cur_code
                        elif cur_code == last_hit_code:
                            pass
                        elif cur_code == last_hit_code + 1:
                            last_hit_code = cur_code
                        else:
                            hit_segments += 1
                            last_hit_code = cur_code
                        out.append(r)
                        if r.code is not None:
                            codes.append(int(r.code))
                    filter_materialize_s += time.perf_counter() - t_mat0
                else:
                    if len(candidate_records) > 0:
                        scalar_candidate_slice_count += 1
                    for r in candidate_records:
                        t_scan0 = time.perf_counter()
                        records_scanned += 1
                        candidate_record_count += 1
                        scalar_candidate_record_count += 1
                        pass_code += 1
                        if r.cell is None:
                            rc = self._quantizer.quantize_points(np.asarray([r.x], dtype=np.float64))[0]
                        else:
                            rc = np.asarray(r.cell, dtype=np.int64)
                        filter_candidate_scan_s += time.perf_counter() - t_scan0
                        t_sem0 = time.perf_counter()
                        semantic_check_count += 1
                        cell_ok = _cell_in_range(rc, low_cell=low_cell, high_cell=high_cell)
                        point_ok = False
                        if cell_ok:
                            point_ok = _point_in_window(r.x, q2)
                        filter_semantic_check_s += time.perf_counter() - t_sem0
                        if not cell_ok:
                            continue
                        if not point_ok:
                            continue
                        pass_semantic += 1
                        accepted_record_count += 1
                        t_mat0 = time.perf_counter()
                        cur_code = int(r.code) if r.code is not None else 0
                        if last_hit_code is None:
                            hit_segments = 1
                            last_hit_code = cur_code
                        elif cur_code == last_hit_code:
                            pass
                        elif cur_code == last_hit_code + 1:
                            last_hit_code = cur_code
                        else:
                            hit_segments += 1
                            last_hit_code = cur_code
                        out.append(r)
                        if r.code is not None:
                            codes.append(int(r.code))
                        filter_materialize_s += time.perf_counter() - t_mat0
            else:
                # Code 二分查找：对每个 code 区间分别处理
                page_range = _page_code_range(page.records)
                if page_range is None:
                    continue
                page_start, page_end = page_range
                candidate_slices: List[Tuple[int, int]] = []
                candidate_total = 0
                for interval in _overlapping_intervals_for_page(intervals, interval_ends, page_start, page_end):
                    page_interval_visits += 1
                    candidate_slice_count += 1
                    t_bisect0 = time.perf_counter()
                    start_idx = bisect.bisect_left(page.records, interval.start,
                                                   key=lambda r: r.code if r.code is not None else -1)
                    end_idx = bisect.bisect_right(page.records, interval.end,
                                                  key=lambda r: r.code if r.code is not None else -1)
                    filter_bisect_s += time.perf_counter() - t_bisect0
                    slice_len = int(end_idx - start_idx)
                    if start_idx == end_idx:
                        empty_slice_count += 1
                    else:
                        nonempty_candidate_slice_count += 1
                        candidate_slice_len_sum_nonempty += slice_len
                        candidate_slice_len_max = max(candidate_slice_len_max, slice_len)
                        page_interval_overlaps += 1
                        candidate_slices.append((int(start_idx), int(end_idx)))
                        candidate_total += slice_len

                if candidate_total >= BATCH_CANDIDATE_MIN_RECORDS:
                    candidate_page_count += 1
                    candidate_page_record_count += page_record_count
                    page_local_span = max(end for _, end in candidate_slices) - min(start for start, _ in candidate_slices)
                    candidate_page_local_span_sum += int(page_local_span)
                    candidate_page_local_span_max = max(candidate_page_local_span_max, int(page_local_span))
                    t_extract0 = time.perf_counter()
                    filter_candidate_extract_s += time.perf_counter() - t_extract0
                    t_array0 = time.perf_counter()
                    entry = self._page_array_entry(pid)
                    candidate_records, cells, xs = _page_slices_records_cells_points(page.records, entry, candidate_slices)
                    filter_candidate_array_s += time.perf_counter() - t_array0
                    n_candidates = len(candidate_records)
                    vectorized_candidate_slice_count += len(candidate_slices)
                    vectorized_candidate_record_count += n_candidates
                    if len(candidate_slices) > 1:
                        batched_candidate_slice_count += len(candidate_slices)
                        batched_candidate_record_count += n_candidates
                    records_scanned += n_candidates
                    candidate_record_count += n_candidates
                    pass_code += n_candidates
                    t_sem0 = time.perf_counter()
                    accept_mask = _semantic_accept_mask(cells, xs, low_cell, high_cell, q2)
                    semantic_check_count += n_candidates
                    filter_semantic_check_s += time.perf_counter() - t_sem0
                    t_mat0 = time.perf_counter()
                    for r, accepted in zip(candidate_records, accept_mask):
                        if not bool(accepted):
                            continue
                        pass_semantic += 1
                        accepted_record_count += 1
                        cur_code = int(r.code)
                        if last_hit_code is None:
                            hit_segments = 1
                            last_hit_code = cur_code
                        elif cur_code == last_hit_code:
                            pass
                        elif cur_code == last_hit_code + 1:
                            last_hit_code = cur_code
                        else:
                            hit_segments += 1
                            last_hit_code = cur_code
                        out.append(r)
                        codes.append(int(r.code))
                    filter_materialize_s += time.perf_counter() - t_mat0
                else:
                    if candidate_total > 0:
                        candidate_page_count += 1
                        candidate_page_record_count += page_record_count
                        page_local_span = max(end for _, end in candidate_slices) - min(start for start, _ in candidate_slices)
                        candidate_page_local_span_sum += int(page_local_span)
                        candidate_page_local_span_max = max(candidate_page_local_span_max, int(page_local_span))
                    else:
                        empty_candidate_page_count += 1
                    for start_idx, end_idx in candidate_slices:
                        if end_idx > start_idx:
                            scalar_candidate_slice_count += 1
                        for r in page.records[start_idx:end_idx]:
                            t_scan0 = time.perf_counter()
                            if r.code is None:
                                filter_candidate_scan_s += time.perf_counter() - t_scan0
                                continue
                            records_scanned += 1
                            candidate_record_count += 1
                            scalar_candidate_record_count += 1
                            pass_code += 1
                            if r.cell is None:
                                rc = self._quantizer.quantize_points(np.asarray([r.x], dtype=np.float64))[0]
                            else:
                                rc = np.asarray(r.cell, dtype=np.int64)
                            filter_candidate_scan_s += time.perf_counter() - t_scan0
                            t_sem0 = time.perf_counter()
                            semantic_check_count += 1
                            cell_ok = _cell_in_range(rc, low_cell=low_cell, high_cell=high_cell)
                            point_ok = False
                            if cell_ok:
                                point_ok = _point_in_window(r.x, q2)
                            filter_semantic_check_s += time.perf_counter() - t_sem0
                            if not cell_ok:
                                continue
                            if not point_ok:
                                continue
                            pass_semantic += 1
                            accepted_record_count += 1
                            t_mat0 = time.perf_counter()
                            cur_code = int(r.code)
                            if last_hit_code is None:
                                hit_segments = 1
                                last_hit_code = cur_code
                            elif cur_code == last_hit_code:
                                pass
                            elif cur_code == last_hit_code + 1:
                                last_hit_code = cur_code
                            else:
                                hit_segments += 1
                                last_hit_code = cur_code
                            out.append(r)
                            codes.append(int(r.code))
                            filter_materialize_s += time.perf_counter() - t_mat0

        t3 = time.perf_counter()
        query_time_ms = (t3 - t0) * 1000.0  # 转换为毫秒
        
        metrics = compute_metrics(
            result_count=len(out),
            page_accesses=len(pages),
            b=self._artifacts.b,
            intervals=intervals,
            codes=codes,
            query_time_ms=query_time_ms,
        )

        candidate_slice_len_mean = float(records_scanned / candidate_slice_count) if candidate_slice_count > 0 else 0.0
        candidate_slice_len_nonempty_mean = (
            float(candidate_slice_len_sum_nonempty / nonempty_candidate_slice_count)
            if nonempty_candidate_slice_count > 0
            else 0.0
        )
        candidate_page_local_span_mean = (
            float(candidate_page_local_span_sum / candidate_page_count)
            if candidate_page_count > 0
            else 0.0
        )
        filter_candidate_scan_s += filter_candidate_extract_s + filter_candidate_array_s
        res = QueryResult(records=tuple(out), intervals=tuple(intervals), metrics=metrics)
        prof = QueryProfile(
            intervals_s=float(t1 - t0),
            lookup_s=float(t2 - t1),
            filter_s=float(t3 - t2),
            total_s=float(t3 - t0),
            interval_count=int(interval_count),
            covered_code_len=int(covered_code_len),
            records_scanned=int(records_scanned),
            pass_code=int(pass_code),
            pass_semantic=int(pass_semantic),
            hit_segments=int(hit_segments),
            page_count=int(page_count),
            touched_page_record_count=int(touched_page_record_count),
            candidate_page_count=int(candidate_page_count),
            candidate_page_record_count=int(candidate_page_record_count),
            empty_candidate_page_count=int(empty_candidate_page_count),
            candidate_page_local_span_sum=int(candidate_page_local_span_sum),
            candidate_page_local_span_mean=float(candidate_page_local_span_mean),
            candidate_page_local_span_max=int(candidate_page_local_span_max),
            page_interval_visits=int(page_interval_visits),
            page_interval_overlaps=int(page_interval_overlaps),
            candidate_slice_count=int(candidate_slice_count),
            candidate_slice_len_mean=float(candidate_slice_len_mean),
            empty_slice_count=int(empty_slice_count),
            nonempty_candidate_slice_count=int(nonempty_candidate_slice_count),
            candidate_slice_len_nonempty_mean=float(candidate_slice_len_nonempty_mean),
            candidate_slice_len_max=int(candidate_slice_len_max),
            vectorized_candidate_slice_count=int(vectorized_candidate_slice_count),
            scalar_candidate_slice_count=int(scalar_candidate_slice_count),
            vectorized_candidate_record_count=int(vectorized_candidate_record_count),
            scalar_candidate_record_count=int(scalar_candidate_record_count),
            batched_candidate_slice_count=int(batched_candidate_slice_count),
            batched_candidate_record_count=int(batched_candidate_record_count),
            candidate_record_count=int(candidate_record_count),
            semantic_check_count=int(semantic_check_count),
            accepted_record_count=int(accepted_record_count),
            filter_bisect_s=float(filter_bisect_s),
            filter_candidate_scan_s=float(filter_candidate_scan_s),
            filter_candidate_extract_s=float(filter_candidate_extract_s),
            filter_candidate_array_s=float(filter_candidate_array_s),
            filter_semantic_check_s=float(filter_semantic_check_s),
            filter_materialize_s=float(filter_materialize_s),
        )
        return QueryProfiledResult(result=res, profile=prof)
