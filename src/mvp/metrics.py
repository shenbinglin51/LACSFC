from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence
import numpy as np

from .types import Interval


@dataclass(frozen=True)
class QueryMetrics:
    """
    新的查询性能指标体系
    
    所有指标都是越小越好（lower is better）
    """
    # 核心性能指标
    mean_seg: float           # 平均段数（越小越好）- SFC code 的连续段数量
    mean_span: float          # 平均跨度（越小越好）- SFC code 的范围跨度
    mean_span_per_cell: float # 平均单元跨度（越小越好）- 每个数据点的平均 code 跨度
    mean_page_amp: float      # 平均页放大率（越小越好）- 访问页数 / 理论最小页数
    query_latency: float      # 查询延迟（毫秒，越小越好）
    physical_pages: int       # 物理页数（越小越好）
    io_cost: int              # I/O 成本（越小越好）
    
    # 辅助信息（用于调试和分析）
    result_count: int         # 结果数量
    segments: int             # 段数（原始值）
    code_span: int            # code 跨度（原始值）


def compute_metrics(
    result_count: int,
    page_accesses: int,
    b: int,
    intervals: Sequence[Interval],
    codes: List[int],
    query_time_ms: float,
) -> QueryMetrics:
    """
    计算查询性能指标
    
    Args:
        result_count: 查询结果数量
        page_accesses: 实际访问的页数
        b: 页大小（每页记录数）
        intervals: SFC code 区间列表
        codes: 结果记录的 SFC codes 列表
        query_time_ms: 查询执行时间（毫秒）
    
    Returns:
        QueryMetrics 对象
    """
    # 1. mean_seg（平均段数）
    segments = len(intervals)
    mean_seg = float(segments)
    
    # 2. mean_span（平均跨度）
    if len(codes) > 0:
        code_span = max(codes) - min(codes)
        mean_span = float(code_span)
    else:
        code_span = 0
        mean_span = 0.0
    
    # 3. mean_span_per_cell（平均单元跨度）
    if result_count > 0:
        mean_span_per_cell = mean_span / result_count
    else:
        mean_span_per_cell = 0.0
    
    # 4. mean_page_amp（平均页放大率）
    if result_count > 0 and b > 0:
        # 理论最小页数 = ceil(result_count / b)
        min_pages = (result_count + b - 1) // b
        mean_page_amp = page_accesses / float(min_pages)
    else:
        mean_page_amp = 1.0
    
    # 5. query_latency（查询延迟）
    query_latency = query_time_ms
    
    # 6. physical_pages（物理页数）
    physical_pages = page_accesses
    
    # 7. io_cost（I/O 成本）
    # 简化实现：假设每个页访问对应一次 I/O
    io_cost = page_accesses
    
    return QueryMetrics(
        mean_seg=mean_seg,
        mean_span=mean_span,
        mean_span_per_cell=mean_span_per_cell,
        mean_page_amp=mean_page_amp,
        query_latency=query_latency,
        physical_pages=physical_pages,
        io_cost=io_cost,
        result_count=result_count,
        segments=segments,
        code_span=code_span,
    )
