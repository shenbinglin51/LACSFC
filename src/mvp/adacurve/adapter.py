"""
数据和查询转换适配器

将 sfcBenchmark 的数据格式转换为 Adacurve 的格式
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

# 添加 Adacurve 项目路径到 sys.path
from .._paths import adacurve_root
ADACURVE_PATH = adacurve_root()
if str(ADACURVE_PATH) not in sys.path:
    sys.path.insert(0, str(ADACURVE_PATH))

from Adacurve.data import Dataset, quantize_points, cell_coords_to_id, build_cell_to_point_ids, enumerate_all_cells
from Adacurve.queries import WindowQuery

from ..types import QueryWindow


def convert_to_adacurve_dataset(
    points: np.ndarray,
    k: int,
) -> Dataset:
    """
    将 sfcBenchmark 的点数据转换为 Adacurve 的 Dataset 格式
    
    Args:
        points: (n, d) 数组，范围 [0, 1)
        k: 量化位数
    
    Returns:
        Adacurve Dataset 对象
    """
    points = np.asarray(points, dtype=np.float32)
    n, d = points.shape
    
    grid_size = 1 << k
    cell_coords = quantize_points(points, k)
    cell_id = cell_coords_to_id(cell_coords, grid_size)
    cell_to_point_ids = build_cell_to_point_ids(cell_id)
    
    all_cell_coords, all_cell_lower, all_cell_upper = enumerate_all_cells(d, k)
    
    return Dataset(
        points=points,
        d=d,
        k=k,
        grid_size=grid_size,
        cell_coords=cell_coords,
        cell_id=cell_id,
        cell_to_point_ids=cell_to_point_ids,
        all_cell_coords=all_cell_coords,
        all_cell_lower=all_cell_lower,
        all_cell_upper=all_cell_upper,
    )


def convert_to_adacurve_queries(
    queries: List[QueryWindow],
) -> List[WindowQuery]:
    """
    将 sfcBenchmark 的查询转换为 Adacurve 的 WindowQuery 格式
    
    Args:
        queries: sfcBenchmark QueryWindow 列表
    
    Returns:
        Adacurve WindowQuery 列表
    """
    adacurve_queries = []
    for q in queries:
        lower = np.asarray(q.low, dtype=np.float32)
        upper = np.asarray(q.high, dtype=np.float32)
        adacurve_queries.append(WindowQuery(lower=lower, upper=upper))
    
    return adacurve_queries


def convert_from_adacurve_query(
    query: WindowQuery,
) -> QueryWindow:
    """
    将 Adacurve 的 WindowQuery 转换为 sfcBenchmark 的 QueryWindow 格式
    
    Args:
        query: Adacurve WindowQuery
    
    Returns:
        sfcBenchmark QueryWindow
    """
    return QueryWindow(
        low=tuple(query.lower.tolist()),
        high=tuple(query.upper.tolist()),
    )
