from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .types import QueryWindow


@dataclass(frozen=True)
class WorkloadConfig:
    points: np.ndarray
    queries: Tuple[QueryWindow, ...]


def _clip01(points: np.ndarray) -> np.ndarray:
    eps = np.finfo(np.float64).eps
    return np.clip(points, 0.0, 1.0 - eps)


def load_points_real(path: str, k: int, max_n: int = 0,
                     require_n: bool = False) -> np.ndarray:
    """Load real points, optionally truncating to ``max_n``.

    When ``require_n`` is True and ``max_n > 0``, raise if the source contains
    fewer points than requested instead of silently using all available points.
    This guards scale experiments against the EXP-0530 Layer C failure mode
    where every requested scale silently reused a smaller cached point set.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    def _check_n(available: int) -> None:
        if require_n and max_n > 0 and available < int(max_n):
            raise ValueError(
                f"requested n={int(max_n)} but points source '{path}' only "
                f"provides {available} points; refusing silent under-fill "
                f"(set require_n=False to allow, or use a larger source)"
            )

    if p.suffix.lower() == ".npy":
        arr = np.load(str(p), allow_pickle=False)
        _check_n(len(arr))
        if max_n > 0 and len(arr) > max_n:
            arr = arr[:max_n]
    elif p.suffix.lower() == ".csv":
        import pandas as pd
        nrows = max_n if max_n > 0 else None
        df = pd.read_csv(str(p), header=None, delimiter=",", nrows=nrows,
                         on_bad_lines="skip", low_memory=False)
        if not pd.api.types.is_float_dtype(df.iloc[0, 0]):
            df = df.iloc[1:].reset_index(drop=True)
        arr = df.values.astype(np.float64)
    elif p.stat().st_size >= 4:
        count_arr = np.fromfile(str(p), dtype="<u4", count=1)
        if count_arr.size != 1:
            raise ValueError("points file must be .npy, .csv, or OSM uint32 binary")
        count = int(count_arr[0])
        expected_size = 4 + count * 2 * 4
        if count <= 0 or expected_size != p.stat().st_size:
            raise ValueError("points file must be .npy, .csv, or OSM uint32 binary")
        _check_n(count)
        mm = np.memmap(str(p), dtype="<u4", mode="r", offset=4, shape=(count, 2))
        col_lo = np.asarray(mm.min(axis=0), dtype=np.float64)
        col_hi = np.asarray(mm.max(axis=0), dtype=np.float64)
        n_read = min(count, int(max_n)) if max_n > 0 else count
        arr = np.asarray(mm[:n_read], dtype=np.float64)
        col_range = col_hi - col_lo
        col_range[col_range == 0] = 1.0
        pts = (arr - col_lo) / col_range
        return _clip01(pts)
    else:
        raise ValueError("points file must be .npy, .csv, or OSM uint32 binary")

    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError("points must have shape (n,d)")

    arr = np.asarray(arr)
    arr64 = arr.astype(np.float64)

    maxv = float(np.max(arr64)) if arr64.size else 0.0
    minv = float(np.min(arr64)) if arr64.size else 0.0

    integer_like = bool(np.all(np.isclose(arr64, np.round(arr64))))

    if np.issubdtype(arr.dtype, np.integer) or (integer_like and maxv > 1.0):
        col_lo = np.percentile(arr64, 1, axis=0)
        col_hi = np.percentile(arr64, 99, axis=0)
        col_range = col_hi - col_lo
        col_range[col_range == 0] = 1.0
        pts = (arr64 - col_lo) / col_range
        return _clip01(pts)

    if minv < 0.0 or maxv > 1.0:
        raise ValueError(
            "real points look like float coordinates but are out of [0,1]; "
            "use integer grid coordinates or normalize to [0,1) first"
        )

    return _clip01(arr64)


def generate_points_uniform(rng: np.random.Generator, n: int, d: int) -> np.ndarray:
    return rng.random((int(n), int(d)), dtype=np.float64)


def generate_points_gmm(
    rng: np.random.Generator,
    n: int,
    d: int,
    components: int = 4,
    sigma: float = 0.06,
) -> np.ndarray:
    """生成高斯混合模型（GMM）分布的点 - 聚类数据"""
    n = int(n)
    d = int(d)
    components = int(components)
    sigma = float(sigma)

    means = rng.random((components, d), dtype=np.float64)
    weights = rng.random((components,), dtype=np.float64)
    weights = weights / np.sum(weights)
    z = rng.choice(components, size=n, replace=True, p=weights)
    noise = rng.normal(loc=0.0, scale=sigma, size=(n, d)).astype(np.float64)
    pts = means[z] + noise
    return _clip01(pts)


def generate_points_normal(
    rng: np.random.Generator,
    n: int,
    d: int,
    mean: float = 0.5,
    sigma: float = 0.15,
) -> np.ndarray:
    """生成正态分布的点 - 集中在中心区域"""
    n = int(n)
    d = int(d)
    mean = float(mean)
    sigma = float(sigma)
    
    pts = rng.normal(loc=mean, scale=sigma, size=(n, d)).astype(np.float64)
    return _clip01(pts)


def generate_points_skewed(
    rng: np.random.Generator,
    n: int,
    d: int,
    skew_factor: float = 2.0,
) -> np.ndarray:
    """生成倾斜分布的点 - 使用 Beta 分布"""
    n = int(n)
    d = int(d)
    skew_factor = float(skew_factor)
    
    # Beta 分布：alpha=1, beta=skew_factor
    # skew_factor 越大，数据越集中在 0 附近
    pts = rng.beta(1.0, skew_factor, size=(n, d)).astype(np.float64)
    return _clip01(pts)


def generate_points_zipf(
    rng: np.random.Generator,
    n: int,
    d: int,
    alpha: float = 1.5,
    grid_size: int = 256,
) -> np.ndarray:
    """生成 Zipf 分布的点 - 幂律分布，模拟真实世界的热点数据"""
    n = int(n)
    d = int(d)
    alpha = float(alpha)
    grid_size = int(grid_size)
    
    pts = np.zeros((n, d), dtype=np.float64)
    for dim in range(d):
        # Zipf 分布生成整数索引
        zipf_samples = rng.zipf(alpha, size=n)
        # 限制在 grid_size 范围内
        zipf_samples = np.clip(zipf_samples, 1, grid_size)
        # 归一化到 [0, 1)
        pts[:, dim] = (zipf_samples - 1) / grid_size
    
    return _clip01(pts)


def generate_points_diagonal(
    rng: np.random.Generator,
    n: int,
    d: int = 2,
    width: float = 0.1,
) -> np.ndarray:
    """生成对角线分布的点 - 沿对角线聚集"""
    if d != 2:
        raise ValueError("Diagonal distribution only supports 2D")
    
    n = int(n)
    width = float(width)
    
    # 沿对角线 y=x 生成点
    t = rng.random(n, dtype=np.float64)  # 参数 t ∈ [0, 1]
    noise = rng.normal(0, width, size=(n, 2)).astype(np.float64)
    
    pts = np.column_stack([t, t]) + noise
    return _clip01(pts)


def generate_queries_fixed_area(
    rng: np.random.Generator,
    qn: int,
    d: int,
    area: float = 0.1,
) -> Tuple[QueryWindow, ...]:
    qn = int(qn)
    d = int(d)
    area = float(area)
    if area <= 0.0 or area > 1.0:
        raise ValueError("area must be in (0,1]")

    side = area ** (1.0 / d)
    half = side / 2.0

    qs: List[QueryWindow] = []
    for _ in range(qn):
        center = rng.random((d,), dtype=np.float64)
        low = center - half
        high = center + half
        low = np.clip(low, 0.0, 1.0)
        high = np.clip(high, 0.0, 1.0)
        qs.append(QueryWindow(low=low, high=high))

    return tuple(qs)


def generate_queries_loguniform_area(
    rng: np.random.Generator,
    qn: int,
    d: int,
    area_min: float = 1e-4,
    area_max: float = 0.2,
) -> Tuple[QueryWindow, ...]:
    qn = int(qn)
    d = int(d)
    area_min = float(area_min)
    area_max = float(area_max)
    if area_min <= 0.0 or area_max <= 0.0 or area_min > area_max:
        raise ValueError("area_min/area_max invalid")

    log_min = float(np.log(area_min))
    log_max = float(np.log(area_max))

    qs: List[QueryWindow] = []
    for _ in range(qn):
        area = float(np.exp(rng.uniform(log_min, log_max)))
        side = area ** (1.0 / d)
        half = side / 2.0

        center = rng.random((d,), dtype=np.float64)
        low = np.clip(center - half, 0.0, 1.0)
        high = np.clip(center + half, 0.0, 1.0)
        qs.append(QueryWindow(low=low, high=high))

    return tuple(qs)


def generate_queries_hotspot_correlated(
    rng: np.random.Generator,
    points: np.ndarray,
    qn: int,
    d: int,
    area_min: float = 1e-4,
    area_max: float = 0.2,
) -> Tuple[QueryWindow, ...]:
    """生成与数据点相关的热点查询"""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != int(d):
        raise ValueError("points shape mismatch")

    qn = int(qn)
    d = int(d)
    area_min = float(area_min)
    area_max = float(area_max)
    if area_min <= 0.0 or area_max <= 0.0 or area_min > area_max:
        raise ValueError("area_min/area_max invalid")

    log_min = float(np.log(area_min))
    log_max = float(np.log(area_max))

    qs: List[QueryWindow] = []
    n = points.shape[0]
    for _ in range(qn):
        idx = int(rng.integers(0, n))
        center = points[idx]

        area = float(np.exp(rng.uniform(log_min, log_max)))
        side = area ** (1.0 / d)
        half = side / 2.0

        low = np.clip(center - half, 0.0, 1.0)
        high = np.clip(center + half, 0.0, 1.0)
        qs.append(QueryWindow(low=low, high=high))

    return tuple(qs)


def generate_queries_gaussian(
    rng: np.random.Generator,
    qn: int,
    d: int,
    area_min: float = 1e-4,
    area_max: float = 0.2,
    center: Optional[np.ndarray] = None,
    sigma: float = 0.15,
) -> Tuple[QueryWindow, ...]:
    """生成高斯分布的查询 - 集中在某个中心区域"""
    qn = int(qn)
    d = int(d)
    area_min = float(area_min)
    area_max = float(area_max)
    sigma = float(sigma)
    
    if center is None:
        center = np.array([0.5] * d, dtype=np.float64)
    else:
        center = np.asarray(center, dtype=np.float64)
    
    if area_min <= 0.0 or area_max <= 0.0 or area_min > area_max:
        raise ValueError("area_min/area_max invalid")
    
    log_min = float(np.log(area_min))
    log_max = float(np.log(area_max))
    
    qs: List[QueryWindow] = []
    for _ in range(qn):
        # 高斯分布的查询中心
        query_center = rng.normal(center, sigma, d).astype(np.float64)
        query_center = np.clip(query_center, 0.0, 1.0)
        
        # 对数均匀分布的查询大小
        area = float(np.exp(rng.uniform(log_min, log_max)))
        side = area ** (1.0 / d)
        half = side / 2.0
        
        low = np.clip(query_center - half, 0.0, 1.0)
        high = np.clip(query_center + half, 0.0, 1.0)
        qs.append(QueryWindow(low=low, high=high))
    
    return tuple(qs)


def generate_queries_skewed(
    rng: np.random.Generator,
    qn: int,
    d: int,
    area_min: float = 1e-4,
    area_max: float = 0.2,
    skew_factor: float = 2.0,
) -> Tuple[QueryWindow, ...]:
    """生成倾斜分布的查询 - 更多查询集中在某些区域"""
    qn = int(qn)
    d = int(d)
    area_min = float(area_min)
    area_max = float(area_max)
    skew_factor = float(skew_factor)
    
    if area_min <= 0.0 or area_max <= 0.0 or area_min > area_max:
        raise ValueError("area_min/area_max invalid")
    
    log_min = float(np.log(area_min))
    log_max = float(np.log(area_max))
    
    qs: List[QueryWindow] = []
    for _ in range(qn):
        # Beta 分布的查询中心（倾斜）
        query_center = rng.beta(1.0, skew_factor, d).astype(np.float64)
        
        # 对数均匀分布的查询大小
        area = float(np.exp(rng.uniform(log_min, log_max)))
        side = area ** (1.0 / d)
        half = side / 2.0
        
        low = np.clip(query_center - half, 0.0, 1.0)
        high = np.clip(query_center + half, 0.0, 1.0)
        qs.append(QueryWindow(low=low, high=high))
    
    return tuple(qs)


def generate_queries_hotspot(
    rng: np.random.Generator,
    qn: int,
    d: int,
    area_min: float = 1e-4,
    area_max: float = 0.2,
    n_hotspots: int = 3,
    hotspot_sigma: float = 0.1,
) -> Tuple[QueryWindow, ...]:
    """生成多个热点区域的查询"""
    qn = int(qn)
    d = int(d)
    area_min = float(area_min)
    area_max = float(area_max)
    n_hotspots = int(n_hotspots)
    hotspot_sigma = float(hotspot_sigma)
    
    if area_min <= 0.0 or area_max <= 0.0 or area_min > area_max:
        raise ValueError("area_min/area_max invalid")
    
    log_min = float(np.log(area_min))
    log_max = float(np.log(area_max))
    
    # 生成热点中心
    hotspots = rng.random((n_hotspots, d), dtype=np.float64)
    
    qs: List[QueryWindow] = []
    for _ in range(qn):
        # 随机选择一个热点
        hotspot_idx = rng.integers(0, n_hotspots)
        hotspot = hotspots[hotspot_idx]
        
        # 在热点附近生成查询中心
        query_center = rng.normal(hotspot, hotspot_sigma, d).astype(np.float64)
        query_center = np.clip(query_center, 0.0, 1.0)
        
        # 对数均匀分布的查询大小
        area = float(np.exp(rng.uniform(log_min, log_max)))
        side = area ** (1.0 / d)
        half = side / 2.0
        
        low = np.clip(query_center - half, 0.0, 1.0)
        high = np.clip(query_center + half, 0.0, 1.0)
        qs.append(QueryWindow(low=low, high=high))
    
    return tuple(qs)


def generate_queries_realistic(
    rng: np.random.Generator,
    qn: int,
    d: int,
    area_min: float = 1e-4,
    area_max: float = 0.2,
    scenario: str = 'mixed',
) -> Tuple[QueryWindow, ...]:
    """
    生成模拟真实场景的查询
    
    场景类型：
    - 'mixed': 混合场景（默认）- 70% 小查询 + 30% 大查询，60% 热点 + 40% 均匀
    - 'gis': GIS 场景 - 70% 热点（城市）+ 30% 均匀（农村）
    - 'scientific': 科学数据 - 高斯分布，集中在感兴趣区域
    - 'multimedia': 多媒体 - 倾斜分布，热门内容更多查询
    - 'iot': IoT 场景 - 双峰分布（早晚高峰）
    """
    qn = int(qn)
    d = int(d)
    area_min = float(area_min)
    area_max = float(area_max)
    
    if scenario == 'mixed':
        # 混合场景：70% 小查询 + 30% 大查询，60% 热点 + 40% 均匀
        n_small = int(qn * 0.7)
        n_large = qn - n_small
        n_hotspot = int(qn * 0.6)
        n_uniform = qn - n_hotspot
        
        # 小查询（热点）
        small_hotspot = generate_queries_hotspot(
            rng, n_small, d, 
            area_min=area_min, area_max=area_min * 10,
            n_hotspots=3, hotspot_sigma=0.08
        )
        
        # 大查询（均匀）
        large_uniform = generate_queries_loguniform_area(
            rng, n_large, d,
            area_min=area_max / 2, area_max=area_max
        )
        
        return small_hotspot + large_uniform
    
    elif scenario == 'gis':
        # GIS 场景：70% 热点（城市）+ 30% 均匀（农村）
        n_hotspot = int(qn * 0.7)
        n_uniform = qn - n_hotspot
        
        hotspot_queries = generate_queries_hotspot(
            rng, n_hotspot, d,
            area_min=area_min, area_max=area_max,
            n_hotspots=5, hotspot_sigma=0.1
        )
        
        uniform_queries = generate_queries_loguniform_area(
            rng, n_uniform, d,
            area_min=area_min, area_max=area_max
        )
        
        return hotspot_queries + uniform_queries
    
    elif scenario == 'scientific':
        # 科学数据：高斯分布
        return generate_queries_gaussian(
            rng, qn, d,
            area_min=area_min, area_max=area_max,
            center=None, sigma=0.2
        )
    
    elif scenario == 'multimedia':
        # 多媒体：倾斜分布（热门内容）
        return generate_queries_skewed(
            rng, qn, d,
            area_min=area_min, area_max=area_max,
            skew_factor=3.0
        )
    
    elif scenario == 'iot':
        # IoT：双峰分布（早晚高峰）
        n_half = qn // 2
        
        queries1 = generate_queries_gaussian(
            rng, n_half, d,
            area_min=area_min, area_max=area_max,
            center=np.array([0.3] * d), sigma=0.1
        )
        
        queries2 = generate_queries_gaussian(
            rng, qn - n_half, d,
            area_min=area_min, area_max=area_max,
            center=np.array([0.7] * d), sigma=0.1
        )
        
        return queries1 + queries2
    
    else:
        # 默认：对数均匀分布
        return generate_queries_loguniform_area(
            rng, qn, d,
            area_min=area_min, area_max=area_max
        )


def generate_points_correlated(
    rng: np.random.Generator,
    n: int,
    d: int,
    noise: float = 0.05,
) -> np.ndarray:
    """生成维度正相关数据点。

    第一维均匀采样，其余维度 = 第一维 + 小噪声，模拟地理轨迹、传感器时序等场景。
    所有坐标截断至 [0, 1)。

    Args:
        rng:   随机数生成器
        n:     点数量
        d:     维度
        noise: 各维度相对第一维的高斯噪声标准差（默认 0.05）
    """
    n = int(n)
    d = int(d)
    noise = float(noise)

    base = rng.random(n).astype(np.float64)
    cols = [base]
    for _ in range(d - 1):
        col = base + rng.normal(0.0, noise, n).astype(np.float64)
        cols.append(col)
    pts = np.stack(cols, axis=1)
    return _clip01(pts)


def generate_points_anti_correlated(
    rng: np.random.Generator,
    n: int,
    d: int,
    noise: float = 0.05,
) -> np.ndarray:
    """生成维度负相关数据点。

    第一维均匀采样，第 i 维 = (1 - 第一维) + 小噪声（奇数维）或 第一维 + 小噪声（偶数维），
    使相邻维度之间呈反相关，模拟多维索引文献中的标准压力测试场景。
    所有坐标截断至 [0, 1)。

    Args:
        rng:   随机数生成器
        n:     点数量
        d:     维度
        noise: 各维度高斯噪声标准差（默认 0.05）
    """
    n = int(n)
    d = int(d)
    noise = float(noise)

    base = rng.random(n).astype(np.float64)
    cols = [base]
    for i in range(1, d):
        if i % 2 == 1:
            col = (1.0 - base) + rng.normal(0.0, noise, n).astype(np.float64)
        else:
            col = base + rng.normal(0.0, noise, n).astype(np.float64)
        cols.append(col)
    pts = np.stack(cols, axis=1)
    return _clip01(pts)


def generate_queries_aspect_ratio(
    rng: np.random.Generator,
    qn: int,
    d: int,
    area_min: float = 1e-4,
    area_max: float = 0.1,
    aspect: float = 8.0,
) -> Tuple[QueryWindow, ...]:
    """生成高纵横比（长条形）查询窗口。

    随机选择一个"拉伸维度"，在该维度上宽度乘以 aspect，其余维度除以
    aspect^(1/(d-1))，保持总面积不变。模拟 GIS 中的"给定时间段内所有位置"
    或"给定区域内所有时间戳"等查询模式。

    Args:
        rng:      随机数生成器
        qn:       查询数量
        d:        维度
        area_min: 最小查询面积
        area_max: 最大查询面积
        aspect:   拉伸维度的放大倍数（默认 8）
    """
    qn = int(qn)
    d = int(d)
    area_min = float(area_min)
    area_max = float(area_max)
    aspect = float(aspect)

    if area_min <= 0.0 or area_max <= 0.0 or area_min > area_max:
        raise ValueError("area_min/area_max invalid")

    log_min = float(np.log(area_min))
    log_max = float(np.log(area_max))
    shrink = aspect ** (1.0 / max(d - 1, 1))

    qs: List[QueryWindow] = []
    for _ in range(qn):
        area = float(np.exp(rng.uniform(log_min, log_max)))
        side = area ** (1.0 / d)

        sides = np.full(d, side / shrink, dtype=np.float64)
        stretch_dim = int(rng.integers(0, d))
        sides[stretch_dim] = side * aspect

        center = rng.random(d).astype(np.float64)
        low = np.clip(center - sides / 2.0, 0.0, 1.0)
        high = np.clip(center + sides / 2.0, 0.0, 1.0)
        qs.append(QueryWindow(low=low, high=high))
    return tuple(qs)
