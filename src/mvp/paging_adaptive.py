"""
Adaptive (Variable-size) Paging for SFC-based indexes.

核心思想：
  标准分页：按固定大小 b 将 SFC-sorted records 分成等大的 page。
  自适应分页：根据 SFC code 的局部"连续性"（code 密度）动态调整页大小。

直觉：
  - 在数据密集区域，SFC code 间隔小（相邻 records 的 code 差值小）
    => 用更小的 page（b_small < b）
    => 一个查询访问的 page 数量虽然多，但每个 page 的 code_span 更小
    => 一个矩形查询命中的 page 数量减少（因为 code_span 小的 page 不会"溢出"到查询边界之外）
  - 在稀疏区域，SFC code 间隔大（相邻 records 的 code 差值大）
    => 用更大的 page（b_large > b）
    => 减少 page 总数，降低索引开销

约束：
  - 平均页大小 ≈ b（总存储成本不变）
  - page 大小在 [b_min, b_max] 范围内
  - page 大小是 2 的幂次（可选，简化索引实现）

算法（Code-Gap 自适应）：
  1. 计算每个 record 的 "局部 code gap" = 其与周围窗口内 records 的平均 code 间隔
  2. gap 小 => 密集区域 => 小页
  3. gap 大 => 稀疏区域 => 大页
  4. 用目标函数: page_size_i = b * (gap_i / mean_gap)^alpha
     alpha 是控制自适应强度的参数（0=固定分页，1=线性自适应）
  5. 归一化确保总 page 数 ≈ n/b

分页的目标：最小化 E[page_amp]
  page_amp_i = pages_accessed_i / ceil(hits_i / b_i)
  
  对于一个矩形查询 q，命中 hits 个点（SFC code 在 [lo, hi] 内）：
  - 标准分页：访问 ceil((hi - lo) / b) 个 page，page_amp ≈ (code_span / b) / (hits / b) = code_span / hits
  - 自适应分页（密集区域页更小）：密集区域一个 page 覆盖更小的 code range
    => 查询只需访问覆盖 [lo, hi] 的 page，不会"多访问"边界 page 中的无关数据
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple
import numpy as np

from .types import Page, Record


def _compute_local_density(
    codes: np.ndarray,
    window: int = 8,
) -> np.ndarray:
    """
    计算每个 record 位置的局部 code 密度（gap 的倒数）。
    
    Args:
        codes: 已排序的 SFC code 数组，shape (n,)
        window: 滑动窗口半径（考虑前后各 window 个 records）
    
    Returns:
        density: shape (n,)，值越大 = 越密集
    """
    n = len(codes)
    if n <= 1:
        return np.ones(n)
    
    # 对每个位置，计算前后 window 个 records 内的平均 code gap
    # gap[i] = (codes[i+window] - codes[i-window]) / (2*window)
    density = np.empty(n, dtype=np.float64)
    
    for i in range(n):
        lo = max(0, i - window)
        hi = min(n - 1, i + window)
        span = max(1, int(codes[hi]) - int(codes[lo]))
        count = hi - lo
        gap = span / max(count, 1)
        density[i] = 1.0 / max(gap, 1e-9)
    
    return density


def _compute_local_density_fast(
    codes: np.ndarray,
    window: int = 16,
) -> np.ndarray:
    """
    向量化版本的局部密度计算（更快）。
    
    density[i] = (2*window) / (codes[min(i+window, n-1)] - codes[max(i-window, 0)] + 1)
    """
    n = len(codes)
    if n <= 1:
        return np.ones(n)
    
    i_arr = np.arange(n)
    lo_arr = np.maximum(0, i_arr - window)
    hi_arr = np.minimum(n - 1, i_arr + window)
    
    spans = (codes[hi_arr] - codes[lo_arr]).astype(np.float64) + 1.0
    counts = (hi_arr - lo_arr).astype(np.float64) + 1.0
    gap = spans / counts  # 平均每个 record 的 code gap
    
    density = 1.0 / np.maximum(gap, 1e-9)
    return density


def build_pages_adaptive(
    records_sorted: Sequence[Record],
    b: int,
    b_min: int = 16,
    b_max: int = 1024,
    alpha: float = 0.6,
    density_window: int = 16,
) -> Tuple[Tuple[Page, ...], np.ndarray]:
    """
    自适应分页：根据局部 SFC code 密度动态调整每页大小。
    
    Args:
        records_sorted: 按 SFC code 排序的 records
        b: 目标平均页大小（与标准分页相同）
        b_min: 最小页大小（密集区域）
        b_max: 最大页大小（稀疏区域）
        alpha: 自适应强度 [0, 1]，0=固定页大小，1=完全自适应
        density_window: 局部密度估计的滑动窗口半径
    
    Returns:
        (pages, page_start_codes)，接口与 build_pages 完全相同
    """
    n = len(records_sorted)
    if n == 0:
        return tuple(), np.empty(0, dtype=np.int64)
    if b <= 0:
        raise ValueError("b must be > 0")
    
    # 提取 codes
    codes = np.array([r.code for r in records_sorted], dtype=np.int64)
    
    # 计算局部密度
    density = _compute_local_density_fast(codes, window=density_window)
    mean_density = float(np.mean(density))
    if mean_density <= 0:
        mean_density = 1.0
    
    # 归一化密度
    rel_density = density / mean_density  # 1.0 = 平均密度
    
    # 目标页大小：密集区域(rel_density > 1) => 更小的页，稀疏区域 => 更大的页
    # page_size = b / rel_density^alpha
    # rel_density > 1 => page_size < b (密集区域)
    # rel_density < 1 => page_size > b (稀疏区域)
    target_sizes = b / np.power(np.maximum(rel_density, 1e-3), alpha)
    target_sizes = np.clip(target_sizes, b_min, b_max)
    
    # 用目标页大小进行分页
    pages: List[Page] = []
    page_start_codes: List[int] = []
    
    i = 0
    while i < n:
        # 取当前位置的目标页大小（取整到最近的整数，至少 b_min）
        page_size = max(b_min, min(b_max, int(round(float(target_sizes[i])))))
        
        end = min(i + page_size, n)
        chunk = tuple(records_sorted[i:end])
        
        page_id = len(pages)
        pages.append(Page(page_id=page_id, records=chunk))
        page_start_codes.append(int(chunk[0].code))
        
        i = end
    
    return tuple(pages), np.asarray(page_start_codes, dtype=np.int64)


def build_pages_adaptive_v2(
    records_sorted: Sequence[Record],
    b: int,
    b_min: int = 16,
    b_max: int = 1024,
    alpha: float = 0.6,
    density_window: int = 16,
) -> Tuple[Tuple[Page, ...], np.ndarray]:
    """
    v2：基于 code-gap 阈值的自适应分页。
    
    策略：扫描 records，当当前 page 内的 code_span 达到阈值时换页。
    
    阈值 = b * mean_gap（使得平均页大小 ≈ b 个 records）
    
    优点：
    - 直接控制每个 page 的 code_span，而不是通过密度间接控制
    - 密集区域（gap 小）：每个 page 覆盖的 code_span = 阈值，因此包含更多 records
    - 稀疏区域（gap 大）：每个 page 覆盖的 code_span = 阈值，但 records 很少
    
    等等，这反过来了——v2 在密集区域会有更大的 page（更多 records 但相同 code span）。
    改为：基于固定 record 数切割，但保证同一 page 内 code_span 不超过阈值。
    """
    n = len(records_sorted)
    if n == 0:
        return tuple(), np.empty(0, dtype=np.int64)
    
    codes = np.array([r.code for r in records_sorted], dtype=np.int64)
    
    # 全局平均 gap = (max_code - min_code) / (n - 1)
    global_span = max(1, int(codes[-1]) - int(codes[0]))
    mean_gap = global_span / max(n - 1, 1)
    
    # code_span 阈值：一个 page 最多覆盖这么大的 code 范围
    # 设为 b * mean_gap（使得密集区域 page 包含更多 records，稀疏区域更少）
    # 注意：这里我们想要的是密集区域小 page（少 records）
    # 所以用固定 record 数 + 额外 code_span 上界来实现

    # 实际上用密度直接控制更清晰，使用 v1 逻辑
    return build_pages_adaptive(
        records_sorted, b=b, b_min=b_min, b_max=b_max,
        alpha=alpha, density_window=density_window
    )


def build_pages_span_capped(
    records_sorted: Sequence[Record],
    b: int,
    span_cap_factor: float = 1.0,
    b_min: int = 8,
) -> Tuple[Tuple[Page, ...], np.ndarray]:
    """
    Span-Capped Paging：每个 page 的 code_span 不超过 cap，同时 record 数不超过 b。
    
    这是最直接的方法：
    - cap = span_cap_factor * mean_page_span（标准分页下的平均 code_span）
    - 在密集区域，一个 page 在达到 cap 之前就已经有 b 个 records => 按 b 切割
    - 在稀疏区域，一个 page 在有 b 个 records 之前就超过 cap => 按 cap 切割 => 小页
    
    效果：
    - 密集区域：页大小 ≤ b（通常 = b，因为密集区域 span < cap 时才达到 b records）
    - 稀疏区域：页大小 < b（span 先达到 cap 就切割）
    - 这等价于：为稀疏区域减小页大小，让每个 page 更"纯净"（span 更小）
    
    注：这会增加总页数（因为稀疏区域页变小了），查询稀疏区域时访问更多 page，
    但密集区域的查询会更精准（不会"溢出"到大 span 的稀疏 page）。
    
    对于热点数据场景：
    - 大部分 page 在热点区域 => 热点 page 大小 ≈ b（正常）
    - 稀疏区域有更多但更小的 page => 热点查询不会"误触"大 span 的稀疏 page
    """
    n = len(records_sorted)
    if n == 0:
        return tuple(), np.empty(0, dtype=np.int64)
    
    codes = np.array([r.code for r in records_sorted], dtype=np.int64)
    
    # 估算标准分页的平均 code_span
    std_pages = max(1, n // b)
    total_span = max(1, int(codes[-1]) - int(codes[0]))
    mean_std_span = total_span / std_pages
    cap = mean_std_span * span_cap_factor
    
    pages: List[Page] = []
    page_start_codes: List[int] = []
    
    i = 0
    while i < n:
        start_code = int(codes[i])
        j = i + 1
        while j < n:
            if (j - i) >= b:
                break  # 达到最大 record 数
            if (int(codes[j]) - start_code) > cap:
                break  # 超过 span cap
            j += 1
        
        # 至少放 b_min 个 records（防止极端碎片化）
        j = max(j, min(i + b_min, n))
        
        chunk = tuple(records_sorted[i:j])
        page_id = len(pages)
        pages.append(Page(page_id=page_id, records=chunk))
        page_start_codes.append(int(chunk[0].code))
        i = j
    
    return tuple(pages), np.asarray(page_start_codes, dtype=np.int64)
