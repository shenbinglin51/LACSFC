"""Scene encoding: convert query workload → per-edge features for the Generator."""
from typing import Optional, Tuple

import numpy as np

from lacsfcf.data.grid import Grid
from lacsfcf.data.query_workload import QueryWorkload


class SceneEncoder:
    """Encode a QueryWorkload as a fixed-size edge feature matrix.

    For each edge (b1, b2, dim) in the dual graph, the feature vector is:

        [density_b1,           # query hit density of b1
         density_b2,           # query hit density of b2
         joint_density,        # fraction of queries hitting both b1 AND b2
         sum_density,          # density_b1 + density_b2
         abs_diff_density,     # |density_b1 - density_b2|
         dim_onehot[0..d-1],   # one-hot for the merge dimension
         neighbor_avg_b1,      # mean density of b1's other neighbors
         neighbor_avg_b2,      # mean density of b2's other neighbors
         
         # 新增查询特征 (8维):
         mean_aspect_ratio,    # 跨越此边的查询的平均长宽比
         std_aspect_ratio,     # 长宽比标准差
         mean_query_size,      # 跨越此边的查询的平均大小(归一化)
         std_query_size,       # 查询大小标准差
         density_cv,           # 密度变异系数 (std/mean)
         edge_crossing_rate,   # 跨越此边的查询比例
         size_weighted_density,# 按查询大小加权的密度
         aspect_weighted_density, # 按长宽比加权的密度
         ]

    Output shape: (num_dual_edges, feat_dim)
    where feat_dim = 5 + d + 2 + 8 = d + 15.
    """

    def __init__(self, grid: Grid):
        self.grid     = grid
        self.feat_dim = grid.d + 15

    # ------------------------------------------------------------------
    def encode(self, workload: QueryWorkload) -> np.ndarray:
        """Return feature matrix of shape (num_dual_edges, feat_dim), float32."""
        grid = self.grid
        d    = grid.d
        bd   = workload.block_density  # shape = block_shape

        # Pre-compute joint density per edge
        joint = self._compute_joint_density(workload)

        # Pre-compute neighbor averages
        avg_nbr = self._neighbor_avg(bd)
        
        # Pre-compute query features per edge
        query_feats = self._compute_query_features(workload)

        feats = np.zeros((grid.num_dual_edges, self.feat_dim), dtype=np.float32)
        for i, (b1, b2, dim) in enumerate(grid.dual_edges):
            db1 = float(bd[b1])
            db2 = float(bd[b2])
            jd  = joint[i]

            row = np.zeros(self.feat_dim, dtype=np.float32)
            # 原有特征 (d+7维)
            row[0] = db1
            row[1] = db2
            row[2] = jd
            row[3] = db1 + db2
            row[4] = abs(db1 - db2)
            row[5 + dim] = 1.0            # dimension one-hot  (indices 5..5+d-1)
            row[5 + d]   = avg_nbr[b1]   # neighbor avg b1
            row[5 + d + 1] = avg_nbr[b2] # neighbor avg b2
            
            # 新增查询特征 (8维)
            base_idx = 5 + d + 2
            row[base_idx:base_idx+8] = query_feats[i]
            
            feats[i] = row

        return feats

    # ------------------------------------------------------------------
    def _compute_joint_density(self, workload: QueryWorkload) -> np.ndarray:
        """joint_density[i] = fraction of queries hitting both b1 and b2.

        For efficiency, uses the cell heatmap and picks the minimum overlap.
        Approximate: joint ≈ min(sum_in_b1, sum_in_b2) / max_hits
        For exact computation use query iteration (slow for large workloads).
        """
        bd    = workload.block_density
        joint = np.zeros(self.grid.num_dual_edges, dtype=np.float32)
        for i, (b1, b2, _) in enumerate(self.grid.dual_edges):
            joint[i] = min(float(bd[b1]), float(bd[b2]))
        return joint

    def _neighbor_avg(self, bd: np.ndarray) -> dict:
        """Compute average density of adjacent blocks for each block."""
        d    = self.grid.d
        B    = self.grid.block_side
        avgs: dict = {}
        for b1, b2, dim in self.grid.dual_edges:
            for blk in (b1, b2):
                if blk in avgs:
                    continue
                nbr_vals = []
                for k in range(d):
                    for delta in (-1, +1):
                        nbr = tuple(blk[j] + delta if j == k else blk[j] for j in range(d))
                        if all(0 <= nbr[j] < B for j in range(d)):
                            nbr_vals.append(float(bd[nbr]))
                avgs[blk] = float(np.mean(nbr_vals)) if nbr_vals else 0.0
        return avgs
    
    def _compute_query_features(self, workload: QueryWorkload) -> np.ndarray:
        """Compute query-specific features for each edge.
        
        Returns array of shape (num_dual_edges, 8) with:
          [mean_aspect_ratio, std_aspect_ratio, mean_query_size, std_query_size,
           density_cv, edge_crossing_rate, size_weighted_density, aspect_weighted_density]
        """
        grid = self.grid
        queries = workload.queries
        n_queries = len(queries)
        
        # 为每条边收集跨越它的查询信息
        edge_queries = [[] for _ in range(grid.num_dual_edges)]
        
        # 计算每个查询的特征
        query_aspects = []
        query_sizes = []
        for q in queries:
            query_aspects.append(q.get_aspect_ratio())
            query_sizes.append(q.get_volume() / grid.num_cells)  # 归一化大小
        
        # 判断每个查询跨越哪些边
        for q_idx, q in enumerate(queries):
            # 获取查询覆盖的块
            blocks_hit = set()
            for cell in self._query_cells(q, grid):
                block, _ = grid.cell_to_block(cell)
                blocks_hit.add(block)
            
            # 检查每条边是否被此查询跨越
            for edge_idx, (b1, b2, _) in enumerate(grid.dual_edges):
                if b1 in blocks_hit and b2 in blocks_hit:
                    edge_queries[edge_idx].append(q_idx)
        
        # 计算每条边的查询特征
        feats = np.zeros((grid.num_dual_edges, 8), dtype=np.float32)
        bd = workload.block_density
        
        for i, (b1, b2, _) in enumerate(grid.dual_edges):
            q_indices = edge_queries[i]
            
            if len(q_indices) > 0:
                # 跨越此边的查询的长宽比统计
                aspects = [query_aspects[j] for j in q_indices]
                feats[i, 0] = float(np.mean(aspects))
                feats[i, 1] = float(np.std(aspects)) if len(aspects) > 1 else 0.0
                
                # 跨越此边的查询的大小统计
                sizes = [query_sizes[j] for j in q_indices]
                feats[i, 2] = float(np.mean(sizes))
                feats[i, 3] = float(np.std(sizes)) if len(sizes) > 1 else 0.0
                
                # 密度变异系数 (反映查询分布的不均匀性)
                densities = [bd[b1], bd[b2]]
                mean_dens = np.mean(densities)
                std_dens = np.std(densities)
                feats[i, 4] = float(std_dens / mean_dens) if mean_dens > 1e-6 else 0.0
                
                # 边缘跨越率
                feats[i, 5] = float(len(q_indices)) / n_queries
                
                # 按查询大小加权的密度
                weighted_dens = sum(sizes[j] * (bd[b1] + bd[b2]) for j in range(len(q_indices)))
                feats[i, 6] = float(weighted_dens / len(q_indices))
                
                # 按长宽比加权的密度 (长宽比大的查询更重要)
                aspect_weighted = sum(aspects[j] * (bd[b1] + bd[b2]) for j in range(len(q_indices)))
                feats[i, 7] = float(aspect_weighted / len(q_indices))
            else:
                # 没有查询跨越此边，使用默认值
                feats[i, 0] = 1.0  # 默认长宽比
                feats[i, 5] = 0.0  # 跨越率为0
        
        return feats
    
    def _query_cells(self, q, grid: Grid):
        """Generate all cells hit by query q (generator for memory efficiency)."""
        # 只采样部分单元格以提高效率
        ranges = [range(q.lo[k], min(q.hi[k] + 1, grid.grid_side)) for k in range(grid.d)]
        # 对于大查询，采样而不是遍历所有单元格
        total_cells = np.prod([len(r) for r in ranges])
        if total_cells > 100:  # 如果查询太大，采样
            sample_size = min(100, int(total_cells))
            for _ in range(sample_size):
                cell = tuple(np.random.choice(r) for r in ranges)
                yield cell
        else:
            import itertools
            for cell in itertools.product(*ranges):
                yield cell

    # ------------------------------------------------------------------
    @property
    def output_dim(self) -> int:
        return self.feat_dim
