"""Surrogate-assisted CMA-ES for LACSFC generalization.

Workflow:
  Offline: run CMA-ES on T diverse tasks → collect (scene_features, optimal_weights) pairs
  Train:   MLP surrogate: scene_features → optimal_weights_init
  Online:  new query set → SceneEncoder → surrogate → warm-start CMA-ES (few evals)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np

from .._paths import lacsfcf_root as _lr
sys.path.insert(0, str(_lr()))

from lacsfcf.data.grid import Grid
from lacsfcf.data.query_workload import QueryWorkload, _sample_one_query
from lacsfcf.data.scene import SceneEncoder
from lacsfcf.core.sfc import weights_to_sfc


# ---------------------------------------------------------------------------
# Task generation: diverse (data_dist × query_dist) combinations
# ---------------------------------------------------------------------------

def generate_diverse_tasks(
    grid: Grid,
    encoder: SceneEncoder,
    points: np.ndarray,
    n_tasks: int = 60,
    n_queries_per_task: int = 100,
    seed: int = 0,
) -> List[dict]:
    """
    Generate T diverse tasks.  Each task has:
      - query_windows: List[lacsfcf.Query]  (for CMA-ES objective)
      - scene_feats:   np.ndarray (num_edges, feat_dim)
      - base_weights:  np.ndarray (num_edges,)  LACSFC-H warm start
    """
    rng = np.random.default_rng(seed)
    tasks = []
    n_pts = len(points)

    for t in range(n_tasks):
        # 随机选查询分布类型
        dist_type = t % 6
        if dist_type == 0:    # 紧凑热点
            hotspot_std, win_lo, win_hi = 0.03, 0.01, 0.08
        elif dist_type == 1:  # 中等热点
            hotspot_std, win_lo, win_hi = 0.06, 0.02, 0.12
        elif dist_type == 2:  # 宽松热点
            hotspot_std, win_lo, win_hi = 0.10, 0.03, 0.18
        elif dist_type == 3:  # 大查询框
            hotspot_std, win_lo, win_hi = 0.08, 0.10, 0.35
        elif dist_type == 4:  # 小查询框
            hotspot_std, win_lo, win_hi = 0.05, 0.005, 0.04
        else:                 # 随机均匀
            hotspot_std, win_lo, win_hi = 0.15, 0.02, 0.25

        # 以数据点为热点中心生成查询
        center_idxs = rng.integers(0, n_pts, n_queries_per_task)
        queries_lacsfcf = []
        for ci in center_idxs:
            c = points[ci]
            q = _sample_one_query(rng, grid, tuple(float(c[d]) for d in range(grid.d)),
                                  win_lo, win_hi)
            queries_lacsfcf.append(q)

        wl_obj = QueryWorkload(queries_lacsfcf, grid)
        scene_feats = encoder.encode(wl_obj)                    # (E, feat_dim)
        base_weights = wl_obj.dual_edge_weights().astype(np.float32)  # (E,)

        tasks.append({
            'task_id':     t,
            'dist_type':   dist_type,
            'queries':     queries_lacsfcf,
            'scene_feats': scene_feats,
            'base_weights': base_weights,
        })

    return tasks


# ---------------------------------------------------------------------------
# Step 1: run CMA-ES on each task, collect (feats, optimal_weights)
# ---------------------------------------------------------------------------

def collect_meta_dataset(
    grid: Grid,
    encoder: SceneEncoder,
    points: np.ndarray,
    n_tasks: int = 60,
    n_queries_per_task: int = 100,
    b: int = 256,
    cmaes_max_evals: int = 1000,
    cmaes_sigma0: float = 0.5,
    cmaes_popsize: int = 20,
    seed: int = 0,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        X: (n_tasks, num_edges * feat_dim)  scene feature vectors (flattened)
        Y: (n_tasks, num_edges)             optimal weight vectors from CMA-ES
    """
    from .trainer import optimize_weights_cmaes
    from src.mvp.types import QueryWindow as _QW
    from src.mvp.quantizer import Quantizer as _Quant

    tasks = generate_diverse_tasks(
        grid=grid, encoder=encoder, points=points,
        n_tasks=n_tasks, n_queries_per_task=n_queries_per_task, seed=seed,
    )

    q_obj = _Quant(k=grid.K, d=grid.d)

    X_list, Y_list = [], []

    for i, task in enumerate(tasks):
        t0 = time.time()

        # 把 lacsfcf.Query → src.mvp.types.QueryWindow（浮点坐标）
        gs = grid.grid_side
        qw_list = []
        for q in task['queries']:
            lo = np.array([q.lo[d] / gs for d in range(grid.d)], dtype=np.float64)
            hi = np.array([(q.hi[d] + 1) / gs for d in range(grid.d)], dtype=np.float64)
            qw_list.append(_QW(low=lo, high=hi))

        # 用 LACSFC-H 权重热启动
        init_w = task['base_weights'].copy()

        w_opt = optimize_weights_cmaes(
            grid=grid,
            workload=task['queries'],
            points=points,
            b=b,
            init_weights=init_w,
            sigma0=cmaes_sigma0,
            max_evals=cmaes_max_evals,
            popsize=cmaes_popsize,
            verbose=False,
            query_windows=qw_list,
            quantizer=q_obj,
        )

        X_list.append(task['scene_feats'].ravel())   # flatten to 1-D
        Y_list.append(w_opt.ravel())

        elapsed = time.time() - t0
        if verbose:
            print(f'  task {i+1:3d}/{n_tasks}  dist_type={task["dist_type"]}  '
                  f'elapsed={elapsed:.1f}s', flush=True)

    X = np.stack(X_list, axis=0).astype(np.float32)
    Y = np.stack(Y_list, axis=0).astype(np.float32)
    return X, Y


# ---------------------------------------------------------------------------
# Step 2: train surrogate MLP
# ---------------------------------------------------------------------------

class SurrogateMLP:
    """Lightweight MLP: scene_feats (flattened) → optimal_weights_init."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256):
        import torch
        import torch.nn as nn
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )
        self._torch = torch

    def train_on_data(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        epochs: int = 500,
        lr: float = 1e-3,
        batch_size: int = 16,
        val_split: float = 0.15,
        verbose: bool = True,
    ) -> dict:
        import torch, torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader

        Xt = torch.tensor(X, dtype=torch.float32)
        Yt = torch.tensor(Y, dtype=torch.float32)

        n_val = max(1, int(len(X) * val_split))
        Xtr, Ytr = Xt[n_val:], Yt[n_val:]
        Xva, Yva = Xt[:n_val],  Yt[:n_val]

        loader = DataLoader(TensorDataset(Xtr, Ytr), batch_size=batch_size, shuffle=True)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        mse = nn.MSELoss()

        history = {'train_loss': [], 'val_loss': []}
        best_val, best_state = float('inf'), None

        for ep in range(epochs):
            self.net.train()
            ep_loss = 0.0
            for xb, yb in loader:
                opt.zero_grad()
                loss = mse(self.net(xb), yb)
                loss.backward()
                opt.step()
                ep_loss += float(loss) * len(xb)
            scheduler.step()

            self.net.eval()
            with torch.no_grad():
                val_loss = float(mse(self.net(Xva), Yva))
            train_loss = ep_loss / len(Xtr)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}

            if verbose and (ep + 1) % 100 == 0:
                print(f'  Epoch {ep+1:4d}/{epochs}  train={train_loss:.4f}  val={val_loss:.4f}',
                      flush=True)

        if best_state is not None:
            self.net.load_state_dict(best_state)

        return {'best_val_loss': best_val, 'history': history}

    def predict(self, x: np.ndarray) -> np.ndarray:
        import torch
        self.net.eval()
        with torch.no_grad():
            xt = torch.tensor(x.ravel()[None], dtype=torch.float32)
            out = self.net(xt).numpy().ravel()
        return out.astype(np.float32)

    def save(self, path: str):
        import torch
        torch.save({'net': self.net.state_dict(),
                    'in_dim': self.in_dim, 'out_dim': self.out_dim}, path)
        print(f'Surrogate saved → {path}')

    @classmethod
    def load(cls, path: str) -> 'SurrogateMLP':
        import torch
        ckpt = torch.load(path, map_location='cpu')
        obj = cls(ckpt['in_dim'], ckpt['out_dim'])
        obj.net.load_state_dict(ckpt['net'])
        return obj


# ---------------------------------------------------------------------------
# Step 3: online warm-start helper
# ---------------------------------------------------------------------------

def predict_init_weights(
    surrogate: SurrogateMLP,
    encoder: SceneEncoder,
    query_windows_lacsfcf: list,
    grid: Grid,
) -> np.ndarray:
    """
    Given a new query workload (lacsfcf.Query list), return predicted init weights
    for CMA-ES warm start.
    """
    wl_obj = QueryWorkload(query_windows_lacsfcf, grid)
    feats = encoder.encode(wl_obj).ravel()
    w_init = surrogate.predict(feats)
    w_init = np.clip(w_init, 0, None)
    return w_init
