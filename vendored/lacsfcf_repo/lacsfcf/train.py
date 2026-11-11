"""LACSFCF training loop.

Two-phase training:
  Phase 1 (Evaluator pre-train):
    For each scene, sample random edge weights, decode the SFC, compute
    true mean_seg, and train QueryEvaluator to predict it.

  Phase 2 (Generator train, Evaluator frozen):
    Generator produces weights → Evaluator predicts mean_seg → backprop
    into Generator to minimize predicted mean_seg.

Usage:
    python -m lacsfcf.train --d 2 --K 8 --epochs 50
"""
import argparse
import os
import random
import time
from typing import List

import numpy as np
import torch
import torch.optim as optim

from lacsfcf.config import LacsfcConfig
from lacsfcf.core.sfc import weights_to_sfc
from lacsfcf.data.grid import Grid
from lacsfcf.data.query_workload import sample_workload
from lacsfcf.data.scene import SceneEncoder
from lacsfcf.metrics.range_query import mean_seg, mean_span
from lacsfcf.model.evaluator import QueryEvaluator
from lacsfcf.model.generator import EdgeWeightGenerator


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SceneDataset:
    """On-the-fly synthetic scene dataset for training.

    Each 'sample' is a (edge_feats, queries) pair from a freshly sampled
    query workload.
    """

    def __init__(self, cfg: LacsfcConfig, grid: Grid, enc: SceneEncoder, size: int = 256):
        self.cfg  = cfg
        self.grid = grid
        self.enc  = enc
        self.size = size
        self._scenes: List = []
        self._build()

    def _build(self):
        rng = np.random.default_rng(self.cfg.seed)
        for i in range(self.size):
            wl    = sample_workload(
                self.grid,
                num_queries  = self.cfg.num_queries,
                num_hotspots = self.cfg.num_hotspots,
                hotspot_std  = self.cfg.hotspot_std,
                win_min_frac = self.cfg.win_min_frac,
                win_max_frac = self.cfg.win_max_frac,
                seed         = int(rng.integers(0, 10**9)),
            )
            feats = self.enc.encode(wl)
            self._scenes.append((feats, wl.queries, wl.dual_edge_weights()))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self._scenes[idx]


# ---------------------------------------------------------------------------
# Evaluator pre-training
# ---------------------------------------------------------------------------

def pretrain_evaluator(
    evaluator:  QueryEvaluator,
    dataset:    SceneDataset,
    cfg:        LacsfcConfig,
    grid:       Grid,
) -> None:
    """Train Evaluator to predict true mean_seg from (edge_feats, weights)."""
    device = torch.device(cfg.device)
    evaluator.to(device).train()
    opt = optim.Adam(evaluator.parameters(), lr=cfg.lr)

    print("=== Phase 1: Evaluator pre-training ===")
    for epoch in range(cfg.evaluator_pretrain_epochs):
        losses = []
        for feats, queries, base_weights in dataset:
            t_feats = torch.tensor(feats, dtype=torch.float32, device=device)

            for sample_idx in range(cfg.evaluator_samples_per_scene):
                # 改进1: 查询驱动的权重采样策略 (6种策略轮流)
                strategy_idx = sample_idx % 6
                
                if strategy_idx == 0:
                    # Strategy 1: Heuristic weights + small noise
                    noise = np.random.normal(0, 0.1, size=base_weights.shape).astype(np.float32)
                    w_np = base_weights + noise
                elif strategy_idx == 1:
                    # Strategy 2: Query-driven hotspot sampling
                    from lacsfcf.data.query_workload import QueryWorkload
                    wl_temp = QueryWorkload(queries, grid)
                    w_np = wl_temp.sample_query_driven_weights("hotspot", rng=np.random.default_rng())
                elif strategy_idx == 2:
                    # Strategy 3: Query-driven gradient sampling
                    from lacsfcf.data.query_workload import QueryWorkload
                    wl_temp = QueryWorkload(queries, grid)
                    w_np = wl_temp.sample_query_driven_weights("gradient", rng=np.random.default_rng())
                elif strategy_idx == 3:
                    # Strategy 4: Query-driven correlation sampling
                    from lacsfcf.data.query_workload import QueryWorkload
                    wl_temp = QueryWorkload(queries, grid)
                    w_np = wl_temp.sample_query_driven_weights("correlation", rng=np.random.default_rng())
                elif strategy_idx == 4:
                    # Strategy 5: Random weights (uniform [-1, 1])
                    w_np = np.random.uniform(-1.0, 1.0, size=base_weights.shape).astype(np.float32)
                else:
                    # Strategy 6: Random weights (normal distribution)
                    w_np = np.random.normal(0.0, 0.5, size=base_weights.shape).astype(np.float32)
                
                t_w = torch.tensor(w_np, dtype=torch.float32, device=device)

                # 改进3: 多目标优化 - 计算两个指标
                path      = weights_to_sfc(grid, w_np)
                code      = grid.code_map_from_path(path)
                true_seg  = float(mean_seg(code, queries))
                true_span = float(mean_span(code, queries))
                
                # 归一化后加权组合 (mean_seg 权重0.7, mean_span 权重0.3)
                true_combined = 0.7 * true_seg + 0.3 * (true_span / 100.0)

                # Predicted metric
                pred    = evaluator(t_feats.detach(), t_w.detach())
                target  = torch.tensor(true_combined, dtype=torch.float32, device=device)
                loss    = (pred - target) ** 2

                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(float(loss.detach()))

        if (epoch + 1) % max(1, cfg.evaluator_pretrain_epochs // 5) == 0:
            print(f"  Epoch {epoch+1:3d}/{cfg.evaluator_pretrain_epochs}  "
                  f"MSE loss = {np.mean(losses):.4f}")

    print()


# ---------------------------------------------------------------------------
# Evaluator online update (改进2)
# ---------------------------------------------------------------------------

def finetune_evaluator_online(
    evaluator:  QueryEvaluator,
    generator:  EdgeWeightGenerator,
    dataset:    SceneDataset,
    grid:       Grid,
    cfg:        LacsfcConfig,
    n_samples:  int = 32,
) -> None:
    """改进2: 用Generator当前生成的权重在线微调Evaluator.
    
    这样可以让Evaluator适应Generator的权重分布变化，避免预测失效。
    """
    device = torch.device(cfg.device)
    evaluator.to(device).train()
    generator.to(device).eval()
    
    opt = optim.Adam(evaluator.parameters(), lr=cfg.lr * 0.1)  # 更小的学习率
    
    losses = []
    for _ in range(n_samples):
        idx = np.random.randint(0, len(dataset))
        feats, queries, _ = dataset[idx]
        t_feats = torch.tensor(feats, dtype=torch.float32, device=device)
        
        # 用Generator生成权重
        with torch.no_grad():
            weights = generator(t_feats)
        w_np = weights.cpu().numpy()
        
        # 计算真实指标
        path      = weights_to_sfc(grid, w_np)
        code      = grid.code_map_from_path(path)
        true_seg  = float(mean_seg(code, queries))
        true_span = float(mean_span(code, queries))
        true_combined = 0.7 * true_seg + 0.3 * (true_span / 100.0)
        
        # 微调Evaluator
        pred   = evaluator(t_feats, weights)
        target = torch.tensor(true_combined, dtype=torch.float32, device=device)
        loss   = (pred - target) ** 2
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    
    evaluator.eval()
    return float(np.mean(losses))


# ---------------------------------------------------------------------------
# Generator training
# ---------------------------------------------------------------------------

def train_generator(
    generator:  EdgeWeightGenerator,
    evaluator:  QueryEvaluator,
    dataset:    SceneDataset,
    cfg:        LacsfcConfig,
    grid:       Grid,
) -> None:
    """Train Generator to minimise evaluator-predicted combined metric."""
    device = torch.device(cfg.device)
    generator.to(device).train()
    evaluator.to(device).eval()
    for p in evaluator.parameters():
        p.requires_grad_(False)

    opt = optim.Adam(generator.parameters(), lr=cfg.lr)

    print("=== Phase 2: Generator training ===")
    for epoch in range(cfg.epochs):
        losses = []
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        for idx in indices[:cfg.batch_size]:
            feats, queries, _ = dataset[idx]
            t_feats = torch.tensor(feats, dtype=torch.float32, device=device)

            # Generator predicts weights → Evaluator predicts cost
            weights   = generator(t_feats)          # (E,) with grad
            pred_cost = evaluator(t_feats, weights)  # scalar

            opt.zero_grad()
            pred_cost.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            opt.step()
            losses.append(float(pred_cost.detach()))

        # 改进2: 每10个epoch在线微调Evaluator
        if (epoch + 1) % 10 == 0:
            for p in evaluator.parameters():
                p.requires_grad_(True)
            
            ft_loss = finetune_evaluator_online(evaluator, generator, dataset, grid, cfg, n_samples=32)
            
            for p in evaluator.parameters():
                p.requires_grad_(False)
            
            print(f"  [Epoch {epoch+1}] Evaluator fine-tuned, loss = {ft_loss:.4f}")

        if (epoch + 1) % cfg.eval_interval == 0:
            avg_loss = float(np.mean(losses))
            print(f"  Epoch {epoch+1:3d}/{cfg.epochs}  "
                  f"predicted combined metric = {avg_loss:.4f}")

    print()


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate_generator(
    generator: EdgeWeightGenerator,
    dataset:   SceneDataset,
    grid:      Grid,
    cfg:       LacsfcConfig,
    n_scenes:  int = 20,
) -> dict:
    """Evaluate Generator on n_scenes using true metrics."""
    device = torch.device(cfg.device)
    generator.to(device).eval()

    true_segs = []
    true_spans = []
    with torch.no_grad():
        for feats, queries, _ in dataset[:n_scenes]:
            t_feats  = torch.tensor(feats, dtype=torch.float32, device=device)
            w_tensor = generator(t_feats)
            w_np     = w_tensor.cpu().numpy()
            path     = weights_to_sfc(grid, w_np)
            code     = grid.code_map_from_path(path)
            true_segs.append(mean_seg(code, queries))
            true_spans.append(mean_span(code, queries))

    return {
        "mean_seg": float(np.mean(true_segs)),
        "mean_span": float(np.mean(true_spans)),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--d",           type=int,   default=2)
    p.add_argument("--K",           type=int,   default=8)
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--pretrain",    type=int,   default=10,
                   help="Evaluator pre-train epochs")
    p.add_argument("--batch_size",  type=int,   default=16)
    p.add_argument("--dataset_size",type=int,   default=128)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--device",      type=str,   default="cpu")
    p.add_argument("--out",         type=str,   default="./output/lacsfcf_train")
    # Model architecture
    p.add_argument("--gen_hidden",  type=int,   default=128,
                   help="Generator hidden dimension")
    p.add_argument("--gen_layers",  type=int,   default=3,
                   help="Generator number of layers")
    p.add_argument("--eval_hidden", type=int,   default=128,
                   help="Evaluator hidden dimension")
    p.add_argument("--eval_layers", type=int,   default=3,
                   help="Evaluator number of layers")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    cfg = LacsfcConfig(
        d                         = args.d,
        K                         = args.K,
        epochs                    = args.epochs,
        evaluator_pretrain_epochs = args.pretrain,
        batch_size                = args.batch_size,
        lr                        = args.lr,
        seed                      = args.seed,
        device                    = args.device,
        out_dir                   = args.out,
        gen_hidden                = args.gen_hidden,
        gen_layers                = args.gen_layers,
        eval_hidden               = args.eval_hidden,
        eval_layers               = args.eval_layers,
    )

    grid = Grid(d=cfg.d, K=cfg.K)
    enc  = SceneEncoder(grid)

    print(f"Building dataset ({args.dataset_size} scenes) …")
    t0 = time.time()
    ds = SceneDataset(cfg, grid, enc, size=args.dataset_size)
    print(f"Done in {time.time()-t0:.1f}s\n")

    generator = EdgeWeightGenerator(
        input_dim  = enc.feat_dim,
        hidden_dim = args.gen_hidden,
        num_layers = args.gen_layers,
    )
    evaluator = QueryEvaluator(
        input_dim  = enc.feat_dim,
        hidden_dim = args.eval_hidden,
        num_layers = args.eval_layers,
    )

    pretrain_evaluator(evaluator, ds, cfg, grid)
    train_generator(generator, evaluator, ds, cfg, grid)

    print("=== Final evaluation (true metrics) ===")
    final_metrics = evaluate_generator(generator, ds, grid, cfg, n_scenes=20)
    print(f"  Neural LACSFCF mean_seg  = {final_metrics['mean_seg']:.4f}")
    print(f"  Neural LACSFCF mean_span = {final_metrics['mean_span']:.4f}\n")

    os.makedirs(args.out, exist_ok=True)
    torch.save(generator.state_dict(), os.path.join(args.out, "generator.pt"))
    torch.save(evaluator.state_dict(), os.path.join(args.out, "evaluator.pt"))
    print(f"Models saved to {args.out}/")


if __name__ == "__main__":
    main()
