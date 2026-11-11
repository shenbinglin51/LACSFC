"""LACSFCF training v3: Joint Evaluator-Generator training with three-weight strategy.

Inspired by NeuralSFC's design:
  - weights_tgt: Heuristic baseline + random perturbation (exploration)
  - weights_gen: Generator output (adversarial learning)
  - weights_gen_avg: Batch-averaged generator output (stability)

Key improvements over v2:
  1. Evaluator sees Generator's output during training (distribution matching)
  2. Real-time adversarial learning (not just pre-training + fine-tuning)
  3. Multi-source weight sampling (diversity + stability)
  4. Joint training loop (E and G evolve together)
"""
import argparse
import os
import numpy as np
import torch
import torch.optim as optim

from typing import List
from lacsfcf.config import LacsfcConfig
from lacsfcf.core.sfc import weights_to_sfc
from lacsfcf.data.grid import Grid
from lacsfcf.data.scene import SceneEncoder
from lacsfcf.data.query_workload import sample_workload
from lacsfcf.metrics.range_query import mean_seg, mean_span
from lacsfcf.model.evaluator import QueryEvaluator
from lacsfcf.model.generator import EdgeWeightGenerator


class SceneDataset:
    """On-the-fly synthetic scene dataset for training."""
    
    def __init__(self, cfg: LacsfcConfig, grid: Grid, enc: SceneEncoder, size: int = 256):
        self.cfg = cfg
        self.grid = grid
        self.enc = enc
        self.size = size
        self._scenes: List = []
        self.build_time = 0.0
        self._build()
    
    def _build(self):
        import time
        start = time.time()
        rng = np.random.default_rng(self.cfg.seed)
        for i in range(self.size):
            wl = sample_workload(
                self.grid,
                num_queries=self.cfg.num_queries,
                num_hotspots=self.cfg.num_hotspots,
                hotspot_std=self.cfg.hotspot_std,
                win_min_frac=self.cfg.win_min_frac,
                win_max_frac=self.cfg.win_max_frac,
                seed=int(rng.integers(0, 10**9)),
            )
            feats = self.enc.encode(wl)
            self._scenes.append((feats, wl.queries, wl.dual_edge_weights()))
        self.build_time = time.time() - start
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self._scenes[i] for i in range(*idx.indices(len(self)))]
        return self._scenes[idx]


def compute_quality(grid: Grid, weights: np.ndarray, queries, metric='mean_seg'):
    """Compute true quality metric for given weights."""
    path = weights_to_sfc(grid, weights)
    code = grid.code_map_from_path(path)
    
    if metric == 'mean_seg':
        return float(mean_seg(code, queries))
    elif metric == 'mean_span':
        return float(mean_span(code, queries))
    elif metric == 'combined':
        seg = float(mean_seg(code, queries))
        span = float(mean_span(code, queries))
        return seg
    else:
        raise ValueError(f"Unknown metric: {metric}")


def sample_target_weights(base_weights: np.ndarray, strategy: str, rng=None):
    """Sample target weights with different strategies (like NeuralSFC's data augmentation)."""
    if rng is None:
        rng = np.random.default_rng()
    
    weights = base_weights.astype(np.float32)
    
    if strategy == 'heuristic_noise':
        noise = rng.normal(0, 0.1, size=weights.shape).astype(np.float32)
        return weights + noise
    
    elif strategy == 'heuristic_flip':
        if rng.random() > 0.5:
            weights = -weights
        return weights
    
    elif strategy == 'heuristic_mix':
        mix_ratio = rng.random()
        random_weights = rng.normal(0, 0.5, size=weights.shape).astype(np.float32)
        return mix_ratio * weights + (1 - mix_ratio) * random_weights
    
    elif strategy == 'random_uniform':
        return rng.uniform(-1.0, 1.0, size=weights.shape).astype(np.float32)
    
    elif strategy == 'random_normal':
        return rng.normal(0.0, 0.5, size=weights.shape).astype(np.float32)
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def joint_train(
    generator: EdgeWeightGenerator,
    evaluator: QueryEvaluator,
    dataset: SceneDataset,
    grid: Grid,
    cfg: LacsfcConfig,
):
    """Joint training with three-weight strategy.
    
    Each iteration:
      1. Sample weights_tgt (exploration)
      2. Generate weights_gen from Generator (adversarial)
      3. Compute weights_gen_avg (stability)
      4. Train Evaluator on all three
      5. Train Generator using Evaluator's gradient
    """
    device = torch.device(cfg.device)
    generator.to(device)
    evaluator.to(device)
    
    opt_g = optim.Adam(generator.parameters(), lr=cfg.lr)
    opt_e = optim.Adam(evaluator.parameters(), lr=cfg.lr * 0.1)
    
    print("=== Joint Evaluator-Generator Training (v3) ===")
    print(f"Strategy: Three-weight (tgt + gen + gen_avg)")
    print(f"Epochs: {cfg.epochs}, Batch size: {cfg.batch_size}")
    print()
    
    strategies = ['heuristic_noise', 'heuristic_flip', 'heuristic_mix', 
                  'random_uniform', 'random_normal']
    
    for epoch in range(cfg.epochs):
        evaluator.train()
        generator.train()
        
        e_losses = []
        g_losses = []
        
        batch_weights_gen = []
        batch_feats = []
        
        for scene_idx, (feats, queries, base_weights) in enumerate(dataset):
            t_feats = torch.tensor(feats, dtype=torch.float32, device=device)
            
            # ============================================================
            # Step 1: Generate weights_gen (Generator output)
            # ============================================================
            with torch.no_grad():
                weights_gen_tensor = generator(t_feats)
                weights_gen = weights_gen_tensor.cpu().numpy()
            
            batch_weights_gen.append(weights_gen)
            batch_feats.append(t_feats)
            
            # ============================================================
            # Step 2: Sample weights_tgt (exploration)
            # ============================================================
            strategy = strategies[scene_idx % len(strategies)]
            weights_tgt = sample_target_weights(base_weights, strategy)
            
            # ============================================================
            # Step 3: Compute true qualities
            # ============================================================
            quality_tgt = compute_quality(grid, weights_tgt, queries, 'mean_seg')
            quality_gen = compute_quality(grid, weights_gen, queries, 'mean_seg')
            
            # ============================================================
            # Step 4: Train Evaluator on both weights
            # ============================================================
            evaluator.train()
            opt_e.zero_grad()
            
            t_weights_tgt = torch.tensor(weights_tgt, dtype=torch.float32, device=device)
            pred_tgt = evaluator(t_feats, t_weights_tgt)
            target_tgt = torch.tensor(quality_tgt, dtype=torch.float32, device=device)
            loss_tgt = (pred_tgt - target_tgt) ** 2
            
            t_weights_gen = torch.tensor(weights_gen, dtype=torch.float32, device=device)
            pred_gen = evaluator(t_feats, t_weights_gen.detach())
            target_gen = torch.tensor(quality_gen, dtype=torch.float32, device=device)
            loss_gen = (pred_gen - target_gen) ** 2
            
            e_loss = loss_tgt + loss_gen
            e_loss.backward()
            opt_e.step()
            e_losses.append(float(e_loss.detach()))
            
            # ============================================================
            # Step 5: Train Generator (every batch_size scenes)
            # ============================================================
            if (scene_idx + 1) % cfg.batch_size == 0 and len(batch_weights_gen) > 0:
                weights_gen_avg = np.mean(batch_weights_gen, axis=0).astype(np.float32)
                
                generator.train()
                evaluator.eval()
                opt_g.zero_grad()
                
                weights_gen_new = generator(t_feats)
                pred_quality = evaluator(t_feats, weights_gen_new)
                
                t_weights_avg = torch.tensor(weights_gen_avg, dtype=torch.float32, device=device)
                pred_avg = evaluator(t_feats, t_weights_avg)
                
                g_loss = pred_quality + 0.5 * pred_avg
                g_loss.backward()
                opt_g.step()
                g_losses.append(float(g_loss.detach()))
                
                batch_weights_gen = []
                batch_feats = []
        
        if (epoch + 1) % 5 == 0:
            avg_e_loss = np.mean(e_losses) if e_losses else 0.0
            avg_g_loss = np.mean(g_losses) if g_losses else 0.0
            print(f"  Epoch {epoch+1:3d}/{cfg.epochs}  "
                  f"E_loss = {avg_e_loss:.4f}  G_loss = {avg_g_loss:.4f}")
    
    print()


def evaluate_generator(
    generator: EdgeWeightGenerator,
    dataset: SceneDataset,
    grid: Grid,
    cfg: LacsfcConfig,
    n_scenes: int = 20,
) -> dict:
    """Evaluate Generator on test scenes."""
    device = torch.device(cfg.device)
    generator.to(device).eval()
    
    segs = []
    spans = []
    
    with torch.no_grad():
        for feats, queries, _ in dataset[:n_scenes]:
            t_feats = torch.tensor(feats, dtype=torch.float32, device=device)
            weights = generator(t_feats).cpu().numpy()
            
            path = weights_to_sfc(grid, weights)
            code = grid.code_map_from_path(path)
            
            segs.append(mean_seg(code, queries))
            spans.append(mean_span(code, queries))
    
    return {
        'mean_seg': float(np.mean(segs)),
        'mean_span': float(np.mean(spans)),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int, default=2)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--dataset_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gen_hidden", type=int, default=128)
    p.add_argument("--gen_layers", type=int, default=3)
    p.add_argument("--eval_hidden", type=int, default=128)
    p.add_argument("--eval_layers", type=int, default=3)
    p.add_argument("--out", type=str, default="./output/v3_joint_training")
    return p.parse_args()


def main():
    args = parse_args()
    
    cfg = LacsfcConfig(
        d=args.d,
        K=args.K,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    
    print(f"Building dataset ({args.dataset_size} scenes) …")
    grid = Grid(d=cfg.d, K=cfg.K)
    enc = SceneEncoder(grid)
    ds = SceneDataset(cfg, grid, enc, size=args.dataset_size)
    print(f"Done in {ds.build_time:.1f}s\n")
    
    generator = EdgeWeightGenerator(
        input_dim=enc.feat_dim,
        hidden_dim=args.gen_hidden,
        num_layers=args.gen_layers,
    )
    evaluator = QueryEvaluator(
        input_dim=enc.feat_dim,
        hidden_dim=args.eval_hidden,
        num_layers=args.eval_layers,
    )
    
    joint_train(generator, evaluator, ds, grid, cfg)
    
    print("=== Final evaluation ===")
    results = evaluate_generator(generator, ds, grid, cfg, n_scenes=20)
    print(f"  mean_seg  = {results['mean_seg']:.4f}")
    print(f"  mean_span = {results['mean_span']:.4f}\n")
    
    os.makedirs(args.out, exist_ok=True)
    torch.save(generator.state_dict(), os.path.join(args.out, "generator.pt"))
    torch.save(evaluator.state_dict(), os.path.join(args.out, "evaluator.pt"))
    print(f"Models saved to {args.out}/")


if __name__ == "__main__":
    main()
