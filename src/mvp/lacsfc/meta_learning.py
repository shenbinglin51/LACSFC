"""Meta-Learning for LACSFC.

Implements meta-learning (MAML-style) pre-training to enable fast adaptation
to new query workloads.
"""
import time
from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.optim as optim
from copy import deepcopy

import sys
from .._paths import lacsfcf_root as _lr
sys.path.insert(0, str(_lr()))

from lacsfcf.core.sfc import weights_to_sfc
from lacsfcf.data.grid import Grid
from lacsfcf.data.scene import SceneEncoder
from lacsfcf.data.query_workload import sample_workload, QueryWorkload
from lacsfcf.model.evaluator import QueryEvaluator
from lacsfcf.model.generator import EdgeWeightGenerator

from .trainer import compute_quality_mvp, PRECOMPUTED_AVAILABLE


class MetaLearningDataset:
    """Dataset for meta-learning with diverse query workloads."""
    
    def __init__(
        self,
        grid: Grid,
        encoder: SceneEncoder,
        n_tasks: int = 50,
        queries_per_task: int = 100,
        seed: int = 42,
    ):
        """
        Args:
            grid: Grid object
            encoder: Scene encoder
            n_tasks: Number of meta-learning tasks
            queries_per_task: Number of queries per task
            seed: Random seed
        """
        self.grid = grid
        self.encoder = encoder
        self.n_tasks = n_tasks
        self.queries_per_task = queries_per_task
        self.tasks = []
        
        self._build_tasks(seed)
    
    def _build_tasks(self, seed: int):
        """Build diverse tasks with different query distributions."""
        rng = np.random.default_rng(seed)
        
        # Define different query distribution types
        distributions = [
            # Single hotspot
            {'num_hotspots': 1, 'hotspot_std': 0.1, 'win_min_frac': 0.01, 'win_max_frac': 0.15},
            # Multiple hotspots
            {'num_hotspots': 3, 'hotspot_std': 0.15, 'win_min_frac': 0.01, 'win_max_frac': 0.2},
            # Dense hotspots
            {'num_hotspots': 5, 'hotspot_std': 0.1, 'win_min_frac': 0.005, 'win_max_frac': 0.1},
            # Large windows
            {'num_hotspots': 2, 'hotspot_std': 0.2, 'win_min_frac': 0.1, 'win_max_frac': 0.4},
            # Small windows
            {'num_hotspots': 2, 'hotspot_std': 0.15, 'win_min_frac': 0.001, 'win_max_frac': 0.05},
            # Sparse hotspots
            {'num_hotspots': 1, 'hotspot_std': 0.2, 'win_min_frac': 0.02, 'win_max_frac': 0.3},
        ]
        
        for task_id in range(self.n_tasks):
            # Select distribution type
            dist_config = distributions[task_id % len(distributions)]
            
            # Generate query workload
            wl = sample_workload(
                self.grid,
                num_queries=self.queries_per_task,
                num_hotspots=dist_config['num_hotspots'],
                hotspot_std=dist_config['hotspot_std'],
                win_min_frac=dist_config['win_min_frac'],
                win_max_frac=dist_config['win_max_frac'],
                seed=int(rng.integers(0, 10**9)),
            )
            
            queries = wl.queries
            wl_obj = QueryWorkload(queries, self.grid)
            feats = self.encoder.encode(wl_obj)
            base_weights = wl_obj.dual_edge_weights()
            
            self.tasks.append({
                'task_id': task_id,
                'feats': feats,
                'queries': queries,
                'base_weights': base_weights,
                'dist_type': task_id % len(distributions),
            })
    
    def __len__(self):
        return len(self.tasks)
    
    def __getitem__(self, idx):
        return self.tasks[idx]
    
    def sample_tasks(self, n: int, rng=None):
        """Sample n random tasks."""
        if rng is None:
            rng = np.random.default_rng()
        indices = rng.choice(len(self.tasks), size=n, replace=False)
        return [self.tasks[i] for i in indices]


def warmstart_generator_to_dual_edge(
    generator: EdgeWeightGenerator,
    meta_dataset: MetaLearningDataset,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 8,
    device: str = 'cpu',
    verbose: bool = True,
) -> EdgeWeightGenerator:
    """D-4 productionization: warm-start generator to mimic `dual_edge_weights()`.

    Rationale: random init for `EdgeWeightGenerator` starts meta-learning far
    from the strong LACSFC-H heuristic (which uses `dual_edge_weights()`).
    Pre-training the generator via MSE regression on ``(feats -> dual_edge_weights)``
    over the meta-dataset places it near the heuristic baseline, so subsequent
    meta-training and fine-tuning only need to *improve* on LACSFC-H rather than
    discover it from scratch. Closes the handicap flagged in the D-4 report.
    """
    device = torch.device(device)
    generator.to(device)
    optimizer = optim.Adam(generator.parameters(), lr=lr)
    mse = torch.nn.MSELoss()

    # Materialize (feats, target_weights) tensors
    feats_list, tgt_list = [], []
    for task in meta_dataset.tasks:
        feats_list.append(torch.tensor(task['feats'], dtype=torch.float32))
        tgt_list.append(torch.tensor(task['base_weights'], dtype=torch.float32))

    if verbose:
        print(f"🔥 Warm-starting generator on {len(feats_list)} tasks "
              f"via dual_edge MSE regression ({epochs} epochs, lr={lr})")

    n = len(feats_list)
    rng = np.random.default_rng(0)
    for ep in range(epochs):
        idx = rng.permutation(n)
        total_loss = 0.0
        for start in range(0, n, batch_size):
            batch = idx[start:start + batch_size]
            loss = torch.zeros(1, device=device)
            for i in batch:
                f = feats_list[i].to(device)
                t = tgt_list[i].to(device)
                w = generator(f)
                # Shape guard: generator may return (E,) or (E,1)
                if w.ndim > t.ndim:
                    w = w.squeeze(-1)
                if t.shape != w.shape:
                    # Trim/pad if feat dim counts differ; best-effort skip
                    continue
                loss = loss + mse(w, t)
            if loss.requires_grad:
                optimizer.zero_grad()
                (loss / max(1, len(batch))).backward()
                optimizer.step()
                total_loss += float(loss.detach()) / max(1, len(batch))
        if verbose and (ep + 1) % max(1, epochs // 5) == 0:
            print(f"  warmstart ep {ep+1:3d}/{epochs}  mse≈{total_loss/(n/batch_size):.4f}")

    if verbose:
        print("✅ Warm-start to dual_edge_weights completed.")
    return generator


def meta_train_step(
    generator: EdgeWeightGenerator,
    evaluator: QueryEvaluator,
    task: dict,
    grid: Grid,
    inner_lr: float = 1e-3,
    inner_steps: int = 5,
    device: str = 'cpu',
) -> Tuple[float, EdgeWeightGenerator]:
    """Single meta-training step (inner loop).
    
    Args:
        generator: Generator model
        evaluator: Evaluator model
        task: Task dictionary
        grid: Grid object
        inner_lr: Inner loop learning rate
        inner_steps: Number of inner loop steps
        device: Device to use
    
    Returns:
        (loss, adapted_generator): Loss and adapted generator
    """
    device = torch.device(device)
    
    # Create a copy of generator for adaptation
    adapted_generator = deepcopy(generator)
    adapted_generator.to(device)
    
    inner_optimizer = optim.SGD(adapted_generator.parameters(), lr=inner_lr)
    
    feats = task['feats']
    queries = task['queries']
    
    t_feats = torch.tensor(feats, dtype=torch.float32, device=device)
    
    # Inner loop: adapt to this task
    for _ in range(inner_steps):
        inner_optimizer.zero_grad()
        
        # Generate weights
        weights_gen = adapted_generator(t_feats)
        
        # Evaluate quality using Evaluator
        pred_quality = evaluator(t_feats, weights_gen)
        
        # Inner loss: minimize predicted quality
        inner_loss = pred_quality
        inner_loss.backward()
        inner_optimizer.step()
    
    # Compute final loss on adapted model
    with torch.no_grad():
        weights_final = adapted_generator(t_feats).cpu().numpy()
        
        # Compute true quality using MVP metrics
        seg, span_per_cell, page_amp = compute_quality_mvp(
            grid, weights_final, queries, b=256, use_precomputed=PRECOMPUTED_AVAILABLE
        )
        # Align with paper representative objective instance (0.60, 0.15, 0.25):
        # C_λ = λ_s·seg + λ_h·span_per_cell + λ_p·page_amp
        true_quality = 0.60 * seg + 0.15 * span_per_cell + 0.25 * page_amp
    
    return true_quality, adapted_generator


def meta_train(
    generator: EdgeWeightGenerator,
    evaluator: QueryEvaluator,
    meta_dataset: MetaLearningDataset,
    grid: Grid,
    meta_epochs: int = 20,
    meta_batch_size: int = 4,
    inner_lr: float = 1e-3,
    outer_lr: float = 1e-4,
    inner_steps: int = 5,
    device: str = 'cpu',
    verbose: bool = True,
) -> Tuple[EdgeWeightGenerator, QueryEvaluator]:
    """Meta-learning training (MAML-style).
    
    Args:
        generator: Generator model to meta-train
        evaluator: Evaluator model (pre-trained)
        meta_dataset: Meta-learning dataset
        grid: Grid object
        meta_epochs: Number of meta-training epochs
        meta_batch_size: Number of tasks per meta-batch
        inner_lr: Inner loop learning rate
        outer_lr: Outer loop learning rate
        inner_steps: Number of inner loop adaptation steps
        device: Device to use
        verbose: Whether to print progress
    
    Returns:
        (meta_generator, evaluator): Meta-trained models
    """
    device = torch.device(device)
    generator.to(device)
    evaluator.to(device)
    evaluator.eval()  # Evaluator is frozen during meta-training
    
    meta_optimizer = optim.Adam(generator.parameters(), lr=outer_lr)
    
    if verbose:
        print("=" * 80)
        print("🧠 Meta-Learning Pre-training (MAML)")
        print("=" * 80)
        print(f"Meta epochs: {meta_epochs}")
        print(f"Meta batch size: {meta_batch_size}")
        print(f"Inner LR: {inner_lr}, Outer LR: {outer_lr}")
        print(f"Inner steps: {inner_steps}")
        print(f"Total tasks: {len(meta_dataset)}")
        print()
    
    rng = np.random.default_rng(42)
    
    for epoch in range(meta_epochs):
        epoch_losses = []
        epoch_qualities = []
        
        # Sample meta-batch of tasks
        tasks = meta_dataset.sample_tasks(meta_batch_size, rng)
        
        meta_optimizer.zero_grad()
        
        for task in tasks:
            # Inner loop: adapt to task
            task_quality, adapted_generator = meta_train_step(
                generator=generator,
                evaluator=evaluator,
                task=task,
                grid=grid,
                inner_lr=inner_lr,
                inner_steps=inner_steps,
                device=device,
            )
            
            epoch_qualities.append(task_quality)
            
            # Outer loop: compute meta-gradient
            # Re-run forward pass with adapted parameters to get gradients
            t_feats = torch.tensor(task['feats'], dtype=torch.float32, device=device)
            
            # Use adapted generator's parameters to compute loss
            weights_adapted = adapted_generator(t_feats)
            pred_quality_adapted = evaluator(t_feats, weights_adapted)
            
            # Meta loss: quality after adaptation
            meta_loss = pred_quality_adapted
            meta_loss.backward()
            
            epoch_losses.append(float(meta_loss.detach()))
        
        # Meta-update
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
        meta_optimizer.step()
        
        if verbose and (epoch + 1) % 5 == 0:
            avg_loss = np.mean(epoch_losses)
            avg_quality = np.mean(epoch_qualities)
            print(f"  Meta-Epoch {epoch+1:3d}/{meta_epochs}  "
                  f"Loss: {avg_loss:.4f}  Avg Quality: {avg_quality:.4f}")
    
    if verbose:
        print()
        print("✅ Meta-learning pre-training completed!")
        print()
    
    return generator, evaluator


def save_pretrained_models(
    generator: EdgeWeightGenerator,
    evaluator: QueryEvaluator,
    save_dir: str,
    model_name: str = "lacsfc_pretrained",
):
    """Save pre-trained models.
    
    Args:
        generator: Generator model
        evaluator: Evaluator model
        save_dir: Directory to save models
        model_name: Base name for saved models
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    gen_path = os.path.join(save_dir, f"{model_name}_generator.pt")
    eval_path = os.path.join(save_dir, f"{model_name}_evaluator.pt")
    
    torch.save(generator.state_dict(), gen_path)
    torch.save(evaluator.state_dict(), eval_path)
    
    print(f"✅ Saved pre-trained models:")
    print(f"  Generator: {gen_path}")
    print(f"  Evaluator: {eval_path}")


def load_pretrained_models(
    generator: EdgeWeightGenerator,
    evaluator: QueryEvaluator,
    save_dir: str,
    model_name: str = "lacsfc_pretrained",
    device: str = 'cpu',
) -> Tuple[EdgeWeightGenerator, QueryEvaluator]:
    """Load pre-trained models.
    
    Args:
        generator: Generator model (architecture)
        evaluator: Evaluator model (architecture)
        save_dir: Directory containing saved models
        model_name: Base name of saved models
        device: Device to load models to
    
    Returns:
        (generator, evaluator): Loaded models
    """
    import os
    
    gen_path = os.path.join(save_dir, f"{model_name}_generator.pt")
    eval_path = os.path.join(save_dir, f"{model_name}_evaluator.pt")
    
    device = torch.device(device)
    
    generator.load_state_dict(torch.load(gen_path, map_location=device))
    evaluator.load_state_dict(torch.load(eval_path, map_location=device))
    
    generator.to(device)
    evaluator.to(device)
    
    print(f"✅ Loaded pre-trained models:")
    print(f"  Generator: {gen_path}")
    print(f"  Evaluator: {eval_path}")
    
    return generator, evaluator


def finetune_from_pretrained(
    generator: EdgeWeightGenerator,
    evaluator: QueryEvaluator,
    target_queries: List,
    grid: Grid,
    encoder: SceneEncoder,
    finetune_epochs: int = 10,
    lr: float = 1e-4,
    device: str = 'cpu',
    verbose: bool = True,
) -> Tuple[EdgeWeightGenerator, QueryEvaluator]:
    """Fine-tune pre-trained models on target workload.
    
    Args:
        generator: Pre-trained generator
        evaluator: Pre-trained evaluator
        target_queries: Target query workload
        grid: Grid object
        encoder: Scene encoder
        finetune_epochs: Number of fine-tuning epochs
        lr: Learning rate
        device: Device to use
        verbose: Whether to print progress
    
    Returns:
        (generator, evaluator): Fine-tuned models
    """
    device = torch.device(device)
    generator.to(device)
    evaluator.to(device)
    
    optimizer = optim.Adam(generator.parameters(), lr=lr)
    
    if verbose:
        print("🔧 Fine-tuning on target workload...")
    
    # Encode target workload
    wl = QueryWorkload(target_queries, grid)
    feats = encoder.encode(wl)
    t_feats = torch.tensor(feats, dtype=torch.float32, device=device)
    
    for epoch in range(finetune_epochs):
        optimizer.zero_grad()
        
        # Generate weights
        weights_gen = generator(t_feats)
        
        # Evaluate quality
        pred_quality = evaluator(t_feats, weights_gen)
        
        # Loss: minimize predicted quality
        loss = pred_quality
        loss.backward()
        optimizer.step()
        
        if verbose and (epoch + 1) % 5 == 0:
            with torch.no_grad():
                weights_np = weights_gen.cpu().numpy()
                seg, span_per_cell, page_amp = compute_quality_mvp(
                    grid, weights_np, target_queries, b=256, use_precomputed=PRECOMPUTED_AVAILABLE
                )
                true_quality = 0.60 * seg + 0.15 * span_per_cell + 0.25 * page_amp
            
            print(f"  Epoch {epoch+1:3d}/{finetune_epochs}  "
                  f"Loss: {float(loss):.4f}  True Quality: {true_quality:.4f}")
    
    if verbose:
        print("✅ Fine-tuning completed!")
        print()
    
    return generator, evaluator


# ==============================================================================
# D-4 v2: Structural EG pretraining + actor-critic meta training (NeuralSFC-style)
# ==============================================================================

def _task_quality(grid: Grid, weights: np.ndarray, queries, b: int = 256) -> float:
    """Scalar real-quality on a task: paper representative (0.60·seg + 0.15·span + 0.25·pa)."""
    seg, span_per_cell, pa = compute_quality_mvp(
        grid, weights.astype(np.float32), queries, b=b,
        use_precomputed=PRECOMPUTED_AVAILABLE,
    )
    return 0.60 * float(seg) + 0.15 * float(span_per_cell) + 0.25 * float(pa)


def pretrain_evaluator_structural(
    evaluator: QueryEvaluator,
    meta_dataset: "MetaLearningDataset",
    grid: Grid,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 8,
    n_perturb_per_task: int = 2,
    perturb_sigma: float = 0.25,
    b: int = 256,
    device: str = 'cpu',
    seed: int = 0,
    verbose: bool = True,
    perturb_scales: Tuple[float, ...] = None,
    n_random_per_task: int = 0,
    perturb_around_hilbert: bool = False,
    interp_steps: int = 0,
) -> Tuple[QueryEvaluator, dict]:
    """T2. Supervised EG pretraining with three weight sources per task:

    (i)   Hilbert-consistent weights (`hilbert_edge_weights(grid)`)
    (ii)  LACSFC-H weights (``task['base_weights']`` = ``dual_edge_weights()``)
    (iii) Perturbations of (ii): ``w + σ · N(0, I)``  (``n_perturb_per_task`` of them)

    Labels are the **real** scalar quality ``0.60·seg + 0.15·span + 0.25·page_amp``
    (paper representative instance) from ``compute_quality_mvp``.

    This addresses the direct root cause of the D-4 failure: the evaluator was
    previously random-initialised and never touched real-metric ground truth.

    Returns
    -------
    (evaluator, info) : (QueryEvaluator, dict)
        ``info`` contains ``final_train_mse``, ``val_pearson``, ``val_spearman``
        so caller can early-abort if the EG fails to learn.
    """
    import torch.nn as nn
    from .hilbert_init import hilbert_edge_weights

    device = torch.device(device)
    evaluator.to(device).train()
    opt = optim.Adam(evaluator.parameters(), lr=lr)
    mse = nn.MSELoss()
    rng = np.random.default_rng(seed)

    # Precompute Hilbert weights once (grid-dependent only).
    w_hilbert = hilbert_edge_weights(grid).astype(np.float32)

    # Build supervised dataset: [(feats_tensor, weights_tensor, y_real_scalar), ...]
    if verbose:
        print("=" * 70)
        print("🧠 Evaluator structural pretraining (T2)")
        print("=" * 70)
        print(f"  tasks={len(meta_dataset.tasks)}  epochs={epochs}  lr={lr}  "
              f"perturb/task={n_perturb_per_task}  σ={perturb_sigma}")
    samples = []
    t_build = time.time()
    for task in meta_dataset.tasks:
        feats = np.asarray(task['feats'], dtype=np.float32)
        queries = task['queries']
        w_dual = np.asarray(task['base_weights'], dtype=np.float32)
        if w_dual.shape[0] != w_hilbert.shape[0]:
            # Shape guard: should not happen when grid matches, but be defensive.
            continue

        # Anchor 1: Hilbert
        samples.append((feats, w_hilbert.copy(),
                        _task_quality(grid, w_hilbert, queries, b)))
        # Anchor 2: LACSFC-H
        samples.append((feats, w_dual.copy(),
                        _task_quality(grid, w_dual, queries, b)))

        dual_std = float(max(1e-3, w_dual.std()))
        hil_std = float(max(1e-3, w_hilbert.std()))
        # Multi-scale perturbations around LACSFC-H (and optionally Hilbert).
        # A wider, multi-scale spread is critical in high-d (e.g. d=3, ~95k
        # edges) where small perturbations barely move quality, collapsing the
        # label variance and starving the evaluator (low Pearson). Default
        # (perturb_scales=None) reproduces the original single-σ behavior.
        scales = perturb_scales if perturb_scales else (perturb_sigma,)
        for s in scales:
            for _ in range(n_perturb_per_task):
                eps = rng.standard_normal(w_dual.shape).astype(np.float32) * (
                    float(s) * dual_std
                )
                w_pert = w_dual + eps
                samples.append((feats, w_pert,
                                _task_quality(grid, w_pert, queries, b)))
            if perturb_around_hilbert:
                for _ in range(n_perturb_per_task):
                    eps = rng.standard_normal(w_hilbert.shape).astype(np.float32) * (
                        float(s) * hil_std
                    )
                    w_pert = w_hilbert + eps
                    samples.append((feats, w_pert,
                                    _task_quality(grid, w_pert, queries, b)))

        # Random-weight anchors: span the *poor* end of the quality range so
        # the evaluator learns a discriminative gradient, not just local jitter.
        dual_mean = float(w_dual.mean())
        for _ in range(n_random_per_task):
            w_rand = (np.abs(rng.standard_normal(w_dual.shape).astype(np.float32))
                      * dual_std + dual_mean)
            samples.append((feats, w_rand,
                            _task_quality(grid, w_rand, queries, b)))

        # Interpolation ramp between LACSFC-H and Hilbert: a smooth quality
        # gradient between two structurally meaningful endpoints.
        for k in range(1, interp_steps + 1):
            a = k / (interp_steps + 1)
            w_interp = ((1.0 - a) * w_dual + a * w_hilbert).astype(np.float32)
            samples.append((feats, w_interp,
                            _task_quality(grid, w_interp, queries, b)))

    if verbose:
        print(f"  supervised set size = {len(samples)} "
              f"(built in {time.time()-t_build:.1f}s)")

    # Z-normalise targets for stable MSE (predict in normalised space, denormalise at eval).
    ys = np.asarray([s[2] for s in samples], dtype=np.float32)
    y_mean, y_std = float(ys.mean()), float(ys.std() + 1e-6)

    # Train/val split (last 10% as held-out, in task-stratified fashion via shuffle-by-sample).
    n = len(samples)
    perm = rng.permutation(n)
    n_val = max(1, n // 10)
    val_idx = set(perm[:n_val].tolist())

    train_tensors = []
    val_tensors = []
    for i, (f, w, y) in enumerate(samples):
        t_f = torch.tensor(f, dtype=torch.float32, device=device)
        t_w = torch.tensor(w, dtype=torch.float32, device=device)
        t_y = torch.tensor((y - y_mean) / y_std, dtype=torch.float32, device=device)
        (val_tensors if i in val_idx else train_tensors).append((t_f, t_w, t_y, y))

    def _eval_pearson_spearman():
        evaluator.eval()
        preds_n, trues = [], []
        with torch.no_grad():
            for t_f, t_w, _, y_raw in val_tensors:
                p = float(evaluator(t_f, t_w))
                preds_n.append(p * y_std + y_mean)   # denorm
                trues.append(float(y_raw))
        evaluator.train()
        preds_n = np.asarray(preds_n)
        trues = np.asarray(trues)
        if preds_n.std() < 1e-8 or trues.std() < 1e-8:
            return 0.0, 0.0
        pear = float(np.corrcoef(preds_n, trues)[0, 1])
        from scipy.stats import spearmanr
        spe = float(spearmanr(preds_n, trues).correlation)
        return pear, spe

    # Training loop
    n_train = len(train_tensors)
    last_loss = float('inf')
    for ep in range(epochs):
        idx = rng.permutation(n_train)
        ep_loss = 0.0
        nbatches = 0
        for start in range(0, n_train, batch_size):
            batch = idx[start:start + batch_size]
            opt.zero_grad()
            loss = torch.zeros(1, device=device)
            for j in batch:
                t_f, t_w, t_y, _ = train_tensors[j]
                p = evaluator(t_f, t_w)
                loss = loss + (p - t_y) ** 2
            loss = loss / max(1, len(batch))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(evaluator.parameters(), 1.0)
            opt.step()
            ep_loss += float(loss.detach())
            nbatches += 1
        last_loss = ep_loss / max(1, nbatches)
        if verbose and ((ep + 1) % max(1, epochs // 6) == 0 or ep == 0):
            pear, spe = _eval_pearson_spearman()
            print(f"  ep {ep+1:3d}/{epochs}  train_mse(norm)={last_loss:.4f}  "
                  f"val_Pearson={pear:+.3f}  val_Spearman={spe:+.3f}")

    pear, spe = _eval_pearson_spearman()
    # Attach denorm params to evaluator so downstream code can denormalise if needed.
    evaluator._pretrain_y_mean = y_mean  # type: ignore[attr-defined]
    evaluator._pretrain_y_std = y_std    # type: ignore[attr-defined]
    info = dict(
        final_train_mse=last_loss, val_pearson=pear, val_spearman=spe,
        y_mean=y_mean, y_std=y_std, n_train=n_train, n_val=len(val_tensors),
    )
    if verbose:
        print(f"✅ EG pretraining done. Pearson={pear:+.3f}  Spearman={spe:+.3f}")
    return evaluator, info


def normalized_residual_np(w_raw, w_dual, alpha, eps: float = 1e-6):
    """Scale-aligned, zero-mean residual combination (numpy).

    The FG output ``w_raw`` typically carries a large DC offset on a scale
    unrelated to the sparse heuristic ``w_dual`` (e.g. FG mean ~0.5 vs w_dual
    mean ~0.001), so the naive ``alpha*w_raw + w_dual`` buries the heuristic's
    discriminative structure. Here we instead z-normalise ``w_raw`` (remove its
    mean, unit its std) and re-scale to ``w_dual``'s own std, so that
    ``alpha`` controls a genuine zero-mean perturbation *around* the heuristic.
    ``alpha -> 0`` recovers ``w_dual`` exactly.
    """
    w_raw = np.asarray(w_raw, dtype=np.float32)
    w_dual = np.asarray(w_dual, dtype=np.float32)
    r = (w_raw - float(w_raw.mean())) / (float(w_raw.std()) + eps)
    return (w_dual + float(alpha) * (float(w_dual.std()) + eps) * r).astype(np.float32)


def meta_train_v2(
    generator: EdgeWeightGenerator,
    evaluator: QueryEvaluator,
    meta_dataset: "MetaLearningDataset",
    grid: Grid,
    outer_epochs: int = 15,
    meta_batch_size: int = 4,
    fg_steps_per_outer: int = 3,
    eg_steps_per_outer: int = 1,
    lr_g: float = 1e-4,
    lr_e: float = 1e-4,
    use_residual: bool = True,
    residual_alpha: float = 0.1,
    residual_mode: str = 'affine',   # 'affine' (legacy α·w_raw+w_dual) or
                                     # 'normalized' (zero-mean, scale-aligned)
    # --- α adaptive (post-PR3 Step α) -------------------------------------
    # If True, α becomes a trainable scalar parameterised as
    #   α = sigmoid(logit_α) · alpha_max,  init so α(0) ≈ residual_alpha.
    # Co-optimised with FG via opt_g; final value returned in log dict.
    learn_alpha: bool = False,
    alpha_max: float = 0.5,
    lr_alpha: float = 1e-2,
    # ----------------------------------------------------------------------
    trust_region_delta: float = 0.05,
    b: int = 256,
    # --- Step 2 (PR2): ES-hybrid gradient ---------------------------------
    use_hybrid_es: bool = False,
    hybrid_beta: float = 0.7,
    es_sigma: float = 0.05,
    es_n_perturb: int = 8,
    es_every: int = 1,
    # ----------------------------------------------------------------------
    device: str = 'cpu',
    seed: int = 0,
    verbose: bool = True,
) -> Tuple[EdgeWeightGenerator, QueryEvaluator, dict]:
    """T5. NeuralSFC-style actor-critic meta-training.

    Pipeline per outer step (analogous to ``neuralsfc/model.py:441-502``):

    E-phase (EG trainable, FG frozen)
        For each sampled task, compute real quality of:
            ``w_dual``         (task['base_weights'], LACSFC-H anchor)
            ``w_gen_detach``   (FG current output, state distribution)
            ``w_avg``          (batch mean of ``w_gen_detach``)
        MSE-regress evaluator to these three real-quality targets.

    G-phase (FG trainable, EG frozen)
        Build ``w_gen = residual_alpha · FG(feats) + w_dual`` (if residual else
        just ``FG(feats)``), take ``g_loss = evaluator(feats, w_gen)`` and step.

    Trust-region gate
        After the G-phase batch, compute the *real* mean quality on the same
        tasks. If it has degraded by more than ``trust_region_delta`` relative
        to the pre-step value, roll the generator back to the snapshot.

    Top-K snapshot
        Retains the single best (lowest real mean quality) generator state
        seen so far. ``meta_train_v2`` returns that snapshot — guaranteeing
        the final generator is at worst the warmstart baseline.

    Returns
    -------
    (generator, evaluator, log) : (model, model, dict)
        ``log`` contains per-outer-epoch real quality, EG train loss,
        rollback events, and the best real quality.
    """
    import torch.nn as nn

    device = torch.device(device)
    generator.to(device)
    evaluator.to(device)
    rng = np.random.default_rng(seed)

    # --- α state ------------------------------------------------------
    if learn_alpha and use_residual:
        init_frac = min(max(residual_alpha / alpha_max, 1e-3), 1.0 - 1e-3)
        init_logit = float(np.log(init_frac / (1.0 - init_frac)))
        logit_alpha = torch.nn.Parameter(
            torch.tensor(init_logit, dtype=torch.float32, device=device)
        )
        opt_g = optim.Adam(
            [{'params': generator.parameters(), 'lr': lr_g},
             {'params': [logit_alpha], 'lr': lr_alpha}]
        )
    else:
        logit_alpha = None
        opt_g = optim.Adam(generator.parameters(), lr=lr_g)
    opt_e = optim.Adam(evaluator.parameters(), lr=lr_e)

    # De-normalisation params from pretraining (if present).
    y_mean = float(getattr(evaluator, '_pretrain_y_mean', 0.0))
    y_std = float(getattr(evaluator, '_pretrain_y_std', 1.0))

    def _current_alpha() -> float:
        if logit_alpha is None:
            return float(residual_alpha)
        return float(torch.sigmoid(logit_alpha).item() * alpha_max)

    def _gen_weights(t_feats: torch.Tensor, t_w_dual: torch.Tensor) -> torch.Tensor:
        w_raw = generator(t_feats)
        if w_raw.ndim > t_w_dual.ndim:
            w_raw = w_raw.squeeze(-1)
        if not use_residual:
            return w_raw
        alpha_t = (torch.sigmoid(logit_alpha) * alpha_max
                   if logit_alpha is not None else residual_alpha)
        if residual_mode == 'normalized':
            # Zero-mean, scale-aligned residual: remove the FG DC offset and
            # rescale to w_dual's own std so α controls a true perturbation
            # around the heuristic (α->0 recovers w_dual exactly).
            r = (w_raw - w_raw.mean()) / (w_raw.std() + 1e-6)
            return t_w_dual + alpha_t * (t_w_dual.std() + 1e-6) * r
        return alpha_t * w_raw + t_w_dual

    # Baseline: FG-current real quality across all tasks (snapshot selection key).
    def _batch_real_quality(tasks) -> float:
        generator.eval()
        qs = []
        with torch.no_grad():
            for task in tasks:
                t_f = torch.tensor(task['feats'], dtype=torch.float32, device=device)
                t_d = torch.tensor(task['base_weights'], dtype=torch.float32,
                                   device=device)
                w = _gen_weights(t_f, t_d).cpu().numpy()
                qs.append(_task_quality(grid, w, task['queries'], b))
        generator.train()
        return float(np.mean(qs))

    # ------------------------------------------------------------------
    # ES gradient helper (Step 2 of PR2).
    # Estimates ∇_θ E_tasks[real_quality(FG_θ)] via NES with antithetic
    # baseline; variance-reduced by subtracting batch mean quality.
    # Returns a list of flat gradients (one per parameter) or ``None`` if
    # ES is disabled.  Cost: ``es_n_perturb * len(batch_tasks)`` real-metric
    # evaluations per call.
    # ------------------------------------------------------------------
    def _es_gradient(batch_tasks):
        if not use_hybrid_es:
            return None
        params = list(generator.parameters())
        orig = [p.data.clone() for p in params]
        eps_list = []
        m_list = []
        generator.eval()
        with torch.no_grad():
            for _ in range(es_n_perturb):
                eps = [torch.randn_like(p) * es_sigma for p in params]
                for p, e, o in zip(params, eps, orig):
                    p.data = o + e
                m = _batch_real_quality(batch_tasks)
                eps_list.append(eps)
                m_list.append(m)
            for p, o in zip(params, orig):
                p.data = o
        generator.train()
        baseline = float(np.mean(m_list))
        denom = max(1e-6, float(np.std(m_list))) * es_n_perturb * es_sigma
        grads = [torch.zeros_like(p) for p in params]
        for eps, m in zip(eps_list, m_list):
            scale = (m - baseline) / denom
            for g, e in zip(grads, eps):
                g.add_(scale * e)
        return grads

    all_tasks = meta_dataset.tasks
    baseline_q = _batch_real_quality(all_tasks)
    if verbose:
        print("=" * 70)
        msg = ("🔁 meta_train_v2 (actor-critic, residual=%s, α=%.3f"
               % (use_residual, residual_alpha))
        if use_hybrid_es:
            msg += ", ES-hybrid β=%.2f, n_perturb=%d, σ=%.3f" % (
                hybrid_beta, es_n_perturb, es_sigma)
        msg += ")"
        print(msg)
        print("=" * 70)
        print(f"  baseline real q (start) = {baseline_q:.4f}")
    best_q = baseline_q
    best_state = deepcopy(generator.state_dict())
    best_alpha = _current_alpha()
    n_rollback = 0
    log_per_epoch = []

    for ep in range(outer_epochs):
        # Sample a meta-batch (with replacement-allowed if dataset small).
        batch_tasks = [
            all_tasks[i] for i in rng.choice(
                len(all_tasks),
                size=min(meta_batch_size, len(all_tasks)), replace=False,
            )
        ]

        # ---------------- E-phase ----------------
        evaluator.train(); generator.eval()
        e_loss_total = 0.0
        for _ in range(eg_steps_per_outer):
            for task in batch_tasks:
                t_f = torch.tensor(task['feats'], dtype=torch.float32, device=device)
                t_d = torch.tensor(task['base_weights'], dtype=torch.float32,
                                   device=device)
                # FG current weights (detached, numpy for real-quality eval)
                with torch.no_grad():
                    w_gen = _gen_weights(t_f, t_d).detach()
                w_gen_np = w_gen.cpu().numpy()
                w_avg_np = np.full_like(w_gen_np, float(w_gen_np.mean()))
                t_w_avg = torch.tensor(w_avg_np, dtype=torch.float32, device=device)

                # Real-quality labels (cheap precomputed path)
                y_dual = _task_quality(grid, t_d.cpu().numpy(), task['queries'], b)
                y_gen = _task_quality(grid, w_gen_np, task['queries'], b)
                y_avg = _task_quality(grid, w_avg_np, task['queries'], b)

                def _norm(y): return (y - y_mean) / y_std
                t_y_dual = torch.tensor(_norm(y_dual), dtype=torch.float32, device=device)
                t_y_gen = torch.tensor(_norm(y_gen), dtype=torch.float32, device=device)
                t_y_avg = torch.tensor(_norm(y_avg), dtype=torch.float32, device=device)

                opt_e.zero_grad()
                p_dual = evaluator(t_f, t_d)
                p_gen = evaluator(t_f, w_gen)
                p_avg = evaluator(t_f, t_w_avg)
                e_loss = ((p_dual - t_y_dual) ** 2
                          + (p_gen - t_y_gen) ** 2
                          + (p_avg - t_y_avg) ** 2) / 3.0
                e_loss.backward()
                torch.nn.utils.clip_grad_norm_(evaluator.parameters(), 1.0)
                opt_e.step()
                e_loss_total += float(e_loss.detach())

        # ---------------- G-phase ----------------
        evaluator.eval(); generator.train()
        for p in evaluator.parameters():
            p.requires_grad_(False)
        g_loss_total = 0.0
        pre_snapshot = deepcopy(generator.state_dict())
        pre_alpha_logit = (logit_alpha.data.clone() if logit_alpha is not None else None)
        pre_q = _batch_real_quality(batch_tasks)

        for fg_step in range(fg_steps_per_outer):
            opt_g.zero_grad()
            loss = torch.zeros(1, device=device)
            for task in batch_tasks:
                t_f = torch.tensor(task['feats'], dtype=torch.float32, device=device)
                t_d = torch.tensor(task['base_weights'], dtype=torch.float32,
                                   device=device)
                w_gen = _gen_weights(t_f, t_d)
                loss = loss + evaluator(t_f, w_gen)
            loss = loss / max(1, len(batch_tasks))
            loss.backward()
            # ES-hybrid gradient mixing (Step 2 of PR2).
            # p.grad currently holds the EG-gradient. We blend in an unbiased
            # ES gradient on the real metric:  ∇ = β·∇EG + (1-β)·∇ES.
            if use_hybrid_es and (ep % max(1, es_every) == 0):
                g_es_list = _es_gradient(batch_tasks)
                if g_es_list is not None:
                    # Match EG-grad scale so that β is a true convex mix.
                    params = list(generator.parameters())
                    eg_norm = float(sum((p.grad.norm() ** 2).item()
                                        for p in params if p.grad is not None) ** 0.5) + 1e-8
                    es_norm = float(sum((g.norm() ** 2).item()
                                        for g in g_es_list) ** 0.5) + 1e-8
                    rescale = eg_norm / es_norm
                    for p, g_es in zip(params, g_es_list):
                        if p.grad is None:
                            p.grad = (1.0 - hybrid_beta) * rescale * g_es
                        else:
                            p.grad.mul_(hybrid_beta).add_(
                                (1.0 - hybrid_beta) * rescale * g_es
                            )
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            opt_g.step()
            g_loss_total += float(loss.detach())

        for p in evaluator.parameters():
            p.requires_grad_(True)

        # ---------------- Trust-region gate ----------------
        post_q = _batch_real_quality(batch_tasks)
        rolled_back = False
        if post_q > (1.0 + trust_region_delta) * pre_q:
            generator.load_state_dict(pre_snapshot)
            if logit_alpha is not None and pre_alpha_logit is not None:
                logit_alpha.data.copy_(pre_alpha_logit)
            rolled_back = True
            n_rollback += 1

        # ---------------- Top-K snapshot ----------------
        full_q = _batch_real_quality(all_tasks)
        cur_alpha = _current_alpha()
        if full_q < best_q:
            best_q = full_q
            best_state = deepcopy(generator.state_dict())
            best_alpha = cur_alpha

        log_per_epoch.append(dict(
            epoch=ep + 1, e_loss=e_loss_total,
            g_loss=g_loss_total, pre_q=pre_q, post_q=post_q,
            rolled_back=rolled_back, full_q=full_q, best_q=best_q,
            alpha=cur_alpha,
        ))
        if verbose and ((ep + 1) % max(1, outer_epochs // 5) == 0 or ep == 0):
            tag = " (rollback)" if rolled_back else ""
            alpha_tag = f"  α={cur_alpha:.3f}" if logit_alpha is not None else ""
            print(f"  ep {ep+1:3d}/{outer_epochs}  "
                  f"e_loss={e_loss_total:.3f}  g_loss={g_loss_total:+.3f}  "
                  f"pre→post={pre_q:.4f}→{post_q:.4f}{tag}  "
                  f"full_q={full_q:.4f}  best={best_q:.4f}{alpha_tag}")

    # Restore best-seen FG (Top-K with K=1) and the α that produced it.
    generator.load_state_dict(best_state)
    if logit_alpha is not None:
        frac = min(max(best_alpha / alpha_max, 1e-3), 1.0 - 1e-3)
        logit_alpha.data.fill_(float(np.log(frac / (1.0 - frac))))
    final_alpha = _current_alpha()
    if verbose:
        alpha_msg = (f"  learned α={final_alpha:.3f}"
                     if logit_alpha is not None else "")
        print(f"✅ meta_train_v2 done. baseline={baseline_q:.4f} → "
              f"best={best_q:.4f} ({(best_q/baseline_q - 1)*100:+.2f}%)  "
              f"rollbacks={n_rollback}/{outer_epochs}{alpha_msg}")

    return generator, evaluator, dict(
        baseline_q=baseline_q, best_q=best_q,
        improvement_pct=(best_q / baseline_q - 1) * 100,
        n_rollback=n_rollback, log=log_per_epoch,
        final_alpha=final_alpha,
    )
