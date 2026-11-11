"""Online Closed-Loop Learning for LACSFC.

Implements online learning mechanisms that continuously improve the model
based on real query feedback.
"""
import time
from typing import List, Tuple, Optional, Dict
from collections import deque
import numpy as np
import torch
import torch.optim as optim

import sys
from .._paths import lacsfcf_root as _lr
sys.path.insert(0, str(_lr()))

from lacsfcf.core.sfc import weights_to_sfc
from lacsfcf.data.grid import Grid
from lacsfcf.data.scene import SceneEncoder
from lacsfcf.data.query_workload import QueryWorkload
from lacsfcf.model.evaluator import QueryEvaluator
from lacsfcf.model.generator import EdgeWeightGenerator

from .trainer import compute_quality_mvp, PRECOMPUTED_AVAILABLE


class OnlineFeedbackBuffer:
    """Buffer for collecting online query feedback."""
    
    def __init__(self, capacity: int = 1000):
        """
        Args:
            capacity: Maximum buffer size
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.query_count = 0
        self.total_quality = 0.0
    
    def add_feedback(
        self,
        queries: List,
        weights: np.ndarray,
        feats: np.ndarray,
        quality: float,
        metrics: Dict[str, float],
    ):
        """Add query feedback to buffer.
        
        Args:
            queries: Query list
            weights: Edge weights used
            feats: Scene features
            quality: Combined quality metric
            metrics: Detailed metrics (mean_seg, mean_span_per_cell, etc.)
        """
        feedback = {
            'queries': queries,
            'weights': weights.copy(),
            'feats': feats.copy(),
            'quality': quality,
            'metrics': metrics.copy(),
            'timestamp': time.time(),
        }
        
        self.buffer.append(feedback)
        self.query_count += len(queries)
        self.total_quality += quality
    
    def get_recent(self, n: int = 100) -> List[Dict]:
        """Get n most recent feedback samples."""
        return list(self.buffer)[-n:]
    
    def get_best(self, n: int = 10) -> List[Dict]:
        """Get n best feedback samples (lowest quality)."""
        sorted_buffer = sorted(self.buffer, key=lambda x: x['quality'])
        return sorted_buffer[:n]
    
    def get_statistics(self) -> Dict[str, float]:
        """Get buffer statistics."""
        if len(self.buffer) == 0:
            return {
                'buffer_size': 0,
                'avg_quality': 0.0,
                'min_quality': 0.0,
                'max_quality': 0.0,
            }
        
        qualities = [fb['quality'] for fb in self.buffer]
        
        return {
            'buffer_size': len(self.buffer),
            'avg_quality': np.mean(qualities),
            'min_quality': np.min(qualities),
            'max_quality': np.max(qualities),
            'std_quality': np.std(qualities),
        }
    
    def clear(self):
        """Clear buffer."""
        self.buffer.clear()
        self.query_count = 0
        self.total_quality = 0.0


class OnlinePerformanceMonitor:
    """Monitor online performance and trigger retraining."""
    
    def __init__(
        self,
        window_size: int = 100,
        degradation_threshold: float = 0.1,
        min_samples_for_retrain: int = 50,
    ):
        """
        Args:
            window_size: Size of sliding window for performance tracking
            degradation_threshold: Threshold for triggering retraining (10% = 0.1)
            min_samples_for_retrain: Minimum samples needed before retraining
        """
        self.window_size = window_size
        self.degradation_threshold = degradation_threshold
        self.min_samples_for_retrain = min_samples_for_retrain
        
        self.quality_history = deque(maxlen=window_size)
        self.baseline_quality = None
        self.retrain_count = 0
    
    def update(self, quality: float):
        """Update with new quality measurement."""
        self.quality_history.append(quality)
        
        # Set baseline on first measurement
        if self.baseline_quality is None and len(self.quality_history) >= 10:
            self.baseline_quality = np.mean(list(self.quality_history)[:10])
    
    def should_retrain(self) -> bool:
        """Check if retraining should be triggered."""
        if len(self.quality_history) < self.min_samples_for_retrain:
            return False
        
        if self.baseline_quality is None:
            return False
        
        # Calculate recent average quality
        recent_quality = np.mean(list(self.quality_history)[-20:])
        
        # Check for degradation
        degradation = (recent_quality - self.baseline_quality) / (self.baseline_quality + 1e-8)
        
        return degradation > self.degradation_threshold
    
    def reset_baseline(self):
        """Reset baseline after retraining."""
        if len(self.quality_history) >= 10:
            self.baseline_quality = np.mean(list(self.quality_history)[-10:])
        self.retrain_count += 1
    
    def get_statistics(self) -> Dict[str, float]:
        """Get monitoring statistics."""
        if len(self.quality_history) == 0:
            return {
                'baseline_quality': 0.0,
                'current_quality': 0.0,
                'degradation': 0.0,
                'retrain_count': self.retrain_count,
            }
        
        current_quality = np.mean(list(self.quality_history)[-10:]) if len(self.quality_history) >= 10 else np.mean(self.quality_history)
        degradation = 0.0
        
        if self.baseline_quality is not None:
            degradation = (current_quality - self.baseline_quality) / (self.baseline_quality + 1e-8)
        
        return {
            'baseline_quality': self.baseline_quality or 0.0,
            'current_quality': current_quality,
            'degradation': degradation,
            'retrain_count': self.retrain_count,
            'samples': len(self.quality_history),
        }


class OnlineLearner:
    """Online closed-loop learner for LACSFC."""
    
    def __init__(
        self,
        generator: EdgeWeightGenerator,
        evaluator: QueryEvaluator,
        grid: Grid,
        encoder: SceneEncoder,
        buffer_capacity: int = 1000,
        batch_size: int = 16,
        lr: float = 1e-4,
        device: str = 'cpu',
        enable_monitoring: bool = True,
        verbose: bool = False,
    ):
        """
        Args:
            generator: Generator model
            evaluator: Evaluator model
            grid: Grid object
            encoder: Scene encoder
            buffer_capacity: Feedback buffer capacity
            batch_size: Batch size for online updates
            lr: Learning rate for online updates
            device: Device to use
            enable_monitoring: Whether to enable performance monitoring
            verbose: Whether to print progress
        """
        self.generator = generator
        self.evaluator = evaluator
        self.grid = grid
        self.encoder = encoder
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.verbose = verbose
        
        # Move models to device
        self.generator.to(self.device)
        self.evaluator.to(self.device)
        
        # Optimizers
        self.gen_optimizer = optim.Adam(self.generator.parameters(), lr=lr)
        self.eval_optimizer = optim.Adam(self.evaluator.parameters(), lr=lr * 0.1)
        
        # Feedback buffer
        self.feedback_buffer = OnlineFeedbackBuffer(capacity=buffer_capacity)
        
        # Performance monitor
        self.monitor = OnlinePerformanceMonitor() if enable_monitoring else None
        
        # Statistics
        self.update_count = 0
        self.retrain_count = 0
    
    def collect_feedback(
        self,
        queries: List,
        weights: np.ndarray,
        metrics: Dict[str, float],
    ):
        """Collect feedback from a query execution.
        
        Args:
            queries: Executed queries
            weights: Edge weights used
            metrics: Query performance metrics
        """
        # Encode queries to features
        wl = QueryWorkload(queries, self.grid)
        feats = self.encoder.encode(wl)
        
        # Paper representative objective: 0.60*seg + 0.15*span_per_cell + 0.25*page_amp.
        quality = (
            0.60 * metrics.get('mean_seg', 0.0) +
            0.15 * metrics.get('mean_span_per_cell', 0.0) +
            0.25 * metrics.get('mean_page_amp', 1.0)
        )
        
        # Add to buffer
        self.feedback_buffer.add_feedback(
            queries=queries,
            weights=weights,
            feats=feats,
            quality=quality,
            metrics=metrics,
        )
        
        # Update monitor
        if self.monitor is not None:
            self.monitor.update(quality)
        
        if self.verbose:
            print(f"[Online] Collected feedback: quality={quality:.4f}, "
                  f"buffer_size={len(self.feedback_buffer.buffer)}")
    
    def online_update(self, n_samples: Optional[int] = None):
        """Perform online update using recent feedback.
        
        Args:
            n_samples: Number of samples to use (default: batch_size)
        """
        if n_samples is None:
            n_samples = self.batch_size
        
        if len(self.feedback_buffer.buffer) < n_samples:
            if self.verbose:
                print(f"[Online] Not enough samples for update "
                      f"({len(self.feedback_buffer.buffer)} < {n_samples})")
            return
        
        # Sample recent feedback
        samples = self.feedback_buffer.get_recent(n_samples)
        
        self.generator.train()
        self.evaluator.train()
        
        total_gen_loss = 0.0
        total_eval_loss = 0.0
        
        for sample in samples:
            feats = sample['feats']
            weights = sample['weights']
            quality = sample['quality']
            
            t_feats = torch.tensor(feats, dtype=torch.float32, device=self.device)
            t_weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
            
            # Update Evaluator
            self.eval_optimizer.zero_grad()
            pred_quality = self.evaluator(t_feats, t_weights)
            target_quality = torch.tensor(quality, dtype=torch.float32, device=self.device)
            eval_loss = (pred_quality - target_quality) ** 2
            eval_loss.backward()
            self.eval_optimizer.step()
            total_eval_loss += float(eval_loss.detach())
            
            # Update Generator
            self.gen_optimizer.zero_grad()
            weights_gen = self.generator(t_feats)
            pred_quality_gen = self.evaluator(t_feats, weights_gen)
            gen_loss = pred_quality_gen  # Minimize predicted quality
            gen_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
            self.gen_optimizer.step()
            total_gen_loss += float(gen_loss.detach())
        
        self.update_count += 1
        
        if self.verbose:
            avg_gen_loss = total_gen_loss / len(samples)
            avg_eval_loss = total_eval_loss / len(samples)
            print(f"[Online] Update #{self.update_count}: "
                  f"Gen Loss={avg_gen_loss:.4f}, Eval Loss={avg_eval_loss:.4f}")
    
    def check_and_retrain(self) -> bool:
        """Check if retraining is needed and perform it.
        
        Returns:
            True if retraining was performed
        """
        if self.monitor is None:
            return False
        
        if not self.monitor.should_retrain():
            return False
        
        if self.verbose:
            stats = self.monitor.get_statistics()
            print(f"\n[Online] Performance degradation detected!")
            print(f"  Baseline: {stats['baseline_quality']:.4f}")
            print(f"  Current:  {stats['current_quality']:.4f}")
            print(f"  Degradation: {stats['degradation']*100:.1f}%")
            print(f"  Triggering retraining...")
        
        # Perform intensive retraining on best samples
        best_samples = self.feedback_buffer.get_best(n=50)
        
        if len(best_samples) < 10:
            if self.verbose:
                print(f"[Online] Not enough samples for retraining")
            return False
        
        self.generator.train()
        self.evaluator.train()
        
        # Multiple passes over best samples
        for epoch in range(10):
            for sample in best_samples:
                feats = sample['feats']
                weights = sample['weights']
                quality = sample['quality']
                
                t_feats = torch.tensor(feats, dtype=torch.float32, device=self.device)
                t_weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
                
                # Update Evaluator
                self.eval_optimizer.zero_grad()
                pred_quality = self.evaluator(t_feats, t_weights)
                target_quality = torch.tensor(quality, dtype=torch.float32, device=self.device)
                eval_loss = (pred_quality - target_quality) ** 2
                eval_loss.backward()
                self.eval_optimizer.step()
                
                # Update Generator
                self.gen_optimizer.zero_grad()
                weights_gen = self.generator(t_feats)
                pred_quality_gen = self.evaluator(t_feats, weights_gen)
                gen_loss = pred_quality_gen
                gen_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
                self.gen_optimizer.step()
        
        # Reset baseline
        self.monitor.reset_baseline()
        self.retrain_count += 1
        
        if self.verbose:
            print(f"[Online] Retraining completed! (Total retrains: {self.retrain_count})")
            print()
        
        return True
    
    def get_statistics(self) -> Dict[str, any]:
        """Get online learning statistics."""
        stats = {
            'update_count': self.update_count,
            'retrain_count': self.retrain_count,
            'buffer_stats': self.feedback_buffer.get_statistics(),
        }
        
        if self.monitor is not None:
            stats['monitor_stats'] = self.monitor.get_statistics()
        
        return stats
    
    def save_checkpoint(self, save_path: str):
        """Save online learner checkpoint."""
        checkpoint = {
            'generator_state': self.generator.state_dict(),
            'evaluator_state': self.evaluator.state_dict(),
            'gen_optimizer_state': self.gen_optimizer.state_dict(),
            'eval_optimizer_state': self.eval_optimizer.state_dict(),
            'update_count': self.update_count,
            'retrain_count': self.retrain_count,
        }
        
        torch.save(checkpoint, save_path)
        
        if self.verbose:
            print(f"[Online] Checkpoint saved: {save_path}")
    
    def load_checkpoint(self, load_path: str):
        """Load online learner checkpoint."""
        checkpoint = torch.load(load_path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state'])
        self.evaluator.load_state_dict(checkpoint['evaluator_state'])
        self.gen_optimizer.load_state_dict(checkpoint['gen_optimizer_state'])
        self.eval_optimizer.load_state_dict(checkpoint['eval_optimizer_state'])
        self.update_count = checkpoint['update_count']
        self.retrain_count = checkpoint['retrain_count']
        
        if self.verbose:
            print(f"[Online] Checkpoint loaded: {load_path}")


def create_online_learner(
    generator: EdgeWeightGenerator,
    evaluator: QueryEvaluator,
    grid: Grid,
    encoder: SceneEncoder,
    config: Optional[Dict] = None,
) -> OnlineLearner:
    """Create an online learner with default or custom configuration.
    
    Args:
        generator: Generator model
        evaluator: Evaluator model
        grid: Grid object
        encoder: Scene encoder
        config: Optional configuration dictionary
    
    Returns:
        OnlineLearner instance
    """
    default_config = {
        'buffer_capacity': 1000,
        'batch_size': 16,
        'lr': 1e-4,
        'device': 'cpu',
        'enable_monitoring': True,
        'verbose': False,
    }
    
    if config is not None:
        default_config.update(config)
    
    return OnlineLearner(
        generator=generator,
        evaluator=evaluator,
        grid=grid,
        encoder=encoder,
        **default_config
    )
