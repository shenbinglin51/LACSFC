"""Surrogate Evaluator for LACSFCF training.

Approximates the non-differentiable mean_seg metric from:
  - scene edge features  (E, scene_feat_dim)
  - edge weights         (E,)

Used as a differentiable proxy for the Generator's training signal.
Trained independently via supervised regression on (scene, weights) → mean_seg.
"""
import torch
import torch.nn as nn
from typing import List


class QueryEvaluator(nn.Module):
    """MLP surrogate: (edge_feats, edge_weights) → predicted mean_seg.

    Input per edge: [scene_feat | weight_scalar]  dim = input_dim + 1
    Global readout: mean-pool over all edges then MLP → scalar.

    Args:
        input_dim:   scene feature dimension per edge
        hidden_dim:  hidden layer width
        num_layers:  number of hidden layers
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 3):
        super().__init__()
        in_dim = input_dim + 1   # +1 for the weight scalar

        # Per-edge encoder
        edge_layers: List[nn.Module] = []
        cur = in_dim
        for _ in range(num_layers):
            edge_layers += [nn.Linear(cur, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()]
            cur = hidden_dim
        edge_layers.append(nn.Linear(cur, hidden_dim))
        self.edge_enc = nn.Sequential(*edge_layers)

        # Readout MLP
        self.readout = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        edge_feats:   torch.Tensor,   # (E, input_dim)
        edge_weights: torch.Tensor,   # (E,)
    ) -> torch.Tensor:
        """Return predicted mean_seg scalar (shape: ())."""
        x = torch.cat([edge_feats, edge_weights.unsqueeze(-1)], dim=-1)  # (E, input_dim+1)
        x = self.edge_enc(x)          # (E, hidden_dim)
        x = x.mean(dim=0)             # (hidden_dim,)   mean-pool over edges
        return self.readout(x).squeeze(-1)  # scalar
