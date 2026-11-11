"""Edge Weight Generator for LACSFCF.

Architecture: per-edge MLP that maps scene edge features → edge weight scalar.

Input:  (num_edges, scene_feat_dim)  – scene encoding from SceneEncoder
Output: (num_edges,)                 – edge weights for the block dual graph

The weights are fed to nx.maximum_spanning_tree to produce the block MST.
"""
import torch
import torch.nn as nn
from typing import List


class EdgeWeightGenerator(nn.Module):
    """MLP per dual-graph edge: scene_features → edge weight.

    Args:
        input_dim:   dimension of per-edge feature vector (SceneEncoder.feat_dim)
        hidden_dim:  hidden layer width
        num_layers:  number of hidden layers (total depth = num_layers + 1)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 3):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, edge_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            edge_feats: (E, input_dim) float tensor

        Returns:
            weights: (E,) float tensor
        """
        return self.net(edge_feats).squeeze(-1)   # (E,)
