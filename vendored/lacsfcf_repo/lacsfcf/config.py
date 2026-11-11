"""Configuration dataclass for LACSFCF."""
from dataclasses import dataclass, field
from typing import List


@dataclass
class LacsfcConfig:
    # Grid parameters
    d: int = 2          # number of spatial dimensions
    K: int = 8          # bits per dimension; grid side = 2^K

    # Derived (auto-set, do not pass)
    @property
    def grid_side(self) -> int:
        return 2 ** self.K

    @property
    def num_cells(self) -> int:
        return self.grid_side ** self.d

    @property
    def block_side(self) -> int:
        return 2 ** (self.K - 1)

    @property
    def num_blocks(self) -> int:
        return self.block_side ** self.d

    @property
    def num_dual_edges(self) -> int:
        """Number of edges in the block dual graph."""
        B = self.block_side
        return self.d * (B ** (self.d - 1)) * (B - 1)

    # Query workload
    num_queries: int = 4000
    num_hotspots: int = 3
    hotspot_std: float = 0.08
    win_min_frac: float = 0.05
    win_max_frac: float = 0.25
    page_size: int = 64

    # Model
    scene_feat_dim: int = 16    # features per edge in scene encoding
    gen_hidden: int = 128
    gen_layers: int = 3
    eval_hidden: int = 128
    eval_layers: int = 3

    # Training
    lr: float = 1e-3
    batch_size: int = 16
    epochs: int = 50
    eval_interval: int = 5
    evaluator_pretrain_epochs: int = 10
    evaluator_samples_per_scene: int = 8

    # Misc
    seed: int = 42
    device: str = "cpu"
    out_dir: str = "./output/lacsfcf"
