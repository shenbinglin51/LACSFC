"""LACSFC: workload-driven adjacency-continuous space-filling curves."""

from .baselines import hilbert_rank, snake_rank, zorder_rank
from .construct import (Curve, MergeSquares, apply_merges, build_curve, build_curve_auto,
                        build_curve_permuted, cycle_phi,
                        cycle_to_path, delta_phi, initial_cycle, is_ac_path,
                        max_spanning_tree, merge_squares)
from .gray import check_facet_assignment, facet_assignment, gray_sequence
from .local import (LocalResult, candidate_moves, iterated_local_search, local_search,
                    neighbor_table)
from .metrics import (avg_segments, expected_box_size, optimality_gap, path_phi,
                      segment_count, upper_bound_ub0)
from .weights import pair_weights, pair_weights_bruteforce
from .workloads import correlated_boxes, random_boxes

__all__ = [
    "Curve", "MergeSquares", "apply_merges", "avg_segments", "build_curve",
    "build_curve_auto", "build_curve_permuted",
    "check_facet_assignment", "cycle_phi", "cycle_to_path", "delta_phi",
    "expected_box_size", "facet_assignment", "gray_sequence", "hilbert_rank",
    "LocalResult", "initial_cycle", "is_ac_path", "local_search", "max_spanning_tree",
    "merge_squares", "neighbor_table", "candidate_moves", "iterated_local_search",
    "optimality_gap", "pair_weights", "pair_weights_bruteforce", "path_phi",
    "random_boxes", "segment_count", "snake_rank", "upper_bound_ub0", "zorder_rank",
]
