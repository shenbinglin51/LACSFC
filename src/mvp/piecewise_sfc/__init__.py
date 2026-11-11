"""
PiecewiseSFC (Learned-BMTree) Plugin for sfcBenchmark MVP

This module provides a simplified integration of the Learned-BMTree algorithm
into the sfcBenchmark framework.
"""

from .plugin import PiecewiseSFCPlugin
from .tree import PiecewiseTree

__all__ = ["PiecewiseSFCPlugin", "PiecewiseTree"]
