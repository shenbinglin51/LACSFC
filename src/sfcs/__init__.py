"""
Space-Filling Curve implementations
"""
from .sfc_base import SFCBase, TraditionalSFC, LearnedSFC
from .traditional import RowMajor, ColumnMajor, ZCurve, HilbertCurve

# Add learned SFCs as they're implemented
from .learned.lacsfc import LACSFC
from .learned.piecewise_sfc import PiecewiseSFC

# Export all SFCs
__all__ = [
    'SFCBase',
    'TraditionalSFC',
    'LearnedSFC',
    'RowMajor',
    'ColumnMajor',
    'ZCurve',
    'HilbertCurve',
    'LACSFC',
    'PiecewiseSFC'
]
