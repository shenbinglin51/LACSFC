"""
LACSFC 学习型曲线适配器模块

将 LACSFC 项目集成到 sfcBenchmark 的 mvp 框架中
"""

from .plugin import LACSFCPlugin, make_lacsfc

__all__ = [
    "LACSFCPlugin",
    "make_lacsfc",
]
