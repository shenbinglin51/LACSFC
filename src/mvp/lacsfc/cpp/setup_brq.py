from __future__ import annotations

import os
from pathlib import Path

from setuptools import Extension, setup

try:
    import pybind11
except ImportError as exc:
    raise SystemExit("pybind11 is required to build the AC-BRQ v2 extension") from exc

# sdsl-lite/BRQ checkout: set SDSL_BRQ_ROOT, or place the tree at <repo>/vendored/sdsl-lite/BRQ.
BRQ_ROOT = Path(os.environ.get("SDSL_BRQ_ROOT",
                               Path(__file__).resolve().parents[4] / "vendored" / "sdsl-lite" / "BRQ"))
SDSL_ROOT = BRQ_ROOT.parent

ext_modules = [
    Extension(
        "lacsfc_brq",
        [
            "brq_cpp_wrapper.cpp",
            str(SDSL_ROOT / "lib" / "bits.cpp"),
            str(SDSL_ROOT / "lib" / "memory_management.cpp"),
            str(SDSL_ROOT / "lib" / "ram_fs.cpp"),
            str(SDSL_ROOT / "lib" / "util.cpp"),
        ],
        include_dirs=[
            pybind11.get_include(),
            str(BRQ_ROOT / "code"),
            str(SDSL_ROOT / "include"),
            str(SDSL_ROOT / "build" / "include"),
        ],
        language="c++",
        extra_compile_args=["-std=c++17", "-O3", "-Wall", "-Wextra"],
    )
]

setup(
    name="lacsfc_brq",
    version="0.1.0",
    ext_modules=ext_modules,
)
