"""Locate external/vendored dependencies.

Resolution order for every dependency: the environment variable wins, then the
copy vendored under ``<repo>/vendored/``. This keeps the code runnable both in
the development workspace (env vars or symlinks) and in the self-contained
reproduction artifact (real directories under vendored/).
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VENDORED = REPO_ROOT / "vendored"


def lacsfcf_root() -> Path:
    """Repo containing the ``lacsfcf`` package (LACSFC core: grid, workloads, weights_to_sfc)."""
    return Path(os.environ.get("LACSFCF_ROOT", VENDORED / "lacsfcf_repo"))


def neuralsfc_root() -> Path:
    """Repo with ``src/`` and ``experiments/`` providing e7_recursive.gbo_rank and ch7_export.build_quilts."""
    return Path(os.environ.get("NEURALSFC_ROOT", VENDORED / "neuralsfc_lacsfc"))


def adacurve_root() -> Path:
    """Optional: the original AdaCurve project (only needed by the alternative engines, not by paper_repro)."""
    return Path(os.environ.get("ADACURVE_ROOT", VENDORED / "adacurve"))


def sdsl_brq_root() -> Path:
    """Optional: sdsl-lite/BRQ checkout needed to build the lacsfc_brq C++ extension."""
    return Path(os.environ.get("SDSL_BRQ_ROOT", VENDORED / "sdsl-lite" / "BRQ"))


def lmsfc_lib_candidates() -> list:
    """Optional: locations of the original LMSFC liblmsfc.so (not needed by the bit-mask MVP path)."""
    env = os.environ.get("LMSFC_LIB")
    out = [env] if env else []
    root = os.environ.get("LMSFC_ROOT")
    if root:
        out += [str(Path(root) / "liblmsfc.so"), str(Path(root) / "build" / "liblmsfc.so")]
    out += ["./liblmsfc.so"]
    return out
