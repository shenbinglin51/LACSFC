"""Compiled C core for the two sequential hot loops of the fast LACSFC pipeline.

Both loops are inherently sequential (each splice mutates nxt/prv and the next
portal search reads them; the cycle traversal follows nxt/prv one step at a
time), so they cannot be vectorized in numpy. This module compiles a tiny C
translation -- performing the EXACT same integer operations in the EXACT same
order as the pure-Python ``apply_mst_fast`` / ``cycle_to_code_map`` -- so the
results stay byte-identical while running 1-2 orders of magnitude faster.

Dependency-free: uses the stdlib ``ctypes`` plus the system ``gcc`` (already
present). If compilation fails for any reason, ``load()`` returns ``None`` and
callers transparently fall back to the pure-Python implementations.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import sys

_C_SOURCE = r"""
#include <stdint.h>

static inline long uf_find(long *parent, long x) {
    while (parent[x] != x) {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    return x;
}

/* Mirrors apply_mst_fast exactly. Returns 0 on success, or (ei+1) if edge ei
 * has no matched portal (caller falls back to the reference merge). */
long merge_run(int32_t *nxt, int32_t *prv, long *parent, long n,
               const long *base_arr, const signed char *k_arr, long m,
               const long *strides, const long *face_off, long fc, long d) {
    for (long ei = 0; ei < m; ++ei) {
        long k = (long)k_arr[ei];
        long base = base_arr[ei];
        long sdim = strides[k];
        const long *fo = face_off + k * fc;  /* sorted face offsets for dim k */

        long pu1 = -1, pv1 = -1, pu2 = -1, pv2 = -1;
        int found = 0;

        for (int require_face = 1; require_face >= 0 && !found; --require_face) {
            for (long j = 0; j < fc && !found; ++j) {
                long u1 = base + fo[j];
                long u2 = u1 + sdim;
                if (u2 >= n) continue;
                if (uf_find(parent, u1) == uf_find(parent, u2)) continue;

                long nb[2];
                nb[0] = nxt[u1];
                nb[1] = prv[u1];
                for (int t = 0; t < 2; ++t) {
                    long v1 = nb[t];
                    if (v1 == -1) continue;
                    if (require_face) {
                        long off = v1 - base;
                        int in_face = 0;
                        for (long jj = 0; jj < fc; ++jj) {
                            if (fo[jj] == off) { in_face = 1; break; }
                        }
                        if (!in_face) continue;
                    }
                    long v2 = v1 + sdim;
                    if (v2 < n && (nxt[u2] == v2 || prv[u2] == v2)
                            && uf_find(parent, v2) == uf_find(parent, u2)) {
                        pu1 = u1; pv1 = v1; pu2 = u2; pv2 = v2;
                        found = 1;
                        break;
                    }
                }
            }
        }

        if (!found) return ei + 1;

        /* remove (u1,v1) and (u2,v2) */
        if (nxt[pu1] == pv1) nxt[pu1] = -1; else prv[pu1] = -1;
        if (nxt[pv1] == pu1) nxt[pv1] = -1; else prv[pv1] = -1;
        if (nxt[pu2] == pv2) nxt[pu2] = -1; else prv[pu2] = -1;
        if (nxt[pv2] == pu2) nxt[pv2] = -1; else prv[pv2] = -1;
        /* add (u1,u2) and (v1,v2) */
        if (nxt[pu1] == -1) nxt[pu1] = pu2; else prv[pu1] = pu2;
        if (nxt[pu2] == -1) nxt[pu2] = pu1; else prv[pu2] = pu1;
        if (nxt[pv1] == -1) nxt[pv1] = pv2; else prv[pv1] = pv2;
        if (nxt[pv2] == -1) nxt[pv2] = pv1; else prv[pv2] = pv1;
        /* union(u1, u2) */
        long ru = uf_find(parent, pu1);
        long rv = uf_find(parent, pu2);
        if (ru != rv) parent[rv] = ru;
    }
    return 0;
}

/* Mirrors cycle_to_code_map: cut start->nxt[start], traverse toward prv.
 * Fills code[id] with sequential position. Returns number of cells visited. */
long cycle_code(const int32_t *nxt, const int32_t *prv, long n, long start,
                int32_t *code) {
    long a0 = nxt[start];
    long b0 = prv[start];
    if (a0 == -1 || b0 == -1) return -1;
    code[start] = 0;
    long prev_id = start;
    long cur = b0;
    long t = 1;
    while (cur != start) {
        code[cur] = (int32_t)t;
        long a = nxt[cur];
        long b = prv[cur];
        long nxt_id = (a != prev_id) ? a : b;
        prev_id = cur;
        cur = nxt_id;
        ++t;
    }
    return t;
}
"""

_lib = None
_loaded = False


def load():
    """Compile (once, cached) and return the ctypes lib, or None on failure."""
    global _lib, _loaded
    if _loaded:
        return _lib
    _loaded = True
    try:
        h = hashlib.sha1(_C_SOURCE.encode()).hexdigest()[:12]
        build_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "_merge_core_build")
        os.makedirs(build_dir, exist_ok=True)
        so_path = os.path.join(build_dir, f"merge_core_{h}.so")
        if not os.path.exists(so_path):
            c_path = os.path.join(build_dir, f"merge_core_{h}.c")
            with open(c_path, "w") as f:
                f.write(_C_SOURCE)
            cc = os.environ.get("CC", "gcc")
            subprocess.run(
                [cc, "-O3", "-march=native", "-shared", "-fPIC",
                 "-o", so_path, c_path],
                check=True, capture_output=True,
            )
        lib = ctypes.CDLL(so_path)
        c_i32p = ctypes.POINTER(ctypes.c_int32)
        c_i64p = ctypes.POINTER(ctypes.c_int64)
        c_i8p = ctypes.POINTER(ctypes.c_int8)
        lib.merge_run.restype = ctypes.c_long
        lib.merge_run.argtypes = [
            c_i32p, c_i32p, c_i64p, ctypes.c_long,
            c_i64p, c_i8p, ctypes.c_long,
            c_i64p, c_i64p, ctypes.c_long, ctypes.c_long,
        ]
        lib.cycle_code.restype = ctypes.c_long
        lib.cycle_code.argtypes = [
            c_i32p, c_i32p, ctypes.c_long, ctypes.c_long, c_i32p,
        ]
        _lib = lib
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[merge_core] C core unavailable, using pure Python: {exc}",
              file=sys.stderr)
        _lib = None
    return _lib
