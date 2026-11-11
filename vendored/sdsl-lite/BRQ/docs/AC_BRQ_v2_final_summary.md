# AC-BRQ v2 Stable Baseline Summary

## Status

AC-BRQ v2 is now frozen as a stable baseline.

This version should be treated as:

```text
AC-BRQ v2 baseline + hybrid query + space/compression analysis
```

No additional backend expansion is required for the stable baseline.

---

## Core Semantics

For an adjacency-continuous space-filling curve:

```text
S : {0, 1, ..., M - 1} -> {0, 1, ..., grid - 1}^d
```

and an axis-aligned query box `Q`, AC-BRQ v2 returns all cells in `Q` as maximal continuous code intervals.

Entrance and exit semantics are:

```text
entrance i: S(i) in Q and (i == 0 or S(i - 1) notin Q)
exit i:     S(i) in Q and (i == M - 1 or S(i + 1) notin Q)
```

The implementation uses boundary events on query-box faces to recover all maximal intervals.

---

## Repository Layout

```text
BRQ/
  docs/    AC-BRQ and BRQ-related markdown documents
  code/    AC-BRQ v2 source headers, examples, reports, and compile scripts
```

---

## Stable Implementation Files

Core implementation:

```text
../code/ac_brq_nd.hpp
../code/ac_brq_nd_example.cpp
../code/compile_ac_brq_nd.sh
```

Stress test:

```text
../code/ac_brq_nd_stress.cpp
../code/compile_ac_brq_nd_stress.sh
```

Benchmark:

```text
../code/ac_brq_nd_benchmark.cpp
../code/compile_ac_brq_nd_benchmark.sh
```

Space report:

```text
../code/ac_brq_nd_space_report.cpp
../code/compile_ac_brq_nd_space_report.sh
```

Compression/low-space prototype:

```text
AC_BRQ_v2_compression_design.md
../code/ac_brq_nd_compression_report.cpp
../code/compile_ac_brq_nd_compression_report.sh
```

Design documentation:

```text
AC_BRQ_v2_design.md
AC_BRQ_v2_final_summary.md
```

Removed legacy prototypes:

The old `ac_window_query*` prototype files and scripts have been deleted to avoid confusion with the stable AC-BRQ v2 implementation.

---

## Implemented Query Interfaces

Main AC-BRQ query:

```cpp
std::vector<std::pair<uint64_t, uint64_t>> querySegments(
    const QueryBoxND<Dim>& box) const;
```

Hybrid query:

```cpp
std::vector<std::pair<uint64_t, uint64_t>> querySegmentsHybrid(
    const QueryBoxND<Dim>& box,
    uint64_t bruteForceVolumeThreshold) const;
```

Hybrid dispatch rule:

```text
if volume(box) <= bruteForceVolumeThreshold:
    use bruteForceSegments()
else:
    use querySegments()
```

Current benchmark threshold:

```text
128 cells
```

---

## Correctness Verification

The stable baseline was verified against brute-force enumeration.

Recommended verification commands:

```bash
cd ../code
bash compile_ac_brq_nd.sh && ./ac_brq_nd_example
bash compile_ac_brq_nd_stress.sh && ./ac_brq_nd_stress
bash compile_ac_brq_nd_benchmark.sh && ./ac_brq_nd_benchmark
bash compile_ac_brq_nd_space_report.sh && ./ac_brq_nd_space_report
bash compile_ac_brq_nd_compression_report.sh && ./ac_brq_nd_compression_report
```

Expected correctness indicators:

```text
AC-BRQ v2 summary: total fails = 0
AC-BRQ v2 stress summary: total fails = 0
AC-BRQ v2 hybrid benchmark mismatches = 0
```

---

## Performance Summary

Observed benchmark behavior:

1. AC-BRQ v2 is much faster than brute-force for medium, large, slab, and random windows.
2. Tiny high-dimensional windows may be faster with brute-force.
3. Hybrid query improves small-window cases by dispatching tiny boxes to brute-force.
4. A fixed threshold is correct but not universally optimal.

---

## Space Summary

The current hyperplane-local index stores one bitvector per:

```text
direction × face_axis × slice
```

Raw space:

```text
2 * d^2 * M bits
```

Raw bits per cell:

```text
d = 2 -> 8
d = 3 -> 18
d = 4 -> 32
d = 5 -> 50
d = 6 -> 72
```

The low-space face-enumeration fallback would reduce this to:

```text
2 * d * M bits
```

but would make query time depend on face area.

---

## Compression Prototype Summary

The compression prototype currently validates:

```text
plain bit_vector
sd_vector
low-space face enumeration fallback theory
```

Current conclusion:

1. Plain `bit_vector` remains the default high-speed implementation.
2. `sd_vector` is useful only for sparse slices and is much slower for rank/select.
3. `sd_vector` can become larger than plain bitvectors at high density.
4. RRR remains pending because the local `rrr_vector_15.hpp` specialization appears modified and breaks the report path.

---

## Known Warnings

Compilation under GCC 13 emits SDSL warnings such as:

```text
std::iterator is deprecated
```

These warnings come from the local SDSL headers and do not affect AC-BRQ v2 correctness.

---

## Frozen Baseline Decision

The following are not part of the stable baseline and should be treated as future work:

1. Adaptive hybrid threshold.
2. Real hyperplane slice density profiling.
3. Optional `sd_vector` backend integration.
4. Clean RRR experiment after isolating or restoring upstream SDSL RRR.
5. Full-direction low-space face enumeration backend.
6. Learned-index or prediction-correction variants.

The stable baseline should prioritize correctness, clarity, and reproducibility over additional backend complexity.
