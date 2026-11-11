# AC-BRQ v2 Compression and Low-Space Design

## Goal

AC-BRQ v2 currently uses hyperplane-local bitvectors:

```text
direction × face_axis × slice -> bitvector
```

The raw space is:

```text
2 * d^2 * M bits
```

where:

```text
M = grid^d
```

This document compares four space-reduction directions:

```text
plain bit_vector
sd_vector
RRR vector
low-space face enumeration fallback
```

---

## 1. Plain bit_vector

Current implementation:

```cpp
sdsl::bit_vector
sdsl::rank_support_v<1>
sdsl::select_support_mcl<1>
```

### Advantages

1. Fast rank/select.
2. Simple implementation.
3. Stable under the current GCC/SDSL environment.
4. Good default for dense or moderately dense bitvectors.

### Disadvantages

1. Stores all zero bits explicitly.
2. Raw space is fixed at `2*d^2*M` bits before support overhead.
3. Not ideal when each hyperplane-local direction bitvector is sparse.

---

## 2. sd_vector

Candidate implementation:

```cpp
sdsl::sd_vector<>
sdsl::sd_vector<>::rank_1_type
sdsl::sd_vector<>::select_1_type
```

`sd_vector` uses Elias-Fano style encoding for sparse bitvectors.

### Advantages

1. Very compact for sparse bitvectors.
2. Provides rank/select support.
3. Usually more robust than the modified local `rrr_vector<15>` code path.

### Disadvantages

1. Can be worse than plain bit_vector for dense bitvectors.
2. Rank/select can be slower.
3. Best suited when the number of 1s is much smaller than the bitvector length.

### Expected AC-BRQ use case

For `direction × face_axis × slice`, each bitvector stores cells whose incoming direction equals one fixed direction.

If directions are roughly balanced, density is about:

```text
1 / (2d)
```

This is:

```text
d = 2 -> 25%
d = 3 -> 16.7%
d = 4 -> 12.5%
d = 5 -> 10%
d = 6 -> 8.3%
```

So `sd_vector` may become more attractive as dimension grows or when actual direction distribution is sparse on many slices.

---

## 3. RRR vector

Candidate implementation:

```cpp
sdsl::rrr_vector<127>
sdsl::rrr_vector<127>::rank_1_type
sdsl::rrr_vector<127>::select_1_type
```

### Advantages

1. Succinct representation based on local entropy.
2. Good for both sparse and moderately dense bitvectors when local structure is compressible.
3. Supports rank/select.

### Disadvantages

1. Rank/select can be slower than plain bit_vector.
2. Construction is more expensive.
3. The local repository's `rrr_vector<15>` path appears modified and should be treated cautiously.

### Recommendation

Use `rrr_vector<127>` for experiments first.

Avoid switching AC-BRQ v2 to `rrr_vector<15>` until the local SDSL specialization is separately verified.

---

## 4. Low-Space Face Enumeration Fallback

Alternative design:

```text
Store only full-space direction bitvectors:
  direction -> bitvector over all cells
```

Raw space:

```text
2 * d * M bits
```

This reduces raw space by a factor of `d` compared with hyperplane-local indexing:

```text
hyperplane-local: 2*d^2*M bits
full-space only:  2*d*M bits
```

### Query method

For each query face, enumerate all cells on that face, compute their code/index, and check the corresponding direction condition.

### Advantages

1. Much lower space.
2. Simple conceptual fallback.
3. Good when memory is more important than query latency.

### Disadvantages

1. Face query time depends on face area.
2. It loses hyperplane-local rank/select acceleration.
3. High-dimensional large faces may be expensive.

### Complexity

Let:

```text
A(Q) = total number of cells on all queried faces
```

Then fallback boundary detection is roughly:

```text
O(A(Q))
```

instead of hyperplane-local:

```text
O(R(Q) + K)
```

where `R(Q)` is the number of local rank/select ranges and `K` is the number of detected boundary events.

---

## 5. Proposed Variants

### Variant A: Current fast index

```text
plain hyperplane-local bit_vector
```

Use when query speed is the priority.

### Variant B: Compressed hyperplane-local index

```text
sd_vector or rrr_vector<127> per hyperplane bitvector
```

Use when memory is constrained but rank/select acceleration is still desired.

### Variant C: Low-space full-direction index

```text
2d full-space direction bitvectors + face enumeration
```

Use when memory is the main bottleneck.

### Variant D: Hybrid index

```text
plain or compressed hyperplane-local index for selected axes/slices
full-space fallback for others
```

Use when workload has predictable query shapes.

---

## 6. Prototype

The prototype report is implemented in:

```text
../code/ac_brq_nd_compression_report.cpp
../code/compile_ac_brq_nd_compression_report.sh
```

It measures:

```text
plain bit_vector + rank/select
sd_vector + rank/select
rrr_vector<127> + rank/select
```

For synthetic bitvectors with different densities.

It also reports theoretical raw-space comparison for:

```text
hyperplane-local index: 2*d^2*M bits
full-space fallback:   2*d*M bits
```

---

## 7. Expected Decision Rule

Initial recommendation:

1. Keep plain `bit_vector` as the default fast implementation.
2. Use `sd_vector` when measured density is below roughly 5-10%.
3. Use `rrr_vector<127>` when local entropy is low but density is not extremely sparse.
4. Use face enumeration fallback when memory must be reduced by a factor of about `d` and slower face queries are acceptable.

The exact thresholds should be chosen from measured prototype results rather than only theory.
