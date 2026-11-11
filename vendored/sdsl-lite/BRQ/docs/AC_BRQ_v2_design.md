# AC-BRQ v2: N-Dimensional Boundary Region Query for AC Space-Filling Curves

Stable baseline summary:

```text
docs/AC_BRQ_v2_final_summary.md
```

Current status:

```text
AC-BRQ v2 baseline + hybrid query + space/compression analysis
```

This version is frozen as the stable baseline. Further backend variants should be treated as future work.

## 1. Problem Definition

Let the discrete grid space be

```text
G = {0, 1, ..., n - 1}^d
```

where `d` is the dimension and `n` is the grid size on each axis. The number of cells is

```text
M = n^d
```

Let an adjacency-continuous space-filling curve be a bijection

```text
S : {0, 1, ..., M - 1} -> G
```

where `S(i)` is the cell visited by curve code `i`.

The AC property requires

```text
∀ i ∈ {1, ..., M - 1}, ||S(i) - S(i - 1)||_1 = 1
```

Given an axis-aligned query box

```text
Q = ∏_{k=0}^{d-1} [low_k, high_k]
```

the goal is to return all curve codes whose cells are inside `Q`, represented as maximal continuous code intervals:

```text
[a_1, b_1], [a_2, b_2], ..., [a_t, b_t]
```

such that

```text
S(i) ∈ Q  iff  i ∈ ⋃_j [a_j, b_j]
```

and each interval is maximal.

---

## 2. Core Observation

The AC property implies that a curve can enter or leave a query box only through the box boundary.

For an entrance code `i`:

```text
S(i) ∈ Q, S(i - 1) ∉ Q
```

Because `S(i - 1)` and `S(i)` are adjacent, `S(i)` must lie on an inner boundary face of `Q`.

For an exit code `i`:

```text
S(i) ∈ Q, S(i + 1) ∉ Q
```

Because `S(i)` and `S(i + 1)` are adjacent, `S(i + 1)` must lie on the outer neighboring hyperplane of some boundary face of `Q`.

Therefore, AC-BRQ does not scan the full query volume. It detects boundary-crossing events using only:

```text
inner boundary faces of Q
outer neighboring hyperplanes of Q
```

---

## 3. Entrance and Exit Definitions

The entrance set is defined as

```text
E = { i | S(i) ∈ Q and (i = 0 or S(i - 1) ∉ Q) }
```

The exit set is defined as

```text
X = { i | S(i) ∈ Q and (i = M - 1 or S(i + 1) ∉ Q) }
```

The endpoint rules are therefore:

```text
if S(0) ∈ Q, add 0 to E
if S(M - 1) ∈ Q, add M - 1 to X
```

The first rule is needed because code `0` has no predecessor. The second rule is needed because code `M - 1` has no successor, so no outside cell can witness the final exit.

---

## 4. Direction Encoding

For each non-start code `i > 0`, define its incoming direction:

```text
dir(i) = S(i - 1) - S(i)
```

Because of the AC property,

```text
dir(i) ∈ { -e_0, +e_0, -e_1, +e_1, ..., -e_{d-1}, +e_{d-1} }
```

The implementation encodes directions as:

```text
directionId(axis, sign) = 2 * axis + (sign > 0 ? 1 : 0)
```

where `sign` is the sign of the predecessor relative to the current cell along `axis`.

Thus:

```text
sign = -1 means predecessor is on the negative side of the current cell
sign = +1 means predecessor is on the positive side of the current cell
```

---

## 5. N-Dimensional Boundary Rules

For each axis `k`, the query box has two faces.

### 5.1 Lower Face

The lower face is

```text
x_k = low_k
```

If `low_k = 0`, there is no grid cell outside the lower side, so ordinary crossing detection for this side is skipped.

Otherwise:

```text
Entrance:
  search inner face x_k = low_k
  required incoming direction = -e_k

Exit:
  search outer hyperplane x_k = low_k - 1
  required incoming direction = +e_k
  map the matched outer cell back to x_k = low_k
```

### 5.2 Upper Face

The upper face is

```text
x_k = high_k
```

If `high_k = n - 1`, there is no grid cell outside the upper side, so ordinary crossing detection for this side is skipped.

Otherwise:

```text
Entrance:
  search inner face x_k = high_k
  required incoming direction = +e_k

Exit:
  search outer hyperplane x_k = high_k + 1
  required incoming direction = -e_k
  map the matched outer cell back to x_k = high_k
```

---

## 6. SDSL Hyperplane Rank/Select Index

A naive row-major bitvector cannot efficiently answer arbitrary N-dimensional face queries because a face is generally not one contiguous row-major interval.

AC-BRQ v2 therefore uses a hyperplane-local index:

```text
direction × axis × slice -> bitvector
```

For each direction `δ`, axis `k`, and slice value `c`, define a local bitvector:

```text
B[δ][k][c]
```

This bitvector stores the `(d - 1)`-dimensional hyperplane

```text
x_k = c
```

using local coordinates obtained by removing axis `k`.

A bit is set iff the corresponding cell has incoming direction `δ`.

Each bitvector is equipped with SDSL:

```text
rank1()
select1()
```

implemented by:

```cpp
sdsl::rank_support_v<1>
sdsl::select_support_mcl<1>
```

A face query is decomposed into one or more contiguous local ranges. For each range `[l, r]`:

```text
cnt = rank1(r + 1) - rank1(l)
positions = select1(rank1(l) + 1), ..., select1(rank1(r + 1))
```

Each selected local position is converted back to an N-dimensional cell coordinate and then to its curve code.

---

## 7. Query Algorithm

Input:

```text
SFC S
query box Q
```

Output:

```text
maximal code intervals inside Q
```

Algorithm:

```text
E = empty set
X = empty set

if S(0) ∈ Q:
    E.add(0)

if S(M - 1) ∈ Q:
    X.add(M - 1)

for each axis k:
    if low_k > 0:
        collect entrances on inner lower face x_k = low_k using direction -e_k
        collect exits on outer lower hyperplane x_k = low_k - 1 using direction +e_k
        map outer exit witnesses back to x_k = low_k

    if high_k < n - 1:
        collect entrances on inner upper face x_k = high_k using direction +e_k
        collect exits on outer upper hyperplane x_k = high_k + 1 using direction -e_k
        map outer exit witnesses back to x_k = high_k

sort and deduplicate E
sort and deduplicate X

return [E_0, X_0], [E_1, X_1], ..., [E_{t-1}, X_{t-1}]
```

---

## 8. Correctness Proof

### Lemma 1: Every true entrance is found

Let `i ∈ E`.

If `i = 0`, then `S(0) ∈ Q`, and the endpoint rule inserts `0` into `E`.

If `i > 0`, then

```text
S(i) ∈ Q and S(i - 1) ∉ Q
```

By AC,

```text
||S(i) - S(i - 1)||_1 = 1
```

Hence `S(i)` must lie on one inner boundary face of `Q`, and its incoming direction points from outside to inside. The corresponding face rule searches exactly this face and direction, so `i` is found.

### Lemma 2: Every reported entrance is true

If the algorithm reports an entrance from an inner face, the matched cell is inside `Q`, and its predecessor lies immediately outside the face. Thus it satisfies

```text
S(i) ∈ Q and S(i - 1) ∉ Q
```

If the algorithm reports `0`, then it checked `S(0) ∈ Q`. Since `0` has no predecessor, it is a valid entrance.

Therefore, the reported entrance set equals the true entrance set.

### Lemma 3: Every true exit is found

Let `i ∈ X`.

If `i = M - 1`, then `S(M - 1) ∈ Q`, and the endpoint rule inserts `M - 1` into `X`.

If `i < M - 1`, then

```text
S(i) ∈ Q and S(i + 1) ∉ Q
```

By AC,

```text
||S(i + 1) - S(i)||_1 = 1
```

Thus `S(i + 1)` lies on the outer neighboring hyperplane of one boundary face of `Q`. Its incoming direction points from the inside cell `S(i)` to the outside cell `S(i + 1)`. The corresponding outer hyperplane rule searches exactly this direction and maps the witness back to `S(i)`, so `i` is found.

### Lemma 4: Every reported exit is true

If the algorithm reports an exit via an outer hyperplane witness, the witness cell is outside `Q`, and its predecessor is the adjacent cell inside `Q`. Therefore the predecessor code `i` satisfies

```text
S(i) ∈ Q and S(i + 1) ∉ Q
```

If the algorithm reports `M - 1`, then it checked `S(M - 1) ∈ Q`. Since `M - 1` has no successor, it is a valid exit.

Therefore, the reported exit set equals the true exit set.

### Lemma 5: Sorted entrances and exits pair one-to-one

Define the Boolean sequence

```text
A[i] = 1 iff S(i) ∈ Q
```

Entrances are exactly the starts of maximal runs of `1` in `A`. Exits are exactly the ends of maximal runs of `1` in `A`.

Therefore, entrances and exits alternate in code order and have the same cardinality:

```text
E_0 <= X_0 < E_1 <= X_1 < ... < E_{t-1} <= X_{t-1}
```

Pairing sorted entrances and exits yields exactly the maximal continuous intervals of codes inside `Q`.

### Theorem

AC-BRQ v2 returns exactly all maximal code intervals whose cells lie inside the query box.

Proof follows directly from Lemmas 1-5.

---

## 9. Complexity Analysis

Let

```text
M = n^d
```

### 9.1 Preprocessing Time

The implementation enumerates all cells and computes each non-start cell's incoming direction:

```text
O(M · d)
```

It also writes the direction event into `d` hyperplane-local indexes, one per possible query axis:

```text
O(M · d)
```

Total preprocessing time:

```text
O(M · d)
```

### 9.2 Space Complexity

There are:

```text
2d directions
```

For each direction, the implementation stores indexes for:

```text
d axes × n slices per axis × n^(d-1) bits per slice
```

Therefore the raw bitvector space is:

```text
2d · d · n · n^(d-1) = 2d^2 · M bits
```

Thus:

```text
O(d^2 M) bits
```

plus SDSL rank/select support overhead.

This v2 index intentionally uses more space than a single global direction bitvector in exchange for efficient hyperplane-local rank/select queries.

### 9.3 Query Time

For each of the `2d` faces, the algorithm performs at most one entrance query and one exit query, so there are at most:

```text
4d hyperplane-local searches
```

Each hyperplane-local face box is decomposed into local contiguous ranges.

Let:

```text
R(Q) = total number of local ranges generated for all searched faces
K = number of reported entrance/exit events
```

Using SDSL rank/select, each local range costs:

```text
O(1 + number of 1s in the range)
```

Thus boundary event detection costs:

```text
O(R(Q) + K)
```

Sorting and deduplication costs:

```text
O(K log K)
```

Total query time:

```text
O(R(Q) + K log K)
```

If events are collected in sorted order or merged by code order, this can be improved toward:

```text
O(R(Q) + K)
```

### 9.4 Comparison with Brute Force

Brute force scans the whole query volume:

```text
O(|Q|) = O(∏_k q_k)
```

where

```text
q_k = high_k - low_k + 1
```

AC-BRQ v2 depends on boundary-crossing events and local face decomposition rather than full volume size.

---

## 10. Implementation Files

The v2 implementation is contained in:

```text
BRQ/code/ac_brq_nd.hpp
BRQ/code/ac_brq_nd_example.cpp
BRQ/code/compile_ac_brq_nd.sh
```

It reuses N-dimensional AC SFC definitions from:

```text
BRQ/code/ac_sfc_nd.hpp
```

The legacy `ac_window_query*` prototype files were removed after AC-BRQ v2 was frozen as the stable baseline, to avoid confusion with the current implementation.

---

## 11. Verification Status

The following command compiles v2:

```bash
cd BRQ/code
bash compile_ac_brq_nd.sh
```

The following command runs the example and brute-force comparison:

```bash
cd BRQ/code
./ac_brq_nd_example
```

Current verified cases:

```text
SnakeND<2>, HilbertND<2>
SnakeND<3>, HilbertND<3>
SnakeND<4>, HilbertND<4>
```

The example covers:

```text
full-space boxes
single-cell boxes
boxes containing code 0
boxes containing maxCode
endpoint neighborhoods
boundary-touching boxes
center boxes
random boxes
```

Observed result:

```text
AC-BRQ v2 summary: total fails = 0
```

The stress test can be compiled and executed by:

```bash
cd BRQ/code
bash compile_ac_brq_nd_stress.sh
./ac_brq_nd_stress
```

The stress test covers more random boxes, degenerate boxes, boundary-touching boxes, and endpoint boxes.

Current stress result:

```text
SnakeND<2>,   grid = 16^2,  boxes = 1017, fails = 0
HilbertND<2>, grid = 16^2,  boxes = 1017, fails = 0
SnakeND<3>,   grid = 8^3,   boxes = 623,  fails = 0
HilbertND<3>, grid = 8^3,   boxes = 623,  fails = 0
SnakeND<4>,   grid = 4^4,   boxes = 429,  fails = 0
HilbertND<4>, grid = 4^4,   boxes = 429,  fails = 0
SnakeND<5>,   grid = 4^5,   boxes = 285,  fails = 0
HilbertND<5>, grid = 4^5,   boxes = 285,  fails = 0

AC-BRQ v2 stress summary: total fails = 0
```

The implementation also exposes lightweight index statistics:

```cpp
uint64_t indexBitvectorCount() const;
uint64_t indexRawBits() const;
```

These report the number of hyperplane-local bitvectors and their raw bit storage before SDSL support overhead.

---

## 12. First Benchmark Results

The benchmark can be compiled and executed by:

```bash
cd BRQ/code
bash compile_ac_brq_nd_benchmark.sh
./ac_brq_nd_benchmark
```

It compares AC-BRQ v2 against brute-force segment generation under five workloads:

```text
small   fixed small boxes
medium  fixed medium boxes
large   fixed large boxes
slab    thin high-volume slab boxes
random  random boxes
```

Current benchmark summary:

```text
SnakeND<2>,   grid = 128^2, queries = 2000, speedup = 3.48x  to 86.25x
HilbertND<2>, grid = 128^2, queries = 2000, speedup = 15.03x to 83.97x

SnakeND<3>,   grid = 32^3,  queries = 1000, speedup = 0.93x  to 16.76x
HilbertND<3>, grid = 32^3,  queries = 1000, speedup = 2.12x  to 19.64x

SnakeND<4>,   grid = 8^4,   queries = 600,  speedup = 0.12x  to 2.96x
HilbertND<4>, grid = 8^4,   queries = 600,  speedup = 0.15x  to 3.37x

AC-BRQ v2 benchmark mismatches = 0
```

Main observations:

1. AC-BRQ v2 is especially effective for medium, large, slab, and random windows where brute force must scan many cells.
2. For very small high-dimensional boxes, brute force can be faster because the number of cells is tiny and AC-BRQ still pays boundary/rank/select overhead.
3. The crossover point depends on dimension, box volume, boundary size, and number of output segments.
4. The benchmark confirms correctness against brute force for every measured query.

---

## 13. Hybrid Query Optimization

The benchmark shows that AC-BRQ v2 is much faster than brute force for medium and large windows, but can be slower for tiny high-dimensional windows.

To handle this case, the implementation provides a hybrid query interface:

```cpp
std::vector<std::pair<uint64_t, uint64_t>> querySegmentsHybrid(
    const QueryBoxND<Dim>& box,
    uint64_t bruteForceVolumeThreshold) const;
```

The rule is:

```text
if volume(box) <= bruteForceVolumeThreshold:
    use bruteForceSegments()
else:
    use querySegments()
```

This does not change the AC-BRQ v2 core algorithm. It is a practical dispatch layer that chooses the cheaper method for very small query boxes.

The benchmark now reports three timings:

```text
brqUs      AC-BRQ v2 only
hybridUs   threshold-based hybrid dispatch
bruteUs    brute-force baseline
```

With `bruteForceVolumeThreshold = 128`, the observed result is:

```text
AC-BRQ v2 hybrid benchmark mismatches = 0
```

Representative observations:

```text
4D Snake small:    brqUs = 293,  hybridUs = 37,  bruteUs = 33
4D Hilbert small:  brqUs = 326,  hybridUs = 54,  bruteUs = 50
4D Snake medium:   brqUs = 990,  hybridUs = 311, bruteUs = 298
4D Hilbert medium: brqUs = 1386, hybridUs = 946, bruteUs = 911
```

So the hybrid strategy significantly improves cases where AC-BRQ has too much overhead relative to query volume.

However, the threshold is workload-dependent:

```text
3D Hilbert small with volume 64:
    brqUs    ~= 3120
    hybridUs ~= 6664
    bruteUs  ~= 6652
```

Here AC-BRQ is already faster than brute force, so forcing brute force with threshold 128 hurts performance.

Conclusion:

1. Hybrid query is correct and preserves exact output.
2. A fixed threshold is simple but not universally optimal.
3. The threshold should be tunable by dimension, curve, grid size, and workload.
4. A future adaptive policy can estimate whether `volume(box)` or `boundary_size(box)` is the better dispatch criterion.

---

## 14. Hyperplane Index Space Report

The space report can be compiled and executed by:

```bash
cd BRQ/code
bash compile_ac_brq_nd_space_report.sh
./ac_brq_nd_space_report
```

For AC-BRQ v2, the hyperplane-local index stores one bitvector for every:

```text
direction × face_axis × slice
```

Therefore:

```text
bitvector_count = 2 * d * d * grid
plane_size      = grid^(d-1)
raw_bits        = 2 * d^2 * grid^d
                = 2 * d^2 * M
```

where:

```text
M = grid^d
```

So the raw index cost per cell is exactly:

```text
2 * d^2 bits/cell
```

For common dimensions:

```text
d = 2  ->  8  raw bits/cell
d = 3  ->  18 raw bits/cell
d = 4  ->  32 raw bits/cell
d = 5  ->  50 raw bits/cell
d = 6  ->  72 raw bits/cell
```

Representative output:

```text
dim  grid    cells       bv_count    plane       bpc   rawMiB  totalMiB*
2    1024    1.05M       8.19K       1.02K       8.0   1.000   1.750
3    100     1.00M       1.80K       10.00K      18.0  2.146   3.755
4    32      1.05M       1.02K       32.77K      32.0  4.000   7.000
5    16      1.05M       800.00      65.54K      50.0  6.250   10.938
6    10      1.00M       720.00      100.00K     72.0  8.583   15.020
```

The `totalMiB*` column is a rough estimate using:

```text
rank support   ~= 25% of raw bits
select support ~= 50% of raw bits
```

This estimate is not a measured SDSL allocation size, but it gives a useful first-order comparison.

Implication:

1. The current index is compact enough for moderate grids.
2. The index grows quadratically with dimension in bits per cell.
3. A compressed or lower-space variant is still useful for high-dimensional or memory-constrained settings.

---

## 15. Compression and Low-Space Prototype

The compression design is documented separately in:

```text
BRQ/docs/AC_BRQ_v2_compression_design.md
```

The prototype report can be compiled and executed by:

```bash
cd BRQ/code
bash compile_ac_brq_nd_compression_report.sh
./ac_brq_nd_compression_report
```

The prototype compares:

```text
plain bit_vector + rank/select
sd_vector + rank/select
rrr_vector
low-space face enumeration fallback
```

### 15.1 Measured plain bit_vector vs sd_vector

The prototype uses synthetic bitvectors of length:

```text
2^20 = 1,048,576 bits
```

and tests densities from `0.5%` to `50%`.

Representative results:

```text
density = 0.005
plain:     180.09 KiB, 1.41 bits/bit, rankUs = 469,  selectUs = 895
sd_vector:   7.28 KiB, 0.06 bits/bit, rankUs = 7295, selectUs = 5188

density = 0.10
plain:     173.49 KiB, 1.36 bits/bit, rankUs = 460,  selectUs = 6785
sd_vector:  95.65 KiB, 0.75 bits/bit, rankUs = 8194, selectUs = 5599

density = 0.50
plain:     175.02 KiB, 1.37 bits/bit, rankUs = 450,  selectUs = 5786
sd_vector: 300.46 KiB, 2.35 bits/bit, rankUs = 7901, selectUs = 6257
```

Observed trend:

1. `sd_vector` is very space-efficient for very sparse bitvectors.
2. `sd_vector` rank is much slower than plain `rank_support_v`.
3. `sd_vector` can become larger than plain bitvectors when density is high.
4. For AC-BRQ v2, `sd_vector` is most promising for high dimensions or sparse slices.

### 15.2 RRR status

`rrr_vector` is theoretically a good candidate for entropy-compressed hyperplane bitvectors.

However, in the current local SDSL tree, `rrr_vector.hpp` includes a locally modified `rrr_vector_15.hpp`, and that specialization currently breaks compilation of the report path.

Therefore:

```text
RRR is a design candidate, but not yet a validated implementation candidate in this local tree.
```

Recommended next step for RRR:

1. Restore or isolate a clean upstream SDSL `rrr_vector` implementation.
2. Re-run the compression report with `rrr_vector<127>`.
3. Only then consider integrating RRR into AC-BRQ v2.

### 15.3 Low-space face enumeration fallback

The low-space fallback stores only full-space direction bitvectors:

```text
direction -> bitvector over all M cells
```

Raw space:

```text
2 * d * M bits
```

Compared with hyperplane-local indexing:

```text
2 * d^2 * M bits
```

Theoretical raw-space saving factor:

```text
d
```

Representative comparison:

```text
dim  hyper_bpc  fallback_bpc  saving
2    8.0        4.0           2.0x
3    18.0       6.0           3.0x
4    32.0       8.0           4.0x
5    50.0       10.0          5.0x
6    72.0       12.0          6.0x
```

This fallback is attractive when memory is the bottleneck, but query time becomes proportional to face area rather than rank/select output size.

### 15.4 Current recommendation

1. Keep plain `bit_vector` as the default high-speed implementation.
2. Add an optional `sd_vector` compressed implementation only after measuring real AC-BRQ slice densities.
3. Treat RRR as pending until the local SDSL specialization is repaired or isolated.
4. Keep face enumeration fallback as a low-memory design path, not as the default fast query path.
