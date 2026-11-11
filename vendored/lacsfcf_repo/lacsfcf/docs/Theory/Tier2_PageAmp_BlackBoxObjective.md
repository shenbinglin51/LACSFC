# Tier 2 Theory: Rigorous Page-Amplification Objective and Black-Box Optimization Boundary

## 1. Purpose and Scope

This document gives the mathematical support for **Tier 2** in the current LACSFC system.

Tier 2 is the direct page-amplification optimization layer:

$$
w
\;\longrightarrow\;
\Phi(w)=\texttt{weights\_to\_sfc(grid, w)}
\;\longrightarrow\;
\pi_w
\;\longrightarrow\;
\text{sorted physical layout}
\;\longrightarrow\;
\text{exact intervals}
\;\longrightarrow\;
\mathrm{PA}.
$$

The rigorous claim of this document is **not** that Tier 2 globally solves the page-amplification minimization problem. The correct claim is:

$$
\boxed{
\text{Tier 2 defines and empirically optimizes a benchmark-aligned page-amplification objective over AC-feasible decoded curves.}
}
$$

The optimization is performed by CMA-ES because the decoder and the measured objective are discrete and non-differentiable.

## 2. Objects and Notation

Let $D=\{x_i\}_{i=1}^n$ be the data records. Each record is mapped to a grid cell

$$
g_i\in\mathcal{G}.
$$

Let $w\in\mathcal{W}_{\mathrm{param}}$ be a parameter vector or edge-weight vector. The AC decoder is

$$
\Phi:w\mapsto \pi_w,
$$

where

$$
\pi_w\in\mathcal{H}_{\mathrm{AC}}.
$$

The code of record $i$ is

$$
z_i(w)=\pi_w(g_i).
$$

Let $\sigma_w$ be a permutation of records such that

$$
z_{\sigma_w(1)}(w)\leq z_{\sigma_w(2)}(w)\leq \cdots \leq z_{\sigma_w(n)}(w).
$$

For page size $b$, define the physical page index of sorted position $r$ as

$$
P_b(r)=\left\lfloor\frac{r-1}{b}\right\rfloor.
$$

Thus the materialized layout is fully determined by $w$, $D$, and $b$.

## 3. Exact Interval Set for a Query

For a query $q$, let $Q(q)\subseteq\mathcal{G}$ be its quantized cell set.

Under curve $\pi_w$, the exact set of code intervals induced by $Q(q)$ is

$$
\mathcal{I}_w(q)=\{[a_j,b_j]\}_{j=1}^{r_w(q)},
$$

where the intervals are maximal contiguous runs of codes whose cells lie in $Q(q)$.

The current implementation computes these intervals using `LACSFCBoundaryQuery`. This affects efficiency, not the mathematical definition of the interval set.

A record $i$ is a code candidate for $q$ if

$$
z_i(w)\in\bigcup_{[a,b]\in\mathcal{I}_w(q)}[a,b].
$$

The exact semantic result set is

$$
R(q)=D\cap q.
$$

## 4. Touched Page Set

Define the touched page set:

$$
\mathcal{P}_w(q)
=
\left\{
P_b(r):
\exists [a,b]\in\mathcal{I}_w(q)
\text{ such that }
 z_{\sigma_w(r)}(w)\in[a,b]
\right\}.
$$

The number of touched pages is

$$
\mathrm{pages}(w,D,q,b)=|\mathcal{P}_w(q)|.
$$

This is the page-level cost used by Tier 2.

## 5. Minimum Possible Page Count and Page Amplification

For a non-empty result set $R(q)$, any page-based layout with page capacity $b$ must touch at least

$$
\left\lceil\frac{|R(q)|}{b}\right\rceil
$$

pages to return all results, because one page can contain at most $b$ records.

Define

$$
\mathrm{minpages}(D,q,b)
=
\max\left(1,\left\lceil\frac{|R(q)|}{b}\right\rceil\right).
$$

Then page amplification is

$$
\boxed{
\mathrm{PA}(w,D,q,b)
=
\frac{\mathrm{pages}(w,D,q,b)}{\mathrm{minpages}(D,q,b)}.
}
$$

For non-empty queries and exact interval processing, the touched pages must include all pages containing returned records. Hence

$$
\mathrm{pages}(w,D,q,b)
\geq
\mathrm{minpages}(D,q,b),
$$

and therefore

$$
\boxed{
\mathrm{PA}(w,D,q,b)\geq 1.
}
$$

The value $1$ is ideal: it means the query touches no more pages than the information-theoretic page-capacity lower bound.

## 6. Population Objective

The Tier 2 population objective is

$$
L_{\mathrm{PA}}(w)
=
\mathbb{E}_{q\sim\mathcal{W}}
[\mathrm{PA}(w,D,q,b)].
$$

The ideal optimization problem is

$$
\boxed{
\min_{w\in\mathcal{W}_{\mathrm{param}}}
L_{\mathrm{PA}}(w)
}
$$

subject to

$$
\Phi(w)\in\mathcal{H}_{\mathrm{AC}}.
$$

The constraint is automatically satisfied because the decoder only returns AC-feasible curves.

## 7. Empirical Objective Used by Implementation

Given training queries $q_1,\ldots,q_m$, define the non-empty index set

$$
S=\{i: |R(q_i)|>0\}.
$$

The empirical objective is

$$
\widehat{L}_{\mathrm{PA}}(w)
=
\frac{1}{|S|}
\sum_{i\in S}
\mathrm{PA}(w,D,q_i,b).
$$

This matches the current implementation pattern, where empty-result queries may be skipped to avoid unstable or uninformative amplification ratios.

Each objective evaluation consists of:

1. decoding $w$ into $\pi_w$ using `weights_to_sfc`;
2. building the code map;
3. sorting data codes and determining page starts;
4. computing exact intervals with `LACSFCBoundaryQuery`;
5. counting touched physical pages;
6. averaging page amplification over training queries.

## 8. Relation to Tier 1 and Why No Exact Equivalence Holds

Tier 1 has an exact identity:

$$
\mathrm{nr}(\pi,Q)=|Q|-\Phi_Q(\pi).
$$

This identity is purely about the number of maximal code intervals. Page amplification is different because it depends on the realized data distribution and physical page boundaries.

In general,

$$
\mathrm{nr}(\pi,Q)<\mathrm{nr}(\pi',Q)
\not\Rightarrow
\mathrm{PA}(\pi,D,q,b)<\mathrm{PA}(\pi',D,q,b).
$$

### Constructive Explanation

A curve with one long interval may scan records spread across many pages. A curve with multiple short intervals may touch fewer pages if the relevant records are concentrated in fewer physical pages.

Thus page amplification is affected by:

$$
\text{interval structure}
+\text{ record density}
+\text{ page boundaries}.
$$

Segment count is a useful proxy, but it is not an exact page-I/O objective.

## 9. Why Tier 2 Is Benchmark-Aligned

The benchmark evaluates how many pages or records are scanned to answer range queries after the SFC order is materialized. The page-amplification objective directly measures the ratio between actual touched pages and the minimum possible page count:

$$
\mathrm{PA}
=
\frac{\text{actual touched pages}}{\text{page-capacity lower bound}}.
$$

Therefore Tier 2 is aligned with the benchmark-level physical objective in a way that pure co-access or segment count is not.

This does not mean page amplification is the same as wall-clock time. Runtime also depends on implementation constants, interval-generation cost, filtering cost, caching, and vectorization. The correct claim is that page amplification is the benchmark-aligned layout-quality objective, not a guaranteed runtime objective.

## 10. Non-Differentiability of the Objective

The mapping

$$
w\mapsto \Phi(w)
$$

contains discrete choices. Small perturbations in $w$ can change the selected path structure. Consequently,

$$
w\mapsto \widehat{L}_{\mathrm{PA}}(w)
$$

is generally non-smooth and non-differentiable.

A gradient of the true decoded objective is not available in the standard sense. Therefore the implementation uses a derivative-free optimizer:

$$
w_{t+1}=\mathrm{CMAESUpdate}(w_t,\widehat{L}_{\mathrm{PA}}).
$$

CMA-ES only requires objective values and is therefore appropriate for this black-box objective.

## 11. Optimization Boundary

Let

$$
w^*\in\arg\min_w L_{\mathrm{PA}}(w)
$$

be a population optimum and

$$
\widehat{w}^*\in\arg\min_w \widehat{L}_{\mathrm{PA}}(w)
$$

be an empirical optimum.

CMA-ES returns an approximate optimizer $\widetilde{w}$, not necessarily $\widehat{w}^*$.

A standard decomposition is

$$
L_{\mathrm{PA}}(\widetilde{w})-L_{\mathrm{PA}}(w^*)
\leq
\underbrace{2\sup_w|L_{\mathrm{PA}}(w)-\widehat{L}_{\mathrm{PA}}(w)|}_{\text{estimation error}}
+
\underbrace{\widehat{L}_{\mathrm{PA}}(\widetilde{w})-\widehat{L}_{\mathrm{PA}}(\widehat{w}^*)}_{\text{optimization error}}.
$$

This equation clarifies what is and is not guaranteed:

- with enough representative queries, estimation error can decrease;
- with more optimization budget, optimization error may decrease;
- neither term is automatically zero in the current implementation.

## 12. Search-Space Parameterizations

Tier 2 can optimize different parameterizations of $w$.

### 12.1 Edge-Space

The optimizer searches directly over edge weights:

$$
w\in\mathbb{R}^{|E_{\mathrm{dual}}|}.
$$

### 12.2 Block-Space

The optimizer searches block scores $z_b$ and maps them to edge weights:

$$
w(b_1,b_2)=f(z_{b_1},z_{b_2}).
$$

This reduces dimension and stabilizes optimization.

### 12.3 Hierarchical Space

A coarse representation is optimized first and then refined. This is a coarse-to-fine approximation strategy.

### 12.4 Function Space

A low-dimensional parameter vector $\theta$ maps to block scores through basis functions:

$$
z=\Psi\theta.
$$

This further restricts the search space.

Each parameterization changes the feasible set searched by CMA-ES, but all decoded candidates remain AC-feasible because feasibility is enforced by $\Phi$.

## 13. Correctness Boundary

The following statements are mathematically valid:

1. Page amplification is well-defined once the data layout, page size, query, and curve are fixed.
2. For non-empty queries under exact interval processing:

$$
\mathrm{PA}(w,D,q,b)\geq 1.
$$

3. The Tier 2 objective is the expected page amplification over the workload:

$$
L_{\mathrm{PA}}(w)=\mathbb{E}_{q\sim\mathcal{W}}[\mathrm{PA}(w,D,q,b)].
$$

4. Every decoded candidate is AC-feasible:

$$
\Phi(w)\in\mathcal{H}_{\mathrm{AC}}.
$$

The following statements should **not** be claimed:

1. CMA-ES guarantees a global optimum.
2. Lower segment count always implies lower page amplification.
3. Lower page amplification always implies lower wall-clock query time.
4. Tier 2 provides a closed-form mathematical solution.

## 14. Summary

Tier 2 replaces the Tier 1 segment-count proxy with the direct physical layout-quality objective:

$$
\boxed{
\min_w
\mathbb{E}_{q\sim\mathcal{W}}
[\mathrm{PA}(w,D,q,b)]
}
$$

where $w$ is decoded into an AC-feasible curve by `weights_to_sfc`.

The mathematical support for Tier 2 is the rigorous definition of page amplification as an end-to-end objective over data distribution, ordering, and page layout. The optimization guarantee is limited: CMA-ES performs derivative-free empirical search, but it does not certify global optimality.
