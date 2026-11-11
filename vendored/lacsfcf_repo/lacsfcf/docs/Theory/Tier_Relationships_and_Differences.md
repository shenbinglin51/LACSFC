# Relationships and Differences among Tier 1, Tier 2, and Tier 3

## 1. Purpose of This Document

This document summarizes the relationship among the three LACSFC construction tiers:

- **Tier 1**: co-access heuristic construction;
- **Tier 2**: page-amplification-driven black-box optimization;
- **Tier 3**: amortized workload-adaptive learning.

The three tiers share the same structural principle:

$$
\text{produce weights } w
\;\longrightarrow\;
\texttt{weights\_to\_sfc(grid, w)}
\;\longrightarrow\;
\text{AC-feasible curve}
\;\longrightarrow\;
\text{boundary-based exact interval recovery}.
$$

They differ only in **how the weight vector $w$ is produced**.

## 2. Common Structural Backbone

All tiers separate feasibility from optimization.

The optimizer or heuristic does not directly construct an arbitrary ordering. Instead, it produces a weight vector over the dual graph or its parameterized representation. The decoder then maps the weights to an AC-feasible curve:

$$
\pi = \Phi(w) = \texttt{weights\_to\_sfc}(\texttt{grid}, w).
$$

This means:

$$
\forall w, \quad \Phi(w) \in \mathcal{H}_{\mathrm{AC}}.
$$

Therefore, feasibility is enforced by construction, while the tier-specific method only determines which feasible curve is selected.

In the current implementation, online query processing uses boundary-based exact interval recovery:

$$
\texttt{LACSFCBoundaryQuery}.
$$

Thus, the three tiers affect layout quality, not query correctness.

## 3. Tier 1: Co-Access Heuristic

### 3.1 Core Idea

Tier 1 estimates which adjacent regions are frequently accessed together by the query workload. These co-access estimates become dual-edge weights.

The construction is:

$$
\mathcal{W}
\;\longrightarrow\;
\texttt{dual\_edge\_weights()}
\;\longrightarrow\;
w^{(1)}
\;\longrightarrow\;
\Phi(w^{(1)}).
$$

### 3.2 Theoretical Support

Tier 1 is supported by the segment-count identity:

$$
\mathrm{nr}(\pi,Q) = |Q| - \Phi_Q(\pi),
$$

where

$$
\Phi_Q(\pi)
=
\sum_{(u,v) \in \mathrm{Edges}(\pi)}
\mathbf{1}[u \in Q \wedge v \in Q].
$$

Taking expectation over the workload gives:

$$
\min_\pi \mathbb{E}[\mathrm{nr}(\pi,Q)]
\Longleftrightarrow
\max_\pi
\sum_{(u,v) \in \mathrm{Edges}(\pi)}
J(u,v),
$$

where $J(u,v)$ is the co-access probability.

### 3.3 Practical Meaning

Tier 1 says:

> If two adjacent regions are often queried together, the curve should prefer to keep them close in its decoded structure.

### 3.4 Strengths

- Fast construction.
- Clear theoretical link to expected segment count.
- Strong heuristic baseline.
- Useful warm start for Tier 2.
- Useful prior or supervised target for Tier 3.

### 3.5 Limitations

- The clean theory is cell-level, while the implementation uses block-level weights.
- It optimizes a segment-count proxy, not final physical page amplification.
- It does not perform global search over all AC-feasible Hamiltonian paths.

## 4. Tier 2: Page-Amplification Black-Box Optimization

### 4.1 Core Idea

Tier 2 directly optimizes the benchmark-aligned system objective: page amplification.

The construction is:

$$
w
\;\longrightarrow\;
\Phi(w)
\;\longrightarrow\;
\text{materialized order}
\;\longrightarrow\;
\text{exact intervals}
\;\longrightarrow\;
\mathrm{page\ amplification}
\;\longrightarrow\;
\text{CMA-ES update}.
$$

### 4.2 Theoretical Support

Tier 2 models the physical objective as:

$$
\min_w
\mathbb{E}_{q \sim \mathcal{W}}
[
\mathrm{PA}(\Phi(w),D,q,b)
],
$$

where:

- $D$ is the data distribution;
- $q$ is the query;
- $b$ is the page size;
- $\Phi(w)$ is the AC decoder;
- $\mathrm{PA}$ is page amplification.

This objective captures the coupling among:

$$
\text{data distribution}
+ 
\text{ordering}
+ 
\text{physical paging}.
$$

### 4.3 Practical Meaning

Tier 2 says:

> Instead of optimizing only a proxy such as segment count, evaluate how many physical pages are actually touched and optimize that directly.

### 4.4 Strengths

- Directly aligned with benchmark page-level I/O cost.
- Captures effects missed by co-access alone.
- Handles interaction between data distribution, ordering, and paging.
- Uses Tier 1 weights as a strong initialization.

### 4.5 Limitations

- More expensive than Tier 1.
- The objective is non-differentiable because decoding is discrete.
- CMA-ES does not guarantee global optimality.
- Each evaluation requires decoding and page-level measurement.

## 5. Tier 3: Amortized Workload-Adaptive Learning

### 5.1 Core Idea

Tier 3 learns how to produce good weights from workload or scene features.

The construction is:

$$
x_T
\;\longrightarrow\;
G_\theta(x_T)
\;\longrightarrow\;
w^{(3)}
\;\longrightarrow\;
\Phi(w^{(3)}),
$$

where $T$ is a workload task and $x_T$ is its feature representation.

### 5.2 Theoretical Support

Tier 3 views each workload as a task:

$$
T \sim \mathcal{P}(T).
$$

For each task, there is a task-specific objective:

$$
L_T(w)
=
\mathbb{E}_{q \sim \mathcal{W}_T}
[
\mathrm{PA}(\Phi(w),D_T,q,b)
].
$$

Tier 3 learns a generator:

$$
G_\theta(x_T) \approx w_T^*,
$$

where

$$
w_T^* \in \arg\min_w L_T(w).
$$

The amortized objective is:

$$
\min_\theta
\mathbb{E}_{T \sim \mathcal{P}(T)}
[
L_T(G_\theta(x_T))].
$$

### 5.3 Practical Meaning

Tier 3 says:

> If workloads share structure, we can learn how to generate good weights instead of running expensive black-box optimization from scratch for every workload.

### 5.4 Strengths

- Can reduce per-workload optimization cost.
- Can learn from Tier 1 heuristics and Tier 2 measurements.
- Supports warm start, surrogate evaluation, and fast adaptation.
- Useful for workload shift scenarios.

### 5.5 Limitations

- Prototype extension in the current system.
- Requires representative training tasks.
- Surrogate predictions may be inaccurate.
- Does not guarantee global optimality.
- Should not be presented as the main deployed evidence unless experimentally validated.

## 6. Relationship among the Three Tiers

The three tiers form a progression from theory-driven heuristic to direct system optimization to learned amortization.

$$
\begin{aligned}
\text{Tier 1:} \quad
&w^{(1)} = \texttt{dual\_edge\_weights}(\mathcal{W}) \\
\text{Tier 2:} \quad
&w^{(2)} \approx \arg\min_w \widehat{L}_{\mathrm{PA}}(w) \\
\text{Tier 3:} \quad
&w^{(3)} = G_\theta(x_T) \text{ or adapted from } G_\theta(x_T)
\end{aligned}
$$

All three are decoded by the same AC-feasible decoder:

$$
\pi^{(i)} = \Phi(w^{(i)}), \quad i \in \{1,2,3\}.
$$

## 7. Tier 1 to Tier 2

Tier 1 provides the initial structural prior:

$$
w_0 = w^{(1)}.
$$

Tier 2 starts from or near this prior and refines it using page amplification:

$$
w^{(2)} \approx \arg\min_w \widehat{L}_{\mathrm{PA}}(w).
$$

So the relation is:

$$
\text{Tier 1 = fast proxy}
\quad \longrightarrow \quad
\text{Tier 2 = direct physical refinement}.
$$

Tier 1 is justified by segment-count co-access theory. Tier 2 is justified by the final page-level system objective.

## 8. Tier 2 to Tier 3

Tier 2 can be expensive because each workload may require many black-box evaluations. Tier 3 tries to amortize this cost.

Tier 2 provides:

- high-quality optimized weights;
- measured page-amplification labels;
- training targets for surrogate or generator models.

Tier 3 uses these signals to learn:

$$
T \mapsto w_T.
$$

So the relation is:

$$
\text{Tier 2 = per-task optimizer}
\quad \longrightarrow \quad
\text{Tier 3 = learned amortized optimizer}.
$$

## 9. Tier 1 to Tier 3

Tier 1 also supports Tier 3 directly. A generator can first imitate Tier 1:

$$
\min_\theta
\sum_T
\|G_\theta(x_T) - w_T^{(1)}\|^2.
$$

Then it can learn residual corrections:

$$
w_T = w_T^{(1)} + \alpha \Delta_\theta(x_T).
$$

So the relation is:

$$
\text{Tier 1 = stable prior}
\quad \longrightarrow \quad
\text{Tier 3 = learned improvement or fast predictor}.
$$

## 10. Comparison Table

| Aspect | Tier 1 | Tier 2 | Tier 3 |
|---|---|---|---|
| Main role | Fast heuristic | Direct objective optimization | Amortized learning / adaptation |
| Weight source | Co-access / dual-edge statistics | CMA-ES search | Generator / surrogate / meta-learning |
| Main objective | Expected segment-count proxy | Page amplification | Predict or adapt good weights across tasks |
| Theoretical basis | $E[\mathrm{nr}]$ to co-access equivalence | End-to-end page-level cost model | Amortized optimization and meta-learning |
| Uses decoder? | Yes | Yes | Yes |
| AC guaranteed? | Yes, by decoder | Yes, by decoder | Yes, by decoder |
| Query exactness guaranteed? | Yes, by boundary query | Yes, by boundary query | Yes, by boundary query |
| Cost | Low | High | Training high, deployment low |
| Global optimality? | No | No | No |
| Current maturity | Baseline / heuristic path | Main optimization path | Prototype extension |

## 11. Recommended Paper-Level Interpretation

A concise way to describe the tiers is:

> The three tiers share the same AC-feasible decoder and differ only in how they produce decoder weights. Tier 1 is theory-driven: it uses the equivalence between expected segment count and adjacent co-access maximization to define workload-aware dual-edge weights. Tier 2 is system-objective-driven: it directly optimizes page amplification, capturing the coupled effects of data distribution, ordering, and paging. Tier 3 is amortization-driven: it learns to predict or adapt good weights across related workloads, reducing the need to rerun expensive black-box optimization from scratch.

## 12. Safe Claims

The following claims are safe for the current implementation:

1. **Shared feasibility**: all tiers rely on the same AC decoder, so all decoded layouts are structurally feasible.
2. **Shared exactness path**: all tiers use boundary-based exact interval recovery after materialization.
3. **Tier 1 theory**: Tier 1 is justified as a co-access proxy for expected segment count.
4. **Tier 2 objective**: Tier 2 directly optimizes benchmark-aligned page amplification.
5. **Tier 3 role**: Tier 3 amortizes or warm-starts weight generation across workloads and should be described as a prototype extension unless supported by final experiments.

## 13. Summary

The tiers are not separate algorithms with separate feasibility rules. They are three ways to produce weights for the same AC-feasible construction pipeline:

$$
\boxed{
\text{Tier-specific weight generation}
\;\longrightarrow\;
\text{common AC decoder}
\;\longrightarrow\;
\text{common exact query processing}
}
$$

Their difference is the source and objective of the weights:

$$
\boxed{
\text{Tier 1: co-access proxy}
\quad | \quad
\text{Tier 2: page-amplification objective}
\quad | \quad
\text{Tier 3: learned amortization}
}
$$
