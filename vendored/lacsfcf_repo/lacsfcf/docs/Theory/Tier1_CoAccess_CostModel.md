# Tier 1 Theory: Rigorous Cost Model from Expected Segment Count to Co-Access Maximization

## 1. Purpose and Scope

This document gives the mathematical support for **Tier 1** in the current LACSFC system.

Tier 1 is the co-access heuristic construction path:

$$
\mathcal{W}
\;\longrightarrow\;
\widehat{J}
\;\longrightarrow\;
\texttt{dual\_edge\_weights()}
\;\longrightarrow\;
w
\;\longrightarrow\;
\texttt{weights\_to\_sfc(grid, w)}
\;\longrightarrow\;
\pi.
$$

The rigorous result proved here is:

$$
\boxed{
\min_{\pi}\mathbb{E}_{q\sim\mathcal{W}}[\mathrm{nr}(\pi,Q(q))]
\Longleftrightarrow
\max_{\pi}\sum_{(u,v)\in E(\pi)}J(u,v)
}
$$

where $J(u,v)$ is the probability that two adjacent cells are covered by the same query.

This equivalence is exact for the **cell-level segment-count objective**. The current implementation uses a block-level dual graph and a decoder, so the implementation is a scalable realization of this principle rather than an exhaustive optimizer over all cell-level AC Hamiltonian paths.

## 2. Formal Setting

Let

$$
\mathcal{G}=\{0,\ldots,G-1\}^d
$$

be a finite $d$-dimensional grid, with

$$
N=|\mathcal{G}|=G^d.
$$

Let $\mathcal{A}$ be the undirected face-adjacency graph over $\mathcal{G}$:

$$
(u,v)\in \mathcal{A}
\quad\Longleftrightarrow\quad
\|u-v\|_1=1.
$$

An AC-feasible SFC is a Hamiltonian path in $\mathcal{A}$:

$$
\pi:\mathcal{G}\to\{0,\ldots,N-1\},
$$

or equivalently its inverse order

$$
c_t=\pi^{-1}(t), \quad t=0,\ldots,N-1,
$$

such that

$$
(c_t,c_{t+1})\in\mathcal{A},\quad t=0,\ldots,N-2.
$$

The curve edge set is

$$
E(\pi)=\{(c_t,c_{t+1}):t=0,\ldots,N-2\}.
$$

For a window query $q$, let $Q(q)\subseteq\mathcal{G}$ be the set of quantized cells covered by the query.

## 3. Definition of Segment Count

For a fixed curve $\pi$ and fixed query cell set $Q$, define the binary sequence

$$
b_t=\mathbf{1}[c_t\in Q],
\quad t=0,\ldots,N-1.
$$

A maximal contiguous segment is a maximal interval $[s,t]$ such that

$$
b_s=b_{s+1}=\cdots=b_t=1
$$

and either $s=0$ or $b_{s-1}=0$, and either $t=N-1$ or $b_{t+1}=0$.

The number of such maximal one-runs is denoted by

$$
\mathrm{nr}(\pi,Q).
$$

## 4. Lemma 1: Run Count by Boundary Transitions

Define the transition count

$$
F(\pi,Q)
=
\sum_{t=0}^{N-2}\mathbf{1}[b_t\neq b_{t+1}].
$$

Then

$$
\boxed{
\mathrm{nr}(\pi,Q)
=
\frac{
F(\pi,Q)+\mathbf{1}[b_0=1]+\mathbf{1}[b_{N-1}=1]
}{2}
}
$$

### Proof

Every maximal one-run has one left boundary and one right boundary.

If a run starts in the interior, its left boundary is a transition $0\to1$. If it starts at $t=0$, its left boundary is counted by $\mathbf{1}[b_0=1]$.

If a run ends in the interior, its right boundary is a transition $1\to0$. If it ends at $t=N-1$, its right boundary is counted by $\mathbf{1}[b_{N-1}=1]$.

Thus the total number of run boundaries is

$$
F(\pi,Q)+\mathbf{1}[b_0=1]+\mathbf{1}[b_{N-1}=1].
$$

Each run has exactly two boundaries, so dividing by $2$ gives the result.

## 5. Lemma 2: Transition Decomposition over Curve Edges

For a curve edge $(c_t,c_{t+1})=(u,v)$,

$$
\mathbf{1}[b_t\neq b_{t+1}]
=
\mathbf{1}[u\in Q]
+
\mathbf{1}[v\in Q]
-
2\mathbf{1}[u\in Q\wedge v\in Q].
$$

### Proof

There are four cases:

| $u\in Q$ | $v\in Q$ | left side | right side |
|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 |
| 0 | 1 | 1 | 1 |
| 1 | 0 | 1 | 1 |
| 1 | 1 | 0 | 0 |

The identity holds in all cases.

Summing over all $t=0,\ldots,N-2$ gives

$$
F(\pi,Q)
=
\sum_{(u,v)\in E(\pi)}
\left(
\mathbf{1}[u\in Q]
+
\mathbf{1}[v\in Q]
-
2\mathbf{1}[u\in Q\wedge v\in Q]
\right).
$$

## 6. Theorem 1: Exact Segment-Co-Access Identity

Define

$$
\Phi_Q(\pi)
=
\sum_{(u,v)\in E(\pi)}
\mathbf{1}[u\in Q\wedge v\in Q].
$$

Then

$$
\boxed{
\mathrm{nr}(\pi,Q)=|Q|-\Phi_Q(\pi)
}
$$

### Proof

Start from Lemma 2:

$$
F(\pi,Q)
=
\sum_{(u,v)\in E(\pi)}
\left(
\mathbf{1}[u\in Q]
+
\mathbf{1}[v\in Q]
\right)
-
2\Phi_Q(\pi).
$$

Each non-endpoint cell in the curve appears in exactly two curve edges. The first cell $c_0$ and last cell $c_{N-1}$ appear in exactly one curve edge. Therefore,

$$
\sum_{(u,v)\in E(\pi)}
\left(
\mathbf{1}[u\in Q]
+
\mathbf{1}[v\in Q]
\right)
=
2|Q|
-
\mathbf{1}[c_0\in Q]
-
\mathbf{1}[c_{N-1}\in Q].
$$

Since $b_0=\mathbf{1}[c_0\in Q]$ and $b_{N-1}=\mathbf{1}[c_{N-1}\in Q]$,

$$
\begin{aligned}
\mathrm{nr}(\pi,Q)
&=
\frac{
F(\pi,Q)+\mathbf{1}[b_0=1]+\mathbf{1}[b_{N-1}=1]
}{2}\\
&=
\frac{
2|Q|
-
\mathbf{1}[c_0\in Q]
-
\mathbf{1}[c_{N-1}\in Q]
-
2\Phi_Q(\pi)
+
\mathbf{1}[c_0\in Q]
+
\mathbf{1}[c_{N-1}\in Q]
}{2}\\
&=|Q|-\Phi_Q(\pi).
\end{aligned}
$$

This completes the proof.

## 7. Theorem 2: Expected Segment Minimization Equals Co-Access Maximization

Define adjacent-cell co-access probability:

$$
J(u,v)
=
\Pr_{q\sim\mathcal{W}}[u\in Q(q)\wedge v\in Q(q)].
$$

Define expected total co-access along $\pi$:

$$
\Phi(\pi)
=
\sum_{(u,v)\in E(\pi)}J(u,v).
$$

Taking expectation in Theorem 1:

$$
\begin{aligned}
\mathbb{E}_{q\sim\mathcal{W}}[\mathrm{nr}(\pi,Q(q))]
&=
\mathbb{E}_{q\sim\mathcal{W}}[|Q(q)|]
-
\mathbb{E}_{q\sim\mathcal{W}}[\Phi_{Q(q)}(\pi)]\\
&=
\mathbb{E}_{q\sim\mathcal{W}}[|Q(q)|]
-
\sum_{(u,v)\in E(\pi)}
\Pr[u\in Q(q)\wedge v\in Q(q)]\\
&=
\mathbb{E}_{q\sim\mathcal{W}}[|Q(q)|]
-
\Phi(\pi).
\end{aligned}
$$

The first term is independent of $\pi$. Hence

$$
\boxed{
\arg\min_{\pi\in\mathcal{H}_{\mathrm{AC}}}
\mathbb{E}[\mathrm{nr}(\pi,Q)]
=
\arg\max_{\pi\in\mathcal{H}_{\mathrm{AC}}}
\Phi(\pi)
}
$$

and equivalently,

$$
\boxed{
\min_{\pi\in\mathcal{H}_{\mathrm{AC}}}
\mathbb{E}[\mathrm{nr}(\pi,Q)]
\Longleftrightarrow
\max_{\pi\in\mathcal{H}_{\mathrm{AC}}}
\sum_{(u,v)\in E(\pi)}J(u,v)
}
$$

This is the exact mathematical basis of Tier 1.

## 8. Why Only Adjacent Co-Access Matters under AC

For any AC-feasible curve,

$$
E(\pi)\subseteq\mathcal{A}.
$$

Therefore the objective

$$
\sum_{(u,v)\in E(\pi)}J(u,v)
$$

can only use co-access values on adjacent cell pairs. Non-adjacent co-access may be semantically meaningful, but it cannot appear as a curve edge in an AC SFC.

Thus, for the segment-count objective restricted to AC curves, the sufficient graph is the adjacent-cell co-access graph.

## 9. Empirical Estimation

Given iid training queries $q_1,\ldots,q_m\sim\mathcal{W}$, define

$$
\widehat{J}(u,v)
=
\frac{1}{m}
\sum_{i=1}^m
\mathbf{1}[u\in Q(q_i)\wedge v\in Q(q_i)].
$$

For a fixed curve $\pi$:

$$
\widehat{\Phi}(\pi)
=
\sum_{(u,v)\in E(\pi)}\widehat{J}(u,v).
$$

By Theorem 1 applied to each query:

$$
\widehat{\Phi}(\pi)
=
\frac{1}{m}\sum_{i=1}^m |Q(q_i)|
-
\frac{1}{m}\sum_{i=1}^m \mathrm{nr}(\pi,Q(q_i)).
$$

Therefore, for the empirical sample:

$$
\boxed{
\arg\max_{\pi}\widehat{\Phi}(\pi)
=
\arg\min_{\pi}
\frac{1}{m}\sum_{i=1}^m\mathrm{nr}(\pi,Q(q_i))
}
$$

This is an exact empirical identity, not an approximation.

## 10. Uniform Convergence Bound

Let $\mathcal{A}$ be the set of adjacent cell pairs. For each $e=(u,v)\in\mathcal{A}$, define the Bernoulli random variable

$$
X_e(q)=\mathbf{1}[u\in Q(q)\wedge v\in Q(q)].
$$

Then

$$
\widehat{J}(e)=\frac{1}{m}\sum_{i=1}^m X_e(q_i),
\quad
J(e)=\mathbb{E}[X_e(q)].
$$

By Hoeffding's inequality, for a fixed edge $e$:

$$
\Pr\left(
|\widehat{J}(e)-J(e)|>\epsilon
\right)
\leq
2\exp(-2m\epsilon^2).
$$

Taking a union bound over all adjacent edges:

$$
\Pr\left(
\sup_{e\in\mathcal{A}}
|\widehat{J}(e)-J(e)|>\epsilon
\right)
\leq
2|\mathcal{A}|\exp(-2m\epsilon^2).
$$

Thus, with probability at least $1-\delta$:

$$
\boxed{
\sup_{e\in\mathcal{A}}
|\widehat{J}(e)-J(e)|
\leq
\sqrt{
\frac{\log(2|\mathcal{A}|/\delta)}{2m}
}
}
$$

For any curve $\pi$ with $N-1$ edges:

$$
|\widehat{\Phi}(\pi)-\Phi(\pi)|
\leq
\sum_{e\in E(\pi)}
|\widehat{J}(e)-J(e)|
\leq
(N-1)\epsilon_J,
$$

where

$$
\epsilon_J=
\sqrt{
\frac{\log(2|\mathcal{A}|/\delta)}{2m}
}.
$$

Let

$$
\widehat{\pi}
\in
\arg\max_{\pi\in\mathcal{H}_{\mathrm{AC}}}
\widehat{\Phi}(\pi),
\quad
\pi^*
\in
\arg\max_{\pi\in\mathcal{H}_{\mathrm{AC}}}
\Phi(\pi).
$$

Then

$$
\Phi(\pi^*)-\Phi(\widehat{\pi})
\leq
2(N-1)\epsilon_J.
$$

Because

$$
\mathbb{E}[\mathrm{nr}(\widehat{\pi},Q)]
-
\mathbb{E}[\mathrm{nr}(\pi^*,Q)]
=
\Phi(\pi^*)-\Phi(\widehat{\pi}),
$$

we also have

$$
\boxed{
\mathbb{E}[\mathrm{nr}(\widehat{\pi},Q)]
-
\mathbb{E}[\mathrm{nr}(\pi^*,Q)]
\leq
2(N-1)\epsilon_J.
}
$$

This bound is pessimistic because it treats all adjacent edge estimates independently and ignores spatial smoothness of window workloads.

## 11. Connection to the Current Implementation

The exact theory above is stated for cell-level co-access. The current system uses a scalable block-level construction:

$$
\mathcal{W}
\;\longrightarrow\;
\texttt{QueryWorkload.dual\_edge\_weights()}
\;\longrightarrow\;
w_{\mathrm{block}}
\;\longrightarrow\;
\texttt{weights\_to\_sfc(grid, }w_{\mathrm{block}}\texttt{)}.
$$

The implementation therefore approximates the ideal cell-level objective:

$$
w_{\mathrm{block}}(e)
\approx
\widehat{J}_{\mathrm{block}}(e).
$$

The decoder `weights_to_sfc` maps these weights to an AC-feasible curve. This guarantees structural feasibility of the decoded curve, but it does not imply that the decoded curve globally maximizes the exact cell-level objective.

## 12. Correctness Boundary

The following statements are mathematically exact:

1. For any curve order and any query cell set:

$$
\mathrm{nr}(\pi,Q)=|Q|-\Phi_Q(\pi).
$$

2. For AC curves, expected segment minimization is equivalent to adjacent co-access maximization:

$$
\min\mathbb{E}[\mathrm{nr}]
\Longleftrightarrow
\max \sum_{e\in E(\pi)}J(e).
$$

3. Empirical co-access maximization is exactly empirical segment-count minimization.

The following statements are implementation-level approximations:

1. `dual_edge_weights()` approximates the ideal cell-level co-access objective at block level.
2. `weights_to_sfc(...)` decodes weights into an AC-feasible curve but is not claimed to solve the global Hamiltonian-path optimization problem.
3. Tier 1 reduces a segment-count proxy, not the final page-amplification objective.

## 13. Summary

Tier 1 has the strongest mathematical guarantee among the three tiers:

$$
\boxed{
\mathrm{nr}(\pi,Q)=|Q|-\sum_{(u,v)\in E(\pi)}
\mathbf{1}[u\in Q\wedge v\in Q]
}
$$

and therefore

$$
\boxed{
\min_{\pi}\mathbb{E}[\mathrm{nr}(\pi,Q)]
\Longleftrightarrow
\max_{\pi}\sum_{(u,v)\in E(\pi)}J(u,v).
}
$$

In the real codebase, Tier 1 is a scalable block-level co-access heuristic using `dual_edge_weights()` and `weights_to_sfc(...)`. Its theory is exact for segment count, while its implementation is an efficient approximation of the ideal cell-level co-access optimizer.
