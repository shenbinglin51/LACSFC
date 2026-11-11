# Tier 3 Theory: Rigorous Amortized Workload-Adaptive Construction

## 1. Purpose and Scope

This document gives the mathematical support for **Tier 3** in the current LACSFC system.

Tier 3 is the learning-based amortization layer. It does not replace the AC decoder and it does not make query processing approximate. Its role is to learn how to produce useful decoder weights for related workloads.

The common structural path remains:

$$
x_T
\;\longrightarrow\;
G_\theta(x_T)
\;\longrightarrow\;
w_T
\;\longrightarrow\;
\Phi(w_T)=\texttt{weights\_to\_sfc(grid, }w_T\texttt{)}
\;\longrightarrow\;
\pi_T.
$$

The rigorous claim of Tier 3 is conditional:

$$
\boxed{
\text{If related workloads share learnable structure, a generator or surrogate can amortize weight search while preserving AC feasibility.}
}
$$

Tier 3 does **not** provide an unconditional guarantee of global optimality.

## 2. Task-Level Formulation

Let a workload task be

$$
T=(\mathcal{W}_T,D_T,b,\mathcal{G}_T),
$$

where:

- $\mathcal{W}_T$ is the query distribution;
- $D_T$ is the data distribution or finite data set;
- $b$ is the page size;
- $\mathcal{G}_T$ is the quantized grid.

For a weight vector $w$, define the decoded curve

$$
\pi_w=\Phi_T(w).
$$

The task-level Tier 2 loss is

$$
L_T(w)
=
\mathbb{E}_{q\sim\mathcal{W}_T}
[\mathrm{PA}(w,D_T,q,b)].
$$

The task-specific optimum is

$$
w_T^*\in\arg\min_{w\in\mathcal{W}_{\mathrm{param}}}L_T(w).
$$

Running Tier 2 separately for each $T$ attempts to approximate $w_T^*$ by black-box search.

## 3. Distribution over Tasks

Tier 3 assumes tasks are sampled from a task distribution:

$$
T\sim\mathcal{P}_{\mathrm{task}}.
$$

Each task has observable features

$$
x_T=\psi(T),
$$

such as query statistics, block densities, scene descriptors, or dual-edge features.

A generator is a function class

$$
G_\theta:\mathcal{X}\to\mathcal{W}_{\mathrm{param}}.
$$

It produces weights

$$
w_T(\theta)=G_\theta(x_T).
$$

The population amortized risk is

$$
\boxed{
R(\theta)
=
\mathbb{E}_{T\sim\mathcal{P}_{\mathrm{task}}}
[L_T(G_\theta(x_T))].
}
$$

Tier 3 seeks

$$
\theta^*\in\arg\min_\theta R(\theta).
$$

This is the formal amortized optimization objective.

## 4. Why This Is Amortization

Per-task optimization solves, approximately,

$$
w_T^*\in\arg\min_w L_T(w)
$$

for every new task $T$.

Amortized optimization instead solves one meta-problem:

$$
\theta^*\in\arg\min_\theta
\mathbb{E}_{T}[L_T(G_\theta(x_T))].
$$

At deployment, a new task only requires evaluating

$$
w_T=G_{\theta^*}(x_T),
$$

possibly followed by a small number of adaptation or CMA-ES refinement steps.

Thus, the expensive search cost is shifted from every new task to an offline training phase.

## 5. Deterministic Feasibility Guarantee

Tier 3 never needs to learn the AC constraint. Feasibility is enforced by the decoder.

For every $\theta$ and every task $T$:

$$
\boxed{
\Phi_T(G_\theta(x_T))\in\mathcal{H}_{\mathrm{AC}}(T).
}
$$

This is a deterministic guarantee, independent of learning quality.

Likewise, exact interval recovery remains outside the learned model. In the current implementation, it is handled by:

$$
\texttt{LACSFCBoundaryQuery}.
$$

Therefore, Tier 3 may affect layout quality, but it does not affect query correctness.

## 6. Relationship to Tier 1

Tier 1 provides a deterministic heuristic target:

$$
w_T^{(1)}=\texttt{dual\_edge\_weights}(T).
$$

A generator can be pretrained to imitate Tier 1:

$$
\min_\theta
\frac{1}{M}\sum_{i=1}^{M}
\|G_\theta(x_{T_i})-w_{T_i}^{(1)}\|^2.
$$

This gives a supervised warm start. It is mathematically a regression problem, not a guarantee of optimality.

A residual version can be written as

$$
w_T(\theta)=w_T^{(1)}+\alpha\Delta_\theta(x_T).
$$

Here Tier 1 supplies a prior and $\Delta_\theta$ learns a correction. This is useful because Tier 1 has a rigorous segment-count basis, while the learned residual can adapt to page-level effects.

## 7. Relationship to Tier 2

Tier 2 defines the high-fidelity page-amplification loss:

$$
L_T(w)=\mathbb{E}_{q\sim\mathcal{W}_T}[\mathrm{PA}(w,D_T,q,b)].
$$

Tier 3 can use Tier 2 in three mathematically distinct ways.

### 7.1 Teacher Weights

If Tier 2 produces approximate optimized weights $\widetilde{w}_T^{(2)}$, Tier 3 can learn

$$
\min_\theta
\frac{1}{M}\sum_{i=1}^{M}
\|G_\theta(x_{T_i})-\widetilde{w}_{T_i}^{(2)}\|^2.
$$

This imitates the optimizer output, but the quality is limited by the quality of the Tier 2 labels.

### 7.2 Direct Loss Minimization

If $L_T(G_\theta(x_T))$ can be evaluated, Tier 3 can directly minimize empirical amortized risk:

$$
\widehat{R}(\theta)
=
\frac{1}{M}\sum_{i=1}^{M}
\widehat{L}_{T_i}(G_\theta(x_{T_i})).
$$

This is closest to the population objective, but expensive.

### 7.3 Warm Start for Tier 2

Tier 3 can provide

$$
w_0=G_\theta(x_T)
$$

as the initialization for CMA-ES. The final optimizer may then return

$$
\widetilde{w}_T=\mathrm{CMAES}(w_0,T).
$$

This use case is conservative: even if the generator is imperfect, Tier 2 can still refine it.

## 8. Surrogate Evaluation

Because $L_T(w)$ is expensive and non-differentiable, Tier 3 may learn a surrogate evaluator:

$$
E_\phi(x_T,w)\approx L_T(w).
$$

Training data may consist of triples

$$
(x_{T_i},w_{ij},\widehat{L}_{T_i}(w_{ij})).
$$

A surrogate training objective is

$$
\min_\phi
\frac{1}{M}
\sum_{i,j}
\left(E_\phi(x_{T_i},w_{ij})-
\widehat{L}_{T_i}(w_{ij})\right)^2.
$$

Then the generator can be trained by minimizing

$$
\min_\theta
\frac{1}{M}
\sum_{i=1}^{M}
E_\phi(x_{T_i},G_\theta(x_{T_i})).
$$

This replaces the true loss with an approximate loss. Therefore it introduces surrogate error.

## 9. Surrogate Error Bound

Suppose the surrogate satisfies, on a relevant domain $\mathcal{D}$,

$$
\sup_{(T,w)\in\mathcal{D}}
|E_\phi(x_T,w)-L_T(w)|\leq \epsilon_{\mathrm{sur}}.
$$

Let

$$
\theta_E\in\arg\min_\theta
\mathbb{E}_{T}[E_\phi(x_T,G_\theta(x_T))]
$$

and

$$
\theta^*\in\arg\min_\theta R(\theta).
$$

Then

$$
R(\theta_E)-R(\theta^*)\leq 2\epsilon_{\mathrm{sur}}.
$$

### Proof

For any $\theta$,

$$
|R(\theta)-R_E(\theta)|\leq\epsilon_{\mathrm{sur}},
$$

where

$$
R_E(\theta)=\mathbb{E}_{T}[E_\phi(x_T,G_\theta(x_T))].
$$

Then

$$
\begin{aligned}
R(\theta_E)
&\leq R_E(\theta_E)+\epsilon_{\mathrm{sur}}\\
&\leq R_E(\theta^*)+\epsilon_{\mathrm{sur}}\\
&\leq R(\theta^*)+2\epsilon_{\mathrm{sur}}.
\end{aligned}
$$

Thus the surrogate optimum is near the true amortized optimum only when surrogate error is small on the relevant domain.

## 10. Statistical Generalization Bound

Assume the task losses are bounded:

$$
0\leq L_T(G_\theta(x_T))\leq B.
$$

Let $\mathcal{F}=\{T\mapsto L_T(G_\theta(x_T)):\theta\in\Theta\}$ be the induced function class over tasks.

The empirical amortized risk over $M$ tasks is

$$
\widehat{R}(\theta)
=
\frac{1}{M}\sum_{i=1}^{M}
L_{T_i}(G_\theta(x_{T_i})).
$$

A standard uniform convergence statement is: with probability at least $1-\delta$,

$$
\sup_{\theta\in\Theta}|R(\theta)-\widehat{R}(\theta)|
\leq
\epsilon_{\mathrm{gen}}(M,\Theta,\delta),
$$

where $\epsilon_{\mathrm{gen}}$ depends on the capacity of the generator class and decreases with the number of independent training tasks.

If

$$
\widehat{\theta}\in\arg\min_\theta\widehat{R}(\theta),
$$

then

$$
R(\widehat{\theta})-R(\theta^*)
\leq
2\epsilon_{\mathrm{gen}}.
$$

This is the statistical basis for Tier 3. It requires representative task sampling and controlled model capacity.

## 11. Full Excess-Risk Decomposition

In the real system, we rarely minimize the exact population risk. Instead, Tier 3 may use finite tasks, finite queries, surrogate labels, and imperfect optimization.

A useful decomposition is:

$$
\begin{aligned}
R(\widetilde{\theta})-R^*
\leq
&\underbrace{\epsilon_{\mathrm{approx}}}_{\text{generator class cannot represent optimum}}\\
&+\underbrace{\epsilon_{\mathrm{task}}}_{\text{finite number of training tasks}}\\
&+\underbrace{\epsilon_{\mathrm{query}}}_{\text{finite queries per task}}\\
&+\underbrace{\epsilon_{\mathrm{sur}}}_{\text{surrogate approximation}}\\
&+\underbrace{\epsilon_{\mathrm{opt}}}_{\text{training optimization error}}.
\end{aligned}
$$

where

$$
R^*=\inf_{\theta}R(\theta).
$$

This decomposition is the correct way to state Tier 3's guarantee. Tier 3 is beneficial when these error terms are small enough that amortized prediction is cheaper than repeated per-task black-box search while maintaining comparable layout quality.

## 12. Meta-Learning Interpretation

Tier 3 can also be written in meta-learning form. Let $\theta$ be an initialization and define an adapted parameter

$$
\theta_T'
=
\theta-\eta\nabla_\theta \widetilde{L}_T(\theta),
$$

where $\widetilde{L}_T$ may be a differentiable surrogate or imitation loss.

The meta-objective is

$$
\min_\theta
\mathbb{E}_{T\sim\mathcal{P}_{\mathrm{task}}}
[L_T(G_{\theta_T'}(x_T))].
$$

The purpose is not to prove that one gradient step finds $w_T^*$. The purpose is to learn an initialization that adapts faster on related tasks.

## 13. Correctness Boundary

The following statements are deterministic and always true if the decoder contract holds:

1. For any generated weights, the decoded curve is AC-feasible:

$$
\Phi_T(G_\theta(x_T))\in\mathcal{H}_{\mathrm{AC}}(T).
$$

2. Query exactness remains handled by boundary-based interval recovery, not by the learned model.

The following statements are conditional statistical claims:

1. A generator can generalize to new tasks if training tasks are representative.
2. A surrogate can guide optimization if surrogate error is small on the relevant domain.
3. Amortization reduces deployment cost if the learned generator is cheaper than repeated Tier 2 search and has acceptable excess risk.

The following statements should **not** be claimed:

1. Tier 3 guarantees global optimality.
2. Tier 3 always outperforms Tier 2.
3. Tier 3 preserves quality under arbitrary workload shift.
4. Surrogate minimization is equivalent to true page-amplification minimization without a surrogate-error assumption.

## 14. Summary

Tier 3 is mathematically an amortized optimization layer:

$$
\boxed{
\min_\theta
\mathbb{E}_{T\sim\mathcal{P}_{\mathrm{task}}}
[L_T(G_\theta(x_T))]
}
$$

Its strict deterministic guarantee is feasibility:

$$
\boxed{
\Phi_T(G_\theta(x_T))\in\mathcal{H}_{\mathrm{AC}}(T).
}
$$

Its quality guarantee is conditional and statistical, governed by approximation, estimation, surrogate, and optimization errors. Therefore Tier 3 should be presented as an amortized workload-adaptation mechanism, not as a closed-form or globally optimal construction method.
