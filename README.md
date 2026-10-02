# Policy churn as Filippov sliding in Double Q-learning

Q-learning targets that select actions by argmax, as in Double Q-learning (van Hasselt et al., 2016)
and its over-relaxed variant Double SOR, make the expected update **discontinuous across
action-tie manifolds**, the parameter sets where two actions have equal value at some state. Each
crossing of such a manifold flips the greedy action: it is an instance of policy churn (Schaul et al.,
2022). This repository shows, on small MDPs with linear features, that

1. the dynamics can be **attracted to a tie manifold and slide along it**, so that the greedy action
   keeps flipping at a steady rate, and
2. the **fraction of time each action is greedy** is predicted by the Filippov weight of the two
   one-sided drifts, as in Borkar's (2026) analysis of stochastic approximation with drift
   discontinuous across a manifold.

## Background and attribution

Borkar (2026) analyses constant-step stochastic gradient descent whose drift is discontinuous across a
lower-dimensional manifold M and points into M from both sides. The iterates are then held near M and
move along it under the Filippov combination of the one-sided drifts h⁺ and h⁻ (Filippov, 1988):

$$
g = \frac{\lVert h^-_n\rVert\, h^+ + \lVert h^+_n\rVert\, h^-}{\lVert h^+_n\rVert + \lVert h^-_n\rVert},
\qquad
\alpha = \frac{\lVert h^-_n\rVert}{\lVert h^+_n\rVert + \lVert h^-_n\rVert},
$$

where h±ₙ are the components normal to M and α is the relative frequency of the '+' side.

Borkar's paper does not discuss reinforcement learning, Q-learning or policy churn. Its main
theorems (Sections 4–5) assume SGD on a loss, whereas the temporal-difference drift is not the gradient
of any loss; only the general sliding dynamics of its Section 3, which hold for any discontinuous
stochastic-approximation drift, are used here. The observation that argmax-based targets create such
discontinuities, and that churn follows the Filippov occupancy, is this repository's.

## Setting

Linear action values Q_θ(s, a) = φ(s, a)ᵀθ, a frozen target network θ̄ (the inner loop between target
synchronisations) and the Double-SOR target

$$
y_\omega(s,a;\theta) = \omega\Bigl(r + \gamma \sum_{s'} P(s'\mid s,a)\, \bar Q\bigl(s', \arg\max_b Q_\theta(s',b)\bigr)\Bigr)
  + (1-\omega)\, \bar Q\bigl(s, \arg\max_c Q_\theta(s,c)\bigr),
$$

with ω = 1 giving Double Q-learning's target and ω ≠ 1 Double SOR's (Kamanchi et al., 2020). The
expected drift h(θ) = Σ d(s, a) φ(s, a) [y_ω − Q_θ(s, a)] jumps across each tie hyperplane
M_s = {Q_θ(s, 1) = Q_θ(s, 2)}, because the argmax selects which target value Q̄ is used.

**Instances.** 300 random MDPs (6 states, 2 actions, Dirichlet(0.5) transitions, normal rewards,
4-dimensional normal features, Dirichlet state–action weighting d, random θ̄ and θ₀), γ = 0.9, and
ω ∈ {0.7, 1.0, 1.3, 1.6}.

**Procedure** (per instance and ω):
1. integrate the expected drift (Euler, step 0.01, 40,000 steps) and detect **codimension-1
   sliding**: in the last quarter, exactly one state's greedy action flips at least 20 times while θ is
   at rest;
2. compute α from the two one-sided drifts at θ projected onto M_s, and check the attraction
   condition h⁺ₙ < 0 < h⁻ₙ;
3. run the **stochastic algorithm** from the same start (sampled (s, a) ~ d and s′ ~ P, step size
   0.002, 300,000 updates) and measure, over the second half, the fraction of updates with action 1
   greedy at the sliding state and the churn rate (greedy flips per update).

## Results

| ω | instances with codimension-1 sliding | ODE occupancy − α (median abs.) | stochastic occupancy − α (median abs.) | correlation (α, stochastic occupancy) | within 0.05 | churn (flips per update, median) |
|---|---|---|---|---|---|---|
| 0.7 | 61 / 300 (20%) | 0.001 | 0.023 | 0.90 | 64% | 0.030 |
| 1.0 | 45 / 300 (15%) | 0.001 | 0.041 | 0.95 | 60% | 0.030 |
| 1.3 | 42 / 300 (14%) | 0.001 | 0.035 | 0.97 | 69% | 0.035 |
| 1.6 | 58 / 300 (19%) | 0.001 | 0.047 | 0.73 | 52% | 0.028 |
| all | 206 / 1200 (17%) | 0.001 | 0.040 | 0.88 | 61% | 0.032 |

| weaker normal push min(‖h⁺ₙ‖, ‖h⁻ₙ‖) | cases | median abs. error (stochastic) |
|---|---|---|
| [0, 0.005) | 48 | 0.103 |
| [0.005, 0.02) | 57 | 0.043 |
| [0.02, inf) | 101 | 0.019 |

Detected chattering cases failing the Filippov attraction condition: 0.

![Occupancy against the Filippov weight](figures/occupancy_vs_alpha.png)

![Example](figures/example_sliding.png)

![Prediction error against the strength of the normal drift](figures/error_vs_push.png)

## Observations

1. **Sliding on tie manifolds occurs in about one instance in six** (14–20% depending on ω). In those
   instances the greedy action at one state flips on about 3% of stochastic updates, sustained by the
   expected dynamics rather than by noise. Every detected case satisfies the Filippov attraction
   condition.
2. **The Filippov weight predicts the split of time between the two actions.** For the expected
   dynamics the agreement is within about 0.001; for the stochastic algorithm the median absolute error
   is 0.04 and the correlation 0.88 (0.90–0.97 for ω ≤ 1.3).
3. **Deviations occur where the small-step limit is not reached.** When the weaker of the two normal
   pushes is strong (> 0.02) the median error is 0.019; when it is weak (< 0.005), noise is comparable
   to the drift at this step size and the median error rises to 0.10.
4. **Over-relaxation does not remove sliding.** Its frequency is non-monotone in ω and the churn
   rate stays at about 3%. Agreement is weaker at ω = 1.6 (correlation 0.73); a plausible reason is
   that SOR's (1 − ω) Q̄(s, argmax) term adds a second discontinuity at the same state.

## Limitations

- The target network is frozen. With a moving target the sliding points move as well; Borkar (2026,
  Section 5) discusses such slowly varying structure, but it is not tested here.
- Linear features and small random MDPs. Neural networks add discontinuities of their own (ReLU
  activation boundaries) that are not studied.
- Only codimension-1 sliding is analysed; instances where several states chatter at once are excluded.
- The detection thresholds (20 flips, θ at rest) are choices that affect the counts in the first table
  but not the occupancy comparison.

## Usage

```bash
pip install -r requirements.txt
python -m pytest              # three checks, about two seconds
python run.py                 # 1,200 runs, about 15 minutes on 3 cores
python analyze.py             # results/summary.md and figures/
```

`results/results.jsonl` holds one record per instance and ω, with the predicted α, the attraction
check, the normal drift components and the measured occupancies and churn.

## Layout

```
filippov/dynamics.py   instances, expected drift, Euler and stochastic dynamics, sliding detection, α
run.py                 the experiment grid
analyze.py             tables and figures
tests/                 a one-dimensional Filippov check and consistency checks
results/, figures/     outputs
```

## References

- Borkar, V. S. (2026). Stochastic gradient descent with discontinuity across a manifold.
  *arXiv:2608.07618*.
- Borkar, V. S. (2022). *Stochastic Approximation: A Dynamical Systems Viewpoint* (2nd ed.). Springer.
- Filippov, A. F. (1988). *Differential Equations with Discontinuous Righthand Sides*. Kluwer
  Academic.
- Kamanchi, C., Diddigi, R. B., & Bhatnagar, S. (2020). Successive over-relaxation Q-learning.
  *IEEE Control Systems Letters*, 4(1).
- Schaul, T., Barreto, A., Quan, J., & Ostrovski, G. (2022). The phenomenon of policy churn.
  *Advances in Neural Information Processing Systems (NeurIPS)*.
- van Hasselt, H., Guez, A., & Silver, D. (2016). Deep reinforcement learning with double
  Q-learning. *AAAI Conference on Artificial Intelligence*.
