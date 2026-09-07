# Adaptive observation trees under fixed calibration scenarios

Status: optional finite-scenario extension. Synthetic examples only. Publication
mechanism MROD, top-level exports and manuscripts are unchanged. This extends
`JOINT_BUDGETED_DESIGN.md` without changing its fixed-bundle comparator.

## Scientific question

Can the outcome of an initial measurement tell us which subsequent measurement
will resolve the declared target? This is different from identifying the target
with the first observation. A zero-singleton-information measurement can have
routing value for a finite-budget policy.

## Input contract and objective

Use the existing `JointCalibrationScenario`: positive world weights, a complete
per-world joint outcome law, finite labelled support, and calibration provenance.
All scenarios share the world rows and candidate/event vocabularies. Do not build
a joint law by multiplying marginal calibrations unless conditional independence
has been justified separately. A single joint law across possible orders further
assumes noninvasive acquisition and an order-invariant underlying state. Destructive
measurements, interventions, drift and action-dependent calibration need a different
model. An input string cannot establish any of those empirical assumptions.

Positive additive integer acquisition costs include the first routing measurement.
Every realized root-to-leaf path must fit the budget; this is NOT an expected-cost
constraint. Each candidate can be measured once per path. The root chooses ONE
policy based on observed outcomes, never on the hidden scenario or true world.

For a tree pi, its transcript contains the actually chosen query names and outcomes.
Under each scenario s, compute

    U_s(pi) = I_s(T; transcript)
            = H_s(T) - sum_h P_s(h) H_s(T | terminal history h).

The oracle uses the same adaptive policy class and the same pathwise budget:

    O_s = max_pi U_s(pi),
    regret(pi) = max_s [O_s - U_s(pi)].

Return one deterministic ex-ante minimax-regret tree (near-ties within the declared
bit tolerance are broken by worst-path cost, then enumeration order). No meta-prior
is assigned to scenarios, and randomized policies are not optimized. A scenario
is held fixed across the entire tree. The ex-ante choice is not claimed to remain
optimal if one replaces its continuation by a newly solved conditional minimax
problem; doing so changes the commitment/decision criterion.

## Why the recursion keeps vectors

A history corresponds to a subset of labelled joint events, remaining queries and
remaining budget. At a terminal history define the UNCONDITIONAL residual
contribution for every scenario:

    L_s(h) = sum_t P_s(t,h) log2[P_s(h)/P_s(t,h)].

A query partitions that event subset by its outcome labels. A feasible continuation
combines one child tree for each supported outcome. Its residual VECTOR is the
sum of the child residual vectors. Keep all nondominated vectors; do not keep only
the child with smallest scalar worst-case residual or regret. Componentwise dominance
is safe because all ancestor operations add the same nonnegative-weight residual
contributions for each fixed scenario. Max over scenarios is taken only at the root.
This preserves coherent whole-tree scenarios instead of allowing a different
adversarial scenario at every branch.

The algorithm exhausts the deterministic tree class, with componentwise Pareto
pruning. It is small-scale: at most 4 candidates, 64 labelled events, 128 worlds
and 8 scenarios. A tree-combination cap raises `AdaptiveSearchLimitError`; it is
not reported as impossibility or as a completed approximate optimum. Logarithms
use floating-point arithmetic, so combinatorial completeness does not mean exact
real arithmetic. No probability tolerance removes a possible outcome branch.

## Routing witness: same two measurements, more information

Four synthetic worlds have a context c in {0,1} and target t in {0,1}. The context
query reveals c. In context 0, assay0 reveals t and assay1 is a fair coin; in context
1 the roles are reversed. A complete joint law is supplied explicitly.

The declared scenarios keep t uniform and give context probabilities 1/2, 3/4,
and 1/4. All three observations cost one unit, including context. At budget 2:

| Quantity | Context balanced | Context 0 common | Context 1 common |
|---|---:|---:|---:|
| Information from context alone | 0 | 0 | 0 |
| Best fixed bundle in each scenario | 0.5 | 0.75 | 0.75 |
| Adaptive tree | 1 | 1 | 1 |

The selected tree is

    measure context
      outcome 0 -> measure assay0 -> stop
      outcome 1 -> measure assay1 -> stop.

Thus the balanced-context information gain is 0.5 bit at the same worst-path cost
of two observations. The gain is not a free calibration measurement. At budget 3,
all observations can be acquired by a fixed bundle and this example's gap vanishes.
The earlier XOR witness also has no adaptive gain beyond its optimal two-test
bundle; adaptivity does not improve every design.

For a fair regret comparison the receipt evaluates fixed bundles against the SAME
adaptive oracle. Comparing each class's regret against its own moving oracle can
misrepresent improvement. Here the best fixed bundle's worst regret against the
adaptive oracle is 0.5 bit; the adaptive tree's is zero.

## Information within scenarios is not identification across scenarios

A separate adverse control has a perfect binary sensor in each of two models, but
one model reverses its labels. It yields 1 bit within BOTH scenarios. Yet the same
observed label corresponds to different targets across those scenarios. The executor
therefore retains the union of positive joint supports and reports two remaining
target values, NOT point identification. Low entropy, high information and zero
regret do not by themselves license an across-model biological conclusion.

Missing joint likelihoods produce no adaptive recommendation. Malformed supplied
likelihoods remain errors. Tiny positive likelihoods are retained, and underflow
raises rather than silently eliminating a world. Structural support, calibration
transport, finite-panel completeness and report reliability remain external.

## Execution and reproduction

    python -m causal_model.adaptive_joint_design
    python -m pytest -q tests/test_adaptive_joint_design.py

API: `plan_adaptive_joint_budget`, `execute_adaptive_policy`. The executor takes a
caller-provided `observe(selected_query_name)` callback and invokes only the query
selected by the actual preceding history. It does not inspect unselected outcomes
or run an instrument. Incompatible outcomes abort. The synthetic callback is not
a field observation, and a final support singleton concerns represented worlds only.

## Prior-art boundary

Adaptive test selection, correlated noisy tests and robust policy optimization are
established topics. This is a small exhaustive integration/audit for the repository's
joint-likelihood and scope contracts, not a novel general-purpose algorithm or an
empirical discovery. See Golovin, Krause & Ray (NIPS 2010), *Near-Optimal Bayesian
Active Learning with Noisy Observations*,
https://papers.nips.cc/paper_files/paper/2010/hash/1e6e0a04d20f50967c64dac2d639a577-Abstract.html;
and Rigter, Lacerda & Hawes (AAAI 2021), *Minimax Regret Optimisation for Robust
Planning in Uncertain Markov Decision Processes*, DOI 10.1609/aaai.v35i13.17417.
Those works' approximation guarantees are not imported into this API.
