# Robust fixed-bundle observation design with an explicit joint likelihood

Status: optional finite-scenario extension of `ROBUST_OBSERVATION_DESIGN.md`.
Publication-facing mechanism MROD, public exports, stopping rules and manuscript
claims are unchanged. All included examples use declared synthetic likelihoods.

## Why another interface is required

Separate matrices P(Q_j|world) do not determine P(Q_1,...,Q_m|world). An informative
combination cannot be constructed by multiplying marginal matrices unless the
relevant conditional independence has been separately justified. This module
therefore requires an explicit JOINT probability table in every named scenario.
It never manufactures a joint law from the existing singleton API.

Supply `candidate_order`, positive additive integer acquisition costs, budget,
finite world rows with a predeclared target, and `JointCalibrationScenario`
objects. Each scenario has positive world weights, joint outcome labels aligned
with candidate_order, one complete probability row per world, and provenance.
Events can omit structurally impossible combinations; this support completeness
is still a scientific assumption, not something a schema can verify. Across
scenarios use the same labelled event vocabulary and encode differing impossibilities
with zero probabilities. Event row order may differ without changing meaning.

## Objective and algorithm

Enumerate every budget-feasible precommitted bundle B, including the empty bundle.
For each scenario s, marginalize its declared joint law to that bundle and compute

    V_s(B)=I_s(T;Q_B).

The oracle benchmark is the best bundle under the SAME budget and scenario:

    regret_s(B)=max_{B' affordable} V_s(B')-V_s(B).

Return all uniformly best bundles when they exist, and all deterministic
minimax-regret bundles under the explicit alternative robust criterion

    argmin_B max_s regret_s(B).

This reuses the existing noisy-score and paired-scenario robust implementation.
The utility scale is raw information bits, not an invented prior over scenarios,
information per dollar, or I/H averaged across scenarios. Regret itself need not
be monotone in budget because the scenario-specific oracle also gains options.

The solver is exhaustive, not a large-scale approximation: at most eight candidate
observations are accepted, giving at most 256 subsets. It does not optimize an
outcome-dependent adaptive decision tree, repeated measurements, changing costs,
or calibration-learning actions. A bundle can be acquired in any order, but this
precommitted optimizer does not reallocate resources based on interim outcomes.

## Budget reverses the correct design

Take four synthetic worlds (u,v), with binary target T=u XOR v. Define three
observations U,V,direct. U and V reveal the corresponding world coordinates;
direct has a 20% symmetric error rate for T. The implementation supplies the full
joint law explicitly. The two declared scenarios use uniform world weights and
weights (9,1,1,9), respectively.

| Bundle | Uniform-target scenario | Rare-target scenario |
|---|---:|---:|
| U alone | 0 bit | 0 bit |
| V alone | 0 bit | 0 bit |
| direct alone | 0.278071905 bit | 0.104818278 bit |
| U,V together | 1 bit | 0.468995594 bit |
| direct,U or direct,V | 0.278071905 bit | 0.104818278 bit |

For one unit of budget, direct is uniformly optimal. For two units, U,V is
uniformly optimal and has zero worst regret. Starting greedily with direct
therefore loses 0.721928095 bits in the uniform scenario when only one extra
unit remains. The initially zero-valued U,V observations should not have been
removed from the candidate vocabulary just because they have zero singleton MI.

This is a property of the declared XOR witness, not an empirical finding about
ecological instruments. It is a regression test for joint-information awareness,
not a claim that all greedy policies fail or that all joint observations help.

A second test holds singleton laws fixed while changing their dependence:

    T=0: (A,B) is 00 or 11 with equal probabilities
    T=1: (A,B) is 01 or 10 with equal probabilities

versus uniformly independent A,B in both targets. All singleton target-information
scores are zero in both models, but joint information is 1 bit in the first
model and zero in the second. Marginal calibration cannot distinguish them.

## Missing data and scope boundaries

A missing joint matrix leaves nonempty bundle scores non-estimable in that
scenario. Incomplete feasible vocabulary produces no full-vocabulary uniform
winner or minimax recommendation. Malformed probabilities, missing target labels,
duplicate names and inconsistent event vocabularies are rejected. Empty bundles
remain zero-information resource choices; at budget zero, selecting the empty
bundle is a budget fact, not a sequence-information limit at larger budgets.

Numerical comparisons use the existing explicit bit tolerance. Tiny roundoff
near zero information is not support elimination. No score or regret by itself
licenses point identification, transport of a measurement model, or a biological
report. Guarantees cover only enumerated scenarios and represented world support,
not their continuous hull or unmodelled calibration uncertainty.

## Prior-art boundary

Non-myopic subset selection and dependent-test design have established literature.
This implementation is a small exact, scenario-robust integration with the current
MROD contracts, not a new general algorithm or new discovery of XOR synergy.
See Krause and Guestrin, *Near-optimal Nonmyopic Value of Information in Graphical
Models* (UAI 2005; author manuscript https://arxiv.org/abs/1207.1394), and Chen,
Hassani and Krause, *Near-optimal Bayesian Active Learning with Correlated and
Noisy Tests* (2016; https://arxiv.org/abs/1605.07334). The latter explicitly studies
dependent noisy tests; its approximation guarantees are not claimed for this API.

## Reproduce

    python -m causal_model.joint_budgeted_design
    python -m pytest -q tests/test_joint_budgeted_design.py

Optional API: `JointCalibrationScenario`, `plan_joint_observation_budget`.
No top-level publication API symbols were added. See also the current README's
singleton-versus-joint information stopping boundary.
