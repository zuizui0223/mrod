# Scenario-wise expected cost does not define a unique adaptive optimum

Status: **internal claim-ceiling audit**. This note evaluates operational cost of registered information-equivalent adaptive trees. It does not change the adaptive planner, public API, active MEE manuscript, frozen G2 results or figures.

## 1. Why a scenario meta-prior matters

The existing expected-cost audit reports one cost per declared calibration/weight scenario:

```text
E_s[C_pi].
```

That is a vector, not a single scalar objective. If two policies exchange which scenario they are cheaper in, then a unique scalar expected-cost preference requires an additional decision rule, such as

- a probability distribution over scenarios;
- a worst-case expected-cost criterion;
- a regret criterion;
- or another explicitly declared utility.

This audit introduces none of those. It reports componentwise ordering and crossing only.

## 2. Crossing witness

Three target states have two unit-cost resolving queries. Either query followed by the other completely identifies the target and both trees have worst-path cost two.

Two scenarios differ only in target mass:

```text
common_heavy: rare/common/other = 0.1 / 0.8 / 0.1
rare_heavy:   rare/common/other = 0.8 / 0.1 / 0.1
```

The expected costs are

| Policy | common_heavy | rare_heavy |
|---|---:|---:|
| rare first | 1.9 | 1.2 |
| common first | 1.2 | 1.9 |

Target information and worst-path budget are the same for both trees in both scenarios. The cost preference reverses.

Therefore

```text
no scenario-wise uniform winner
+ no declared scenario meta-prior
=> no scalar expected-cost ranking is licensed by this audit.
```

For example, if one later declared a meta-weight `alpha` on `common_heavy`, the two cross-scenario mean costs would be

```text
rare first   = 1.2 + 0.7*alpha
common first = 1.9 - 0.7*alpha,
```

which cross at `alpha=0.5`. That calculation illustrates why the missing meta-rule matters; the repository does not adopt such a meta-prior automatically.

## 3. A positive control: uniform componentwise ordering

A second witness keeps the common target more frequent in every scenario:

```text
common_very_heavy:       0.1 / 0.8 / 0.1
common_moderately_heavy: 0.2 / 0.6 / 0.2.
```

Then

| Policy | very heavy | moderately heavy |
|---|---:|---:|
| rare first | 1.9 | 1.8 |
| common first | 1.2 | 1.4 |

`common first` is no more expensive in every declared scenario and strictly cheaper in both. A uniform componentwise statement is therefore licensed **among these registered policies** without averaging scenarios.

This still does not prove global expected-cost optimality over every possible adaptive tree.

## 4. Relation to the current planner

The current adaptive planner optimizes target-information regret under a hard pathwise budget. It does not optimize the scenario-wise expected-cost vector. Equal-information trees can therefore be selected according to search/frontier order rather than operational cost.

The distinction is now:

```text
target-information value
!= routing value
!= expected acquisition cost
!= a robust/scenario-averaged cost objective.
```

Each requires a separate reporting contract.

## 5. Claim boundaries

- Scenario weights describe world uncertainty **within** each scenario; they are not probabilities over scenarios.
- A policy can be cheaper in one scenario and more expensive in another.
- No scalar expected-cost optimum is claimed when componentwise preferences cross.
- A uniform winner among the compared policies is not proof of global policy-class optimality.
- The scenario likelihoods and abstract resource costs remain supplied assumptions, not field calibration.
- No biological, causal, fitness or adaptation conclusion follows from an operational cost comparison.

Cost-sensitive active learning, robust design, stochastic optimization and decision-making under model uncertainty are established fields. This audit adds an executable claim boundary to the existing MROD extensions; it does not claim a new robust-cost optimizer.

## 6. Reproduce

```bash
python -m examples.adaptive_cost_scenario_tradeoff_report
python -m pytest -q tests/test_adaptive_cost_scenario_tradeoff.py
```
