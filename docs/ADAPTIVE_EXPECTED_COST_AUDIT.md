# Expected acquisition cost is a separate adaptive-design output

Status: **internal finite-model audit** on top of the existing adaptive and fixed joint-likelihood planners. It does not alter the active MEE manuscript, publication objective, public exports, frozen G2 results or figures.

## 1. Why pathwise budget is not expected cost

The current adaptive planner enforces a hard pathwise budget. If the selected tree is `pi`, each internal node `v` has a declared acquisition cost `c(v)`. Under one declared joint scenario `s`, define

```text
E_s[C_pi]
= sum_v P_s(reach v) c(v).
```

Equivalently, enumerate positive-probability joint outcomes and average the realized path cost. The implementation checks both the scenario probability normalization and

```text
E_s[C_pi] <= max positive path cost <= declared pathwise budget.
```

This is an **evaluation of the already-selected tree**, not a new optimization objective. The existing planner minimizes information regret and uses worst-path cost only as a tie-break among near-equal root regrets. It does not search for the minimum expected-cost tree at fixed information.

Expected costs are reported separately for every calibration scenario. No probability distribution over scenarios is invented.

## 2. Information advantage and cost saving are different axes

For each scenario, the audit also finds the cheapest precommitted fixed bundle within the same budget whose target information is at least the selected adaptive tree's information. If such a bundle exists, report

```text
fixed acquisition cost - adaptive expected acquisition cost.
```

If no fixed bundle under the budget matches the adaptive information, the cost-saving comparison is left undefined rather than treating a lower-information fixed bundle as equivalent.

A fixed bundle pays its full declared additive cost because it is precommitted. If a supposedly fixed protocol conditionally omits later measurements after seeing earlier outcomes, it is an adaptive tree and should be modelled as one.

## 3. Attribute the saving to measurements that were avoided

Suppose a cheapest information-matched fixed reference bundle `F` contains every query that the adaptive tree can ever acquire with positive probability in the scenario. Because the adaptive tree never repeats a query,

```text
E_s[C_pi]
= sum_{q in F} c_q Pr_s(q acquired).
```

Therefore linearity of expectation gives the exact bookkeeping identity

```text
C(F) - E_s[C_pi]
= sum_{q in F} c_q [1-Pr_s(q acquired)].
```

No independence between query-acquisition indicators is required. The audit reports each acquisition probability, skip probability and expected saving contribution.

This attribution is withheld when no cheapest information-matched fixed bundle contains the adaptive query support. A numerical cost gap can still exist in that case, but labelling it as savings from specific omitted measurements would be misleading because the two policies may use different measurement vocabularies.

## 4. Early-stop witness: same information, lower expected cost

Three equally weighted targets are observed through two unit-cost measurements.

```text
screen:
    target zero -> zero
    target one/two -> rest

resolve:
    target zero/one -> one
    target two -> two
```

Neither singleton completely identifies all three targets. The pair does. An adaptive budget-2 tree can stop after one observation on one one-third-probability branch and uses two observations otherwise.

Therefore

```text
selected adaptive information = log2(3) bits
best fixed information         = log2(3) bits
adaptive information gap       = 0
adaptive expected cost         = 5/3
cheapest information-matched fixed cost = 2
expected cost saving           = 1/3 cost unit.
```

Whichever measurement the selected tree places second is skipped on the one-third early-stop branch. The attribution therefore returns one skipped unit-cost measurement with probability `1/3`; the other is always acquired. This is operational value without an information advantage.

## 5. Context-routing witness: the two axes change with budget

The existing routing witness has `context`, `assay0`, and `assay1`, each with unit cost. Context itself has zero direct target information but tells the adaptive tree which assay is relevant.

For the balanced-context scenario:

| Pathwise budget | Adaptive information gap over best fixed | Selected adaptive expected cost | Cheapest fixed cost matching selected information | Cost saving |
|---:|---:|---:|---:|---:|
| 1 | 0 bit | 1 | 1 | 0 |
| 2 | 0.5 bit | 2 | none within budget | not comparable |
| 3 | 0 bit | 2 | 3 | 1 |

At budget 2, adaptivity has an information advantage but no fixed policy reaches the same information, so there is no like-for-like cost saving to report. At budget 3, the fixed class catches up in information by acquiring all three measurements, while the adaptive tree still measures context and only the relevant assay. Information gap is zero but operational saving remains one cost unit.

In the balanced scenario at budget 3,

```text
Pr(context acquired)=1
Pr(assay0 acquired)=1/2
Pr(assay1 acquired)=1/2.
```

Relative to the fixed bundle `(context, assay0, assay1)`, the skipped-assay contributions are `1/2 + 1/2 = 1` cost unit.

This extends the existing budget-local routing result:

```text
adaptive information value can disappear when budget grows,
while outcome-contingent avoidance of unnecessary measurements can still save resources.
```

## 6. Unequal costs make the saving scenario-dependent

Keep the same routing likelihood but set abstract costs to

```text
context = 2
assay0  = 1
assay1  = 3.
```

A fixed full-information bundle costs `6`. The adaptive route costs `2` plus only the relevant assay. Hence expected cost depends on the scenario's context frequency:

| Scenario | Expected adaptive cost | Information-matched fixed cost | Saving |
|---|---:|---:|---:|
| balanced context | 4.0 | 6 | 2.0 |
| context0 common (3/4) | 3.5 | 6 | 2.5 |
| context1 common (3/4) | 4.5 | 6 | 1.5 |

For the balanced scenario, assay0 and assay1 are each skipped half the time, contributing `0.5*1 + 0.5*3 = 2` units of expected saving. For the context0-common scenario, the expensive assay1 is skipped three-quarters of the time, so the expected saving is larger. This is why expected cost is scenario-specific even when the selected tree and its information are unchanged.

## 7. Equal information and equal worst-path cost do not imply expected-cost optimality

The planner does not use expected acquisition cost in its objective or in frontier pruning. This matters even when two adaptive trees have identical target information and identical worst-path cost.

A controlled three-target witness uses target masses

```text
rare   = 0.1
common = 0.8
other  = 0.1.
```

Two unit-cost measurements are available. `rare_split` isolates the rare target and `common_split` isolates the common target. Either measurement followed by the other fully resolves the target, so both two-stage trees have the same target information and worst-path cost `2`.

If `rare_split` is first, only the 0.1-mass branch terminates after one observation:

```text
E[C | rare_split first] = 1 + 0.9 = 1.9.
```

If `common_split` is first, the 0.8-mass branch terminates immediately:

```text
E[C | common_split first] = 1 + 0.2 = 1.2.
```

The current planner can retain different equal-information trees when the candidate order is reversed because expected cost is not part of the optimization or tie-break. This is an executable **claim ceiling**, not a bug fix to the planner:

```text
same target information
+ same worst-path budget
!= same expected operational cost.
```

Therefore the expected-cost audit must not be read as evidence that the selected adaptive tree is resource-optimal. A future cost-aware planner would be a distinct optimization problem and belongs to the established literature on cost-sensitive active learning, adaptive stochastic optimization and optimal stopping.

## 8. Claim boundaries

- Expected cost is conditional on the supplied scenario weights and complete joint observation law.
- The audit does not verify laboratory/field costs, likelihood calibration, intervention compatibility or natural-system exhaustiveness.
- `acquisition_cost` is an abstract positive resource unit unless the caller supplies a scientifically defensible mapping to money, time, samples or another resource.
- The selected adaptive tree is **not** claimed to minimize expected cost.
- Candidate ordering can change which equal-information/equal-worst-cost tree is retained; such changes are operationally relevant but do not change the information objective.
- A cost saving against an information-matched fixed bundle is not a utility optimum; another adaptive tree may dominate it on cost, information, or both.
- Skipped-measurement attribution is issued only when an information-matched fixed reference bundle contains the adaptive query support.
- Scenario-specific expected costs are not averaged across calibration scenarios without a declared meta-distribution.
- A lower expected cost does not license a biological or causal conclusion.

Cost-sensitive active learning, adaptive stochastic optimization, sequential Bayesian design and optimal stopping are established research areas. Relevant examples include Golovin & Krause (2011), *Adaptive Submodularity: Theory and Applications in Active Learning and Stochastic Optimization*, JAIR 42:427-486, and Cheng & Huan (2025), *Optimal Stopping for Sequential Bayesian Experimental Design*. This audit does not claim to invent cost-aware or stopping-aware design.

## 9. Reproduce

```bash
python -m examples.adaptive_expected_cost_report > adaptive_expected_cost_report.json
python -m pytest -q tests/test_adaptive_expected_cost_audit.py
```

The implementation is `causal_model/adaptive_expected_cost_audit.py`. It reuses the same joint scenarios, adaptive tree and fixed-bundle planner already used by the routing/budget audits; no separate marginal-likelihood approximation is introduced.
