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

## 3. Early-stop witness: same information, lower expected cost

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

This is operational value without an information advantage.

## 4. Context-routing witness: the two axes change with budget

The existing routing witness has `context`, `assay0`, and `assay1`, each with unit cost. Context itself has zero direct target information but tells the adaptive tree which assay is relevant.

For the balanced-context scenario:

| Pathwise budget | Adaptive information gap over best fixed | Selected adaptive expected cost | Cheapest fixed cost matching selected information | Cost saving |
|---:|---:|---:|---:|---:|
| 1 | 0 bit | 1 | 1 | 0 |
| 2 | 0.5 bit | 2 | none within budget | not comparable |
| 3 | 0 bit | 2 | 3 | 1 |

At budget 2, adaptivity has an information advantage but no fixed policy reaches the same information, so there is no like-for-like cost saving to report. At budget 3, the fixed class catches up in information by acquiring all three measurements, while the adaptive tree still measures context and only the relevant assay. Information gap is zero but operational saving remains one cost unit.

This extends the existing budget-local routing result:

```text
adaptive information value can disappear when budget grows,
while outcome-contingent avoidance of unnecessary measurements can still save resources.
```

## 5. Claim boundaries

- Expected cost is conditional on the supplied scenario weights and complete joint observation law.
- The audit does not verify laboratory/field costs, likelihood calibration, intervention compatibility or natural-system exhaustiveness.
- `acquisition_cost` is an abstract positive resource unit unless the caller supplies a scientifically defensible mapping to money, time, samples or another resource.
- The selected adaptive tree is **not** claimed to minimize expected cost.
- A cost saving against an information-matched fixed bundle is not a utility optimum; another adaptive tree may dominate it on cost, information, or both.
- Scenario-specific expected costs are not averaged across calibration scenarios without a declared meta-distribution.
- A lower expected cost does not license a biological or causal conclusion.

Cost-sensitive active learning, adaptive stochastic optimization, sequential Bayesian design and optimal stopping are established research areas. Relevant examples include Golovin & Krause (2011), *Adaptive Submodularity: Theory and Applications in Active Learning and Stochastic Optimization*, JAIR 42:427-486, and Cheng & Huan (2025), *Optimal Stopping for Sequential Bayesian Experimental Design*. This audit does not claim to invent cost-aware or stopping-aware design.

## 6. Reproduce

```bash
python -m examples.adaptive_expected_cost_report > adaptive_expected_cost_report.json
python -m pytest -q tests/test_adaptive_expected_cost_audit.py
```

The implementation is `causal_model/adaptive_expected_cost_audit.py`. It reuses the same joint scenarios, adaptive tree and fixed-bundle planner already used by the routing/budget audits; no separate marginal-likelihood approximation is introduced.
