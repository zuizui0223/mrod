# Budget-local value of adaptive routing

Status: optional diagnostic on the existing adaptive and fixed joint-likelihood planners. All registered examples are synthetic.

## Class-oracle comparison

At every declared pathwise budget `B` and scenario `s`, compare the best value available to the adaptive policy class with the best fixed bundle under the **same** budget:

```text
gap_s(B)
= max_adaptive I_s(T; transcript)
  - max_fixed I_s(T; bundle).
```

The gap is nonnegative because a fixed bundle is a special case available to the adaptive class. This is a scenario-specific class comparison, not the information of the single cross-scenario minimax-regret policy selected for deployment.

Implementation: `causal_model/adaptivity_budget_profile.py`.

## Context-routing witness

The existing context/assay example has three unit-cost observations:

```text
context, assay0, assay1.
```

The budget profile is:

| Budget | Adaptive gap, balanced context | Adaptive gap, 3/4-1/4 context |
|---:|---:|---:|
| 0 | 0 | 0 |
| 1 | 0 | 0 |
| 2 | 0.5 bit | 0.25 bit |
| 3 | 0 | 0 |

Thus the routing advantage exists only at the intermediate budget `B=2`.

Interpretation:

```text
B=1:
there is no resource left to use the route-dependent assay,
so paying for context alone cannot beat the best direct assay.

B=2:
context can route the second measurement,
so adaptive acquisition is strictly better.

B=3:
a fixed bundle can acquire context + both assays,
so it catches up to the adaptive class.
```

The result is therefore not "adaptivity is always better". Its value is budget-local.

## XOR adverse control

The XOR complementarity example has zero singleton target information for `U` and `V`, but the same second observation is needed regardless of the first outcome. The best fixed and adaptive classes coincide at every budget 0 through 3.

So

```text
zero singleton information
+
positive joint information
```

can exist without an adaptivity window.

## Claim boundary

- The profile uses the declared finite scenario set and explicit joint observation laws.
- It does not assign a probability over scenarios.
- It compares class oracles, not expected acquisition cost or randomized policies.
- A positive gap at one budget does not imply a positive gap at larger budgets.
- The profile is not evidence that a natural measurement has the synthetic likelihood used by the witness.
- Report licensing and identification across calibration scenarios remain separate.

## Reproduce

```bash
python -m pytest -q tests/test_adaptivity_budget_profile.py
```
