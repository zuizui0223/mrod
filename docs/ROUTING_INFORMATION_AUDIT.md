# Routing information versus direct target information

Status: optional diagnostic on top of the existing adaptive joint-likelihood design. It does not replace mechanism MROD, target information value, or the adaptive planner objective.

## Three quantities that must not be collapsed

For a selected adaptive policy with root observation `Q1` and full transcript `H`, target information obeys the ordinary chain rule

```text
I(T;H) = I(T;Q1) + [I(T;H)-I(T;Q1)].
```

The implementation reports the first term as **root direct target information** and the bracketed term as **conditional continuation target information**.

A third quantity is deliberately different. Root outcomes can select different immediate continuation actions. Let `A2` be the next query name chosen by the already-selected policy, with `STOP` as an action when a branch terminates. The scenario-specific

```text
H(A2)
```

is reported as **routing-action entropy**. It is entropy of an action identity, not Shannon information about the biological target and not a new MROD utility.

Implementation: `causal_model/adaptive_routing_audit.py`.

## Registered routing witness

The existing synthetic context/assay witness has three observations:

```text
context, assay0, assay1
```

The target is independent of context. Therefore, in every declared scenario,

```text
I(T;context)=0.
```

The selected budget-2 adaptive tree is

```text
context=0 -> assay0
context=1 -> assay1.
```

For the balanced-context scenario:

```text
root direct target information       = 0 bit
full adaptive-policy target info      = 1 bit
conditional continuation information = 1 bit
routing-action entropy                = 1 bit
best fixed budget-2 information       = 0.5 bit
selected adaptive gain over fixed     = 0.5 bit.
```

For context probabilities 3/4 versus 1/4, routing-action entropy is

```text
h2(1/4)=0.8112781244591328... bit,
```

while the adaptive policy still gets 1 bit and the best fixed pair gets 0.75 bit.

Thus the first observation is useful here without directly informing the target: its outcome chooses which assay is informative.

## Why zero direct information is not enough

The XOR adverse control is equally important. Observations `U` and `V` each have

```text
I(T;U)=I(T;V)=0,
```

but a budget-2 policy can measure `U` and then `V` on every branch. The immediate second action does **not** depend on the root outcome. Hence

```text
routing-action entropy = 0
branch-dependent continuation = false
adaptive gain over the best fixed pair = 0.
```

Continuation information is positive, but it is ordinary complementary information rather than adaptive routing value.

Therefore

```text
zero direct target information
+
positive later information
```

is not sufficient to claim that adaptivity helped.

## Operational routing witness

`audit_adaptive_routing` labels a selected tree as `routing_without_direct_target_information` only when all of the following hold:

1. the selected root has zero direct target information in every declared scenario within the numerical comparison tolerance;
2. at least two possible root outcomes select different immediate next actions;
3. at least one declared scenario has positive conditional continuation target information.

The receipt separately reports the selected policy's signed gain relative to the best fixed bundle in each scenario. A routing witness does not imply that this gain is positive in every possible robust-decision problem.

## Claim boundaries

- Scenario likelihoods and world weights are supplied inputs, not empirically validated by this audit.
- `H(A2)` is not target information, mechanism information, causal information, or a report licence.
- High routing entropy need not imply a large adaptive gain; action identities can differ while their scientific values are nearly equal.
- A zero routing entropy at the root does not prove that deeper branches cannot adapt later.
- The audit concerns the selected ex-ante policy. It does not prove a general theorem that routing observations are novel or universally optimal.
- Point identification across calibration scenarios, support completeness and evidence-admissible reporting remain separate checks.

## Reproduce

```bash
python -m pytest -q tests/test_adaptive_routing_audit.py
```

Related files:

```text
causal_model/adaptive_joint_design.py
causal_model/joint_budgeted_design.py
docs/ADAPTIVE_JOINT_DESIGN.md
```
