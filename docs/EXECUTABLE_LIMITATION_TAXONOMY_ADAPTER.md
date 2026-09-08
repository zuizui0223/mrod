# Executable projection into the shared limitation-to-action taxonomy

Status: **internal architecture / regression adapter**. This does not change the active MEE manuscript, public API, planner objective, frozen G2 results or figures.

The shared Boundary/MROD taxonomy in `docs/limitation_action_taxonomy_v1.json` is a canonical **state/action vocabulary**, not a new optimizer and not a replacement for the richer diagnostic receipts already emitted by individual modules.

## 1. Why an adapter is needed

Before this adapter, the core `limitation_action_report.py` already emitted several action strings directly, while later replication, calibration, routing and operational audits were connected to the shared taxonomy mainly by documentation and regression markers.

`causal_model/limitation_taxonomy_adapter.py` makes that relationship executable. It reads the canonical JSON and projects existing diagnostic receipts onto canonical `(axis, state, action)` triples. It does **not** recompute mutual information, choose a candidate, invent a scenario prior, or collapse multiple diagnostics to one severity.

Each projection contains two different objects:

- `signals`: canonical taxonomy state/action triples;
- `detail_actions`: more specific implementation messages already emitted by the underlying diagnostic.

This distinction matters. For example, an exhausted-budget core report can retain the canonical action

```text
retain_best_candidate_for_future_budget
```

while also preserving the implementation-level message

```text
report_budget_limit.
```

The latter is not silently promoted to a new canonical taxonomy state.

## 2. Core MROD projection

A `SpecificationStatus` can project simultaneously to several axes. For example,

```text
current_state_estimability = estimable
target_identification = target_unresolved
candidate_prediction_coverage = complete
information_horizon = positive_singleton
resource_operational = budget_exhausted_best_known
```

is a coherent state: the scientific target is unresolved, a best next measurement is known, but it cannot be acquired under the current budget.

Zero singleton information is projected according to the already-declared joint-information status:

```text
joint not audited -> audit_joint_candidate_information_before_sequence_limit
joint positive     -> use_nonmyopic_bundle_or_sequence_design
joint zero         -> redesign_or_expand_measurement_vocabulary
```

No precedence is imposed among these coordinates.

## 3. Specification sensitivity

`LimitationActionReport` already distinguishes target-state stability from recommendation stability across a declared specification set. The adapter issues one of the shared robustness states only when both coordinates are interpretable.

Examples:

```text
stable target status + stable common best
    -> conclusion_stable_recommendation_stable

stable target status + candidate ranking changes
    -> conclusion_stable_recommendation_sensitive
```

If a recommendation is simply unavailable, the adapter does not manufacture a robustness classification.

## 4. Replication versus switching

For a `ReplicationSwitchAudit`, two actions can coexist:

```text
one repeat still has positive target information
    -> replication_still_informative
    -> compare_repeat_vs_switch

an alternative exceeds the whole repeat-only information ceiling
    -> alternative_beats_repeat_ceiling
    -> switch_observation_type
```

The second statement is stronger, but it does not erase the first. The taxonomy therefore records both rather than using a single winner-takes-all status.

## 5. Adaptive information and operational cost

For an already-selected adaptive tree, `AdaptiveExpectedCostAudit` can license independently:

```text
adaptive information advantage
    -> report_information_gain_not_cost_optimality

positive expected cost saving versus an information-matched fixed bundle
    -> report_skipped_measurements_if_attributable
```

The selected tree is still not claimed to minimize expected cost.

For `ScenarioCostTradeoffAudit`, equal-information/equal-worst-path policies with different expected costs trigger

```text
equal_information_cost_tie_sensitive
    -> do_not_call_selected_tree_cost_optimal.
```

If expected-cost preference crosses across declared scenarios, the additional state is

```text
cost_preference_crosses_scenarios
    -> declare_meta_prior_or_robust_cost_rule_before_scalar_cost_ranking.
```

These are claim ceilings, not a new robust-cost design rule.

## 6. Composition rule

`combine_taxonomy_projections` takes the union of canonical signals and detail actions. It deliberately has no priority ordering.

A single study can therefore report at once that:

- the target remains unresolved;
- a positive singleton candidate exists;
- the current budget is exhausted;
- expected operational cost preference is scenario-sensitive.

Those statements answer different questions and should not be compressed into one `limitation severity`.

## 7. Scope guard

The adapter does not establish that:

- the declared mechanism family is exhaustive;
- a canonical action is globally optimal outside its diagnostic assumptions;
- information, causal value and operational cost share a utility scale;
- scenario-specific expected costs may be averaged without a declared meta-rule;
- evidence-integrity states owned by REC/TNOA are automatically inferred by MROD;
- observation-structure states owned by Boundary are reconstructed from MROD outputs alone.

It is only an executable interface between diagnostics that already exist and the shared cross-program state/action vocabulary.
