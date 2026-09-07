# MROD alignment to the cross-program limitation-to-action taxonomy

Status: **internal architecture / claim-ceiling alignment**. This file does not alter the active MEE manuscript, public API, frozen G2 protocol/results or planner objective.

Machine-readable contract: `docs/limitation_action_taxonomy_v1.json`  
Canonical SHA-256 shared with `zuizui0223/boundary`: `04f81c3400300cd30c05504477cea5a446b7b1839f9b49e4bea5c65db73a8002`

## 1. Why this contract exists

MROD now contains several logically distinct diagnostics: current-region estimability, target versus full-mechanism resolution, predictive candidate coverage, singleton and joint information, specification sensitivity, replication versus switching, calibration uncertainty, adaptive routing and operational cost audits.

These must not be collapsed into one `limitation severity`. The cross-program contract treats the scientific state as an orthogonal vector and maps each active coordinate to one or more licensed actions.

The active MEE method remains much narrower:

```text
current admissible mechanism region
-> validated singleton I(S;Q|A)/K
-> maximum-current positive candidate
-> observe outcome
-> condition and recompute.
```

The taxonomy is a reporting and extension contract, not a replacement optimizer.

## 2. Existing MROD action strings are part of the shared contract

The machine-readable taxonomy deliberately reuses the actions already emitted by `causal_model/limitation_action_report.py` where that prototype owns the state:

```text
repair_or_reestimate_current_admissible_region
stop_fully_resolved
stop_declared_target_resolved_report_residual_ambiguity
expand_candidate_vocabulary
identify_candidate_outcome_models
resolve_nonestimable_candidates_before_global_ranking
measure_best_candidate
audit_joint_candidate_information_before_sequence_limit
use_nonmyopic_bundle_or_sequence_design
redesign_or_expand_measurement_vocabulary
retain_best_candidate_for_future_budget
```

This makes the architecture testable: the conceptual taxonomy cannot silently rename the operational actions while the implementation continues returning older semantics.

## 3. Extension coordinates remain separate from the mainline selector

The shared taxonomy also names later internal claim ceilings that are not outputs of the mainline greedy API:

- `compare_repeat_vs_switch` and `switch_observation_type` belong to replication/switch audits;
- `report_finite_budget_recommendation_not_universal_impossibility` belongs to calibration/observation-law sensitivity;
- `report_information_gain_not_cost_optimality` and `report_skipped_measurements_if_attributable` belong to adaptive operational audits;
- `do_not_call_selected_tree_cost_optimal` records the equal-information tie counterexample;
- `declare_meta_prior_or_robust_cost_rule_before_scalar_cost_ranking` records the scenario-wise expected-cost crossing result.

These actions prevent overclaiming; they do not add a cost-aware or robust optimizer to MROD.

## 4. Composition, not precedence

Multiple actions can be simultaneously licensed. Examples:

- budget exhausted + validated best candidate -> report the resource limit **and** retain the best candidate for a future budget;
- partial predictive coverage + positive estimable candidate -> resolve missing candidate models **and** report only a provisional best among estimable candidates;
- target resolved + residual full-mechanism entropy -> stop for the declared question **and** report residual out-of-target ambiguity;
- zero singleton information + positive joint information -> do not call an information limit; use a non-myopic bundle or sequence design;
- conclusion sensitivity + stable recommendation -> report the identification breakdown **and** the stable follow-up candidate;
- expected-cost preference crossing scenarios -> keep the scenario-wise cost vector and require an explicit meta-decision rule before scalar ranking.

No generic priority ordering is imposed across these axes.

## 5. Boundary interface

The same JSON is stored in the Boundary repository. Boundary primarily owns the observation-structure and target-identification coordinates, including exact redundancy, target factorization on observation fibres and identification sensitivity. MROD primarily owns predictive coverage, information horizon, recommendation stability and resource/operational diagnostics.

The shared rule is:

> preserve the type of limitation, state what inference it blocks, and name the next scientific action without converting unknown, zero information, budget exhaustion and specification dependence into one label.

## 6. Scope guard

This contract does not claim that:

- the axes are independent random variables;
- every study needs every axis;
- a state/action pair is globally optimal outside the declared model and candidate family;
- operational cost, target information and causal value have a common utility scale;
- the active MEE manuscript validates the optional routing, replication, calibration or cost extensions;
- cost-aware, robust, goal-oriented or non-myopic design is new to this repository.

The taxonomy is an internal interface and regression contract for keeping the growing limitation logic logically separated.
