from pathlib import Path

from causal_model.adaptive_cost_scenario_tradeoff import (
    crossing_scenario_witness,
    uniform_scenario_witness,
)
from causal_model.adaptive_expected_cost_audit import synthetic_example as cost_example
from causal_model.limitation_action_report import (
    CandidateScore,
    SpecificationInput,
    build_limitation_action_report,
    classify_specification,
)
from causal_model.limitation_taxonomy_adapter import (
    combine_taxonomy_projections,
    project_adaptive_expected_cost,
    project_limitation_action_report,
    project_replication_switch,
    project_scenario_cost_tradeoff,
    project_specification_status,
)
from causal_model.replication_switch_audit import audit_replication_switch
from examples.replication_switch_report import current_weights, three_program_model


def _states(projection):
    return set(projection.canonical_states)


def _actions(projection):
    return set(projection.canonical_actions)


def test_core_status_projects_orthogonal_states_and_keeps_detail_actions_separate():
    status = classify_specification(
        SpecificationInput(
            "budgeted",
            1.0,
            (CandidateScore("q", True, 0.5),),
            budget_remaining=False,
        )
    )
    projection = project_specification_status(status)
    assert {
        ("current_state_estimability", "estimable"),
        ("target_identification", "target_unresolved"),
        ("candidate_prediction_coverage", "complete"),
        ("information_horizon", "positive_singleton"),
        ("resource_operational", "budget_exhausted_best_known"),
    } <= _states(projection)
    assert "retain_best_candidate_for_future_budget" in _actions(projection)
    assert "report_budget_limit" in projection.detail_actions
    assert "report_budget_limit" not in projection.canonical_actions


def test_zero_singleton_joint_positive_maps_to_nonmyopic_action():
    status = classify_specification(
        SpecificationInput(
            "synergy",
            1.0,
            (
                CandidateScore("q1", True, 0.0),
                CandidateScore("q2", True, 0.0),
            ),
            joint_candidate_information_value=1.0,
        )
    )
    projection = project_specification_status(status)
    assert ("information_horizon", "zero_singleton_joint_positive") in _states(projection)
    assert "use_nonmyopic_bundle_or_sequence_design" in _actions(projection)
    assert "report_joint_information_despite_zero_singletons" in projection.detail_actions


def test_cross_specification_ranking_sensitivity_projects_canonical_robustness_state():
    report = build_limitation_action_report(
        (
            SpecificationInput(
                "a",
                1.0,
                (CandidateScore("q1", True, 0.8), CandidateScore("q2", True, 0.2)),
            ),
            SpecificationInput(
                "b",
                1.0,
                (CandidateScore("q1", True, 0.2), CandidateScore("q2", True, 0.8)),
            ),
        )
    )
    projection = project_limitation_action_report(report)
    assert (
        "specification_robustness",
        "conclusion_stable_recommendation_sensitive",
    ) in _states(projection)
    assert "report_no_unique_next_observation" in _actions(projection)


def test_replication_switch_can_license_both_repeat_and_switch_actions():
    rows, contact, physiology = three_program_model()
    weights, _ = current_weights(("present",))
    audit = audit_replication_switch(
        rows,
        contact,
        (physiology,),
        target_columns=("program",),
        weights=weights,
        support_reference="synthetic adapter test",
        weight_reference="conditioned on contact-present",
        conditional_iid_reference="fresh contact readings iid given fixed program",
        future_likelihood_reference="both channels independent of past readings given full world",
    )
    projection = project_replication_switch(audit)
    assert ("information_horizon", "replication_still_informative") in _states(projection)
    assert ("information_horizon", "alternative_beats_repeat_ceiling") in _states(projection)
    assert {"compare_repeat_vs_switch", "switch_observation_type"} <= _actions(projection)


def test_expected_cost_audit_separates_information_and_operational_actions():
    early, routing, _, _ = cost_example()
    early_projection = project_adaptive_expected_cost(early)
    assert ("resource_operational", "expected_cost_saving") in _states(early_projection)
    assert ("resource_operational", "adaptive_information_advantage") not in _states(early_projection)

    budget_two_projection = project_adaptive_expected_cost(routing[1])
    assert ("resource_operational", "adaptive_information_advantage") in _states(
        budget_two_projection
    )
    assert ("resource_operational", "expected_cost_saving") not in _states(
        budget_two_projection
    )


def test_scenario_cost_tradeoff_projects_tie_and_meta_rule_claim_ceilings():
    crossing = project_scenario_cost_tradeoff(crossing_scenario_witness())
    assert ("resource_operational", "equal_information_cost_tie_sensitive") in _states(crossing)
    assert ("resource_operational", "cost_preference_crosses_scenarios") in _states(crossing)
    assert {
        "do_not_call_selected_tree_cost_optimal",
        "declare_meta_prior_or_robust_cost_rule_before_scalar_cost_ranking",
    } <= _actions(crossing)

    uniform = project_scenario_cost_tradeoff(uniform_scenario_witness())
    assert ("resource_operational", "equal_information_cost_tie_sensitive") in _states(uniform)
    assert ("resource_operational", "cost_preference_crosses_scenarios") not in _states(uniform)


def test_projection_composition_is_union_not_priority_ordering():
    status = classify_specification(
        SpecificationInput(
            "budgeted",
            1.0,
            (CandidateScore("q", True, 0.5),),
            budget_remaining=False,
        )
    )
    core = project_specification_status(status)
    cost = project_scenario_cost_tradeoff(crossing_scenario_witness())
    combined = combine_taxonomy_projections(core, cost)
    assert ("information_horizon", "positive_singleton") in _states(combined)
    assert ("resource_operational", "budget_exhausted_best_known") in _states(combined)
    assert ("resource_operational", "cost_preference_crosses_scenarios") in _states(combined)
    assert "report_budget_limit" in combined.detail_actions


def test_adapter_remains_internal_and_off_submission_surface():
    root = Path(__file__).resolve().parents[1]
    public_api = (root / "causal_model" / "__init__.py").read_text(encoding="utf-8")
    manuscript = (root / "paper" / "manuscript.md").read_text(encoding="utf-8")
    assert "limitation_taxonomy_adapter" not in public_api
    assert "limitation_taxonomy_adapter" not in manuscript
