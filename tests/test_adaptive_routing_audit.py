from itertools import product

from causal_model.adaptive_joint_design import routing_witness
from causal_model.adaptive_routing_audit import audit_adaptive_routing
from causal_model.joint_budgeted_design import JointCalibrationScenario


def test_routing_witness_has_zero_direct_target_information_but_positive_continuation():
    rows, scenarios = routing_witness()
    audit = audit_adaptive_routing(
        rows,
        scenarios,
        candidate_order=("context", "assay0", "assay1"),
        acquisition_costs={"context": 1, "assay0": 1, "assay1": 1},
        budget=2,
        target_columns=("target",),
        support_reference="synthetic routing audit",
    )
    assert audit.status == "routing_audit_complete"
    assert audit.root_query == "context"
    assert audit.branch_dependent_continuation
    assert audit.root_direct_target_information_zero_all_scenarios
    assert audit.routing_without_direct_target_information
    assert set(audit.next_action_by_root_outcome) == {
        ("0", "assay0"),
        ("1", "assay1"),
    }
    by_name = {row.scenario: row for row in audit.scenario_audits}
    assert abs(by_name["uniform_context"].root_direct_target_information_bits) < 1e-12
    assert abs(by_name["uniform_context"].total_policy_target_information_bits - 1.0) < 1e-12
    assert abs(by_name["uniform_context"].conditional_continuation_target_information_bits - 1.0) < 1e-12
    assert abs(by_name["uniform_context"].routing_action_entropy_bits - 1.0) < 1e-12
    assert abs(by_name["uniform_context"].selected_policy_gain_over_best_fixed_bits - 0.5) < 1e-12
    for name in ("context0_common", "context1_common"):
        assert abs(by_name[name].total_policy_target_information_bits - 1.0) < 1e-12
        assert abs(by_name[name].root_direct_target_information_bits) < 1e-12
        assert abs(by_name[name].conditional_continuation_target_information_bits - 1.0) < 1e-12
        assert abs(by_name[name].routing_action_entropy_bits - 0.8112781244591328) < 1e-12
        assert abs(by_name[name].selected_policy_gain_over_best_fixed_bits - 0.25) < 1e-12


def _xor_models():
    worlds = tuple(product((0, 1), repeat=2))
    rows = tuple({"target": u ^ v} for u, v in worlds)
    outcomes = tuple(product(("0", "1"), repeat=3))
    matrix = tuple(tuple(
        (0.8 if int(z) == (u ^ v) else 0.2)
        if (int(a), int(b)) == (u, v)
        else 0.0
        for a, b, z in outcomes
    ) for u, v in worlds)
    scenarios = (
        JointCalibrationScenario(
            "uniform", (1, 1, 1, 1), outcomes, matrix,
            "synthetic explicit XOR law",
        ),
        JointCalibrationScenario(
            "target_rare", (9, 1, 1, 9), outcomes, matrix,
            "synthetic explicit XOR law",
        ),
    )
    return rows, scenarios


def test_zero_direct_information_with_branch_invariant_continuation_is_not_routing_witness():
    rows, scenarios = _xor_models()
    audit = audit_adaptive_routing(
        rows,
        scenarios,
        candidate_order=("U", "V", "direct"),
        acquisition_costs={"U": 1, "V": 1, "direct": 1},
        budget=2,
        target_columns=("target",),
        support_reference="synthetic XOR routing adverse control",
    )
    assert audit.root_query in {"U", "V"}
    assert audit.root_direct_target_information_zero_all_scenarios
    assert not audit.branch_dependent_continuation
    assert not audit.routing_without_direct_target_information
    for row in audit.scenario_audits:
        assert row.conditional_continuation_target_information_bits > 0.0
        assert abs(row.routing_action_entropy_bits) < 1e-12
        assert abs(row.selected_policy_gain_over_best_fixed_bits) < 1e-12


def test_one_step_policy_has_no_routing_continuation():
    rows, scenarios = routing_witness()
    audit = audit_adaptive_routing(
        rows,
        scenarios,
        candidate_order=("context", "assay0", "assay1"),
        acquisition_costs={"context": 1, "assay0": 1, "assay1": 1},
        budget=1,
        target_columns=("target",),
        support_reference="synthetic one-step control",
    )
    # With only one acquisition, the robust planner selects an assay rather than
    # paying for a zero-target-information routing query with no continuation.
    assert audit.root_query in {"assay0", "assay1"}
    assert not audit.branch_dependent_continuation
    assert not audit.routing_without_direct_target_information
