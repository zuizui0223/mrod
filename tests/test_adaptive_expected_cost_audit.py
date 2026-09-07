from math import log2
from pathlib import Path

import pytest

from causal_model.adaptive_expected_cost_audit import (
    audit_adaptive_expected_cost,
    early_stop_witness,
    operational_tie_witness,
    synthetic_example,
)
from causal_model.adaptive_joint_design import routing_witness
from causal_model.joint_budgeted_design import JointCalibrationScenario


def _uniform(result):
    return next(row for row in result.scenario_results if row.scenario == "uniform_context")


def _scenario(result, name):
    return next(row for row in result.scenario_results if row.scenario == name)


def test_early_stop_matches_full_information_with_lower_expected_cost_than_fixed_bundle():
    rows, scenarios = early_stop_witness()
    receipt = audit_adaptive_expected_cost(
        rows,
        scenarios,
        candidate_order=("screen", "resolve"),
        acquisition_costs={"screen": 1, "resolve": 1},
        budget=2,
        target_columns=("target",),
        support_reference="synthetic three-target early-stop test",
    )
    assert receipt.complete_search
    assert not receipt.expected_cost_optimized
    row = receipt.scenario_results[0]
    assert row.selected_information_bits == pytest.approx(log2(3), abs=1e-12)
    assert row.adaptive_information_gain_over_fixed_oracle_bits == pytest.approx(0.0, abs=1e-12)
    assert row.expected_acquisition_cost == pytest.approx(5 / 3, abs=1e-12)
    assert row.selected_tree_worst_path_cost == 2
    assert row.scenario_worst_positive_path_cost == 2
    assert row.probability_below_selected_tree_worst_path == pytest.approx(1 / 3, abs=1e-12)
    assert tuple(x.acquisition_cost for x in row.path_cost_distribution) == (1, 2)
    assert tuple(x.probability for x in row.path_cost_distribution) == pytest.approx(
        (1 / 3, 2 / 3), abs=1e-12
    )
    assert row.minimum_fixed_cost_matching_selected_information == 2
    assert row.expected_cost_saving_vs_information_matched_fixed == pytest.approx(1 / 3, abs=1e-12)
    assert set(row.attribution_reference_fixed_bundle or ()) == {"screen", "resolve"}
    assert row.attributed_expected_cost_saving == pytest.approx(1 / 3, abs=1e-12)
    assert sum(item.expected_cost_saving for item in row.skipped_measurement_savings) == pytest.approx(1 / 3)
    assert sorted(item.skip_probability for item in row.skipped_measurement_savings) == pytest.approx([0.0, 1 / 3])


def test_routing_information_gap_and_expected_cost_saving_are_different_budget_local_axes():
    _, routing, _, _ = synthetic_example()
    by_budget = {receipt.budget: receipt for receipt in routing}

    b1 = _uniform(by_budget[1])
    assert b1.adaptive_information_gain_over_fixed_oracle_bits == pytest.approx(0.0, abs=1e-12)
    assert b1.expected_acquisition_cost == pytest.approx(1.0, abs=1e-12)
    assert b1.minimum_fixed_cost_matching_selected_information == 1
    assert b1.expected_cost_saving_vs_information_matched_fixed == pytest.approx(0.0, abs=1e-12)

    b2 = _uniform(by_budget[2])
    assert b2.adaptive_information_gain_over_fixed_oracle_bits == pytest.approx(0.5, abs=1e-12)
    assert b2.expected_acquisition_cost == pytest.approx(2.0, abs=1e-12)
    assert b2.minimum_fixed_cost_matching_selected_information is None
    assert b2.expected_cost_saving_vs_information_matched_fixed is None
    assert b2.attribution_reference_fixed_bundle is None

    b3 = _uniform(by_budget[3])
    assert b3.adaptive_information_gain_over_fixed_oracle_bits == pytest.approx(0.0, abs=1e-12)
    assert b3.selected_information_bits == pytest.approx(1.0, abs=1e-12)
    assert b3.expected_acquisition_cost == pytest.approx(2.0, abs=1e-12)
    assert b3.minimum_fixed_cost_matching_selected_information == 3
    assert b3.expected_cost_saving_vs_information_matched_fixed == pytest.approx(1.0, abs=1e-12)
    assert b3.attributed_expected_cost_saving == pytest.approx(1.0, abs=1e-12)
    by_query = {item.candidate: item for item in b3.skipped_measurement_savings}
    assert by_query["context"].skip_probability == pytest.approx(0.0)
    assert by_query["assay0"].skip_probability == pytest.approx(0.5)
    assert by_query["assay1"].skip_probability == pytest.approx(0.5)
    assert b3.probability_below_selected_tree_worst_path == pytest.approx(0.0, abs=1e-12)


def test_unequal_cost_routing_makes_scenario_specific_savings_explicit():
    _, _, unequal, _ = synthetic_example()
    uniform = _scenario(unequal, "uniform_context")
    context0 = _scenario(unequal, "context0_common")
    context1 = _scenario(unequal, "context1_common")

    assert uniform.expected_acquisition_cost == pytest.approx(4.0)
    assert context0.expected_acquisition_cost == pytest.approx(3.5)
    assert context1.expected_acquisition_cost == pytest.approx(4.5)
    assert uniform.minimum_fixed_cost_matching_selected_information == 6
    assert context0.minimum_fixed_cost_matching_selected_information == 6
    assert context1.minimum_fixed_cost_matching_selected_information == 6
    assert uniform.attributed_expected_cost_saving == pytest.approx(2.0)
    assert context0.attributed_expected_cost_saving == pytest.approx(2.5)
    assert context1.attributed_expected_cost_saving == pytest.approx(1.5)

    uniform_use = {item.candidate: item.acquisition_probability for item in uniform.query_use_probabilities}
    assert uniform_use == pytest.approx({"context": 1.0, "assay0": 0.5, "assay1": 0.5})


def test_equal_information_equal_worst_cost_trees_can_have_different_expected_costs():
    rare_rows, rare_scenarios, rare_order = operational_tie_witness(common_first=False)
    common_rows, common_scenarios, common_order = operational_tie_witness(common_first=True)
    rare = audit_adaptive_expected_cost(
        rare_rows,
        rare_scenarios,
        candidate_order=rare_order,
        acquisition_costs={"rare_split": 1, "common_split": 1},
        budget=2,
        target_columns=("target",),
        support_reference="synthetic rare-first operational tie",
    )
    common = audit_adaptive_expected_cost(
        common_rows,
        common_scenarios,
        candidate_order=common_order,
        acquisition_costs={"rare_split": 1, "common_split": 1},
        budget=2,
        target_columns=("target",),
        support_reference="synthetic common-first operational tie",
    )
    rare_row, common_row = rare.scenario_results[0], common.scenario_results[0]
    assert rare_row.selected_information_bits == pytest.approx(common_row.selected_information_bits, abs=1e-12)
    assert rare_row.selected_information_bits == pytest.approx(
        -(0.1 * log2(0.1) + 0.8 * log2(0.8) + 0.1 * log2(0.1)), abs=1e-12
    )
    assert rare_row.selected_tree_worst_path_cost == common_row.selected_tree_worst_path_cost == 2
    assert rare_row.expected_acquisition_cost == pytest.approx(1.9, abs=1e-12)
    assert common_row.expected_acquisition_cost == pytest.approx(1.2, abs=1e-12)
    assert rare_row.expected_acquisition_cost - common_row.expected_acquisition_cost == pytest.approx(0.7)
    assert not rare.expected_cost_optimized
    assert not common.expected_cost_optimized


def test_synthetic_report_registers_both_operational_tie_orders():
    _, _, _, ties = synthetic_example()
    assert len(ties) == 2
    costs = [receipt.scenario_results[0].expected_acquisition_cost for receipt in ties]
    assert costs == pytest.approx([1.9, 1.2], abs=1e-12)


def test_expected_cost_is_scenario_specific_and_never_exceeds_selected_pathwise_budget():
    rows, scenarios = routing_witness()
    receipt = audit_adaptive_expected_cost(
        rows,
        scenarios,
        candidate_order=("context", "assay0", "assay1"),
        acquisition_costs={"context": 1, "assay0": 1, "assay1": 1},
        budget=3,
        target_columns=("target",),
        support_reference="synthetic routing expected-cost bounds",
    )
    assert len(receipt.scenario_results) == 3
    for row in receipt.scenario_results:
        assert 0 <= row.expected_acquisition_cost <= row.selected_tree_worst_path_cost <= 3
        assert sum(item.probability for item in row.path_cost_distribution) == pytest.approx(1.0)
        assert sum(item.expected_cost_contribution for item in row.query_use_probabilities) == pytest.approx(
            row.expected_acquisition_cost
        )


def test_missing_joint_likelihood_fails_closed_without_expected_cost_claim():
    rows = ({"target": 0}, {"target": 1})
    scenario = JointCalibrationScenario(
        "missing",
        (1.0, 1.0),
        (("0",), ("1",)),
        None,
        "missing synthetic likelihood",
    )
    receipt = audit_adaptive_expected_cost(
        rows,
        (scenario,),
        candidate_order=("q",),
        acquisition_costs={"q": 1},
        budget=1,
        target_columns=("target",),
        support_reference="missing joint expected-cost test",
    )
    assert not receipt.complete_search
    assert receipt.scenario_results == ()


def test_audit_remains_internal_and_does_not_change_submission_surface():
    root = Path(__file__).resolve().parents[1]
    public_api = (root / "causal_model" / "__init__.py").read_text(encoding="utf-8")
    manuscript = (root / "paper" / "manuscript.md").read_text(encoding="utf-8")
    assert "adaptive_expected_cost_audit" not in public_api
    assert "expected acquisition cost saving" not in manuscript.lower()
