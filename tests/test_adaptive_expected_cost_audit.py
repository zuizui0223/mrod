from math import log2
from pathlib import Path

import pytest

from causal_model.adaptive_expected_cost_audit import (
    audit_adaptive_expected_cost,
    early_stop_witness,
    synthetic_example,
)
from causal_model.adaptive_joint_design import routing_witness
from causal_model.joint_budgeted_design import JointCalibrationScenario


def _uniform(result):
    return next(row for row in result.scenario_results if row.scenario == "uniform_context")


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
    assert [(x.acquisition_cost, x.probability) for x in row.path_cost_distribution] == pytest.approx(
        [(1, 1 / 3), (2, 2 / 3)], abs=1e-12
    )
    assert row.minimum_fixed_cost_matching_selected_information == 2
    assert row.expected_cost_saving_vs_information_matched_fixed == pytest.approx(1 / 3, abs=1e-12)


def test_routing_information_gap_and_expected_cost_saving_are_different_budget_local_axes():
    _, routing = synthetic_example()
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

    b3 = _uniform(by_budget[3])
    assert b3.adaptive_information_gain_over_fixed_oracle_bits == pytest.approx(0.0, abs=1e-12)
    assert b3.selected_information_bits == pytest.approx(1.0, abs=1e-12)
    assert b3.expected_acquisition_cost == pytest.approx(2.0, abs=1e-12)
    assert b3.minimum_fixed_cost_matching_selected_information == 3
    assert b3.expected_cost_saving_vs_information_matched_fixed == pytest.approx(1.0, abs=1e-12)
    assert b3.probability_below_selected_tree_worst_path == pytest.approx(0.0, abs=1e-12)


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
