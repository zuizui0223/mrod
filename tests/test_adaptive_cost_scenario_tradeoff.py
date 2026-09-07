from pathlib import Path

import pytest

from causal_model.adaptive_cost_scenario_tradeoff import (
    crossing_scenario_witness,
    uniform_scenario_witness,
)


def _vector(audit, label):
    return next(row for row in audit.policy_vectors if row.policy_label == label)


def test_expected_cost_preference_can_reverse_across_declared_scenarios():
    audit = crossing_scenario_witness()
    rare = _vector(audit, "rare_first")
    common = _vector(audit, "common_first")

    assert audit.information_equivalent_by_scenario
    assert audit.same_worst_path_cost
    assert audit.pairwise_cost_preference_crosses
    assert audit.uniformly_lowest_expected_cost_policies == ()
    assert not audit.scalar_expected_cost_ranking_defined
    assert not audit.scenario_meta_prior_used

    assert rare.expected_cost_by_scenario == pytest.approx(
        {"common_heavy": 1.9, "rare_heavy": 1.2}, abs=1e-12
    )
    assert common.expected_cost_by_scenario == pytest.approx(
        {"common_heavy": 1.2, "rare_heavy": 1.9}, abs=1e-12
    )
    assert rare.worst_path_cost == common.worst_path_cost == 2
    for scenario in audit.scenario_names:
        assert rare.information_bits_by_scenario[scenario] == pytest.approx(
            common.information_bits_by_scenario[scenario], abs=1e-12
        )


def test_uniform_componentwise_cost_order_can_be_reported_without_meta_prior():
    audit = uniform_scenario_witness()
    rare = _vector(audit, "rare_first")
    common = _vector(audit, "common_first")

    assert audit.information_equivalent_by_scenario
    assert audit.same_worst_path_cost
    assert not audit.pairwise_cost_preference_crosses
    assert audit.uniformly_lowest_expected_cost_policies == ("common_first",)
    assert audit.scalar_expected_cost_ranking_defined
    assert not audit.scenario_meta_prior_used

    assert common.expected_cost_by_scenario == pytest.approx(
        {"common_very_heavy": 1.2, "common_moderately_heavy": 1.4}, abs=1e-12
    )
    assert rare.expected_cost_by_scenario == pytest.approx(
        {"common_very_heavy": 1.9, "common_moderately_heavy": 1.8}, abs=1e-12
    )


def test_tradeoff_audit_remains_internal_and_off_submission_surface():
    root = Path(__file__).resolve().parents[1]
    public_api = (root / "causal_model" / "__init__.py").read_text(encoding="utf-8")
    manuscript = (root / "paper" / "manuscript.md").read_text(encoding="utf-8")
    assert "adaptive_cost_scenario_tradeoff" not in public_api
    assert "scenario-wise expected-cost" not in manuscript.lower()
