"""Tests for the post-frozen static-information claim-ceiling diagnostic."""
import json
from pathlib import Path

from causal_model.generality_sweep import (
    INFORMATION_GUIDED_POLICY,
    RANDOM_ORDER_POLICY,
    STATIC_INITIAL_INFORMATION_POLICY,
    run_static_information_diagnostic,
)

ROOT = Path(__file__).resolve().parents[1]


def test_static_information_diagnostic_is_post_frozen_and_matched():
    result = run_static_information_diagnostic(
        seeds=(101, 202),
        budgets=(2, 4),
        n_systems_per_seed=20,
        n_attempts=450,
        K_choices=(4, 5, 6),
        confound_choices=(1, 2),
        min_sub_size=8,
        n_distractors=2,
    )
    assert result["status"] == "post_frozen_claim_ceiling_diagnostic"
    assert result["preregistered_g2_modified"] is False
    assert result["policies"] == [
        INFORMATION_GUIDED_POLICY,
        STATIC_INITIAL_INFORMATION_POLICY,
        RANDOM_ORDER_POLICY,
    ]
    assert result["budgets"] == [2, 4]
    keys = {(row["policy"], row["budget"]) for row in result["aggregate"]}
    assert keys == {
        (policy, budget)
        for policy in result["policies"]
        for budget in result["budgets"]
    }
    for row in result["aggregate"]:
        assert 0.0 <= row["frac_converged_mean"] <= 1.0
        assert 0.0 <= row["mean_frac_resolved_mean"] <= 1.0
        assert 0.0 <= row["false_exclusion_rate_mean"] <= 1.0
        assert 0.0 <= row["mean_distractors_selected_mean"] <= row["mean_steps_mean"]


def test_frozen_g2_policy_firewall_remains_two_policy_preregistration():
    protocol = json.loads((ROOT / "paper/g2_frozen_benchmark_protocol.json").read_text())
    assert protocol["selection_validation"]["policies"] == ["rach_seq", "random_order"]
    assert STATIC_INITIAL_INFORMATION_POLICY not in protocol["selection_validation"]["policies"]
