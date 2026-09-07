"""Shared Boundary/MROD limitation-to-action taxonomy regression guards."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from causal_model.limitation_action_report import (
    CandidateScore,
    SpecificationInput,
    classify_specification,
)


ROOT = Path(__file__).resolve().parents[1]
TAXONOMY = ROOT / "docs" / "limitation_action_taxonomy_v1.json"
NOTE = ROOT / "docs" / "LIMITATION_ACTION_TAXONOMY.md"
MANUSCRIPT = ROOT / "paper" / "manuscript.md"
EXPECTED_SHA256 = "04f81c3400300cd30c05504477cea5a446b7b1839f9b49e4bea5c65db73a8002"


def _actions():
    data = json.loads(TAXONOMY.read_text(encoding="utf-8"))
    return {
        state["action"]
        for axis in data["axes"]
        for state in axis["states"]
    }


def test_taxonomy_matches_frozen_cross_repo_hash_and_axes():
    raw = TAXONOMY.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == EXPECTED_SHA256
    data = json.loads(raw)
    assert data["schema"] == "limitation_action_taxonomy"
    assert data["version"] == "1.0"
    assert "do_not_replace_the_vector_with_one_severity_score" in data["composition_rule"]
    assert {axis["id"] for axis in data["axes"]} == {
        "evidence_integrity",
        "observation_structure",
        "current_state_estimability",
        "target_identification",
        "candidate_prediction_coverage",
        "information_horizon",
        "specification_robustness",
        "resource_operational",
    }


def test_existing_limitation_report_core_actions_are_in_shared_taxonomy():
    actions = _actions()
    required = {
        "repair_or_reestimate_current_admissible_region",
        "stop_fully_resolved",
        "stop_declared_target_resolved_report_residual_ambiguity",
        "expand_candidate_vocabulary",
        "identify_candidate_outcome_models",
        "resolve_nonestimable_candidates_before_global_ranking",
        "measure_best_candidate",
        "audit_joint_candidate_information_before_sequence_limit",
        "use_nonmyopic_bundle_or_sequence_design",
        "redesign_or_expand_measurement_vocabulary",
        "retain_best_candidate_for_future_budget",
    }
    assert required <= actions

    nonestimable = classify_specification(
        SpecificationInput("bad_current", float("nan"), ())
    )
    assert nonestimable.recommended_actions[0] in actions

    resolved = classify_specification(
        SpecificationInput("resolved", 0.0, ())
    )
    assert resolved.recommended_actions == ("stop_fully_resolved",)
    assert resolved.recommended_actions[0] in actions

    positive = classify_specification(
        SpecificationInput(
            "positive",
            1.0,
            (CandidateScore("q", True, 0.5),),
        )
    )
    assert "measure_best_candidate" in positive.recommended_actions
    assert "measure_best_candidate" in actions


def test_extension_claim_ceiling_actions_remain_explicit_and_off_mainline():
    actions = _actions()
    for action in (
        "compare_repeat_vs_switch",
        "switch_observation_type",
        "report_finite_budget_recommendation_not_universal_impossibility",
        "report_information_gain_not_cost_optimality",
        "report_skipped_measurements_if_attributable",
        "do_not_call_selected_tree_cost_optimal",
        "declare_meta_prior_or_robust_cost_rule_before_scalar_cost_ranking",
    ):
        assert action in actions

    note = NOTE.read_text(encoding="utf-8")
    assert "Composition, not precedence" in note
    assert "same JSON is stored in the Boundary repository" in note

    manuscript = MANUSCRIPT.read_text(encoding="utf-8")
    assert "limitation_action_taxonomy_v1" not in manuscript
    assert "cost_preference_crosses_scenarios" not in manuscript
