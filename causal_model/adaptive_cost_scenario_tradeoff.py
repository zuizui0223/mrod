"""Internal scenario-wise expected-cost comparison for information-equivalent trees.

The adaptive planner does not optimize expected acquisition cost. This module
compares already-selected trees scenario by scenario and deliberately refuses to
invent a probability distribution over calibration/weight scenarios.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping, Sequence

from .adaptive_expected_cost_audit import (
    AdaptiveExpectedCostAudit,
    audit_adaptive_expected_cost,
)
from .joint_budgeted_design import JointCalibrationScenario


@dataclass(frozen=True)
class PolicyCostVector:
    policy_label: str
    candidate_order: tuple[str, ...]
    expected_cost_by_scenario: dict[str, float]
    information_bits_by_scenario: dict[str, float]
    worst_path_cost: int | None


@dataclass(frozen=True)
class ScenarioCostTradeoffAudit:
    policy_vectors: tuple[PolicyCostVector, ...]
    scenario_names: tuple[str, ...]
    information_equivalent_by_scenario: bool
    same_worst_path_cost: bool
    uniformly_lowest_expected_cost_policies: tuple[str, ...]
    pairwise_cost_preference_crosses: bool
    scalar_expected_cost_ranking_defined: bool
    scenario_meta_prior_used: bool
    comparison_tolerance: float
    scope: str = "registered_policy_cost_vectors_no_scenario_meta_prior_no_cost_optimality"


def compare_expected_cost_receipts(
    policies: Mapping[str, AdaptiveExpectedCostAudit], *,
    comparison_tolerance: float = 1e-12,
) -> ScenarioCostTradeoffAudit:
    """Compare expected-cost vectors without inventing a scenario average.

    A uniformly lowest policy must have expected cost no larger than every other
    registered policy in every scenario. If cost preference crosses across
    scenarios, no scalar expected-cost ranking is returned because that would
    require an additional decision rule such as a scenario meta-prior, max-cost,
    or regret criterion. Those choices are outside this audit.
    """
    tol = float(comparison_tolerance)
    if not isfinite(tol) or tol < 0:
        raise ValueError("comparison_tolerance must be finite and non-negative")
    if not policies:
        raise ValueError("at least one policy receipt is required")
    labels = tuple(policies)
    receipts = tuple(policies[label] for label in labels)
    if len(set(labels)) != len(labels) or any(not isinstance(label, str) or not label.strip() for label in labels):
        raise ValueError("policy labels must be unique non-empty strings")
    if any(not receipt.complete_search or not receipt.scenario_results for receipt in receipts):
        raise ValueError("every policy must have a complete expected-cost audit")

    scenario_names = tuple(row.scenario for row in receipts[0].scenario_results)
    if len(set(scenario_names)) != len(scenario_names):
        raise ValueError("scenario names must be unique")
    if any(set(row.scenario for row in receipt.scenario_results) != set(scenario_names) for receipt in receipts[1:]):
        raise ValueError("all policy receipts must cover the same named scenarios")

    vectors = []
    for label, receipt in zip(labels, receipts):
        by_name = {row.scenario: row for row in receipt.scenario_results}
        vectors.append(
            PolicyCostVector(
                label,
                receipt.candidate_order,
                {name: by_name[name].expected_acquisition_cost for name in scenario_names},
                {name: by_name[name].selected_information_bits for name in scenario_names},
                receipt.selected_tree_worst_path_cost,
            )
        )

    information_equivalent = all(
        max(vector.information_bits_by_scenario[name] for vector in vectors)
        - min(vector.information_bits_by_scenario[name] for vector in vectors)
        <= tol
        for name in scenario_names
    )
    same_worst = len({vector.worst_path_cost for vector in vectors}) == 1

    uniformly_lowest = []
    for candidate in vectors:
        if all(
            candidate.expected_cost_by_scenario[name]
            <= other.expected_cost_by_scenario[name] + tol
            for other in vectors
            for name in scenario_names
        ):
            uniformly_lowest.append(candidate.policy_label)

    crosses = False
    for i, left in enumerate(vectors):
        for right in vectors[i + 1 :]:
            left_better = any(
                left.expected_cost_by_scenario[name] < right.expected_cost_by_scenario[name] - tol
                for name in scenario_names
            )
            right_better = any(
                right.expected_cost_by_scenario[name] < left.expected_cost_by_scenario[name] - tol
                for name in scenario_names
            )
            crosses |= left_better and right_better

    # A scalar ranking is licensed here only by uniform componentwise ordering.
    # A meta-prior or robust scalarization would be an additional declared rule.
    scalar_defined = bool(uniformly_lowest) and not crosses
    return ScenarioCostTradeoffAudit(
        tuple(vectors),
        scenario_names,
        information_equivalent,
        same_worst,
        tuple(uniformly_lowest),
        crosses,
        scalar_defined,
        False,
        tol,
    )


def _three_target_receipt(order: tuple[str, str], weights_by_scenario: Mapping[str, tuple[float, float, float]]):
    rows = ({"target": "rare"}, {"target": "common"}, {"target": "other"})
    if order == ("rare_split", "common_split"):
        events = (("rare", "rest"), ("rest", "common"), ("rest", "rest"))
    elif order == ("common_split", "rare_split"):
        events = (("rest", "rare"), ("common", "rest"), ("rest", "rest"))
    else:
        raise ValueError("witness order must contain rare_split and common_split")
    matrix = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    scenarios = tuple(
        JointCalibrationScenario(
            name,
            weights,
            events,
            matrix,
            "synthetic deterministic scenario-wise operational tie law",
        )
        for name, weights in weights_by_scenario.items()
    )
    return audit_adaptive_expected_cost(
        rows,
        scenarios,
        candidate_order=order,
        acquisition_costs={"rare_split": 1, "common_split": 1},
        budget=2,
        target_columns=("target",),
        support_reference="synthetic three-target scenario-wise expected-cost panel",
    )


def crossing_scenario_witness() -> ScenarioCostTradeoffAudit:
    """Cost preference reverses when the common versus rare target mass reverses."""
    weights = {
        "common_heavy": (1.0, 8.0, 1.0),
        "rare_heavy": (8.0, 1.0, 1.0),
    }
    rare_first = _three_target_receipt(("rare_split", "common_split"), weights)
    common_first = _three_target_receipt(("common_split", "rare_split"), weights)
    return compare_expected_cost_receipts(
        {"rare_first": rare_first, "common_first": common_first}
    )


def uniform_scenario_witness() -> ScenarioCostTradeoffAudit:
    """Common-first is uniformly cheaper when common remains more frequent."""
    weights = {
        "common_very_heavy": (1.0, 8.0, 1.0),
        "common_moderately_heavy": (1.0, 3.0, 1.0),
    }
    rare_first = _three_target_receipt(("rare_split", "common_split"), weights)
    common_first = _three_target_receipt(("common_split", "rare_split"), weights)
    return compare_expected_cost_receipts(
        {"rare_first": rare_first, "common_first": common_first}
    )
