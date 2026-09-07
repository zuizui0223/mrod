"""Internal expected-cost audit for an already selected adaptive observation tree.

The adaptive planner optimizes target information/regret under a pathwise budget;
it does NOT optimize expected acquisition cost.  This module therefore evaluates
expected cost as a separate operational diagnostic under each declared scenario.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import fsum, isfinite
from typing import Mapping, Sequence

from .adaptive_joint_design import (
    AdaptiveJointReceipt,
    AdaptivePolicyNode,
    plan_adaptive_joint_budget,
    routing_witness,
)
from .empirical_observation_contract import _weights
from .joint_budgeted_design import (
    JointCalibrationScenario,
    _joint_candidate,
    plan_joint_observation_budget,
)


@dataclass(frozen=True)
class PathCostMass:
    acquisition_cost: int
    probability: float


@dataclass(frozen=True)
class ScenarioExpectedCost:
    scenario: str
    selected_information_bits: float
    adaptive_information_gain_over_fixed_oracle_bits: float
    expected_acquisition_cost: float
    scenario_worst_positive_path_cost: int
    selected_tree_worst_path_cost: int
    probability_below_selected_tree_worst_path: float
    path_cost_distribution: tuple[PathCostMass, ...]
    minimum_fixed_cost_matching_selected_information: int | None
    minimum_cost_information_matched_fixed_bundles: tuple[tuple[str, ...], ...]
    expected_cost_saving_vs_information_matched_fixed: float | None


@dataclass(frozen=True)
class AdaptiveExpectedCostAudit:
    status: str
    budget: int
    candidate_order: tuple[str, ...]
    scenario_results: tuple[ScenarioExpectedCost, ...]
    selected_tree_worst_path_cost: int | None
    support_reference: str
    comparison_tolerance_bits: float
    complete_search: bool
    expected_cost_optimized: bool = False
    scenario_meta_prior_used: bool = False
    scope: str = "selected_tree_scenario_specific_expected_acquisition_cost_not_cost_optimality"


def _path_cost(node: AdaptivePolicyNode, event: tuple[str, ...], positions: Mapping[str, int]) -> int:
    used = 0
    queried: set[str] = set()
    current = node
    while current.query is not None:
        query = current.query
        if query in queried or query not in positions:
            raise ValueError("selected policy contains a repeated or unknown query")
        if type(current.acquisition_cost) is not int or current.acquisition_cost < 1:
            raise ValueError("selected policy contains an invalid acquisition cost")
        queried.add(query)
        used += current.acquisition_cost
        branches = dict(current.branches)
        label = event[positions[query]]
        if label not in branches:
            raise ValueError("selected policy lacks a branch for declared positive joint support")
        current = branches[label]
    return used


def audit_adaptive_expected_cost(
    accepted_rows: Sequence[Mapping],
    scenarios: Sequence[JointCalibrationScenario],
    *,
    candidate_order: Sequence[str],
    acquisition_costs: Mapping[str, int],
    budget: int,
    target_columns: Sequence[str],
    support_reference: str,
    comparison_tolerance_bits: float = 1e-12,
    max_tree_combinations: int = 200_000,
) -> AdaptiveExpectedCostAudit:
    """Evaluate expected acquisition cost of the selected adaptive tree.

    The same explicit joint likelihoods are passed to the existing adaptive and
    fixed-bundle planners.  Expected cost is computed separately inside each
    scenario; no probability over scenarios is invented.  A fixed bundle is a
    precommitted acquisition and therefore pays its full declared additive cost.

    ``expected_cost_saving_vs_information_matched_fixed`` compares the selected
    tree with the cheapest fixed bundle that reaches at least the selected tree's
    target information in that scenario.  If no fixed bundle within the same
    pathwise budget matches the information, the cost comparison is left None.
    This routine does not claim that the selected tree minimizes expected cost
    among adaptive trees with the same information.
    """
    tol = float(comparison_tolerance_bits)
    if not isfinite(tol) or tol < 0:
        raise ValueError("comparison_tolerance_bits must be finite and non-negative")
    rows, models, order = tuple(accepted_rows), tuple(scenarios), tuple(candidate_order)
    adaptive = plan_adaptive_joint_budget(
        rows,
        models,
        candidate_order=order,
        acquisition_costs=acquisition_costs,
        budget=budget,
        target_columns=target_columns,
        support_reference=support_reference,
        comparison_tolerance_bits=tol,
        max_tree_combinations=max_tree_combinations,
    )
    if adaptive.selected_policy is None or not adaptive.complete_search:
        return AdaptiveExpectedCostAudit(
            adaptive.status,
            budget,
            order,
            (),
            adaptive.worst_path_cost,
            support_reference,
            tol,
            adaptive.complete_search,
        )

    fixed = plan_joint_observation_budget(
        rows,
        models,
        candidate_order=order,
        acquisition_costs=acquisition_costs,
        budget=budget,
        target_columns=target_columns,
        support_reference=support_reference,
        comparison_tolerance_bits=tol,
    )
    positions = {query: i for i, query in enumerate(order)}
    results: list[ScenarioExpectedCost] = []
    for model in models:
        events, matrix = _joint_candidate(model, len(rows), len(order))
        if matrix is None:
            raise ValueError("selected adaptive tree cannot be cost-audited without a joint likelihood")
        weights = _weights(model.weights, len(rows))
        cost_mass: dict[int, float] = {}
        for j, event in enumerate(events):
            probability = fsum(weights[i] * matrix[i][j] for i in range(len(rows)))
            if probability <= 0:
                continue
            cost = _path_cost(adaptive.selected_policy, event, positions)
            if cost > budget:
                raise ArithmeticError("positive-probability adaptive path exceeds declared budget")
            cost_mass[cost] = cost_mass.get(cost, 0.0) + probability
        total = fsum(cost_mass.values())
        if abs(total - 1.0) > 1e-10:
            raise ArithmeticError("positive joint support does not sum to one in expected-cost audit")
        distribution = tuple(PathCostMass(cost, mass) for cost, mass in sorted(cost_mass.items()))
        expected = fsum(item.acquisition_cost * item.probability for item in distribution)
        scenario_worst = max(cost_mass)
        selected_worst = adaptive.worst_path_cost
        if selected_worst is None or expected > selected_worst + 1e-10 or scenario_worst > selected_worst:
            raise ArithmeticError("expected-cost audit exceeded selected policy worst-path cost")
        early_mass = fsum(mass for cost, mass in cost_mass.items() if cost < selected_worst)

        selected_info = adaptive.selected_information_bits[model.name]
        matching = tuple(
            score
            for score in fixed.bundle_scores
            if score.information_by_scenario[model.name] is not None
            and score.information_by_scenario[model.name] + tol >= selected_info
        )
        if matching:
            min_cost = min(score.acquisition_cost for score in matching)
            matched_names = tuple(
                score.candidate_names for score in matching if score.acquisition_cost == min_cost
            )
            saving = min_cost - expected
        else:
            min_cost = None
            matched_names = ()
            saving = None
        results.append(
            ScenarioExpectedCost(
                model.name,
                selected_info,
                adaptive.adaptive_information_gain_over_fixed_oracle_bits[model.name],
                expected,
                scenario_worst,
                selected_worst,
                early_mass,
                distribution,
                min_cost,
                matched_names,
                saving,
            )
        )
    return AdaptiveExpectedCostAudit(
        adaptive.status,
        budget,
        order,
        tuple(results),
        adaptive.worst_path_cost,
        support_reference,
        tol,
        adaptive.complete_search,
    )


def early_stop_witness():
    """Three targets: one branch resolves after one query, the other needs two."""
    rows = ({"target": "zero"}, {"target": "one"}, {"target": "two"})
    events = (("zero", "one"), ("rest", "one"), ("rest", "two"))
    matrix = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    scenarios = (
        JointCalibrationScenario(
            "uniform",
            (1.0, 1.0, 1.0),
            events,
            matrix,
            "synthetic deterministic early-stop law",
        ),
    )
    return rows, scenarios


def synthetic_example() -> dict:
    early_rows, early_scenarios = early_stop_witness()
    early = audit_adaptive_expected_cost(
        early_rows,
        early_scenarios,
        candidate_order=("screen", "resolve"),
        acquisition_costs={"screen": 1, "resolve": 1},
        budget=2,
        target_columns=("target",),
        support_reference="synthetic three-target early-stop panel",
    )
    route_rows, route_scenarios = routing_witness()
    routing = tuple(
        audit_adaptive_expected_cost(
            route_rows,
            route_scenarios,
            candidate_order=("context", "assay0", "assay1"),
            acquisition_costs={"context": 1, "assay0": 1, "assay1": 1},
            budget=budget,
            target_columns=("target",),
            support_reference="synthetic four-world routing panel",
        )
        for budget in (1, 2, 3)
    )
    return early, routing
