"""Budget-indexed value of the adaptive policy class relative to fixed bundles.

This optional diagnostic compares scenario-specific *class oracles* at each
pathwise budget.  It is distinct from the selected robust minimax-regret policy:
for every scenario s and budget B,

    gap_s(B) = max_adaptive I_s(T;H) - max_fixed I_s(T;Q_bundle) >= 0.

No meta-prior over scenarios is introduced.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .adaptive_joint_design import plan_adaptive_joint_budget
from .joint_budgeted_design import JointCalibrationScenario


@dataclass(frozen=True)
class BudgetAdaptivityRow:
    budget: int
    adaptive_oracle_information_bits: dict[str, float]
    fixed_oracle_information_bits: dict[str, float]
    adaptivity_gap_bits: dict[str, float]
    positive_gap_scenarios: tuple[str, ...]


@dataclass(frozen=True)
class AdaptivityBudgetProfile:
    rows: tuple[BudgetAdaptivityRow, ...]
    positive_gap_budgets_any_scenario: tuple[int, ...]
    positive_gap_budgets_all_scenarios: tuple[int, ...]
    peak_gap_bits_by_scenario: dict[str, float]
    peak_gap_budgets_by_scenario: dict[str, tuple[int, ...]]
    scope: str = (
        "scenario_specific_adaptive_class_oracle_minus_fixed_bundle_oracle_"
        "over_declared_integer_pathwise_budgets"
    )


def adaptivity_budget_profile(
    accepted_rows: Sequence[Mapping],
    scenarios: Sequence[JointCalibrationScenario],
    *,
    candidate_order: Sequence[str],
    acquisition_costs: Mapping[str, int],
    budgets: Sequence[int],
    target_columns: Sequence[str],
    support_reference: str,
    comparison_tolerance_bits: float = 1e-12,
    max_tree_combinations: int = 200_000,
) -> AdaptivityBudgetProfile:
    """Evaluate where outcome-contingent routing adds value over fixed bundles.

    Budgets must be distinct nonnegative integers.  The comparison is fair:
    adaptive and fixed classes use the same joint scenarios, target, costs and
    budget.  The adaptive value is the per-scenario oracle already computed by
    ``plan_adaptive_joint_budget``; it is not the information of the selected
    cross-scenario minimax-regret tree.
    """
    values = tuple(budgets)
    if not values or any(type(b) is not int or b < 0 for b in values):
        raise ValueError("budgets must be a nonempty sequence of nonnegative integers")
    if len(set(values)) != len(values):
        raise ValueError("budgets must be unique")
    values = tuple(sorted(values))
    models = tuple(scenarios)
    names = tuple(model.name for model in models)
    rows = []
    tol = float(comparison_tolerance_bits)
    for budget in values:
        receipt = plan_adaptive_joint_budget(
            accepted_rows,
            models,
            candidate_order=candidate_order,
            acquisition_costs=acquisition_costs,
            budget=budget,
            target_columns=target_columns,
            support_reference=support_reference,
            comparison_tolerance_bits=tol,
            max_tree_combinations=max_tree_combinations,
        )
        adaptive = dict(receipt.oracle_information_bits)
        fixed = dict(receipt.per_scenario_best_fixed_information_bits)
        gaps = {}
        for name in names:
            gap = adaptive[name] - fixed[name]
            if gap < 0.0 and abs(gap) <= max(tol, 1e-10):
                gap = 0.0
            if gap < -max(tol, 1e-10):
                raise ArithmeticError("adaptive class oracle fell below fixed bundle oracle")
            gaps[name] = gap
        positive = tuple(name for name in names if gaps[name] > tol)
        rows.append(BudgetAdaptivityRow(budget, adaptive, fixed, gaps, positive))

    any_budgets = tuple(row.budget for row in rows if row.positive_gap_scenarios)
    all_budgets = tuple(
        row.budget for row in rows
        if names and len(row.positive_gap_scenarios) == len(names)
    )
    peaks = {
        name: max(row.adaptivity_gap_bits[name] for row in rows)
        for name in names
    }
    peak_budgets = {
        name: tuple(
            row.budget for row in rows
            if abs(row.adaptivity_gap_bits[name] - peaks[name]) <= tol
        )
        for name in names
    }
    return AdaptivityBudgetProfile(
        tuple(rows), any_budgets, all_budgets, peaks, peak_budgets
    )
