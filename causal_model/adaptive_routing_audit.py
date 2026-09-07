"""Audit direct target information, continuation information and routing diversity.

This optional diagnostic sits on top of the existing exact adaptive joint design.
It does not define a new publication-facing MROD score.  In particular,
``routing_action_entropy_bits`` is entropy of the *next action identity* induced
by the selected policy; it is not information about the biological target.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import fsum, log2
from typing import Mapping, Sequence

from .adaptive_joint_design import AdaptiveJointReceipt, plan_adaptive_joint_budget
from .empirical_observation_contract import _weights
from .joint_budgeted_design import (
    JointCalibrationScenario,
    _joint_candidate,
    plan_joint_observation_budget,
)


@dataclass(frozen=True)
class RoutingScenarioAudit:
    scenario: str
    total_policy_target_information_bits: float
    root_direct_target_information_bits: float
    conditional_continuation_target_information_bits: float
    routing_action_entropy_bits: float
    selected_policy_gain_over_best_fixed_bits: float


@dataclass(frozen=True)
class AdaptiveRoutingAudit:
    status: str
    root_query: str | None
    next_action_by_root_outcome: tuple[tuple[str, str], ...]
    branch_dependent_continuation: bool
    root_direct_target_information_zero_all_scenarios: bool
    routing_without_direct_target_information: bool
    scenario_audits: tuple[RoutingScenarioAudit, ...]
    adaptive_receipt: AdaptiveJointReceipt
    scope: str = (
        "selected_adaptive_tree_direct_target_information_chain_rule_and_"
        "next_action_entropy_under_declared_joint_scenarios"
    )


def _entropy(probabilities) -> float:
    return -fsum(p * log2(p) for p in probabilities if p > 0.0)


def audit_adaptive_routing(
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
) -> AdaptiveRoutingAudit:
    """Decompose a selected adaptive policy without inventing a routing utility.

    For each declared scenario the chain-rule accounting is

        I(T; full transcript)
        = I(T; root observation)
          + [I(T; full transcript)-I(T; root observation)].

    The bracketed term is reported as conditional continuation target
    information.  Separately, root outcomes are mapped to the selected policy's
    immediate next action (or ``__stop__``).  Entropy of that action identity
    describes how branch-dependent the continuation is under each scenario.

    A zero root target-information value can therefore coexist with positive
    continuation information.  It becomes a routing witness only when at least
    two root outcomes actually select different next actions.  This distinction
    prevents ordinary zero-MI synergy with a branch-invariant continuation from
    being relabelled as adaptive routing value.
    """
    models = tuple(scenarios)
    order = tuple(candidate_order)
    receipt = plan_adaptive_joint_budget(
        accepted_rows,
        models,
        candidate_order=order,
        acquisition_costs=acquisition_costs,
        budget=budget,
        target_columns=target_columns,
        support_reference=support_reference,
        comparison_tolerance_bits=comparison_tolerance_bits,
        max_tree_combinations=max_tree_combinations,
    )
    root = receipt.selected_policy
    if root is None or root.query is None:
        return AdaptiveRoutingAudit(
            "no_nonterminal_selected_root",
            None,
            (),
            False,
            False,
            False,
            (),
            receipt,
        )

    root_query = root.query
    root_index = order.index(root_query)
    next_by_outcome = tuple(
        (label, child.query if child.query is not None else "__stop__")
        for label, child in root.branches
    )
    next_lookup = dict(next_by_outcome)
    branch_dependent = len(set(next_lookup.values())) > 1

    fixed = plan_joint_observation_budget(
        accepted_rows,
        models,
        candidate_order=order,
        acquisition_costs=acquisition_costs,
        budget=budget,
        target_columns=target_columns,
        support_reference=support_reference,
        comparison_tolerance_bits=comparison_tolerance_bits,
    )
    singleton = next(
        score for score in fixed.bundle_scores
        if score.candidate_names == (root_query,)
    )

    audits = []
    tol = float(comparison_tolerance_bits)
    for model in models:
        direct = singleton.information_by_scenario[model.name]
        if direct is None:
            raise ArithmeticError("selected adaptive root lost its singleton information value")
        total = receipt.selected_information_bits[model.name]
        continuation = total - direct
        if continuation < 0.0 and abs(continuation) <= max(tol, 1e-10):
            continuation = 0.0
        if continuation < -max(tol, 1e-10):
            raise ArithmeticError("chain-rule continuation information became negative")

        events, matrix = _joint_candidate(model, len(tuple(accepted_rows)), len(order))
        if matrix is None:
            raise ArithmeticError("adaptive recommendation exists despite missing joint likelihood")
        weights = _weights(model.weights, len(tuple(accepted_rows)))
        outcome_mass: dict[str, list[float]] = {}
        for weight, row in zip(weights, matrix):
            for event, probability in zip(events, row):
                label = event[root_index]
                if label not in next_lookup:
                    # A positive-probability root outcome absent from the policy
                    # would mean the planner dropped declared joint support.
                    if probability > 0.0:
                        raise ArithmeticError("selected policy omitted a positive root outcome")
                    continue
                outcome_mass.setdefault(label, []).append(weight * probability)
        action_mass: dict[str, list[float]] = {}
        for label, pieces in outcome_mass.items():
            mass = fsum(pieces)
            action_mass.setdefault(next_lookup[label], []).append(mass)
        action_probabilities = tuple(fsum(pieces) for pieces in action_mass.values())
        routing_entropy = _entropy(action_probabilities)
        best_fixed = receipt.per_scenario_best_fixed_information_bits[model.name]
        audits.append(RoutingScenarioAudit(
            model.name,
            total,
            direct,
            continuation,
            routing_entropy,
            total - best_fixed,
        ))

    zero_direct = bool(audits) and all(
        audit.root_direct_target_information_bits <= tol for audit in audits
    )
    routing_witness = (
        branch_dependent
        and zero_direct
        and any(audit.conditional_continuation_target_information_bits > tol for audit in audits)
    )
    return AdaptiveRoutingAudit(
        "routing_audit_complete",
        root_query,
        next_by_outcome,
        branch_dependent,
        zero_direct,
        routing_witness,
        tuple(audits),
        receipt,
    )
