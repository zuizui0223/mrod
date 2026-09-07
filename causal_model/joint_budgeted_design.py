"""Robust FIXED-BUNDLE design from an explicitly declared joint observation law.

Marginal likelihoods do not determine complementarity. Supply a joint outcome
matrix for each scenario, including dependence. No conditional independence,
scenario meta-prior, adaptive-policy optimality, or report licence is invented.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
import json
from math import fsum, isfinite
from typing import Mapping, Sequence

from .empirical_observation_contract import LikelihoodCandidate
from .robust_observation_design import CalibrationScenario, robust_likelihood_design


@dataclass(frozen=True)
class JointCalibrationScenario:
    name: str
    weights: tuple[float, ...]
    joint_outcomes: tuple[tuple[str, ...], ...]
    probabilities: tuple[tuple[float, ...], ...] | None
    calibration_reference: str


@dataclass(frozen=True)
class BudgetedBundleScore:
    candidate_names: tuple[str, ...]
    acquisition_cost: int
    information_by_scenario: dict[str, float | None]
    worst_regret_bits: float | None


@dataclass(frozen=True)
class JointBudgetReceipt:
    budget: int
    candidate_order: tuple[str, ...]
    bundle_scores: tuple[BudgetedBundleScore, ...]
    uniformly_best_bundles: tuple[tuple[str, ...], ...]
    minimax_regret_bundles: tuple[tuple[str, ...], ...]
    minimum_worst_regret_bits: float | None
    complete_vocabulary: bool
    ranking_scope: str
    utility_units: str = "raw_target_information_bits"
    scope: str = "enumerated_joint_scenarios_budget_feasible_nonadaptive_bundles"
    adaptive_policy_optimality_claimed: bool = False


def _integer(value: int, name: str, minimum: int = 0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _joint_candidate(model: JointCalibrationScenario, n_worlds: int, n_candidates: int):
    events = tuple(tuple(event) for event in model.joint_outcomes)
    if (not events or len(set(events)) != len(events)
            or any(len(event) != n_candidates for event in events)
            or any(not isinstance(v, str) or not v.strip() for event in events for v in event)):
        raise ValueError("unique joint events must align with candidate_order and contain nonempty labels")
    names = tuple(f"joint_{j}" for j in range(len(events)))
    candidate = LikelihoodCandidate("__joint_validation__", names, model.probabilities,
                                    model.calibration_reference)
    # Reuse existing probability checks, including normalization and finite rows.
    from .empirical_observation_contract import _matrix
    return events, _matrix(candidate, n_worlds)


def plan_joint_observation_budget(
    accepted_rows: Sequence[Mapping], scenarios: Sequence[JointCalibrationScenario], *,
    candidate_order: Sequence[str], acquisition_costs: Mapping[str, int], budget: int,
    target_columns: Sequence[str], support_reference: str,
    comparison_tolerance_bits: float = 1e-12,
) -> JointBudgetReceipt:
    """Enumerate all affordable fixed bundles and apply paired robust regret.

    Matrices give P(Q_1,...,Q_m | world), not separate marginal predictions.
    Events can omit structurally impossible combinations, but each row must be a
    complete declared joint law. This completeness is an input assumption, not
    empirical validation. All models share the same joint event support labels;
    zero probabilities can differ. Costs are additive positive integer units.
    Missing joint matrices make nonempty bundles non-estimable; no authoritative
    full-vocabulary choice is returned. The empty bundle is explicitly feasible.
    """
    B = _integer(budget, "budget")
    if isinstance(candidate_order, (str, bytes)):
        raise ValueError("candidate_order must be a sequence, not a bare string")
    order, rows, models = tuple(candidate_order), tuple(accepted_rows), tuple(scenarios)
    if (not order or len(set(order)) != len(order)
            or any(not isinstance(v, str) or not v.strip() for v in order)):
        raise ValueError("declare a nonempty unique candidate order")
    if len(order) > 8:
        raise ValueError("exact bundle enumeration permits at most 8 candidate observations")
    if set(acquisition_costs) != set(order):
        raise ValueError("supply exactly one acquisition cost per candidate")
    costs = tuple(_integer(acquisition_costs[v], "acquisition cost", 1) for v in order)
    if not rows or not models:
        raise ValueError("nonempty world rows and joint scenarios required")
    tol = float(comparison_tolerance_bits)
    if not isfinite(tol) or tol < 0:
        raise ValueError("comparison tolerance must be finite and nonnegative")
    # Every bundle is precommitted. The numbering carries identity, not data.
    subsets = tuple(s for s in range(1 << len(order))
                    if sum(costs[i] for i in range(len(order)) if s & (1 << i)) <= B)
    bundle_names = tuple(tuple(order[i] for i in range(len(order)) if s & (1 << i)) for s in subsets)
    bundle_costs = tuple(sum(acquisition_costs[v] for v in names) for names in bundle_names)
    scenario_inputs = []
    event_support = None
    for model in models:
        events, matrix = _joint_candidate(model, len(rows), len(order))
        if event_support is None:
            event_support = set(events)
        elif set(events) != event_support:
            raise ValueError("scenarios must share joint outcome support labels; encode impossible events with zero")
        candidates = []
        for j, subset in enumerate(subsets):
            indices = tuple(i for i in range(len(order)) if subset & (1 << i))
            projected = tuple(tuple(event[i] for i in indices) for event in events)
            states = tuple(dict.fromkeys(projected))
            labels = tuple(f"outcome_{i}" for i in range(len(states)))
            if subset == 0:
                probabilities = ((1.0,),)*len(rows)
            elif matrix is None:
                probabilities = None
            else:
                probabilities = tuple(tuple(fsum(row[k] for k, event in enumerate(projected) if event == state)
                                            for state in states) for row in matrix)
            candidates.append(LikelihoodCandidate(f"bundle_{j}", labels, probabilities,
                                                   model.calibration_reference))
        scenario_inputs.append(CalibrationScenario(model.name, model.weights, tuple(candidates),
                                                   model.calibration_reference))
    robust = robust_likelihood_design(rows, scenario_inputs, target_columns=target_columns,
        support_reference=support_reference, comparison_tolerance_bits=tol)
    lookup = {f"bundle_{j}": names for j, names in enumerate(bundle_names)}
    scores = tuple(BudgetedBundleScore(bundle_names[j], bundle_costs[j],
                                      s.information_by_scenario, s.worst_regret_bits)
                   for j, s in enumerate(robust.scores))
    return JointBudgetReceipt(B, order, scores,
        tuple(lookup[n] for n in robust.uniformly_best_names),
        tuple(lookup[n] for n in robust.minimax_regret_names),
        robust.minimum_worst_regret_bits, robust.complete_vocabulary, robust.ranking_scope)


def synthetic_example() -> dict:
    # Explicit complete joint law: U,V are deterministic world coordinates;
    # direct is a noisy report of T=U XOR V. No products of fitted marginals.
    worlds = tuple(product((0, 1), repeat=2))
    rows = tuple({"target": u ^ v} for u, v in worlds)
    outcomes = tuple(product(("0", "1"), repeat=3))
    matrix = tuple(tuple((0.8 if int(z) == (u ^ v) else 0.2)
                         if (int(a), int(b)) == (u, v) else 0.0
                         for a, b, z in outcomes) for u, v in worlds)
    scenarios = (JointCalibrationScenario("uniform", (1, 1, 1, 1), outcomes, matrix,
                                          "synthetic explicit joint law"),
                 JointCalibrationScenario("target_rare", (9, 1, 1, 9), outcomes, matrix,
                                          "synthetic target-prior sensitivity"))
    results = []
    for budget in (1, 2):
        r = plan_joint_observation_budget(rows, scenarios, candidate_order=("U", "V", "direct"),
            acquisition_costs={"U": 1, "V": 1, "direct": 1}, budget=budget,
            target_columns=("target",), support_reference="synthetic four-world XOR panel")
        results.append(asdict(r))
    return {"data_kind": "synthetic_explicit_joint_likelihood", "budgets": results}


if __name__ == "__main__":
    print(json.dumps(synthetic_example(), indent=2, allow_nan=False))
