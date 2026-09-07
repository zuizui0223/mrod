"""Small exhaustive adaptive designs under a pathwise budget and fixed scenarios.

The input is the existing explicit JOINT observation law. The selected policy is
an outcome-contingent tree, not a fixed bundle or a scenario-aware oracle. Regret
is evaluated ex ante with one scenario held fixed over the entire tree. There is
no meta-prior, branchwise adversarial scenario switch, or randomized policy.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from itertools import product
import json
from math import fsum, isfinite, log2
from typing import Callable, Mapping, Sequence

from .empirical_observation_contract import _weights, target_state
from .joint_budgeted_design import (
    JointCalibrationScenario, _integer, _joint_candidate,
    plan_joint_observation_budget,
)


class AdaptiveSearchLimitError(RuntimeError):
    """The exhaustive search limit was reached; no optimality receipt is issued."""


@dataclass(frozen=True)
class AdaptivePolicyNode:
    query: str | None
    acquisition_cost: int
    branches: tuple[tuple[str, 'AdaptivePolicyNode'], ...]
    remaining_target_image_size: int


@dataclass(frozen=True)
class AdaptiveJointReceipt:
    status: str
    budget: int
    support_reference: str
    candidate_order: tuple[str, ...]
    scenario_names: tuple[str, ...]
    selected_policy: AdaptivePolicyNode | None
    selected_information_bits: dict[str, float]
    oracle_information_bits: dict[str, float]
    per_scenario_best_fixed_information_bits: dict[str, float]
    adaptive_information_gain_over_fixed_oracle_bits: dict[str, float]
    worst_regret_bits: float | None
    best_fixed_bundle_against_adaptive_oracle: tuple[str, ...] | None
    best_fixed_worst_regret_against_adaptive_oracle_bits: float | None
    uniformly_optimal_policy_exists: bool
    worst_path_cost: int | None
    retained_root_vectors: int
    search_states: int
    evaluated_tree_combinations: int
    comparison_tolerance_bits: float
    scope: str = 'finite_joint_scenarios_deterministic_adaptive_trees_pathwise_budget_ex_ante_regret'
    complete_search: bool = True
    continuous_scenario_region_certified: bool = False
    randomized_policies_optimized: bool = False
    conditional_reoptimization_guaranteed: bool = False


@dataclass(frozen=True)
class AdaptiveExecutionReceipt:
    observations: tuple[tuple[str, str], ...]
    acquisition_cost: int
    remaining_target_image_size: int
    represented_target_point_identified: bool
    scope: str = 'selected_tree_positive_joint_support_union_not_biological_report_licensing'


@dataclass(frozen=True)
class _Option:
    residuals: tuple[float, ...]
    node: AdaptivePolicyNode
    worst_cost: int


def _frontier(options: Sequence[_Option]) -> tuple[_Option, ...]:
    """Pareto prune residual vectors, not a scalar worst-scenario value.

    Exact floating comparisons are used here; numerical tolerance is only used
    for root near-ties. Thus an information difference is not rounded to zero
    to eliminate a support branch or identify a target.
    """
    kept: list[_Option] = []
    for option in options:
        if any(k.residuals == option.residuals and k.worst_cost <= option.worst_cost
               for k in kept):
            continue
        if any(all(a <= b for a, b in zip(k.residuals, option.residuals))
               and any(a < b for a, b in zip(k.residuals, option.residuals)) for k in kept):
            continue
        kept = [k for k in kept if not (
            (k.residuals == option.residuals and option.worst_cost < k.worst_cost)
            or (all(a <= b for a, b in zip(option.residuals, k.residuals))
                and any(a < b for a, b in zip(option.residuals, k.residuals))))]
        kept.append(option)
    return tuple(kept)


def plan_adaptive_joint_budget(
    accepted_rows: Sequence[Mapping], scenarios: Sequence[JointCalibrationScenario], *,
    candidate_order: Sequence[str], acquisition_costs: Mapping[str, int], budget: int,
    target_columns: Sequence[str], support_reference: str,
    comparison_tolerance_bits: float = 1e-12, max_tree_combinations: int = 200_000,
) -> AdaptiveJointReceipt:
    """Optimize an ex-ante deterministic minimax-regret contingent policy.

    Costs are positive additive integer resource costs. EVERY realized path must
    fit the budget. Each observation can be used at most once; the complete joint
    law must be invariant to order and noninvasive acquisition. Scenarios remain
    fixed, including their world weights, throughout policy evaluation.

    The objective is I_s(T;transcript), equivalently H_s(T) minus expected terminal
    entropy. Scenario-specific oracles use the SAME adaptive policy class/budget.
    Nondominated scenario vectors, rather than locally optimal scalar choices,
    are combined across outcome branches. Max regret is taken ONLY at the root.

    Exact search means exhaustive combinatorics, not exact real-valued logarithms.
    At most 4 observations, 64 labelled events and 128 worlds are supported. A
    resource-limit exception is not an impossibility or approximate optimum.
    """
    if isinstance(candidate_order, (str, bytes)) or isinstance(target_columns, (str, bytes)):
        raise ValueError('candidate_order and target_columns must be sequences')
    order, columns = tuple(candidate_order), tuple(target_columns)
    rows, models = tuple(accepted_rows), tuple(scenarios)
    if len(order) > 4:
        raise ValueError('adaptive exhaustive solver permits at most 4 observations')
    if len(rows) > 128 or len(models) > 8:
        raise ValueError('adaptive solver permits at most 128 worlds and 8 scenarios')
    limit = _integer(max_tree_combinations, 'max_tree_combinations', 1)
    # The preceding API validates costs, target labels, scenario identity, event
    # vocabularies, probability normalization and provenance; it is also the
    # nonadaptive comparator, not a new independent-marginal implementation.
    fixed = plan_joint_observation_budget(
        rows, models, candidate_order=order, acquisition_costs=acquisition_costs,
        budget=budget, target_columns=columns, support_reference=support_reference,
        comparison_tolerance_bits=comparison_tolerance_bits,
    )
    tol = float(comparison_tolerance_bits)
    names = tuple(s.name for s in models)
    costs = tuple(acquisition_costs[q] for q in order)
    states = tuple(target_state(row, columns) for row in rows)
    joint_models = []
    canonical = None
    for model in models:
        events, matrix = _joint_candidate(model, len(rows), len(order))
        if len(events) > 64:
            raise ValueError('adaptive solver permits at most 64 labelled joint events')
        if matrix is None:
            return AdaptiveJointReceipt(
                'missing_joint_likelihood_no_adaptive_recommendation', budget,
                support_reference, order, names, None, {}, {}, {}, {}, None,
                None, None, False, None, 0, 0, 0, tol, complete_search=False,
            )
        if canonical is None:
            canonical = events
        positions = {event: i for i, event in enumerate(events)}
        weights = _weights(model.weights, len(rows))
        masses = []
        for weight, row in zip(weights, matrix):
            masses_row = []
            for event in canonical:
                probability = row[positions[event]]
                mass = weight*probability
                if weight > 0 and probability > 0 and mass == 0:
                    raise ValueError('positive joint mass underflow; no world may silently disappear')
                masses_row.append(mass)
            masses.append(tuple(masses_row))
        joint_models.append(tuple(masses))
    assert canonical is not None
    active = tuple(i for i in range(len(canonical))
                   if any(matrix[w][i] > 0 for matrix in joint_models for w in range(len(rows))))
    root_mask = sum(1 << i for i in active)
    counts = {'combinations': 0}

    @lru_cache(None)
    def at(mask: int):
        indices = tuple(i for i in active if mask & (1 << i))
        target_support = set()
        residuals = []
        for matrix in joint_models:
            by_target = {}
            for target in set(states):
                mass = fsum(matrix[w][i] for w, t in enumerate(states) if t == target for i in indices)
                if mass > 0:
                    target_support.add(target)
                    by_target[target] = mass
            total = fsum(by_target.values())
            # Unconditional mass-weighted entropy, so branch contributions are
            # summed without conditioning each scenario onto a new meta-prior.
            residuals.append(fsum(m*(log2(total)-log2(m)) for m in by_target.values()) if total else 0.0)
        return tuple(residuals), len(target_support)

    @lru_cache(None)
    def search(mask: int, remaining: int, allowance: int):
        entropy, image_size = at(mask)
        stop = _Option(entropy, AdaptivePolicyNode(None, 0, (), image_size), 0)
        if image_size == 1:
            return (stop,)
        options = [stop]
        for j, q in enumerate(order):
            if not (remaining & (1 << j)) or costs[j] > allowance:
                continue
            splits: dict[str, int] = {}
            for i in active:
                if mask & (1 << i):
                    label = canonical[i][j]
                    splits[label] = splits.get(label, 0) | (1 << i)
            # A constant observation consumes resources without changing the
            # information state. Discarding it does not discard zero-MI synergy.
            if len(splits) <= 1:
                continue
            labels = tuple(sorted(splits))
            alternatives = tuple(search(splits[label], remaining ^ (1 << j), allowance-costs[j])
                                 for label in labels)
            for children in product(*alternatives):
                counts['combinations'] += 1
                if counts['combinations'] > limit:
                    raise AdaptiveSearchLimitError('adaptive tree search limit reached; optimality not certified')
                residual = tuple(fsum(child.residuals[s] for child in children) for s in range(len(models)))
                node = AdaptivePolicyNode(q, costs[j], tuple((label, c.node) for label, c in zip(labels, children)), image_size)
                options.append(_Option(residual, node, costs[j]+max(c.worst_cost for c in children)))
            options = list(_frontier(options))
        return _frontier(options)

    options = search(root_mask, (1 << len(order))-1, budget)
    initial_h, _ = at(root_mask)
    oracle_residual = tuple(min(o.residuals[s] for o in options) for s in range(len(models)))
    regrets = tuple(max(o.residuals[s]-oracle_residual[s] for s in range(len(models))) for o in options)
    best = min(regrets)
    near = tuple(i for i, regret in enumerate(regrets) if regret <= best+tol)
    selected_index = min(near, key=lambda i: (options[i].worst_cost, i))
    selected = options[selected_index]
    oracle = {s: initial_h[j]-oracle_residual[j] for j, s in enumerate(names)}
    selected_info = {s: max(0.0, initial_h[j]-selected.residuals[j]) for j, s in enumerate(names)}
    fixed_oracle = {s: max(b.information_by_scenario[s] for b in fixed.bundle_scores) for s in names}
    # Compare fixed and adaptive policies against ONE oracle, not each class's
    # own moving regret baseline.
    fixed_regrets = tuple(max(oracle[s]-b.information_by_scenario[s] for s in names) for b in fixed.bundle_scores)
    best_fixed = min(fixed_regrets)
    fixed_i = min((i for i, r in enumerate(fixed_regrets) if r <= best_fixed+tol),
                  key=lambda i: (fixed.bundle_scores[i].acquisition_cost, i))
    if any(oracle[s] < fixed_oracle[s]-max(tol, 1e-10) for s in names):
        raise ArithmeticError('adaptive policy class lost a feasible fixed-bundle policy')
    return AdaptiveJointReceipt(
        'adaptive_search_complete', budget, support_reference, order, names,
        selected.node, selected_info, oracle, fixed_oracle,
        {s: max(0.0, oracle[s]-fixed_oracle[s]) for s in names},
        regrets[selected_index], fixed.bundle_scores[fixed_i].candidate_names,
        max(0.0, fixed_regrets[fixed_i]),
        any(all(o.residuals[j]-oracle_residual[j] <= tol for j in range(len(names))) for o in options),
        selected.worst_cost, len(options), search.cache_info().currsize,
        counts['combinations'], tol,
    )


def execute_adaptive_policy(receipt: AdaptiveJointReceipt,
                            observe: Callable[[str], str]) -> AdaptiveExecutionReceipt:
    """Request only the chosen observation at each node; never inspect future data.

    `observe` is supplied by the caller. This function does not operate an
    instrument or verify an observation's authenticity. Branch incompatibility
    raises instead of selecting the nearest or most convenient outcome.
    """
    if receipt.selected_policy is None or not receipt.complete_search:
        raise ValueError('no complete adaptive plan is available to execute')
    if not callable(observe):
        raise ValueError('observe must be a callable accepting a selected query name')
    node, used, trace = receipt.selected_policy, 0, []
    queried = set()
    while node.query is not None:
        if node.query in queried or node.acquisition_cost < 1 or used+node.acquisition_cost > receipt.budget:
            raise ValueError('invalid policy: repeated query or pathwise budget violation')
        queried.add(node.query)
        outcome = observe(node.query)
        if not isinstance(outcome, str):
            raise ValueError('observation must be an explicitly labelled string outcome')
        branches = dict(node.branches)
        if outcome not in branches:
            raise ValueError('observed outcome contradicts the declared joint support on this history')
        trace.append((node.query, outcome))
        used += node.acquisition_cost
        node = branches[outcome]
    if used > receipt.budget:
        raise ArithmeticError('executed policy exceeded its declared pathwise budget')
    return AdaptiveExecutionReceipt(tuple(trace), used, node.remaining_target_image_size,
                                    node.remaining_target_image_size == 1)


def routing_witness():
    """Synthetic context chooses which of two assays is informative about target."""
    worlds = tuple(product((0, 1), repeat=2))  # (context,target)
    rows = tuple({'target': t} for c, t in worlds)
    events = tuple(product(('0', '1'), repeat=3))  # context, assay0, assay1
    matrix = tuple(tuple(0.5 if int(route) == c and int((left, right)[c]) == t else 0.0
                         for route, left, right in events) for c, t in worlds)
    scenarios = tuple(JointCalibrationScenario(name, weights, events, matrix, 'synthetic fixed joint routing law')
                      for name, weights in (('uniform_context', (1, 1, 1, 1)),
                                            ('context0_common', (3, 3, 1, 1)),
                                            ('context1_common', (1, 1, 3, 3))))
    return rows, scenarios


def synthetic_example() -> dict:
    rows, scenarios = routing_witness()
    receipt = plan_adaptive_joint_budget(rows, scenarios,
        candidate_order=('context', 'assay0', 'assay1'),
        acquisition_costs={'context': 1, 'assay0': 1, 'assay1': 1}, budget=2,
        target_columns=('target',), support_reference='synthetic four-world routing panel')
    trace = execute_adaptive_policy(receipt, {'context': '1', 'assay1': '0'}.__getitem__)
    return {'data_kind': 'synthetic_declared_joint_routing_witness', 'plan': asdict(receipt),
            'example_execution_not_field_data': asdict(trace)}


if __name__ == '__main__':
    print(json.dumps(synthetic_example(), indent=2, allow_nan=False))
