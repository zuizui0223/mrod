from dataclasses import replace
from itertools import product
import pytest

from causal_model.adaptive_joint_design import (
    AdaptiveSearchLimitError, AdaptivePolicyNode, _Option, _frontier,
    plan_adaptive_joint_budget, execute_adaptive_policy, routing_witness,
)
from causal_model.joint_budgeted_design import JointCalibrationScenario, plan_joint_observation_budget


def plan(rows=None, models=None, **changes):
    base_rows, base_models = routing_witness()
    kwargs = dict(candidate_order=('context', 'assay0', 'assay1'),
                  acquisition_costs={'context': 1, 'assay0': 1, 'assay1': 1}, budget=2,
                  target_columns=('target',), support_reference='synthetic test support')
    kwargs.update(changes)
    return plan_adaptive_joint_budget(base_rows if rows is None else rows,
                                     base_models if models is None else models, **kwargs)


def test_routing_doubles_information_at_same_pathwise_budget():
    r = plan()
    assert r.selected_policy.query == 'context'
    assert r.worst_path_cost == 2
    assert r.selected_information_bits == {s: 1.0 for s in r.scenario_names}
    assert r.per_scenario_best_fixed_information_bits == {
        'uniform_context': 0.5, 'context0_common': 0.75, 'context1_common': 0.75}
    assert r.worst_regret_bits == 0
    assert r.best_fixed_worst_regret_against_adaptive_oracle_bits == 0.5
    assert r.uniformly_optimal_policy_exists
    assert r.complete_search
    assert not r.conditional_reoptimization_guaranteed


def test_first_query_has_zero_singleton_target_information():
    rows, models = routing_witness()
    fixed = plan_joint_observation_budget(rows, models,
        candidate_order=('context', 'assay0', 'assay1'),
        acquisition_costs={'context': 1, 'assay0': 1, 'assay1': 1}, budget=1,
        target_columns=('target',), support_reference='synthetic')
    context = next(b for b in fixed.bundle_scores if b.candidate_names == ('context',))
    assert set(context.information_by_scenario.values()) == {0.0}
    assert plan().selected_policy.query == 'context'


@pytest.mark.parametrize('context,target', product(('0', '1'), repeat=2))
def test_executor_only_requests_actual_path(context, target):
    trace = []
    available = {'context': context, f'assay{context}': target}
    def observe(q):
        trace.append(q)
        return available[q]  # An unselected assay has no value to read.
    result = execute_adaptive_policy(plan(), observe)
    assert trace == ['context', f'assay{context}']
    assert result.acquisition_cost == 2
    assert result.represented_target_point_identified


def test_unknown_outcome_is_not_silently_rerouted():
    with pytest.raises(ValueError, match='support'):
        execute_adaptive_policy(plan(), lambda q: 'not_in_support')


def test_zero_budget_uses_no_callback_and_does_not_identify_target():
    r = plan(budget=0)
    result = execute_adaptive_policy(r, lambda q: pytest.fail('no query allowed'))
    assert result.acquisition_cost == 0
    assert not result.represented_target_point_identified
    assert r.worst_path_cost == 0
    assert all(x == 0 for x in r.selected_information_bits.values())


def test_target_already_identified_stops_without_spending():
    rows, models = routing_witness()
    r = plan(rows=tuple({'target': 'same'} for _ in rows), models=models)
    result = execute_adaptive_policy(r, lambda q: pytest.fail('target already identified'))
    assert result.represented_target_point_identified
    assert result.acquisition_cost == 0


def test_three_query_budget_removes_information_gap_for_this_witness():
    r = plan(budget=3)
    assert r.adaptive_information_gain_over_fixed_oracle_bits == {s: 0 for s in r.scenario_names}
    assert r.worst_path_cost <= 3


def test_no_missing_prediction_becomes_zero_information():
    rows, models = routing_witness()
    r = plan(rows, (models[0], replace(models[1], probabilities=None)))
    assert not r.complete_search
    assert r.selected_policy is None
    assert r.status.startswith('missing_joint')
    with pytest.raises(ValueError, match='no complete'):
        execute_adaptive_policy(r, lambda q: '0')


def test_joint_event_order_has_no_semantic_effect():
    rows, models = routing_witness()
    reordered = tuple(replace(m, joint_outcomes=m.joint_outcomes[::-1],
                             probabilities=tuple(row[::-1] for row in m.probabilities))
                      if j % 2 else m for j, m in enumerate(models))
    assert plan(rows, reordered).selected_information_bits == plan(rows, models).selected_information_bits


def test_scenarios_are_compared_not_averaged_or_relabelled():
    rows, models = routing_witness()
    a, b = plan(rows, models), plan(rows, models[::-1])
    assert a.selected_information_bits == b.selected_information_bits
    assert a.worst_regret_bits == b.worst_regret_bits


def test_vector_frontier_keeps_tradeoffs_for_later_branch_combination():
    node = AdaptivePolicyNode(None, 0, (), 2)
    options = (_Option((0, 1), node, 1), _Option((1, 0), node, 1),
               _Option((0.6, 0.6), node, 1), _Option((1, 1), node, 1))
    assert {o.residuals for o in _frontier(options)} == {(0, 1), (1, 0), (0.6, 0.6)}


def test_tiny_positive_probability_never_becomes_point_identification():
    rows = ({'target': 0}, {'target': 1})
    m = JointCalibrationScenario('tiny_error', (1, 1), (('0',), ('1',)),
        ((1.0, 1e-16), (1e-16, 1.0)), 'synthetic tiny error')
    r = plan(rows, (m,), candidate_order=('q',), acquisition_costs={'q': 1}, budget=1)
    result = execute_adaptive_policy(r, lambda q: '0')
    assert result.remaining_target_image_size == 2
    assert not result.represented_target_point_identified


def test_unequal_costs_obey_every_path_not_only_expected_budget():
    r = plan(acquisition_costs={'context': 1, 'assay0': 1, 'assay1': 3}, budget=2)
    assert r.worst_path_cost <= 2
    assert r.oracle_information_bits['uniform_context'] < 1


@pytest.mark.parametrize('changes', [
    {'budget': True}, {'budget': -1}, {'target_columns': 'target'},
    {'candidate_order': 'context'}, {'comparison_tolerance_bits': float('nan')},
    {'max_tree_combinations': 0},
    {'acquisition_costs': {'context': 0, 'assay0': 1, 'assay1': 1}},
])
def test_invalid_contract_fails(changes):
    with pytest.raises(ValueError):
        plan(**changes)


def test_missing_target_is_not_a_singleton_label():
    rows, models = routing_witness()
    with pytest.raises(ValueError, match='target'):
        plan(({},)+rows[1:], models)


def test_malformed_likelihood_is_not_a_missing_prediction():
    rows, models = routing_witness()
    malformed = replace(models[0], probabilities=((2.0,)*8,)*4)
    with pytest.raises(ValueError):
        plan(rows, (malformed,))


def test_search_cap_is_not_a_structural_impossibility():
    with pytest.raises(AdaptiveSearchLimitError, match='not certified'):
        plan(max_tree_combinations=1)


def test_xor_complementarity_is_preserved():
    worlds = tuple(product((0, 1), repeat=2))
    events = tuple(product(('0', '1'), repeat=3))
    matrix = tuple(tuple((0.8 if int(z) == u ^ v else 0.2)
                         if (int(a), int(b)) == (u, v) else 0.0
                         for a, b, z in events) for u, v in worlds)
    model = JointCalibrationScenario('xor', (1,)*4, events, matrix, 'synthetic')
    r = plan(tuple({'target': u ^ v} for u, v in worlds), (model,),
             candidate_order=('U', 'V', 'direct'), acquisition_costs={'U': 1, 'V': 1, 'direct': 1})
    assert r.selected_information_bits['xor'] == 1
    assert r.adaptive_information_gain_over_fixed_oracle_bits['xor'] == 0


def test_per_scenario_information_one_does_not_identify_target_across_unknown_scenarios():
    rows = ({'target': 0}, {'target': 1})
    events = (('0',), ('1',))
    models = (JointCalibrationScenario('normal', (1, 1), events, ((1, 0), (0, 1)), 'synthetic'),
              JointCalibrationScenario('reversed', (1, 1), events, ((0, 1), (1, 0)), 'synthetic'))
    r = plan(rows, models, candidate_order=('q',), acquisition_costs={'q': 1}, budget=1)
    assert r.selected_information_bits == {'normal': 1, 'reversed': 1}
    result = execute_adaptive_policy(r, lambda q: '0')
    assert result.remaining_target_image_size == 2
    assert not result.represented_target_point_identified
