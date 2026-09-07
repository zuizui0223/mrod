"""Independent information, degradation, perturbation and input checks."""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal, localcontext
from itertools import product
from math import fsum, log2
import random

import pytest

from causal_model.calibration_switch_audit import (
    audit_binary_error_calibration, binary_error_candidate, target_information_continuity_bound,
)
from causal_model.empirical_observation_contract import LikelihoodCandidate, score_likelihood_candidates
from causal_model.replication_information_audit import replication_information_profile
from examples.calibration_switch_report import (
    REFERENCES, build_report, current_model, error_audit, finite_budget_guarantee, law_split_report,
)


def entropy(values):
    return -fsum(p*log2(p) for p in values if p > 0)


def direct_binary_information(targets, weights, probabilities):
    total = fsum(weights)
    w = [v/total for v in weights]
    mean = fsum(v*p for v, p in zip(w, probabilities))
    result = entropy((mean, 1-mean))
    for t in set(targets):
        mass = fsum(v for s, v in zip(targets, w) if s == t)
        p = fsum(v*q for s, v, q in zip(targets, w, probabilities) if s == t)/mass
        result -= mass*entropy((p, 1-p))
    return result


def decimal_threshold(benchmark):
    with localcontext() as ctx:
        ctx.prec = 60
        def h(p):
            return -sum(x*x.ln()/Decimal(2).ln() for x in (p, 1-p) if x)
        a = Decimal(10)/19
        lo, hi = Decimal(0), Decimal('0.5')
        for _ in range(150):
            mid = (lo+hi)/2
            information = h(mid+(1-2*mid)*a)-h(mid)
            if information > benchmark(h):
                lo = mid
            else:
                hi = mid
        return float((lo+hi)/2)


def test_thresholds_match_independent_60_digit_calculation():
    report = error_audit((0.1, 0.18))
    next_root = decimal_threshold(lambda h: h(Decimal(163)/190)-h(Decimal('0.1')))
    ceiling_root = decimal_threshold(lambda h: h(Decimal(1)/19))
    assert next_root == pytest.approx(0.298090224610244, abs=1e-14)
    assert ceiling_root == pytest.approx(0.190138065920987, abs=1e-14)
    for row, truth in zip(report.comparisons, (next_root, ceiling_root)):
        lo, hi = row.equality_error_bracket
        assert lo <= truth <= hi
        assert hi-lo <= report.root_width
    assert report.information_at_lower_error_bits == pytest.approx(0.5297251850260735)


@pytest.mark.parametrize('interval,expected', [
    ((0.1, 0.18), ('uniformly_above_with_margin', 'uniformly_above_with_margin')),
    ((0.1, 0.2), ('uniformly_above_with_margin', 'crosses_or_touches_at_tolerance')),
    ((0.2, 0.3), ('crosses_or_touches_at_tolerance', 'uniformly_below_with_margin')),
    ((0.3, 0.4), ('uniformly_below_with_margin', 'uniformly_below_with_margin')),
])
def test_interval_results_are_not_grid_interpolation(interval, expected):
    report = error_audit(interval)
    assert tuple(c.interval_status for c in report.comparisons) == expected


def test_all_binary_world_maps_against_output_entropy_oracle():
    # 16 target maps x 16 ideal maps x 4 errors = 1,024 independent comparisons.
    for targets, bits in product(product((0, 1), repeat=4), repeat=2):
        rows = [dict(T=t) for t in targets]
        values = []
        for error in (0.0, 0.1, 0.25, 0.5):
            candidate = binary_error_candidate('Q', bits, error, calibration_reference='synthetic')
            score = score_likelihood_candidates(
                rows, [candidate], target_columns=['T'], weights=[1, 2, 3, 4],
                support_reference='enumerated', weight_reference='fixed unequal weights',
            ).scores[0].information_bits
            truth = direct_binary_information(targets, [1, 2, 3, 4],
                                              [error+(1-2*error)*b for b in bits])
            assert score == pytest.approx(truth, abs=1e-12)
            values.append(score)
        assert all(a+1e-12 >= b for a, b in zip(values, values[1:]))


def audit_kwargs():
    rows, repeat, weights = current_model()
    return dict(accepted_rows=rows, repeat_candidate=repeat, weights=weights,
                ideal_readings=(0, 1, 1), target_columns=['program'], error_interval=(0.1, 0.2),
                **REFERENCES, future_likelihood_reference='fixed future conditional model',
                calibration_reference='synthetic independent binary error')


def test_missing_repeat_is_not_zero_and_alternative_is_still_scored():
    kwargs = audit_kwargs()
    kwargs['repeat_candidate'] = LikelihoodCandidate('missing', ('no', 'yes'), None)
    result = audit_binary_error_calibration(**kwargs)
    assert result.information_at_lower_error_bits > 0
    assert all(c.benchmark_bits is None and c.interval_status == 'non_estimable'
               for c in result.comparisons)
    assert all(c.equality_error_bracket is None for c in result.comparisons)


def test_zero_information_and_truly_resolved_targets_do_not_get_fake_roots():
    kwargs = audit_kwargs()
    kwargs['ideal_readings'] = (1, 1, 1)
    result = audit_binary_error_calibration(**kwargs)
    assert result.information_at_lower_error_bits == pytest.approx(0, abs=1e-12)
    assert all(c.equality_error_bracket is None for c in result.comparisons)
    kwargs['accepted_rows'] = [dict(program='same') for _ in range(3)]
    kwargs['ideal_readings'] = (0, 1, 1)
    result = audit_binary_error_calibration(**kwargs)
    assert result.target_identified_in_declared_pool
    assert all(c.interval_status == 'near_tie_throughout_at_tolerance' for c in result.comparisons)


def test_zero_benchmark_has_analytic_half_error_endpoint():
    kwargs = audit_kwargs()
    kwargs['repeat_candidate'] = LikelihoodCandidate('uninformative', ('no', 'yes'), ((0.5, 0.5),)*3)
    result = audit_binary_error_calibration(**kwargs)
    assert all(c.equality_error_bracket == (0.5, 0.5) for c in result.comparisons)


@pytest.mark.parametrize('field,value', [
    ('error_interval', (-0.01, 0.2)), ('error_interval', (0.3, 0.2)),
    ('error_interval', (0.1, 0.51)), ('error_interval', (0.1, float('nan'))),
    ('error_interval', 'ab'), ('error_interval', (0.1,)),
    ('ideal_readings', (0, None, 1)), ('ideal_readings', (0, 2, 1)),
    ('ideal_readings', '011'), ('ideal_readings', (0, 1)),
    ('target_columns', 'program'), ('target_columns', []), ('target_columns', ['missing']),
    ('weights', (1, 0, 1)), ('weights', (1, float('inf'), 1)),
    ('conditional_iid_reference', ''), ('future_likelihood_reference', ''),
    ('calibration_reference', ''), ('root_width', 0), ('root_width', True),
    ('information_tolerance_bits', -1), ('candidate_name', 'contact'),
])
def test_bad_inputs_fail_closed(field, value):
    kwargs = audit_kwargs()
    kwargs[field] = value
    with pytest.raises(ValueError):
        audit_binary_error_calibration(**kwargs)


def test_inputs_are_not_mutated():
    kwargs = audit_kwargs()
    before = deepcopy(kwargs)
    audit_binary_error_calibration(**kwargs)
    assert kwargs == before


def test_splitting_an_exact_law_changes_the_limit_not_the_finite_preference():
    base = law_split_report(0.0)
    altered = law_split_report(1e-6)
    assert base['repeat_audit']['irreducible_target_entropy_bits'] == pytest.approx(18/19)
    assert altered['repeat_audit']['irreducible_target_entropy_bits'] == 0
    assert len(base['repeat_audit']['law_classes']) == 2
    assert len(altered['repeat_audit']['law_classes']) == 3
    assert base['margin_over_unlimited_ceiling_bits'] > 0.23
    assert altered['margin_over_unlimited_ceiling_bits'] < 0
    assert altered['one_step_switch_margin_bits'] > 0.4
    assert altered['margin_over_50_repeats_bits'] > 0.23
    # Historical evidence is reconditioned instead of retaining stale weights.
    assert altered['current_weights'][2] == pytest.approx((0.9+1e-6)/(1.9+1e-6))
    assert altered['current_weights'] != base['current_weights']


def test_finite_budget_uniform_bound_and_independent_scenarios():
    guarantee = finite_budget_guarantee()
    bound = guarantee['switch_margin_lower_bound_bits']
    assert bound == pytest.approx(0.2304795603415861, abs=1e-12)
    assert guarantee['dominates_every_audited_budget_with_margin']
    for delta in (0, 1e-9, 0.25e-6, 0.75e-6, 1e-6):
        rows, repeat, weights = current_model(delta)
        profile = replication_information_profile(rows, repeat, target_columns=['program'],
                    weights=weights, **REFERENCES, horizons=(0, 1, 2, 5, 10, 50))
        alt = direct_binary_information([r['program'] for r in rows], weights, (0.1, 0.9, 0.9))
        for horizon in profile.horizons:
            assert alt-horizon.information_bits >= bound
    # A larger perturbation/budget may make the conservative guarantee fail;
    # failure to certify is not evidence that switching is actually worse.
    assert not finite_budget_guarantee(0.05, 50)['dominates_every_audited_budget_with_margin']


def test_information_continuity_bound_with_independent_joint_tables():
    def mi(table):
        px = [fsum(row) for row in table]
        py = [fsum(row[j] for row in table) for j in range(4)]
        return entropy(px)+entropy(py)-entropy([x for row in table for x in row])
    rng = random.Random(1709)
    for _ in range(200):
        a, b = [[rng.random() for _ in range(12)] for _ in range(2)]
        a = [x/fsum(a) for x in a]
        b = [x/fsum(b) for x in b]
        mix = rng.random()
        b = [(1-mix)*x+mix*y for x, y in zip(a, b)]
        tv = fsum(abs(x-y) for x, y in zip(a, b))/2
        gap = abs(mi([a[i:i+4] for i in (0, 4, 8)])-mi([b[i:i+4] for i in (0, 4, 8)]))
        assert gap <= target_information_continuity_bound(tv, 3)+1e-12
    assert target_information_continuity_bound(0, 3) == 0
    assert target_information_continuity_bound(0.3, 1) == 0
    assert target_information_continuity_bound(0.8, 3) == log2(3)


@pytest.mark.parametrize('tv,k', [(-1, 3), (float('nan'), 3), (2, 3), (0.1, 0), (0.1, True)])
def test_continuity_bound_rejects_bad_assumptions(tv, k):
    with pytest.raises(ValueError):
        target_information_continuity_bound(tv, k)


def test_example_serializes_and_keeps_scope_explicit():
    import json
    report = build_report()
    json.dumps(report, allow_nan=False)
    assert report['data_kind'].startswith('synthetic')
    assert all(a['feasible_domain_exhaustiveness'] == 'not_certified'
               for a in report['alternative_error_intervals'])
    assert all(a['scope'] == 'pairwise_fixed_current_state_symmetric_error_family_only'
               for a in report['alternative_error_intervals'])
