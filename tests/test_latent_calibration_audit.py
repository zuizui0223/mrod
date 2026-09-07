"""Independent finite joint-table oracle, aliasing and input/scope regressions."""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, replace
from itertools import product
import json
from math import log2
import random

import pytest

from causal_model.empirical_observation_contract import LikelihoodCandidate
from causal_model.latent_calibration_audit import audit_latent_calibration
from examples.latent_calibration_report import (
    audit_history, build_report, shared_background_model, weights_after_history,
)


def run(rows=None, candidates=None, **kwargs):
    base_rows, response, standard = shared_background_model()
    options = dict(target_columns=["process"], calibration_columns=["calibration"],
                   weights=[1] * len(base_rows if rows is None else rows),
                   support_reference="synthetic finite support", joint_weight_reference="declared joint prior",
                   conditional_likelihood_reference="fresh given fixed full world")
    options.update(kwargs)
    return audit_latent_calibration(base_rows if rows is None else rows,
                                   [response, standard] if candidates is None else candidates, **options)


def direct_mi(joint, x_indices, y_indices):
    """Independent direct probability-ratio definition; not an entropy subtraction."""
    xy, px, py = defaultdict(float), defaultdict(float), defaultdict(float)
    total = sum(joint.values())
    for state, probability in joint.items():
        x, y = tuple(state[i] for i in x_indices), tuple(state[i] for i in y_indices)
        p = probability / total
        xy[x, y] += p
        px[x] += p
        py[y] += p
    return sum(p * log2(p / (px[x] * py[y])) for (x, y), p in xy.items() if p > 0)


def direct_cmi(joint, x_indices, y_indices, z_indices):
    groups = defaultdict(dict)
    for state, p in joint.items():
        if p > 0:
            groups[tuple(state[i] for i in z_indices)][state] = p
    total = sum(joint.values())
    return sum(sum(group.values()) / total * direct_mi(group, x_indices, y_indices)
               for group in groups.values())


def compare_to_oracle(rows, weights, matrix):
    candidate = LikelihoodCandidate("candidate", tuple(f"q{i}" for i in range(len(matrix[0]))), tuple(matrix))
    audit = run(rows, [candidate], weights=weights)
    joint = defaultdict(float)
    for row, weight, probabilities in zip(rows, weights, matrix):
        for q, probability in enumerate(probabilities):
            joint[row["process"], row["calibration"], q] += weight * probability
    marginal = direct_mi(joint, (0,), (2,))
    oracle = direct_cmi(joint, (0,), (2,), (1,))
    dependence = direct_mi(joint, (0,), (1,))
    post_dependence = direct_cmi(joint, (0,), (1,), (2,))
    value = audit.candidates[0]
    assert value.marginalized_target_information_bits == pytest.approx(marginal, abs=2e-12)
    assert value.known_calibration_average_information_bits == pytest.approx(oracle, abs=2e-12)
    assert audit.current_target_calibration_information_bits == pytest.approx(dependence, abs=2e-12)
    assert value.expected_posterior_target_calibration_information_bits == pytest.approx(post_dependence, abs=2e-12)
    assert value.oracle_minus_marginal_bits == pytest.approx(post_dependence-dependence, abs=2e-12)
    assert abs(value.chain_rule_error_bits) < 2e-12


def test_all_4096_binary_maps_against_independent_joint_table():
    maps = tuple(product((0, 1), repeat=4))
    for t, l, q in product(maps, repeat=3):
        rows = tuple({"process": a, "calibration": b} for a, b in zip(t, l))
        matrix = tuple((1-outcome, outcome) for outcome in q)
        compare_to_oracle(rows, [1] * 4, matrix)


def test_200_weighted_noisy_joint_models_against_independent_oracle():
    rng = random.Random(20260907)
    for _ in range(200):
        rows = tuple({"process": rng.randrange(3), "calibration": rng.randrange(2)} for _ in range(6))
        weights = [rng.uniform(0.01, 4) for _ in rows]
        matrix = []
        for _ in rows:
            raw = [rng.random() for _ in range(3)]
            matrix.append(tuple(p / sum(raw) for p in raw))
        compare_to_oracle(rows, weights, matrix)


def test_known_calibration_score_is_not_unknown_calibration_score():
    prior = run()
    response, standard = prior.candidates
    assert response.marginalized_target_information_bits == pytest.approx(0.1187091007693073)
    assert response.known_calibration_average_information_bits == pytest.approx(0.14679310243605204)
    assert response.oracle_minus_marginal_bits > 0
    assert standard.calibration_only_likelihood_in_pool
    assert standard.marginalized_target_information_bits == pytest.approx(0)
    assert prior.current_target_calibration_information_bits == pytest.approx(0)


def test_past_data_can_make_calibration_only_measurement_target_informative():
    history = audit_history(("present", "absent") * 5)
    response, standard, unlinked = history["audit"]["candidates"]
    assert standard["calibration_only_likelihood_in_pool"]
    assert not response["calibration_only_likelihood_in_pool"]
    assert standard["marginalized_target_information_bits"] == pytest.approx(0.5159454284095983)
    assert response["marginalized_target_information_bits"] == pytest.approx(0.00001667688706797, abs=1e-12)
    assert standard["known_calibration_average_information_bits"] == pytest.approx(0, abs=1e-12)
    assert response["known_calibration_average_information_bits"] > 0
    assert standard["oracle_minus_marginal_bits"] < 0
    assert history["audit"]["best_positive_marginalized_candidates"] == ("shared_standard",)
    assert unlinked["marginalized_target_information_bits"] == pytest.approx(0)


def test_single_standard_beats_entire_response_only_ceiling_in_this_model():
    history = audit_history(("present", "absent") * 5)
    assert history["response_repeat_ceiling_bits"] == pytest.approx(0.006010275760803929)
    assert history["standard_minus_entire_response_repeat_ceiling_bits"] > 0.5099
    assert history["same_response_law_different_target_pair"] == (1, 2)
    assert history["response_repeat_residual_floor_bits"] > 0.9939


def test_full_world_mixture_not_product_of_target_marginals():
    report = build_report()
    known = report["fixed_known_calibration"]
    assert all(row["asymptotic_target_entropy_bits"] == 0 for row in known)
    persistent = report["unknown_persistent_calibration"]
    redrawn = report["independently_redrawn_calibration_different_protocol"]
    assert persistent["irreducible_target_entropy_bits"] == pytest.approx(0.5)
    assert redrawn["irreducible_target_entropy_bits"] == 0
    assert persistent["horizons"][1]["information_bits"] == pytest.approx(redrawn["horizons"][1]["information_bits"])
    assert persistent["horizons"][2]["information_bits"] != pytest.approx(redrawn["horizons"][2]["information_bits"])


def test_reference_information_is_not_exact_identification():
    posterior = build_report()["illustrative_selected_standard_outcome_not_used_in_ranking"]
    assert posterior["remaining_target_image_size"] == 2
    assert not posterior["target_point_identified"]
    assert all(p > 0 for p in posterior["posterior_weights"])
    assert posterior["target_entropy_bits"] == pytest.approx(0.48405457159040166)


def test_joint_law_refinement_requires_informative_shared_standard():
    rows, response, standard = shared_background_model()
    pairs = [(a[1], b[1]) for a, b in zip(response.probabilities, standard.probabilities)]
    assert len(set(pairs)) == 4
    _, _, useless = shared_background_model(0.5)
    pairs = [(a[1], b[1]) for a, b in zip(response.probabilities, useless.probabilities)]
    assert pairs[1] == pairs[2] and rows[1]["process"] != rows[2]["process"]
    report = audit_history(("present", "absent") * 5, reference_error=0.5)
    assert report["audit"]["best_positive_marginalized_candidates"] == ("response",)


def test_target_calibration_correlation_must_not_be_forced_to_zero():
    rows = ({"process": 0, "calibration": 0}, {"process": 1, "calibration": 1})
    candidate = LikelihoodCandidate("reference", ("zero", "one", "impossible"), ((1, 0, 0), (0, 1, 0)))
    audit = run(rows, [candidate])
    value = audit.candidates[0]
    assert audit.current_target_calibration_information_bits == pytest.approx(1)
    assert value.marginalized_target_information_bits == pytest.approx(1)
    assert value.known_calibration_average_information_bits == pytest.approx(0)
    assert value.oracle_minus_marginal_bits == pytest.approx(-1)
    assert value.expected_posterior_target_calibration_information_bits == pytest.approx(0)
    assert value.calibration_only_likelihood_in_pool


def test_resolution_and_zero_candidate_values_are_not_equivalent():
    rows = ({"process": 0, "calibration": 0}, {"process": 0, "calibration": 1})
    constant = LikelihoodCandidate("uninformative", ("zero", "one"), ((0.5, 0.5),) * 2)
    resolved = run(rows, [constant])
    assert resolved.target_identified_in_declared_pool
    rows = ({"process": 0, "calibration": 0}, {"process": 1, "calibration": 1})
    unresolved = run(rows, [constant])
    assert not unresolved.target_identified_in_declared_pool
    assert unresolved.best_positive_marginalized_candidates == ()
    assert not hasattr(unresolved, "sequence_information_limit")


def test_partial_vocabulary_retains_nonestimable_not_zero():
    report = audit_history(("present", "absent") * 5, include_unmodelled=True)["audit"]
    assert not report["complete_singleton_coverage"]
    assert report["ranking_scope"] == "provisional_estimable_subset"
    assert report["best_positive_marginalized_candidates"] == ("shared_standard",)
    value = report["candidates"][-1]
    assert not value["estimable"]
    assert value["marginalized_target_information_bits"] is None
    assert value["oracle_minus_marginal_bits"] is None
    assert value["calibration_only_likelihood_in_pool"] is None


def test_empty_candidates_and_all_missing_do_not_certify_information_limit():
    audit = run(candidates=[])
    assert not audit.complete_singleton_coverage
    assert audit.best_positive_marginalized_candidates == ()
    missing = LikelihoodCandidate("missing", ("no", "yes"), None)
    audit = run(candidates=[missing])
    assert not audit.complete_singleton_coverage
    assert not audit.candidates[0].estimable


@pytest.mark.parametrize("field,bad", [
    ("target_columns", "process"), ("target_columns", []),
    ("calibration_columns", "calibration"), ("calibration_columns", []),
    ("calibration_columns", ["calibration", "calibration"]),
    ("calibration_columns", ["process"]),
    ("weights", [1, 1, 1, 0]), ("weights", [1, 1]),
    ("weights", [1, 1, 1, float("inf")]),
    ("support_reference", ""), ("joint_weight_reference", ""),
    ("conditional_likelihood_reference", ""),
    ("information_tolerance_bits", -1), ("information_tolerance_bits", True),
    ("information_tolerance_bits", float("nan")),
])
def test_invalid_contract_is_rejected(field, bad):
    with pytest.raises(ValueError):
        run(**{field: bad})


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), [], (1, None)])
def test_invalid_calibration_labels_not_treated_as_known(value):
    rows, _, _ = shared_background_model()
    rows = [dict(row) for row in rows]
    rows[0]["calibration"] = value
    with pytest.raises(ValueError):
        run(rows, [])


def test_missing_calibration_is_rejected_before_missing_candidate():
    rows, _, _ = shared_background_model()
    rows = [dict(row) for row in rows]
    del rows[0]["calibration"]
    with pytest.raises(ValueError):
        run(rows, [LikelihoodCandidate("unknown", ("x", "y"), None)])


def test_duplicate_candidate_and_outcome_names_rejected():
    _, response, _ = shared_background_model()
    with pytest.raises(ValueError):
        run(candidates=[response, response])
    with pytest.raises(ValueError):
        run(candidates=[replace(response, outcomes=("same", "same"))])
    with pytest.raises(ValueError):
        run(candidates=[replace(response, outcomes="ab")])


def test_underflow_is_not_a_structural_zero():
    rows = ({"process": 0, "calibration": 0}, {"process": 1, "calibration": 1})
    candidate = LikelihoodCandidate("tiny", ("a", "b"), ((1.0, 1e-200), (0.5, 0.5)))
    with pytest.raises(ValueError, match="underflow"):
        run(rows, [candidate], weights=[1e-300, 1.0])


def test_ties_no_mutation_order_invariance_and_json():
    rows, response, _ = shared_background_model()
    rows = [dict(row) for row in rows]
    candidates = [response, replace(response, name="same_response")]
    before = deepcopy((rows, candidates))
    first = run(rows, candidates)
    second = run(rows[::-1], [replace(c, probabilities=c.probabilities[::-1]) for c in candidates[::-1]])
    assert first.best_positive_marginalized_candidates == ("response", "same_response")
    assert first.best_positive_marginalized_candidates == second.best_positive_marginalized_candidates
    assert {v.candidate: v.marginalized_target_information_bits for v in first.candidates} == pytest.approx(
        {v.candidate: v.marginalized_target_information_bits for v in second.candidates})
    assert (rows, candidates) == before
    assert json.loads(json.dumps(asdict(first), allow_nan=False))["feasible_domain_exhaustiveness"] == "not_certified"


def test_actual_history_update_matches_independent_likelihood_product():
    weights, probability = weights_after_history(("present", "absent") * 5)
    raw = [0.09**5, 0.25**5, 0.25**5, 0.09**5]
    assert weights == pytest.approx([p / sum(raw) for p in raw])
    assert probability == pytest.approx(sum(raw) / 4)
