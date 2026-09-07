"""Internal audit of unknown calibration, not an oracle or a new design policy.

A single joint weighted ensemble carries both the question target and persistent
calibration states. Supplied weights and future likelihoods must be conditional
on current data. This module reuses existing scoring and conditioning; it does
not convert a sensitivity-scenario list into an invented probability distribution.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import fsum, isfinite
from typing import Mapping, Sequence

from causal_model.empirical_observation_contract import (
    LikelihoodCandidate, _matrix, _weights, condition_on_selected,
    score_likelihood_candidates, target_state,
)


@dataclass(frozen=True)
class LatentCalibrationValue:
    candidate: str
    estimable: bool
    reason: str
    calibration_reference: str | None
    marginalized_target_information_bits: float | None
    known_calibration_average_information_bits: float | None
    oracle_minus_marginal_bits: float | None
    expected_posterior_target_calibration_information_bits: float | None
    chain_rule_error_bits: float | None
    calibration_only_likelihood_in_pool: bool | None


@dataclass(frozen=True)
class LatentCalibrationAudit:
    target_columns: tuple[str, ...]
    calibration_columns: tuple[str, ...]
    target_entropy_bits: float
    calibration_entropy_bits: float
    current_target_calibration_information_bits: float
    target_image_size: int
    target_identified_in_declared_pool: bool
    candidates: tuple[LatentCalibrationValue, ...]
    best_positive_marginalized_candidates: tuple[str, ...]
    complete_singleton_coverage: bool
    ranking_scope: str
    support_reference: str
    joint_weight_reference: str
    conditional_likelihood_reference: str
    information_tolerance_bits: float
    feasible_domain_exhaustiveness: str = "not_certified"
    scope: str = "finite_joint_ensemble_expected_information_not_causal_or_cost_guarantee"


def audit_latent_calibration(
    accepted_rows: Sequence[Mapping], candidates: Sequence[LikelihoodCandidate], *,
    target_columns: Sequence[str], calibration_columns: Sequence[str],
    weights: Sequence[float], support_reference: str, joint_weight_reference: str,
    conditional_likelihood_reference: str, information_tolerance_bits: float = 1e-10,
) -> LatentCalibrationAudit:
    """Compare actual target MI to the distinct known-calibration oracle quantity.

    The returned signed gap obeys
    I(T;Q|L,D)-I(T;Q|D) = I(T;L|Q,D)-I(T;L|D).
    It need not be positive after current data induce dependence between T and L.
    A calibration-only likelihood can then inform T without measuring T directly.

    All per-outcome calculations are prospective integrations, not access to an
    unselected observation's realised outcome. A supplied joint distribution is
    required; robustness over a set of scenarios is a different task.
    """
    if (isinstance(information_tolerance_bits, bool)
            or not isfinite(information_tolerance_bits) or information_tolerance_bits < 0):
        raise ValueError("information_tolerance_bits must be finite and non-negative")
    if (not isinstance(conditional_likelihood_reference, str)
            or not conditional_likelihood_reference.strip()):
        raise ValueError("conditional_likelihood_reference must be declared")
    if isinstance(target_columns, (str, bytes)) or isinstance(calibration_columns, (str, bytes)):
        raise ValueError("column declarations must be sequences, not bare strings")
    rows, ts, ls = tuple(accepted_rows), tuple(target_columns), tuple(calibration_columns)
    supplied_weights, vocabulary = tuple(weights), tuple(candidates)
    # Validate declarations and every label even if no candidates can be scored.
    for row in rows:
        target_state(row, ts)
        target_state(row, ls)
    if set(ts) & set(ls):
        raise ValueError("target and calibration column names must be disjoint")
    for candidate in vocabulary:
        if not isinstance(candidate.name, str) or not candidate.name.strip():
            raise ValueError("candidate names must be non-empty strings")
        if isinstance(candidate.outcomes, (str, bytes)):
            raise ValueError("candidate outcomes must be a sequence, not a bare string")
    provenance = dict(support_reference=support_reference, weight_reference=joint_weight_reference)

    def score(subrows, subweights, columns, cands=()):
        return score_likelihood_candidates(
            subrows, cands, target_columns=columns, weights=subweights, **provenance,
        )

    def dependence(subrows, subweights):
        ht = score(subrows, subweights, ts).target_entropy_bits
        hl = score(subrows, subweights, ls).target_entropy_bits
        hj = score(subrows, subweights, ts + ls).target_entropy_bits
        value = ht + hl - hj
        if value < -1e-10:
            raise ArithmeticError("negative target-calibration mutual information")
        return max(0.0, value)

    current = score(rows, supplied_weights, ts, vocabulary)
    w = _weights(supplied_weights, len(rows))
    calibration_h = score(rows, w, ls).target_entropy_bits
    current_dependence = dependence(rows, w)
    calibration_states = tuple(target_state(row, ls) for row in rows)
    groups = {}
    for i, state in enumerate(calibration_states):
        groups.setdefault(state, []).append(i)
    values = []
    for candidate, marginal in zip(vocabulary, current.scores):
        matrix = _matrix(candidate, len(rows))
        if matrix is None:
            values.append(LatentCalibrationValue(
                candidate.name, False, "missing_per_world_predictive_likelihood",
                candidate.calibration_reference, None, None, None, None, None, None,
            ))
            continue
        terms = []
        reference_only = True
        for indices in groups.values():
            reference_only &= all(matrix[i] == matrix[indices[0]] for i in indices)
            restricted = LikelihoodCandidate(
                candidate.name, tuple(candidate.outcomes), tuple(matrix[i] for i in indices),
                candidate.calibration_reference,
            )
            value = score(tuple(rows[i] for i in indices), tuple(w[i] for i in indices),
                          ts, [restricted]).scores[0].information_bits
            if value is None:
                raise ArithmeticError("known candidate lost its conditional information")
            terms.append(fsum(w[i] for i in indices) * value)
        known_average = fsum(terms)
        posterior_terms = []
        for q, outcome in enumerate(candidate.outcomes):
            if not any(row[q] > 0 for row in matrix):
                continue  # Impossible outcomes carry no prospective expectation.
            posterior = condition_on_selected(
                rows, candidate, outcome, target_columns=ts, weights=w,
            )
            # Explicit restriction after exact zero likelihoods, never by epsilon.
            indices = tuple(i for i, probability in enumerate(posterior.posterior_weights)
                            if probability > 0)
            dep = dependence(tuple(rows[i] for i in indices),
                             tuple(posterior.posterior_weights[i] for i in indices))
            posterior_terms.append(posterior.outcome_probability * dep)
        expected_dependence = fsum(posterior_terms)
        marginal_bits = marginal.information_bits
        if marginal_bits is None:
            raise ArithmeticError("known candidate lost its marginalized information")
        gap = known_average - marginal_bits
        identity_error = gap - (expected_dependence - current_dependence)
        if abs(identity_error) > 1e-9:
            raise ArithmeticError("latent-calibration chain-rule identity failed")
        if reference_only and (known_average > 1e-9 or marginal_bits > current_dependence + 1e-9):
            raise ArithmeticError("calibration-only information violated its dependence bound")
        values.append(LatentCalibrationValue(
            candidate.name, True, "conditional_on_declared_joint_model_not_verified_calibration",
            candidate.calibration_reference, marginal_bits, known_average, gap,
            expected_dependence, identity_error, reference_only,
        ))
    positive = [v for v in values if v.marginalized_target_information_bits is not None
                and v.marginalized_target_information_bits > information_tolerance_bits]
    best = max((v.marginalized_target_information_bits for v in positive), default=None)
    names = tuple(sorted(v.candidate for v in positive
                         if abs(v.marginalized_target_information_bits - best)
                         <= information_tolerance_bits))
    return LatentCalibrationAudit(
        ts, ls, current.target_entropy_bits, calibration_h, current_dependence,
        current.target_image_size, current.target_point_identified, tuple(values), names,
        current.complete_vocabulary, current.ranking_scope, support_reference,
        joint_weight_reference, conditional_likelihood_reference, information_tolerance_bits,
    )
