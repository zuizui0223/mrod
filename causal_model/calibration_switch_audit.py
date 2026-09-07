"""Internal binary-error sensitivity audit, not a new design objective.

Only the alternative's independent symmetric error varies. Current weights,
ideal per-world readings and the old protocol stay fixed. Theorems are exact
within that family; floating-point brackets are not interval-arithmetic proofs.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log2
from numbers import Integral, Real
from typing import Callable, Mapping, Sequence

from causal_model.empirical_observation_contract import LikelihoodCandidate, score_likelihood_candidates
from causal_model.replication_information_audit import replication_information_profile


def _number(value, name: str, lower: float, upper: float) -> float:
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not isfinite(value) or not lower <= value <= upper):
        raise ValueError(f"{name} must be finite in [{lower}, {upper}]")
    return float(value)


def binary_error_candidate(
    name: str, ideal_readings: Sequence[int], error: float, *, calibration_reference: str,
) -> LikelihoodCandidate:
    """XOR a fixed ideal bit with independent Bernoulli(error) noise."""
    e = _number(error, "error", 0.0, 0.5)
    if not isinstance(name, str) or not name.strip():
        raise ValueError("candidate name must be non-empty")
    if not isinstance(calibration_reference, str) or not calibration_reference.strip():
        raise ValueError("calibration_reference must declare the symmetric-error model")
    if isinstance(ideal_readings, (str, bytes)):
        raise ValueError("ideal_readings must be a sequence of binary integers")
    bits = tuple(ideal_readings)
    if not bits or any(not isinstance(b, Integral) or b not in (0, 1) for b in bits):
        raise ValueError("ideal_readings must contain binary integers")
    return LikelihoodCandidate(name, ("absent", "present"),
                               tuple((1-e, e) if b == 0 else (e, 1-e) for b in bits),
                               calibration_reference)


@dataclass(frozen=True)
class CalibrationComparison:
    benchmark: str
    benchmark_bits: float | None
    worst_advantage_bits: float | None
    best_advantage_bits: float | None
    interval_status: str
    equality_error_bracket: tuple[float, float] | None
    threshold_status: str


@dataclass(frozen=True)
class BinaryErrorCalibrationAudit:
    candidate: str
    repeat_candidate: str
    target_columns: tuple[str, ...]
    target_entropy_bits: float
    target_identified_in_declared_pool: bool
    error_interval: tuple[float, float]
    information_at_lower_error_bits: float
    information_at_upper_error_bits: float
    comparisons: tuple[CalibrationComparison, ...]
    support_reference: str
    weight_reference: str
    conditional_iid_reference: str
    future_likelihood_reference: str
    calibration_reference: str
    information_tolerance_bits: float
    root_width: float
    scope: str = "pairwise_fixed_current_state_symmetric_error_family_only"
    numerical_status: str = "ordinary_float_evaluations_not_validated_interval_arithmetic"
    feasible_domain_exhaustiveness: str = "not_certified"


def _comparison(
    label: str, benchmark: float | None, information: Callable[[float], float],
    best: float, worst: float, width: float, tolerance: float,
) -> CalibrationComparison:
    if benchmark is None:
        return CalibrationComparison(label, None, None, None, "non_estimable",
                                     None, "missing_repeat_likelihood")
    low_gap, high_gap = worst-benchmark, best-benchmark
    if low_gap > tolerance:
        status = "uniformly_above_with_margin"
    elif high_gap < -tolerance:
        status = "uniformly_below_with_margin"
    elif max(abs(low_gap), abs(high_gap)) <= tolerance:
        status = "near_tie_throughout_at_tolerance"
    else:
        status = "crosses_or_touches_at_tolerance"
    # Root of I(error)=benchmark, not of I(error)=benchmark+tolerance.
    if information(0.0) <= benchmark+tolerance:
        bracket, threshold = None, "no_strict_advantage_detected_at_tolerance"
    elif benchmark == 0.0:
        bracket, threshold = (0.5, 0.5), "analytic_zero_at_half_error"
    else:
        lo, hi = 0.0, 0.5
        for _ in range(64):
            if hi-lo <= width:
                break
            mid = (lo+hi)/2
            if information(mid) > benchmark:
                lo = mid
            else:
                hi = mid
        bracket, threshold = (lo, hi), "numerical_equality_bracket"
    return CalibrationComparison(label, benchmark, low_gap, high_gap, status, bracket, threshold)


def audit_binary_error_calibration(
    accepted_rows: Sequence[Mapping], repeat_candidate: LikelihoodCandidate, *,
    ideal_readings: Sequence[int], target_columns: Sequence[str], weights: Sequence[float],
    error_interval: Sequence[float], support_reference: str, weight_reference: str,
    conditional_iid_reference: str, future_likelihood_reference: str,
    calibration_reference: str, candidate_name: str = "alternative",
    root_width: float = 1e-10, information_tolerance_bits: float = 1e-10,
) -> BinaryErrorCalibrationAudit:
    """Compare an entire ordered error interval with one repeat and its ceiling.

    The monotone endpoint result follows from binary-channel degradation, not
    a finite grid. It is conditional on FIXED current weights and repeat laws.
    If varied calibration also affected past data, update weights per scenario
    instead; this ordered-family audit alone then no longer certifies the range.
    Non-estimable repeat predictions remain non-estimable, never zero.
    """
    width = _number(root_width, "root_width", 1e-12, 0.5)
    tolerance = _number(information_tolerance_bits, "information_tolerance_bits", 0, 1)
    if not isinstance(future_likelihood_reference, str) or not future_likelihood_reference.strip():
        raise ValueError("future_likelihood_reference must declare predictions after current data")
    if isinstance(error_interval, (str, bytes)):
        raise ValueError("error_interval must contain two ordered errors")
    interval = tuple(error_interval)
    if len(interval) != 2:
        raise ValueError("error_interval must contain two ordered errors")
    lower, upper = (_number(x, "error_interval", 0, 0.5) for x in interval)
    if lower > upper:
        raise ValueError("error_interval must be ordered")
    if isinstance(target_columns, (str, bytes)):
        raise ValueError("target_columns must be a sequence, not a bare string")
    rows, columns, supplied_weights = tuple(accepted_rows), tuple(target_columns), tuple(weights)
    ideal = binary_error_candidate(candidate_name, ideal_readings, 0.0,
                                   calibration_reference=calibration_reference)
    if len(ideal.probabilities) != len(rows):
        raise ValueError("ideal_readings must cover every current world")
    if candidate_name == repeat_candidate.name:
        raise ValueError("alternative and repeat candidate names must differ")
    bits = tuple(int(row[1]) for row in ideal.probabilities)
    profile = replication_information_profile(
        rows, repeat_candidate, target_columns=columns, weights=supplied_weights,
        support_reference=support_reference, weight_reference=weight_reference,
        conditional_iid_reference=conditional_iid_reference, horizons=(1,),
    )
    cache: dict[float, float] = {}
    def information(error: float) -> float:
        if error not in cache:
            candidate = binary_error_candidate(candidate_name, bits, error,
                                               calibration_reference=calibration_reference)
            value = score_likelihood_candidates(
                rows, [candidate], target_columns=columns, weights=supplied_weights,
                support_reference=support_reference, weight_reference=weight_reference,
            ).scores[0].information_bits
            if value is None:
                raise ArithmeticError("complete binary-error model became non-estimable")
            cache[error] = value
        return cache[error]
    best, worst = information(lower), information(upper)
    if worst > best+1e-9:
        raise ArithmeticError("binary-channel degradation increased target information")
    one = profile.horizons[0].information_bits if profile.estimable else None
    comparisons = tuple(_comparison(label, value, information, best, worst, width, tolerance)
                        for label, value in (("one_repeat", one),
                                             ("all_repeat_ceiling", profile.asymptotic_information_bits)))
    return BinaryErrorCalibrationAudit(
        candidate_name, repeat_candidate.name, columns, profile.target_entropy_bits,
        profile.target_identified_in_declared_pool, (lower, upper), best, worst,
        comparisons, support_reference, weight_reference, conditional_iid_reference,
        future_likelihood_reference, calibration_reference, tolerance, width,
    )


def target_information_continuity_bound(tv_bound: float, target_states: int) -> float:
    """Conservative |I_P(T;Y)-I_Q(T;Y)| bound from joint total variation.

    On a common finite target alphabet of size k, the common-part mixture
    argument bounds each entropy change by eta*log2(k)+h2(eta). For a TV
    upper bound <=1/2 this increases with eta. Larger bounds return log2(k).
    A supplied TV bound is an assumption/calculation, not calibration by this API.
    """
    eta = _number(tv_bound, "tv_bound", 0, 1)
    if (isinstance(target_states, bool) or not isinstance(target_states, Integral)
            or target_states < 1):
        raise ValueError("target_states must be a positive integer")
    maximum = log2(target_states)
    if eta == 0 or target_states == 1:
        return 0.0
    if eta > 0.5:
        return maximum
    binary_h = -eta*log2(eta)-(1-eta)*log2(1-eta)
    return min(maximum, 2*(eta*maximum+binary_h))
