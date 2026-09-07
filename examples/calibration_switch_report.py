"""Synthetic calibration thresholds and fragile infinite-repeat dominance.

Run: python -m examples.calibration_switch_report
"""
from __future__ import annotations

from dataclasses import asdict
import json

from causal_model.calibration_switch_audit import (
    audit_binary_error_calibration, binary_error_candidate, target_information_continuity_bound,
)
from causal_model.empirical_observation_contract import (
    LikelihoodCandidate, condition_on_selected, score_likelihood_candidates,
)
from causal_model.replication_information_audit import replication_information_profile


REFERENCES = dict(
    support_reference="three synthetic programs; natural-system exhaustiveness not certified",
    weight_reference="equal initial weights reconditioned on the same contact-present result",
    conditional_iid_reference="fresh contact readings iid given a fixed full program",
)


def current_model(delta=0.0):
    """Perturb the combined program's contact law, then recondition past evidence."""
    if isinstance(delta, bool) or not 0 <= delta < 0.1:
        raise ValueError("delta must lie in [0, 0.1)")
    rows = tuple({"program": label} for label in ("pollination_only", "abiotic_only", "combined"))
    contact = LikelihoodCandidate(
        "contact", ("absent", "present"),
        ((0.1, 0.9), (0.9, 0.1), (0.1-delta, 0.9+delta)),
        f"synthetic combined-law perturbation delta={delta}; not field calibration",
    )
    posterior = condition_on_selected(rows, contact, "present", target_columns=["program"],
                                      weights=(1, 1, 1))
    return rows, contact, posterior.posterior_weights


def error_audit(interval):
    rows, contact, weights = current_model()
    return audit_binary_error_calibration(
        rows, contact, ideal_readings=(0, 1, 1), target_columns=["program"], weights=weights,
        error_interval=interval, candidate_name="physiology", **REFERENCES,
        future_likelihood_reference="alternative is independent of past contact conditional on full world",
        calibration_reference="only alternative symmetric error varies; old law and current weights fixed",
    )


def law_split_report(delta):
    rows, contact, weights = current_model(delta)
    profile = replication_information_profile(
        rows, contact, target_columns=["program"], weights=weights, **REFERENCES, horizons=(1, 50),
    )
    alternative = binary_error_candidate("physiology", (0, 1, 1), 0.1,
                                          calibration_reference="fixed synthetic alternative error=0.1")
    score = score_likelihood_candidates(
        rows, [alternative], target_columns=["program"], weights=weights,
        support_reference=REFERENCES["support_reference"], weight_reference=REFERENCES["weight_reference"],
    ).scores[0]
    return dict(
        delta=delta, observed_history=["present"], current_weights=weights,
        alternative_information_bits=score.information_bits,
        one_step_switch_margin_bits=score.information_bits-profile.horizons[0].information_bits,
        margin_over_50_repeats_bits=score.information_bits-profile.horizons[1].information_bits,
        margin_over_unlimited_ceiling_bits=score.information_bits-profile.asymptotic_information_bits,
        repeat_audit=asdict(profile),
    )


def finite_budget_guarantee(delta_upper=1e-6, max_repeats=50):
    """Analytic coupling bound for delta in [0,delta_upper], not a grid claim.

    Past contact-present probabilities are (.9,.1,.9+delta). Their normalized
    weights change in TV by delta/[1.9*(1.9+delta)] <= delta/1.9.
    Couple future Bernoulli readings: m repeats disagree with probability at
    most m*delta. The alternative law stays fixed, so only its weights change.
    """
    if isinstance(delta_upper, bool) or not 0 <= delta_upper < 0.1:
        raise ValueError("delta_upper must lie in [0, 0.1)")
    if isinstance(max_repeats, bool) or not isinstance(max_repeats, int) or max_repeats < 1:
        raise ValueError("max_repeats must be a positive integer")
    baseline = error_audit((0.1, 0.1))
    ceiling_gap = baseline.comparisons[1].worst_advantage_bits
    weights_tv = delta_upper/1.9  # Conservative, intentionally not sharp.
    repeated_joint_tv = min(1.0, weights_tv+max_repeats*delta_upper)
    loss = (target_information_continuity_bound(weights_tv, 3)
            + target_information_continuity_bound(repeated_joint_tv, 3))
    lower_margin = ceiling_gap-loss
    return dict(
        scope="all delta in the declared interval and all integer repeat budgets 0..max_repeats",
        delta_interval=(0.0, delta_upper), max_repeats=max_repeats,
        baseline_switch_over_ceiling_margin_bits=ceiling_gap,
        weight_tv_upper=weights_tv, repeat_joint_tv_upper=repeated_joint_tv,
        information_continuity_penalty_bits=loss,
        switch_margin_lower_bound_bits=lower_margin,
        dominates_every_audited_budget_with_margin=lower_margin > 1e-10,
        numerical_status="analytic_bound_evaluated_in_float_not_machine_verified_interval_arithmetic",
        exclusions="not arbitrary calibration variation, unknown-calibration mixing, cost or realised utility",
    )


def build_report():
    return dict(
        data_kind="synthetic_calibration_breakdown_and_law_equivalence_audit",
        current_history=["contact_present"],
        alternative_error_intervals=[asdict(error_audit(interval)) for interval in
                                     ((0.1, 0.18), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4))],
        law_equality_sensitivity=[law_split_report(delta) for delta in (0.0, 1e-3, 1e-6, 1e-9)],
        finite_budget_uniform_bound=finite_budget_guarantee(),
        warning="Infinite-repeat dominance requires an exact old-law restriction; finite-budget superiority need not.",
    )


if __name__ == "__main__":
    print(json.dumps(build_report(), indent=2, allow_nan=False))
