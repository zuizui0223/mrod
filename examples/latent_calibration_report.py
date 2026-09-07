"""Synthetic shared-calibration example: python -m examples.latent_calibration_report."""
from __future__ import annotations

from dataclasses import asdict
import json
from math import isfinite

from causal_model.empirical_observation_contract import LikelihoodCandidate, condition_on_selected
from causal_model.latent_calibration_audit import audit_latent_calibration
from causal_model.replication_information_audit import replication_information_profile

SUPPORT = "four synthetic process/background worlds; not an exhaustive natural-system model"
FUTURE = "fresh response and standard readings independent given the same persistent full world"


def shared_background_model(reference_error=0.1):
    """The background shifts detection without being the process target itself."""
    if isinstance(reference_error, bool) or not isfinite(reference_error) or not 0 <= reference_error <= 0.5:
        raise ValueError("reference_error must lie in [0, 0.5]")
    rows = tuple({"process": t, "calibration": l} for t in (0, 1) for l in (0, 1))
    response = LikelihoodCandidate(
        "response", ("absent", "present"),
        ((0.9, 0.1), (0.5, 0.5), (0.5, 0.5), (0.1, 0.9)),
        "synthetic p(response)=0.1+0.4*process+0.4*calibration; no field calibration",
    )
    e = reference_error
    standard = LikelihoodCandidate(
        "shared_standard", ("low_background", "high_background"),
        tuple((1-e, e) if row["calibration"] == 0 else (e, 1-e) for row in rows),
        f"synthetic known-standard response depends only on shared calibration; error={e}",
    )
    return rows, response, standard


def weights_after_history(history):
    """Update the joint ensemble on observed response results, not on future data."""
    rows, response, _ = shared_background_model()
    weights, probability = (1.0, 1.0, 1.0, 1.0), 1.0
    for outcome in tuple(history):
        posterior = condition_on_selected(
            rows, response, outcome, target_columns=["process"], weights=weights,
        )
        weights = posterior.posterior_weights
        probability *= posterior.outcome_probability
    total = sum(weights)
    return tuple(w / total for w in weights), probability


def _profile(rows, candidate, weights, reference, horizons=(1,)):
    return replication_information_profile(
        rows, candidate, target_columns=["process"], weights=weights,
        support_reference=SUPPORT, weight_reference=reference,
        conditional_iid_reference=FUTURE, horizons=horizons,
    )


def audit_history(history, *, reference_error=0.1, include_unmodelled=False):
    history = tuple(history)
    rows, response, standard = shared_background_model(reference_error)
    weights, history_probability = weights_after_history(history)
    unlinked = LikelihoodCandidate(
        "unlinked_standard", standard.outcomes, ((0.5, 0.5),) * len(rows),
        "synthetic new independent calibration state, not shared with the response record",
    )
    candidates = [response, standard, unlinked]
    if include_unmodelled:
        candidates.append(LikelihoodCandidate("unmodelled_followup", ("no", "yes"), None))
    reference = f"independent uniform initial T,L updated jointly on {history!r}"
    audit = audit_latent_calibration(
        rows, candidates, target_columns=["process"], calibration_columns=["calibration"],
        weights=weights, support_reference=SUPPORT, joint_weight_reference=reference,
        conditional_likelihood_reference=FUTURE,
    )
    profile = _profile(rows, response, weights, reference)
    target_information = next(v.marginalized_target_information_bits for v in audit.candidates
                              if v.candidate == standard.name)
    return {
        "observed_response_history": list(history),
        "probability_of_this_ordered_history": history_probability,
        "current_joint_world_weights": list(weights),
        "audit": asdict(audit),
        "response_repeat_ceiling_bits": profile.asymptotic_information_bits,
        "response_repeat_residual_floor_bits": profile.irreducible_target_entropy_bits,
        "same_response_law_different_target_pair": profile.same_law_different_target_pair,
        "standard_minus_entire_response_repeat_ceiling_bits":
            target_information - profile.asymptotic_information_bits,
    }


def build_report():
    rows, response, standard = shared_background_model()
    fixed = []
    for state in (0, 1):
        indices = tuple(i for i, row in enumerate(rows) if row["calibration"] == state)
        candidate = LikelihoodCandidate(
            response.name, response.outcomes, tuple(response.probabilities[i] for i in indices),
            response.calibration_reference,
        )
        profile = _profile(tuple(rows[i] for i in indices), candidate, (1, 1),
                           f"oracle calibration known to be {state}")
        fixed.append({"known_calibration": state,
                      "asymptotic_target_entropy_bits": profile.irreducible_target_entropy_bits})
    persistent = _profile(rows, response, (1, 1, 1, 1), "independent uniform T,L",
                          horizons=(0, 1, 2, 10, 50))
    # This is a DIFFERENT experiment, not a valid shortcut for persistent L.
    redrawn = _profile(
        ({"process": 0}, {"process": 1}),
        LikelihoodCandidate("redrawn_background_response", response.outcomes,
                            ((0.7, 0.3), (0.3, 0.7)), "background redrawn independently each reading"),
        (1, 1), "uniform target; independent uniform background redrawn at every reading",
        horizons=(0, 1, 2, 10, 50),
    )
    balanced = ("present", "absent") * 5
    weights, _ = weights_after_history(balanced)
    posterior = condition_on_selected(
        rows, standard, "high_background", target_columns=["process"], weights=weights,
    )
    return {
        "data_kind": "synthetic_latent_shared_calibration_not_field_evidence",
        "fixed_known_calibration": fixed,
        "unknown_persistent_calibration": asdict(persistent),
        "independently_redrawn_calibration_different_protocol": asdict(redrawn),
        "current_data_comparisons": [audit_history(h) for h in
                                     ((), ("present",), ("present", "absent"), balanced)],
        "incomplete_vocabulary": audit_history(balanced, include_unmodelled=True),
        "uninformative_standard": audit_history(balanced, reference_error=0.5),
        "illustrative_selected_standard_outcome_not_used_in_ranking": asdict(posterior),
        "scope": "joint model-relative expected information; not exact resolution, calibration validation or cost optimality",
    }


if __name__ == "__main__":
    print(json.dumps(build_report(), indent=2, allow_nan=False))
