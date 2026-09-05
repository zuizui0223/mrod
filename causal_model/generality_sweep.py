"""Truth-peek-free controlled benchmark for information-guided observation selection.

This is the publication-facing reference surface for the frozen G2 benchmark.
The historical implementation is isolated in ``_compat_generality_sweep`` so the
frozen machine-level policy key can remain reproducible without defining the
current method vocabulary.

The preregistered benchmark compares information-guided selection with uniform
random ordering on identical generated systems, hidden truths, candidate
vocabularies and budgets. A separate post-frozen diagnostic implemented here
compares adaptive information-guided selection with a strong nonadaptive policy
that ranks candidates once by their initial information values.

The frozen JSON stores the information-guided policy under its historical key
``rach_seq``. That string is preserved only for exact protocol/result lookup.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import random
import statistics
from pathlib import Path
from typing import Iterable

from . import _compat_generality_sweep as _impl
from .mechanism_equivalence import mechanism_equivalence_structure
from .mechanism_region import mechanism_resolvability
from .sequential_observation import filter_by_outcome, sequential_candidate_value

HISTORICAL_INFORMATION_GUIDED_POLICY_KEY = "rach_seq"
INFORMATION_GUIDED_POLICY = "information_guided"
STATIC_INITIAL_INFORMATION_POLICY = "static_initial_information"
RANDOM_ORDER_POLICY = "random_order"

Policy = _impl.Policy
SystemRecord = _impl.SystemRecord
SweepResult = _impl.SweepResult
BudgetSummary = _impl.BudgetSummary

# Benchmark-construction helpers retained because the frozen runner and integrity
# tests intentionally exercise the exact generator used to create G2.
_abc_accept = _impl._abc_accept
_candidates_for_system = _impl._candidates_for_system
_make_random_system = _impl._make_random_system
_sample_driver_coefficients = _impl._sample_driver_coefficients
_truth_magnitude = _impl._truth_magnitude
_truth_outcome_overrides = _impl._truth_outcome_overrides
_outcome_by_name = _impl._outcome_by_name
_truth_retained = _impl._truth_retained
_summarize = _impl._summarize
_run_random_order = _impl._run_random_order
_run_information_guided_policy = _impl._run_rach_policy

run_generality_sweep = _impl.run_generality_sweep
run_budget_sweep = _impl.run_budget_sweep
save_budget_table = _impl.save_budget_table
make_figure = _impl.make_figure


def display_policy_name(policy: str) -> str:
    """Translate a frozen machine policy key into current presentation vocabulary."""
    if policy == HISTORICAL_INFORMATION_GUIDED_POLICY_KEY:
        return INFORMATION_GUIDED_POLICY
    return policy


def run_information_guided_sweep(*args, **kwargs):
    """Run the frozen information-guided policy using descriptive caller vocabulary."""
    kwargs = dict(kwargs)
    kwargs["policy"] = HISTORICAL_INFORMATION_GUIDED_POLICY_KEY
    return _impl.run_generality_sweep(*args, **kwargs)


@dataclass(frozen=True)
class _PreparedDiagnosticSystem:
    K: int
    n_confounds: int
    switches: tuple
    drivers: tuple
    truth_driver: tuple
    driver_coeffs: tuple[float, float]
    accepted: tuple[dict, ...]
    candidates: tuple
    outcome_overrides: dict[str, str]
    sequence_seed: int
    n_distractors: int


def _prepare_diagnostic_systems(
    *,
    n_systems: int,
    seed: int,
    n_attempts: int,
    K_choices: tuple[int, ...],
    confound_choices: tuple[int, ...],
    min_sub_size: int,
    n_distractors: int,
) -> list[_PreparedDiagnosticSystem]:
    """Generate each system once so all diagnostic policies share one carrier."""
    master = random.Random(seed)
    prepared: list[_PreparedDiagnosticSystem] = []
    for _ in range(n_systems):
        sys_rng = random.Random(master.randrange(1 << 30))
        K = sys_rng.choice(K_choices)
        n_confounds = min(sys_rng.choice(confound_choices), K // 2)
        switches, drivers, truth_driver = _make_random_system(sys_rng, K, n_confounds)
        driver_coeffs = _sample_driver_coefficients(sys_rng)
        distractor_truth = [bool(sys_rng.getrandbits(1)) for _ in range(n_distractors)]
        accepted = _abc_accept(
            sys_rng,
            switches,
            drivers,
            n_attempts,
            driver_coeffs=driver_coeffs,
            n_distractors=n_distractors,
        )
        if len(accepted) < min_sub_size:
            continue
        candidates = _candidates_for_system(
            drivers,
            accepted,
            driver_coeffs=driver_coeffs,
            n_distractors=n_distractors,
        )
        overrides = _truth_outcome_overrides(
            drivers,
            truth_driver,
            distractor_truth=distractor_truth,
        )
        prepared.append(_PreparedDiagnosticSystem(
            K=K,
            n_confounds=n_confounds,
            switches=tuple(switches),
            drivers=tuple(drivers),
            truth_driver=tuple(truth_driver),
            driver_coeffs=driver_coeffs,
            accepted=tuple(accepted),
            candidates=tuple(candidates),
            outcome_overrides=overrides,
            sequence_seed=sys_rng.randrange(1 << 30),
            n_distractors=n_distractors,
        ))
    return prepared


def _run_static_initial_information(
    prepared: _PreparedDiagnosticSystem,
    *,
    budget: int,
    min_sub_size: int,
):
    """Follow one initial information ranking without branchwise recomputation."""
    current_rows = list(prepared.accepted)
    switches = list(prepared.switches)
    candidates = list(prepared.candidates)
    initial_structure = mechanism_equivalence_structure(current_rows, switches)
    current_structure = initial_structure

    ranked: list[tuple[float, str, object]] = []
    for candidate in candidates:
        score, _ = sequential_candidate_value(
            candidate,
            current_rows,
            switches,
            current_structure,
            min_sub_size=min_sub_size,
        )
        ranked.append((float(score), candidate.name, candidate))
    ranked.sort(key=lambda item: (-item[0], item[1]))

    observations: list[str] = []
    for score, _, candidate in ranked:
        if len(observations) >= budget or not current_structure.edges:
            break
        if score <= 0.0:
            break
        outcome_name = prepared.outcome_overrides.get(candidate.name)
        if outcome_name is None:
            raise RuntimeError(
                f"diagnostic requires a pre-generated hidden outcome for {candidate.name!r}"
            )
        outcome = _outcome_by_name(candidate, outcome_name)
        current_rows = filter_by_outcome(current_rows, outcome.extra_pattern_rows)
        observations.append(candidate.name)
        current_structure = mechanism_equivalence_structure(current_rows, switches)
        if len(current_rows) < min_sub_size:
            break

    initial_ids = {(edge.a, edge.b) for edge in initial_structure.edges}
    final_ids = {(edge.a, edge.b) for edge in current_structure.edges}
    final_R = mechanism_resolvability(current_rows, switches) if current_rows else float("nan")
    return _impl._SequenceOutcome(
        final_rows=current_rows,
        n_resolved=len(initial_ids - final_ids),
        n_unresolved=len(current_structure.edges),
        converged=not bool(current_structure.edges),
        steps_taken=len(observations),
        R_final=final_R,
        observations_taken=observations,
    )


def _diagnostic_record(
    prepared: _PreparedDiagnosticSystem,
    *,
    policy: str,
    budget: int,
    min_sub_size: int,
) -> SystemRecord:
    accepted = list(prepared.accepted)
    switches = list(prepared.switches)
    candidates = list(prepared.candidates)
    initial = mechanism_equivalence_structure(accepted, switches)
    R0 = mechanism_resolvability(accepted, switches)

    if policy == INFORMATION_GUIDED_POLICY:
        outcome = _run_information_guided_policy(
            accepted,
            switches,
            candidates,
            budget=budget,
            min_sub_size=min_sub_size,
            seed=prepared.sequence_seed,
            outcome_overrides=prepared.outcome_overrides,
        )
    elif policy == STATIC_INITIAL_INFORMATION_POLICY:
        outcome = _run_static_initial_information(
            prepared,
            budget=budget,
            min_sub_size=min_sub_size,
        )
    elif policy == RANDOM_ORDER_POLICY:
        outcome = _run_random_order(
            accepted,
            switches,
            candidates,
            budget=budget,
            min_sub_size=min_sub_size,
            seed=prepared.sequence_seed,
            outcome_overrides=prepared.outcome_overrides,
        )
    else:
        raise ValueError(f"unknown diagnostic policy: {policy!r}")

    distractors_selected = sum(
        name.startswith("measure_decoy") for name in outcome.observations_taken
    )
    return SystemRecord(
        K=prepared.K,
        n_confounds=prepared.n_confounds,
        n_initial_edges=len(initial.edges),
        n_resolved=outcome.n_resolved,
        n_unresolved=outcome.n_unresolved,
        converged=outcome.converged,
        steps_taken=outcome.steps_taken,
        R0=round(R0, 4),
        R_final=(round(outcome.R_final, 4) if math.isfinite(outcome.R_final) else outcome.R_final),
        truth_retained=_truth_retained(
            outcome.final_rows,
            list(prepared.drivers),
            list(prepared.truth_driver),
        ),
        truth_peek_free=True,
        driver_coeff_a=prepared.driver_coeffs[0],
        driver_coeff_b=prepared.driver_coeffs[1],
        policy=policy,
        n_distractors=prepared.n_distractors,
        distractors_selected=distractors_selected,
    )


def run_static_information_diagnostic(
    *,
    seeds: Iterable[int],
    budgets: Iterable[int] = (2, 4),
    n_systems_per_seed: int = 200,
    n_attempts: int = 1500,
    K_choices: tuple[int, ...] = (4, 5, 6),
    confound_choices: tuple[int, ...] = (1, 2),
    min_sub_size: int = 8,
    n_distractors: int = 2,
) -> dict:
    """Compare adaptive, static-initial and random policies on matched systems.

    This is a post-frozen claim-ceiling diagnostic. It never edits the frozen G2
    protocol and it must not be represented as part of the preregistered two-policy
    comparison.
    """
    policies = (
        INFORMATION_GUIDED_POLICY,
        STATIC_INITIAL_INFORMATION_POLICY,
        RANDOM_ORDER_POLICY,
    )
    seeds = tuple(int(seed) for seed in seeds)
    budgets = tuple(int(budget) for budget in budgets)
    per_seed: list[dict] = []

    for seed in seeds:
        prepared = _prepare_diagnostic_systems(
            n_systems=n_systems_per_seed,
            seed=seed,
            n_attempts=n_attempts,
            K_choices=K_choices,
            confound_choices=confound_choices,
            min_sub_size=min_sub_size,
            n_distractors=n_distractors,
        )
        for policy in policies:
            for budget in budgets:
                result = SweepResult(n_systems=n_systems_per_seed, policy=policy)
                result.records = [
                    _diagnostic_record(
                        system,
                        policy=policy,
                        budget=budget,
                        min_sub_size=min_sub_size,
                    )
                    for system in prepared
                ]
                _summarize(result)
                per_seed.append({
                    "seed": seed,
                    "policy": policy,
                    "budget": budget,
                    "n_systems": len(result.records),
                    "frac_converged": result.frac_converged,
                    "mean_frac_resolved": result.mean_frac_resolved,
                    "mean_steps": result.mean_steps,
                    "mean_distractors_selected": result.mean_distractors_selected,
                    "false_exclusion_rate": result.false_exclusion_rate,
                })

    metrics = (
        "frac_converged",
        "mean_frac_resolved",
        "mean_steps",
        "mean_distractors_selected",
        "false_exclusion_rate",
    )
    aggregate: list[dict] = []
    for policy in policies:
        for budget in budgets:
            rows = [row for row in per_seed if row["policy"] == policy and row["budget"] == budget]
            summary = {
                "policy": policy,
                "budget": budget,
                "n_seeds": len(rows),
                "total_systems": sum(row["n_systems"] for row in rows),
            }
            for metric in metrics:
                values = [float(row[metric]) for row in rows]
                summary[f"{metric}_mean"] = statistics.mean(values)
                summary[f"{metric}_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
            aggregate.append(summary)

    return {
        "status": "post_frozen_claim_ceiling_diagnostic",
        "preregistered_g2_modified": False,
        "policies": list(policies),
        "seeds": list(seeds),
        "budgets": list(budgets),
        "n_systems_per_seed_requested": n_systems_per_seed,
        "per_seed": per_seed,
        "aggregate": aggregate,
    }


def run_static_information_diagnostic_from_frozen_protocol(
    protocol_path: str | Path,
    *,
    budgets: Iterable[int] = (2, 4),
) -> dict:
    """Run the claim-ceiling diagnostic with the generator settings frozen for G2."""
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    sweep = protocol["sweep"]
    return run_static_information_diagnostic(
        seeds=sweep["seeds"],
        budgets=budgets,
        n_systems_per_seed=int(sweep["n_systems_per_seed"]),
        n_attempts=int(sweep["n_attempts"]),
        K_choices=tuple(int(value) for value in sweep["K_choices"]),
        confound_choices=tuple(int(value) for value in sweep["confound_choices"]),
        min_sub_size=int(sweep["min_sub_size"]),
        n_distractors=int(protocol["generator"]["distractor_candidates"]["count"]),
    )


def print_report(result: SweepResult) -> None:
    """Print the controlled benchmark summary with descriptive policy labels."""
    print("=" * 76)
    print(
        "Mechanism-Resolving Observation Design controlled selection benchmark "
        f"— policy={display_policy_name(result.policy)}"
    )
    print("=" * 76)
    print(f"systems run              : {len(result.records)} / {result.n_systems}")
    print(f"systems with >=1 edge    : {result.systems_with_edges}")
    if not result.records:
        print("no systems produced a usable admissible region")
        return
    print(f"fully converged          : {result.frac_converged * 100:.1f}%")
    print(f"edges resolved (mean)    : {result.mean_frac_resolved * 100:.1f}%")
    print(f"resolvability R          : {result.mean_R0:.3f} -> {result.mean_R_final:.3f}")
    print(f"observations taken       : {result.mean_steps:.2f}")
    print(f"nuisance selections      : {result.mean_distractors_selected:.2f}")
    print(f"false exclusion rate     : {result.false_exclusion_rate * 100:.2f}%")


def print_budget_table(summaries) -> None:
    """Print budget summaries with the historical guided key translated for display."""
    print("policy                 budget  converged  resolved  steps  nuisance  false_exclusion")
    for summary in summaries:
        print(
            f"{display_policy_name(summary.policy):22s} {summary.budget:>6d}  "
            f"{summary.frac_converged:>9.3f}  {summary.mean_frac_resolved:>8.3f}  "
            f"{summary.mean_steps:>5.2f}  {summary.mean_distractors_selected:>8.2f}  "
            f"{summary.false_exclusion_rate:>15.3f}"
        )


def __getattr__(name: str):
    """Delegate non-public historical helpers needed by frozen support code."""
    return getattr(_impl, name)


if __name__ == "__main__":  # pragma: no cover - historical CLI compatibility
    _impl.main()
