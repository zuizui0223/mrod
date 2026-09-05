"""Truth-peek-free controlled benchmark for RACH-SEQ observation selection.

The benchmark separates two questions that were conflated in the earlier pilot:

1. can an explicit resolving observation reduce a known confounding structure?;
2. can RACH-SEQ *select* that observation efficiently when equally available
   mechanism-uninformative distractors compete for a limited budget?

Every random system contains one or two disjoint two-driver confounds. A
quantitative magnitude observation is available for each confound and separates
A-only, B-only, and both-on states. The frozen G2 protocol additionally includes
binary nuisance measurements generated independently of the mechanism vector.
Those nuisance outcomes are valid, fully observed predictive partitions but
carry no designed mechanism information.

Two policies can be evaluated on exactly the same generated systems and hidden
truths:

``rach_seq``
    greedily selects by expected confounding-edge cuts, with predictive outcome
    probabilities recomputed from current ``A_epsilon``;

``random_order``
    selects one remaining candidate uniformly at random before hidden truth is
    materialised. It is a no-information selection baseline, not a straw-man
    model-selection method.

Hidden truth is used only *after* either policy has selected a candidate, solely
to materialise the realised benchmark outcome. No favourable performance
threshold is encoded here; policy differences are outputs of the frozen run.
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence

from causal_model.causal_admissibility import (
    CandidateObservation,
    CandidateOutcome,
    causal_resolvability,
)
from causal_model.mechanism_equivalence import mechanism_equivalence_structure
from causal_model.rach_seq import (
    filter_by_outcome,
    predictive_outcome_distribution,
    rach_seq,
)


Policy = Literal["rach_seq", "random_order"]
_VALID_POLICIES: tuple[str, ...] = ("rach_seq", "random_order")

# Compatibility defaults for direct helper calls. The publication sweep samples
# a fresh pair per system. With theta in [0.8, 1.2], non-overlapping magnitude
# bands require 1.5 < coeff_b / coeff_a < 2.0.
_DEFAULT_DRIVER_COEFFS = (0.35, 0.60)
_THETA_LO, _THETA_HI = 0.8, 1.2
_SLOPE_TOL = 0.05


class _SW:
    """Minimal switch object; the inference layer needs only ``.name``."""

    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = name


@dataclass
class SystemRecord:
    """One policy outcome for one generated mechanism system."""

    K: int
    n_confounds: int
    n_initial_edges: int
    n_resolved: int
    n_unresolved: int
    converged: bool
    steps_taken: int
    R0: float
    R_final: float
    truth_retained: bool = True
    truth_peek_free: bool = True
    driver_coeff_a: float = float("nan")
    driver_coeff_b: float = float("nan")
    policy: str = "rach_seq"
    n_distractors: int = 0
    distractors_selected: int = 0

    @property
    def frac_resolved(self) -> float:
        if self.n_initial_edges == 0:
            return 1.0
        return self.n_resolved / self.n_initial_edges


@dataclass
class SweepResult:
    n_systems: int
    policy: str = "rach_seq"
    records: list[SystemRecord] = field(default_factory=list)
    frac_converged: float = float("nan")
    mean_frac_resolved: float = float("nan")
    median_frac_resolved: float = float("nan")
    mean_R0: float = float("nan")
    mean_R_final: float = float("nan")
    mean_steps: float = float("nan")
    mean_distractors_selected: float = float("nan")
    systems_with_edges: int = 0
    false_exclusion_rate: float = float("nan")


@dataclass(frozen=True)
class BudgetSummary:
    policy: str
    budget: int
    n_systems: int
    systems_with_edges: int
    frac_converged: float
    mean_frac_resolved: float
    mean_steps: float
    false_exclusion_rate: float
    mean_distractors_selected: float = 0.0


@dataclass
class _SequenceOutcome:
    final_rows: list[dict]
    n_resolved: int
    n_unresolved: int
    converged: bool
    steps_taken: int
    R_final: float
    observations_taken: list[str]


def _sample_driver_coefficients(rng: random.Random) -> tuple[float, float]:
    """Sample a discriminable driver pair before data/truth are inspected."""
    a = rng.uniform(0.28, 0.42)
    ratio = rng.uniform(1.60, 1.85)
    b = a * ratio
    return round(a, 6), round(b, 6)


def _make_random_system(rng: random.Random, K: int, n_confounds: int):
    """Build one random confounded system with one hidden driver per confound."""
    names = [f"s{i}" for i in range(K)]
    switches = [_SW(n) for n in names]
    pool = names[:]
    rng.shuffle(pool)
    drivers_per_trait: list[tuple[str, str]] = []
    for t in range(n_confounds):
        drivers_per_trait.append((pool[2 * t], pool[2 * t + 1]))
    truth_driver = [rng.choice(pair) for pair in drivers_per_trait]
    return switches, drivers_per_trait, truth_driver


def _abc_accept(
    rng: random.Random,
    switches,
    drivers_per_trait,
    n_attempts: int,
    driver_coeffs: tuple[float, float] = _DEFAULT_DRIVER_COEFFS,
    *,
    n_distractors: int = 0,
):
    """Generate ``A_epsilon`` from ordinal-positive observations only.

    Distractor columns are binary nuisance measurements sampled independently of
    the mechanism state after a row has passed the ordinal acceptance rule. They
    are observed outputs, not switches in ``S``.
    """
    if n_distractors < 0:
        raise ValueError("n_distractors must be non-negative")
    coeff_a, coeff_b = driver_coeffs
    names = [sw.name for sw in switches]
    accepted: list[dict] = []
    for _ in range(n_attempts):
        state = {name: (rng.random() < 0.5) for name in names}
        theta = rng.uniform(_THETA_LO, _THETA_HI)
        magnitudes: list[float] = []
        ok = True
        for a, b in drivers_per_trait:
            slope = theta * (
                coeff_a * int(state[a])
                + coeff_b * int(state[b])
            )
            magnitudes.append(slope)
            if slope <= _SLOPE_TOL:
                ok = False
                break
        if not ok:
            continue

        row = dict(state)
        row["theta"] = theta
        # Raw values are retained so analytic outcome bands remain exact
        # predictive partitions rather than being damaged by display rounding.
        for t, magnitude in enumerate(magnitudes):
            row[f"trait{t}_mag"] = magnitude
        for d in range(n_distractors):
            row[f"decoy{d}_marker"] = 1.0 if rng.random() < 0.5 else 0.0
        accepted.append(row)
    return accepted


def _truth_magnitude(
    driver_name: str,
    pair: tuple[str, str],
    driver_coeffs: tuple[float, float] = _DEFAULT_DRIVER_COEFFS,
) -> float:
    """Magnitude at theta=1 when exactly the hidden true driver is on."""
    return driver_coeffs[0] if driver_name == pair[0] else driver_coeffs[1]


def _mode_name(a_on: bool, b_on: bool) -> str:
    if a_on and not b_on:
        return "driver_a_only"
    if b_on and not a_on:
        return "driver_b_only"
    if a_on and b_on:
        return "both_on"
    return "neither_on"


def _mode_coefficient(mode: str, driver_coeffs: tuple[float, float]) -> float:
    coeff_a, coeff_b = driver_coeffs
    if mode == "driver_a_only":
        return coeff_a
    if mode == "driver_b_only":
        return coeff_b
    if mode == "both_on":
        return coeff_a + coeff_b
    raise ValueError(f"unsupported mode: {mode}")


def _absolute_band_pattern(
    trait_index: int,
    mode: str,
    driver_coeffs: tuple[float, float],
) -> dict:
    """Return an absolute-summary pattern covering one full theta-scaled band."""
    coeff = _mode_coefficient(mode, driver_coeffs)
    lower = coeff * _THETA_LO
    upper = coeff * _THETA_HI
    centre = (lower + upper) / 2.0
    # rach_seq interprets absolute_summary as |sim-observed| <= 2*scale.
    scale = (upper - lower) / 4.0 + 1e-6
    return {
        "type": "absolute_summary",
        "variable": "mag",
        "population": f"trait{trait_index}",
        "observed_value": f"{centre:.12f}",
        "scale": f"{scale:.12f}",
    }


def _binary_marker_pattern(decoy_index: int, value: int) -> dict:
    return {
        "type": "absolute_summary",
        "variable": "marker",
        "population": f"decoy{decoy_index}",
        "observed_value": str(float(value)),
        "scale": "0.1",
    }


def _resolving_candidates(
    drivers_per_trait,
    accepted_rows: list[dict],
    driver_coeffs: tuple[float, float],
) -> list[CandidateObservation]:
    candidates: list[CandidateObservation] = []
    for t, pair in enumerate(drivers_per_trait):
        a, b = pair
        counts = {"driver_a_only": 0, "driver_b_only": 0, "both_on": 0}
        for row in accepted_rows:
            mode = _mode_name(bool(row.get(a)), bool(row.get(b)))
            if mode in counts:
                counts[mode] += 1
        total = sum(counts.values())
        if total == 0:
            continue
        outcomes = [
            CandidateOutcome(
                name=mode,
                description=f"Trait {t} magnitude falls in the {mode} band.",
                prior_probability=counts[mode] / total,
                extra_pattern_rows=[_absolute_band_pattern(t, mode, driver_coeffs)],
            )
            for mode in ("driver_a_only", "driver_b_only", "both_on")
            if counts[mode] > 0
        ]
        candidates.append(CandidateObservation(
            name=f"measure_trait{t}_magnitude",
            description=f"Measure the quantitative magnitude of trait {t}.",
            target_switches=list(pair),
            rationale=(
                f"The current admissible-region outcome distribution separates which "
                f"member of {pair} drives the ordinal-positive trait."
            ),
            pattern_type="absolute_summary",
            outcomes=outcomes,
        ))
    return candidates


def _distractor_candidates(
    accepted_rows: list[dict],
    n_distractors: int,
) -> list[CandidateObservation]:
    """Return valid predictive measurements designed to be mechanism-uninformative."""
    candidates: list[CandidateObservation] = []
    for d in range(n_distractors):
        n0 = sum(float(row[f"decoy{d}_marker"]) == 0.0 for row in accepted_rows)
        n1 = len(accepted_rows) - n0
        if n0 == 0 or n1 == 0:
            # Extremely unlikely at benchmark sample sizes; omit rather than
            # inventing a missing outcome probability.
            continue
        candidates.append(CandidateObservation(
            name=f"measure_decoy{d}_marker",
            description=(
                f"Measure synthetic nuisance marker {d}, generated independently of "
                "the mechanism vector."
            ),
            target_switches=[],
            rationale=(
                "Negative-control candidate for observation-selection efficiency; "
                "its predictive outcome map is valid but has no designed causal link."
            ),
            pattern_type="absolute_summary",
            outcomes=[
                CandidateOutcome(
                    name="marker_0",
                    description="Nuisance marker equals zero.",
                    prior_probability=n0 / len(accepted_rows),
                    extra_pattern_rows=[_binary_marker_pattern(d, 0)],
                ),
                CandidateOutcome(
                    name="marker_1",
                    description="Nuisance marker equals one.",
                    prior_probability=n1 / len(accepted_rows),
                    extra_pattern_rows=[_binary_marker_pattern(d, 1)],
                ),
            ],
        ))
    return candidates


def _candidates_for_system(
    drivers_per_trait,
    accepted_rows: list[dict],
    driver_coeffs: tuple[float, float] = _DEFAULT_DRIVER_COEFFS,
    *,
    n_distractors: int = 0,
):
    """Construct candidate distributions from current ``A_epsilon``, never truth."""
    if not accepted_rows:
        return []
    return [
        *_resolving_candidates(drivers_per_trait, accepted_rows, driver_coeffs),
        *_distractor_candidates(accepted_rows, n_distractors),
    ]


def _truth_outcome_overrides(
    drivers_per_trait,
    truth_driver,
    distractor_truth: Sequence[bool] = (),
) -> dict[str, str]:
    """Materialise hidden outcomes only after a policy selects a candidate."""
    overrides: dict[str, str] = {}
    for t, (pair, true_driver) in enumerate(zip(drivers_per_trait, truth_driver)):
        overrides[f"measure_trait{t}_magnitude"] = (
            "driver_a_only" if true_driver == pair[0] else "driver_b_only"
        )
    for d, value in enumerate(distractor_truth):
        overrides[f"measure_decoy{d}_marker"] = "marker_1" if value else "marker_0"
    return overrides


def _outcome_by_name(candidate: CandidateObservation, name: str) -> CandidateOutcome:
    matches = [outcome for outcome in candidate.outcomes if outcome.name == name]
    if not matches:
        raise ValueError(f"unknown outcome {name!r} for candidate {candidate.name!r}")
    return matches[0]


def _replay_final_rows(accepted_rows, candidates, observations) -> list[dict]:
    rows = list(accepted_rows)
    by_name = {candidate.name: candidate for candidate in candidates}
    for candidate_name, outcome_name in observations:
        candidate = by_name[candidate_name]
        outcome = _outcome_by_name(candidate, outcome_name)
        rows = filter_by_outcome(rows, outcome.extra_pattern_rows)
    return rows


def _truth_retained(rows, drivers_per_trait, truth_driver) -> bool:
    """Whether a surviving row contains every hidden one-driver mechanism mode."""
    for row in rows:
        ok = True
        for pair, true_driver in zip(drivers_per_trait, truth_driver):
            other = pair[1] if true_driver == pair[0] else pair[0]
            if not bool(row.get(true_driver)) or bool(row.get(other)):
                ok = False
                break
        if ok:
            return True
    return False


def _random_outcome_if_unoverridden(
    candidate: CandidateObservation,
    current_rows: list[dict],
    rng: random.Random,
) -> str:
    distribution = predictive_outcome_distribution(candidate, current_rows)
    probabilities = distribution.probabilities
    draw = rng.random()
    cumulative = 0.0
    chosen = candidate.outcomes[-1].name
    for outcome in candidate.outcomes:
        cumulative += probabilities.get(outcome.name, 0.0)
        if draw <= cumulative:
            chosen = outcome.name
            break
    return chosen


def _run_random_order(
    accepted_rows: list[dict],
    switches,
    candidates: list[CandidateObservation],
    *,
    budget: int,
    min_sub_size: int,
    seed: int,
    outcome_overrides: dict[str, str],
) -> _SequenceOutcome:
    """Uniform-random candidate selection with truth materialised after selection."""
    rng = random.Random(seed)
    current_rows = list(accepted_rows)
    initial_structure = mechanism_equivalence_structure(current_rows, switches)
    current_structure = initial_structure
    used: set[str] = set()
    observations: list[tuple[str, str]] = []

    for _ in range(budget):
        if not current_structure.edges:
            break
        available = [candidate for candidate in candidates if candidate.name not in used]
        if not available:
            break
        candidate = rng.choice(available)
        used.add(candidate.name)
        outcome_name = outcome_overrides.get(candidate.name)
        if outcome_name is None:
            outcome_name = _random_outcome_if_unoverridden(candidate, current_rows, rng)
        outcome = _outcome_by_name(candidate, outcome_name)
        filtered = filter_by_outcome(current_rows, outcome.extra_pattern_rows)
        observations.append((candidate.name, outcome_name))
        if len(filtered) < min_sub_size:
            # The observation was taken; an over-restrictive outcome is retained as
            # an error-control event rather than silently resampled.
            current_rows = filtered
            current_structure = mechanism_equivalence_structure(current_rows, switches)
            break
        current_rows = filtered
        current_structure = mechanism_equivalence_structure(current_rows, switches)

    initial_ids = {(edge.a, edge.b) for edge in initial_structure.edges}
    final_ids = {(edge.a, edge.b) for edge in current_structure.edges}
    n_resolved = len(initial_ids - final_ids)
    final_R = causal_resolvability(current_rows, switches) if current_rows else float("nan")
    return _SequenceOutcome(
        final_rows=current_rows,
        n_resolved=n_resolved,
        n_unresolved=len(current_structure.edges),
        converged=not bool(current_structure.edges),
        steps_taken=len(observations),
        R_final=final_R,
        observations_taken=[name for name, _ in observations],
    )


def _run_rach_policy(
    accepted_rows: list[dict],
    switches,
    candidates: list[CandidateObservation],
    *,
    budget: int,
    min_sub_size: int,
    seed: int,
    outcome_overrides: dict[str, str],
) -> _SequenceOutcome:
    seq = rach_seq(
        accepted_rows,
        switches,
        candidates,
        budget=budget,
        min_sub_size=min_sub_size,
        seed=seed,
        outcome_overrides=outcome_overrides,
    )
    observations = [
        (step.observation_taken, step.outcome_observed)
        for step in seq.steps[1:]
        if step.observation_taken and step.outcome_observed
    ]
    final_rows = _replay_final_rows(accepted_rows, candidates, observations)
    return _SequenceOutcome(
        final_rows=final_rows,
        n_resolved=len(seq.edges_resolved),
        n_unresolved=len(seq.edges_unresolved),
        converged=seq.converged,
        steps_taken=len(seq.observations_taken),
        R_final=seq.steps[-1].R,
        observations_taken=list(seq.observations_taken),
    )


def run_generality_sweep(
    n_systems: int = 200,
    seed: int = 0,
    *,
    n_attempts: int = 1500,
    K_choices: tuple[int, ...] = (4, 5, 6),
    confound_choices: tuple[int, ...] = (1, 2),
    budget: int = 4,
    min_sub_size: int = 8,
    n_distractors: int = 0,
    policy: Policy = "rach_seq",
) -> SweepResult:
    """Run one selection policy over a deterministic seed-defined system family."""
    if policy not in _VALID_POLICIES:
        raise ValueError(f"unknown policy: {policy!r}")
    if budget < 0:
        raise ValueError("budget must be non-negative")
    master = random.Random(seed)
    result = SweepResult(n_systems=n_systems, policy=policy)

    for _ in range(n_systems):
        sys_rng = random.Random(master.randrange(1 << 30))
        K = sys_rng.choice(K_choices)
        n_confounds = min(sys_rng.choice(confound_choices), K // 2)
        switches, drivers, truth_driver = _make_random_system(sys_rng, K, n_confounds)
        driver_coeffs = _sample_driver_coefficients(sys_rng)
        # Hidden distractor truths are sampled before candidate construction but
        # are not supplied to either selection policy until after candidate choice.
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

        initial = mechanism_equivalence_structure(accepted, switches)
        R0 = causal_resolvability(accepted, switches)
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
        sequence_seed = sys_rng.randrange(1 << 30)

        if policy == "rach_seq":
            outcome = _run_rach_policy(
                accepted,
                switches,
                candidates,
                budget=budget,
                min_sub_size=min_sub_size,
                seed=sequence_seed,
                outcome_overrides=overrides,
            )
        else:
            outcome = _run_random_order(
                accepted,
                switches,
                candidates,
                budget=budget,
                min_sub_size=min_sub_size,
                seed=sequence_seed,
                outcome_overrides=overrides,
            )

        distractors_selected = sum(
            name.startswith("measure_decoy") for name in outcome.observations_taken
        )
        result.records.append(SystemRecord(
            K=K,
            n_confounds=n_confounds,
            n_initial_edges=len(initial.edges),
            n_resolved=outcome.n_resolved,
            n_unresolved=outcome.n_unresolved,
            converged=outcome.converged,
            steps_taken=outcome.steps_taken,
            R0=round(R0, 4),
            R_final=(round(outcome.R_final, 4) if math.isfinite(outcome.R_final) else outcome.R_final),
            truth_retained=_truth_retained(outcome.final_rows, drivers, truth_driver),
            truth_peek_free=True,
            driver_coeff_a=driver_coeffs[0],
            driver_coeff_b=driver_coeffs[1],
            policy=policy,
            n_distractors=n_distractors,
            distractors_selected=distractors_selected,
        ))

    _summarize(result)
    return result


def _summarize(result: SweepResult) -> None:
    records = result.records
    if not records:
        return
    with_edges = [record for record in records if record.n_initial_edges > 0]
    result.systems_with_edges = len(with_edges)
    base = with_edges or records
    result.frac_converged = sum(record.converged for record in base) / len(base)
    result.mean_frac_resolved = statistics.mean(record.frac_resolved for record in base)
    result.median_frac_resolved = statistics.median(record.frac_resolved for record in base)
    result.mean_R0 = statistics.mean(record.R0 for record in records)
    finite_rfinal = [record.R_final for record in records if math.isfinite(record.R_final)]
    result.mean_R_final = statistics.mean(finite_rfinal) if finite_rfinal else float("nan")
    result.mean_steps = statistics.mean(record.steps_taken for record in base)
    result.mean_distractors_selected = statistics.mean(
        record.distractors_selected for record in base
    )
    result.false_exclusion_rate = (
        sum(not record.truth_retained for record in records) / len(records)
    )


def run_budget_sweep(
    budgets: Sequence[int] = (0, 1, 2, 3, 4),
    *,
    n_systems: int = 200,
    seed: int = 0,
    n_attempts: int = 1500,
    K_choices: tuple[int, ...] = (4, 5, 6),
    confound_choices: tuple[int, ...] = (1, 2),
    min_sub_size: int = 8,
    n_distractors: int = 0,
    policies: Sequence[str] = ("rach_seq",),
) -> list[BudgetSummary]:
    """Evaluate fixed policies on the same seed-defined systems at each budget."""
    summaries: list[BudgetSummary] = []
    for policy in policies:
        if policy not in _VALID_POLICIES:
            raise ValueError(f"unknown policy: {policy!r}")
        for budget in budgets:
            if budget < 0:
                raise ValueError("budgets must be non-negative")
            result = run_generality_sweep(
                n_systems=n_systems,
                seed=seed,
                n_attempts=n_attempts,
                K_choices=K_choices,
                confound_choices=confound_choices,
                budget=int(budget),
                min_sub_size=min_sub_size,
                n_distractors=n_distractors,
                policy=policy,  # type: ignore[arg-type]
            )
            summaries.append(BudgetSummary(
                policy=policy,
                budget=int(budget),
                n_systems=len(result.records),
                systems_with_edges=result.systems_with_edges,
                frac_converged=result.frac_converged,
                mean_frac_resolved=result.mean_frac_resolved,
                mean_steps=result.mean_steps,
                false_exclusion_rate=result.false_exclusion_rate,
                mean_distractors_selected=result.mean_distractors_selected,
            ))
    return summaries


def save_budget_table(summaries: Sequence[BudgetSummary], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "policy",
        "budget",
        "n_systems",
        "systems_with_edges",
        "frac_converged",
        "mean_frac_resolved",
        "mean_steps",
        "false_exclusion_rate",
        "mean_distractors_selected",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({field: getattr(summary, field) for field in fields})
    return output


def print_report(result: SweepResult) -> None:
    print("=" * 76)
    print(f"RACH-SEQ controlled selection benchmark — policy={result.policy}")
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
    print(f"distractors selected     : {result.mean_distractors_selected:.2f}")
    print(f"false exclusion rate     : {result.false_exclusion_rate * 100:.2f}%")


def print_budget_table(summaries: Sequence[BudgetSummary]) -> None:
    print("policy        budget  converged  resolved  steps  decoys  false_exclusion")
    for summary in summaries:
        print(
            f"{summary.policy:12s} {summary.budget:>6d}  "
            f"{summary.frac_converged:>9.3f}  {summary.mean_frac_resolved:>8.3f}  "
            f"{summary.mean_steps:>5.2f}  {summary.mean_distractors_selected:>6.2f}  "
            f"{summary.false_exclusion_rate:>15.3f}"
        )


def make_figure(result: SweepResult, path: str) -> str | None:
    """Write a compact diagnostic without changing benchmark logic."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib unavailable — skipping figure")
        return None

    records = [record for record in result.records if record.n_initial_edges > 0]
    if not records:
        return None
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.scatter([r.R0 for r in records], [r.R_final for r in records], s=16, alpha=0.5)
    finite = [r.R_final for r in records if math.isfinite(r.R_final)]
    lim = max([r.R0 for r in records] + finite + [0.1]) * 1.05
    ax.plot([0, lim], [0, lim], "--", linewidth=1)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("R before")
    ax.set_ylabel("R after sequence")
    ax.set_title(f"Truth-peek-free sequence: {result.policy}")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return str(output)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Truth-peek-free RACH-SEQ controlled selection benchmark."
    )
    parser.add_argument("--n-systems", type=int, default=200)
    parser.add_argument("--n-attempts", type=int, default=1500)
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-distractors", type=int, default=0)
    parser.add_argument("--policy", choices=_VALID_POLICIES, default="rach_seq")
    parser.add_argument("--figure", default="")
    parser.add_argument(
        "--budget-sweep",
        default="",
        help="Comma-separated budgets; when supplied, print a policy/budget table.",
    )
    parser.add_argument(
        "--policies",
        default="rach_seq",
        help="Comma-separated policies for --budget-sweep (rach_seq,random_order).",
    )
    parser.add_argument("--budget-table", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.budget_sweep:
        budgets = [int(value) for value in args.budget_sweep.split(",") if value.strip()]
        policies = [value.strip() for value in args.policies.split(",") if value.strip()]
        summaries = run_budget_sweep(
            budgets,
            n_systems=args.n_systems,
            seed=args.seed,
            n_attempts=args.n_attempts,
            n_distractors=args.n_distractors,
            policies=policies,
        )
        print_budget_table(summaries)
        if args.budget_table:
            print(f"budget table written: {save_budget_table(summaries, args.budget_table)}")
        return 0

    result = run_generality_sweep(
        n_systems=args.n_systems,
        seed=args.seed,
        n_attempts=args.n_attempts,
        budget=args.budget,
        n_distractors=args.n_distractors,
        policy=args.policy,
    )
    print_report(result)
    if args.figure:
        output = make_figure(result, args.figure)
        if output:
            print(f"figure written: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BudgetSummary",
    "SweepResult",
    "SystemRecord",
    "_abc_accept",
    "_candidates_for_system",
    "_make_random_system",
    "_sample_driver_coefficients",
    "_truth_magnitude",
    "main",
    "make_figure",
    "print_budget_table",
    "print_report",
    "run_budget_sweep",
    "run_generality_sweep",
    "save_budget_table",
]
