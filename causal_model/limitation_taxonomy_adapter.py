"""Internal adapter from MROD diagnostic receipts to the shared action taxonomy.

The canonical state/action vocabulary lives in ``docs/limitation_action_taxonomy_v1.json``
and is byte-identical to the Boundary copy.  This module does not define a new
score, priority order, optimizer or public API.  It projects already-computed
MROD diagnostics onto canonical taxonomy coordinates while retaining any more
detailed implementation actions separately.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

from .adaptive_cost_scenario_tradeoff import ScenarioCostTradeoffAudit
from .adaptive_expected_cost_audit import AdaptiveExpectedCostAudit
from .limitation_action_report import LimitationActionReport, SpecificationStatus
from .replication_switch_audit import ReplicationSwitchAudit


_TAXONOMY = Path(__file__).resolve().parents[1] / "docs" / "limitation_action_taxonomy_v1.json"


@dataclass(frozen=True)
class TaxonomySignal:
    axis: str
    state: str
    action: str
    source: str


@dataclass(frozen=True)
class TaxonomyProjection:
    signals: tuple[TaxonomySignal, ...]
    detail_actions: tuple[str, ...]
    sources: tuple[str, ...]

    @property
    def canonical_actions(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(signal.action for signal in self.signals))

    @property
    def canonical_states(self) -> tuple[tuple[str, str], ...]:
        return tuple((signal.axis, signal.state) for signal in self.signals)


def _registry() -> dict[tuple[str, str], str]:
    data = json.loads(_TAXONOMY.read_text(encoding="utf-8"))
    if data.get("schema") != "limitation_action_taxonomy" or data.get("version") != "1.0":
        raise ValueError("unsupported limitation-action taxonomy contract")
    registry: dict[tuple[str, str], str] = {}
    for axis in data["axes"]:
        axis_id = axis["id"]
        for item in axis["states"]:
            key = (axis_id, item["state"])
            if key in registry:
                raise ValueError(f"duplicate taxonomy state {key!r}")
            registry[key] = item["action"]
    return registry


def _signal(axis: str, state: str, source: str) -> TaxonomySignal:
    registry = _registry()
    try:
        action = registry[(axis, state)]
    except KeyError as exc:
        raise ValueError(f"state {(axis, state)!r} is absent from the canonical taxonomy") from exc
    return TaxonomySignal(axis, state, action, source)


def _projection(
    signals: Iterable[TaxonomySignal], *,
    detail_actions: Iterable[str] = (),
    sources: Iterable[str] = (),
) -> TaxonomyProjection:
    unique: dict[tuple[str, str, str], TaxonomySignal] = {}
    for signal in signals:
        unique.setdefault((signal.axis, signal.state, signal.action), signal)
    return TaxonomyProjection(
        tuple(unique.values()),
        tuple(dict.fromkeys(str(action) for action in detail_actions)),
        tuple(dict.fromkeys(str(source) for source in sources)),
    )


def project_specification_status(status: SpecificationStatus) -> TaxonomyProjection:
    """Project one core limitation-report status onto canonical taxonomy states."""
    source = f"limitation_action_report:{status.name}"
    signals: list[TaxonomySignal] = []

    if status.current_state_status == "nonestimable":
        signals.append(_signal("current_state_estimability", "nonestimable", source))
    elif status.current_state_status == "estimable":
        signals.append(_signal("current_state_estimability", "estimable", source))
    else:
        raise ValueError(f"unknown current_state_status {status.current_state_status!r}")

    if status.full_mechanism_status == "fully_resolved":
        signals.append(_signal("target_identification", "fully_resolved", source))
    elif status.declared_target_status == "resolved":
        signals.append(
            _signal("target_identification", "target_identified_full_mechanism_ambiguous", source)
        )
    elif status.declared_target_status == "unresolved":
        signals.append(_signal("target_identification", "target_unresolved", source))
    elif status.declared_target_status != "nonestimable":
        raise ValueError(f"unknown declared_target_status {status.declared_target_status!r}")

    coverage_map = {
        "none_declared": "none_declared",
        "none_estimable": "none_estimable",
        "partial": "partial",
        "complete": "complete",
    }
    if status.candidate_coverage in coverage_map:
        signals.append(
            _signal(
                "candidate_prediction_coverage",
                coverage_map[status.candidate_coverage],
                source,
            )
        )

    if status.candidate_coverage == "complete":
        if status.validated_information_status == "positive":
            signals.append(_signal("information_horizon", "positive_singleton", source))
        elif status.validated_information_status == "zero":
            joint_map = {
                "not_audited": "zero_singleton_joint_not_audited",
                "positive": "zero_singleton_joint_positive",
                "zero": "joint_zero",
            }
            if status.joint_information_status not in joint_map:
                raise ValueError(
                    f"zero singleton status has invalid joint state {status.joint_information_status!r}"
                )
            signals.append(
                _signal("information_horizon", joint_map[status.joint_information_status], source)
            )

    if (
        status.budget_status == "exhausted"
        and status.recommendation_status == "validated_best"
        and status.best_candidates
    ):
        signals.append(
            _signal("resource_operational", "budget_exhausted_best_known", source)
        )

    return _projection(
        signals,
        detail_actions=status.recommended_actions,
        sources=(source,),
    )


def project_limitation_action_report(report: LimitationActionReport) -> TaxonomyProjection:
    """Compose per-specification projections and any licensed robustness state."""
    pieces = [project_specification_status(status) for status in report.per_specification]
    signals = [signal for piece in pieces for signal in piece.signals]
    details = [action for piece in pieces for action in piece.detail_actions]
    sources = [source for piece in pieces for source in piece.sources]

    conclusion_stable = report.target_resolution_stability.startswith("stable_target_")
    conclusion_sensitive = report.target_resolution_stability.startswith("specification_sensitive")
    recommendation_stable = report.recommendation_stability == "stable_common_best"
    recommendation_sensitive = report.recommendation_stability.startswith("specification_sensitive")

    robustness_state = None
    if conclusion_stable and recommendation_stable:
        robustness_state = "conclusion_stable_recommendation_stable"
    elif conclusion_stable and recommendation_sensitive:
        robustness_state = "conclusion_stable_recommendation_sensitive"
    elif conclusion_sensitive and recommendation_stable:
        robustness_state = "conclusion_sensitive_recommendation_stable"
    elif conclusion_sensitive and recommendation_sensitive:
        robustness_state = "conclusion_sensitive_recommendation_sensitive"
    if robustness_state is not None:
        signals.append(
            _signal("specification_robustness", robustness_state, "limitation_action_report:specifications")
        )
        sources.append("limitation_action_report:specifications")

    return _projection(signals, detail_actions=details, sources=sources)


def project_replication_switch(audit: ReplicationSwitchAudit) -> TaxonomyProjection:
    """Project repeat-versus-switch diagnostics without inventing a sequence policy."""
    source = "replication_switch_audit"
    signals: list[TaxonomySignal] = []
    tol = audit.information_tolerance_bits
    if (
        audit.repeat_estimable
        and audit.one_repeat_information_bits is not None
        and audit.one_repeat_information_bits > tol
    ):
        signals.append(_signal("information_horizon", "replication_still_informative", source))
    if audit.ceiling_dominant_alternatives:
        signals.append(_signal("information_horizon", "alternative_beats_repeat_ceiling", source))
    return _projection(signals, sources=(source,))


def project_adaptive_expected_cost(
    audit: AdaptiveExpectedCostAudit,
    *,
    tolerance: float = 1e-12,
) -> TaxonomyProjection:
    """Project information/cost consequences of an already-selected adaptive tree."""
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    source = "adaptive_expected_cost_audit"
    signals: list[TaxonomySignal] = []
    if any(
        row.adaptive_information_gain_over_fixed_oracle_bits > tolerance
        for row in audit.scenario_results
    ):
        signals.append(_signal("resource_operational", "adaptive_information_advantage", source))
    if any(
        row.expected_cost_saving_vs_information_matched_fixed is not None
        and row.expected_cost_saving_vs_information_matched_fixed > tolerance
        for row in audit.scenario_results
    ):
        signals.append(_signal("resource_operational", "expected_cost_saving", source))
    return _projection(signals, sources=(source,))


def project_scenario_cost_tradeoff(
    audit: ScenarioCostTradeoffAudit,
) -> TaxonomyProjection:
    """Project equal-information cost ties and cross-scenario preference reversals."""
    source = "adaptive_cost_scenario_tradeoff"
    signals: list[TaxonomySignal] = []
    tol = audit.comparison_tolerance
    costs_differ = any(
        max(vector.expected_cost_by_scenario[name] for vector in audit.policy_vectors)
        - min(vector.expected_cost_by_scenario[name] for vector in audit.policy_vectors)
        > tol
        for name in audit.scenario_names
    )
    if audit.information_equivalent_by_scenario and audit.same_worst_path_cost and costs_differ:
        signals.append(
            _signal("resource_operational", "equal_information_cost_tie_sensitive", source)
        )
    if audit.pairwise_cost_preference_crosses:
        signals.append(
            _signal("resource_operational", "cost_preference_crosses_scenarios", source)
        )
    return _projection(signals, sources=(source,))


def combine_taxonomy_projections(*projections: TaxonomyProjection) -> TaxonomyProjection:
    """Union orthogonal signals without imposing a priority ordering."""
    return _projection(
        (signal for projection in projections for signal in projection.signals),
        detail_actions=(
            action for projection in projections for action in projection.detail_actions
        ),
        sources=(source for projection in projections for source in projection.sources),
    )


__all__ = [
    "TaxonomySignal",
    "TaxonomyProjection",
    "project_specification_status",
    "project_limitation_action_report",
    "project_replication_switch",
    "project_adaptive_expected_cost",
    "project_scenario_cost_tradeoff",
    "combine_taxonomy_projections",
]
