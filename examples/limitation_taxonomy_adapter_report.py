"""Synthetic projections of existing MROD diagnostics into the shared taxonomy.

Run from repository root:
    python -m examples.limitation_taxonomy_adapter_report
"""
from __future__ import annotations

from dataclasses import asdict
import json

from causal_model.adaptive_cost_scenario_tradeoff import crossing_scenario_witness
from causal_model.adaptive_expected_cost_audit import synthetic_example as cost_example
from causal_model.limitation_action_report import (
    CandidateScore,
    SpecificationInput,
    build_limitation_action_report,
    classify_specification,
)
from causal_model.limitation_taxonomy_adapter import (
    combine_taxonomy_projections,
    project_adaptive_expected_cost,
    project_limitation_action_report,
    project_replication_switch,
    project_scenario_cost_tradeoff,
    project_specification_status,
)
from causal_model.replication_switch_audit import audit_replication_switch
from examples.replication_switch_report import current_weights, three_program_model


def _replication_projection():
    rows, contact, physiology = three_program_model()
    weights, _ = current_weights(("present",))
    audit = audit_replication_switch(
        rows,
        contact,
        (physiology,),
        target_columns=("program",),
        weights=weights,
        support_reference="synthetic shared-taxonomy adapter example",
        weight_reference="equal initial weights conditioned on contact-present",
        conditional_iid_reference="fresh contact readings iid given fixed program",
        future_likelihood_reference="both channels independent of past readings given full world",
    )
    return project_replication_switch(audit)


def build_report() -> dict:
    budgeted = project_specification_status(
        classify_specification(
            SpecificationInput(
                "budgeted_positive",
                1.0,
                (CandidateScore("q", True, 0.5),),
                budget_remaining=False,
            )
        )
    )
    sensitivity = project_limitation_action_report(
        build_limitation_action_report(
            (
                SpecificationInput(
                    "spec_a",
                    1.0,
                    (CandidateScore("q1", True, 0.8), CandidateScore("q2", True, 0.2)),
                ),
                SpecificationInput(
                    "spec_b",
                    1.0,
                    (CandidateScore("q1", True, 0.2), CandidateScore("q2", True, 0.8)),
                ),
            )
        )
    )
    replication = _replication_projection()
    early, routing, _, _ = cost_example()
    operational = combine_taxonomy_projections(
        project_adaptive_expected_cost(early),
        project_adaptive_expected_cost(routing[1]),
        project_scenario_cost_tradeoff(crossing_scenario_witness()),
    )
    combined = combine_taxonomy_projections(
        budgeted,
        sensitivity,
        replication,
        operational,
    )
    return {
        "data_kind": "synthetic_shared_limitation_taxonomy_projection",
        "scope": (
            "existing diagnostic receipts projected to canonical taxonomy states; "
            "no new optimizer, scenario meta-prior, field calibration or biological claim"
        ),
        "budgeted_core": asdict(budgeted),
        "specification_sensitivity": asdict(sensitivity),
        "replication_switch": asdict(replication),
        "operational_extensions": asdict(operational),
        "composed_vector": asdict(combined),
    }


if __name__ == "__main__":
    print(json.dumps(build_report(), indent=2, allow_nan=False))
