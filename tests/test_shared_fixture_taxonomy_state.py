"""Cross-repository target-state semantics on the existing shared fixture."""
from __future__ import annotations

import json
from pathlib import Path

from causal_model.limitation_action_report import SpecificationInput, classify_specification
from causal_model.limitation_taxonomy_adapter import project_specification_status


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "target_factorization_v1.json"


def test_shared_factorization_fixture_maps_to_same_target_taxonomy_semantics():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert data["contract_id"] == "boundary-mrod-target-factorization-v1"

    for case in data["cases"]:
        # The shared fixture already independently validates `expected.factors`
        # against MROD target information and Boundary fibre factorization. Here
        # we test only the downstream taxonomy meaning of that result.
        target_resolved = bool(case["expected"]["factors"])
        status = classify_specification(
            SpecificationInput(
                case["name"],
                current_entropy_bits=1.0,  # retain finer represented ambiguity
                candidates=(),
                declared_target_resolved=target_resolved,
            )
        )
        projection = project_specification_status(status)
        target_states = {
            state
            for axis, state in projection.canonical_states
            if axis == "target_identification"
        }
        expected_state = (
            "target_identified_full_mechanism_ambiguous"
            if target_resolved
            else "target_unresolved"
        )
        assert target_states == {expected_state}, case["name"]
