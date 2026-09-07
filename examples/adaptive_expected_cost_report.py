"""Synthetic expected-cost diagnostics for selected adaptive trees.

Run from repository root:
    python -m examples.adaptive_expected_cost_report
"""
from __future__ import annotations

from dataclasses import asdict
import json

from causal_model.adaptive_expected_cost_audit import synthetic_example


def build_report() -> dict:
    early, routing, unequal = synthetic_example()
    return {
        "data_kind": "synthetic_adaptive_expected_cost_audit",
        "scope": (
            "scenario-specific expected acquisition cost of already selected trees; "
            "not expected-cost optimization, field calibration, causal evidence, or a scenario meta-prior"
        ),
        "early_stop_witness": asdict(early),
        "routing_budget_profile": [asdict(receipt) for receipt in routing],
        "unequal_cost_routing": asdict(unequal),
    }


if __name__ == "__main__":
    print(json.dumps(build_report(), indent=2, allow_nan=False))
