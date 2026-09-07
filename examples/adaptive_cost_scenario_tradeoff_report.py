"""Synthetic scenario-wise expected-cost tradeoff diagnostics.

Run from repository root:
    python -m examples.adaptive_cost_scenario_tradeoff_report
"""
from __future__ import annotations

from dataclasses import asdict
import json

from causal_model.adaptive_cost_scenario_tradeoff import (
    crossing_scenario_witness,
    uniform_scenario_witness,
)


def build_report() -> dict:
    return {
        "data_kind": "synthetic_adaptive_cost_scenario_tradeoff",
        "scope": (
            "registered information-equivalent policy cost vectors; no scenario meta-prior, "
            "no expected-cost optimization, no field calibration or biological claim"
        ),
        "crossing_scenario_witness": asdict(crossing_scenario_witness()),
        "uniform_componentwise_witness": asdict(uniform_scenario_witness()),
    }


if __name__ == "__main__":
    print(json.dumps(build_report(), indent=2, allow_nan=False))
