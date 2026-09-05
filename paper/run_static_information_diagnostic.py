"""Run the post-frozen static-information claim-ceiling diagnostic."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from causal_model.generality_sweep import (
    run_static_information_diagnostic,
    run_static_information_diagnostic_from_frozen_protocol,
)

ROOT = Path(__file__).resolve().parents[1]
FROZEN_PROTOCOL = ROOT / "paper" / "g2_frozen_benchmark_protocol.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        payload = run_static_information_diagnostic(
            seeds=(101, 202),
            budgets=(2, 4),
            n_systems_per_seed=30,
            n_attempts=500,
            K_choices=(4, 5, 6),
            confound_choices=(1, 2),
            min_sub_size=8,
            n_distractors=2,
        )
    else:
        payload = run_static_information_diagnostic_from_frozen_protocol(
            FROZEN_PROTOCOL,
            budgets=(2, 4),
        )

    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
