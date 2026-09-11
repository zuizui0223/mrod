#!/usr/bin/env python3
"""Build presentation-only Markdown sources for MEE PDF export.

The scientific manuscript and Supporting Information remain canonical in
``paper/manuscript.md`` and ``paper/supporting_information.md``. This builder only
appends the frozen figure images required by the submission manifest.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "submission_pdf"
FIG = ROOT / "outputs" / "g5" / "figures"

MAIN = ROOT / "paper" / "manuscript.md"
SI = ROOT / "paper" / "supporting_information.md"

MAIN_FIGURES = [
    ("Figure 1", "figure1_controlled_confounding.png"),
    ("Figure 2", "figure2_g2_frozen_v2.png"),
    ("Figure 3", "figure3_information_value_calibration.png"),
]
SI_FIGURES = [("Figure S1", "figureS1_known_truth.png")]


def _append_figures(text: str, figures: list[tuple[str, str]]) -> str:
    chunks = [text.rstrip(), "", "\\newpage", "", "# Figure pages", ""]
    for label, filename in figures:
        path = FIG / filename
        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
        rel = path.relative_to(ROOT).as_posix()
        chunks += [f"## {label}", "", f"![]({rel}){{ width=92% }}", "", "\\newpage", ""]
    return "\n".join(chunks).rstrip() + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    main_text = _append_figures(MAIN.read_text(encoding="utf-8"), MAIN_FIGURES)
    si_text = _append_figures(SI.read_text(encoding="utf-8"), SI_FIGURES)
    (OUT / "MROD_MAIN_WITH_FIGURES.md").write_text(main_text, encoding="utf-8")
    (OUT / "MROD_SUPPORTING_INFORMATION_WITH_FIGURE.md").write_text(si_text, encoding="utf-8")
    print("MROD_SUBMISSION_SOURCE PASS")


if __name__ == "__main__":
    main()
