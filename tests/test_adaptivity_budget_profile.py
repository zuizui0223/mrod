from itertools import product

from causal_model.adaptive_joint_design import routing_witness
from causal_model.adaptivity_budget_profile import adaptivity_budget_profile
from causal_model.joint_budgeted_design import JointCalibrationScenario


def test_context_routing_gain_exists_only_at_intermediate_budget_two():
    rows, scenarios = routing_witness()
    profile = adaptivity_budget_profile(
        rows,
        scenarios,
        candidate_order=("context", "assay0", "assay1"),
        acquisition_costs={"context": 1, "assay0": 1, "assay1": 1},
        budgets=(0, 1, 2, 3),
        target_columns=("target",),
        support_reference="synthetic context-routing budget profile",
    )
    assert profile.positive_gap_budgets_any_scenario == (2,)
    assert profile.positive_gap_budgets_all_scenarios == (2,)
    by_budget = {row.budget: row for row in profile.rows}
    assert all(abs(v) < 1e-12 for v in by_budget[0].adaptivity_gap_bits.values())
    assert all(abs(v) < 1e-12 for v in by_budget[1].adaptivity_gap_bits.values())
    assert abs(by_budget[2].adaptivity_gap_bits["uniform_context"] - 0.5) < 1e-12
    assert abs(by_budget[2].adaptivity_gap_bits["context0_common"] - 0.25) < 1e-12
    assert abs(by_budget[2].adaptivity_gap_bits["context1_common"] - 0.25) < 1e-12
    assert all(abs(v) < 1e-12 for v in by_budget[3].adaptivity_gap_bits.values())
    assert profile.peak_gap_budgets_by_scenario == {
        "uniform_context": (2,),
        "context0_common": (2,),
        "context1_common": (2,),
    }


def _xor_models():
    worlds = tuple(product((0, 1), repeat=2))
    rows = tuple({"target": u ^ v} for u, v in worlds)
    outcomes = tuple(product(("0", "1"), repeat=3))
    matrix = tuple(tuple(
        (0.8 if int(z) == (u ^ v) else 0.2)
        if (int(a), int(b)) == (u, v)
        else 0.0
        for a, b, z in outcomes
    ) for u, v in worlds)
    scenarios = (
        JointCalibrationScenario(
            "uniform", (1, 1, 1, 1), outcomes, matrix,
            "synthetic explicit XOR law",
        ),
        JointCalibrationScenario(
            "target_rare", (9, 1, 1, 9), outcomes, matrix,
            "synthetic explicit XOR law",
        ),
    )
    return rows, scenarios


def test_complementary_xor_has_no_adaptive_class_gap_at_any_budget():
    rows, scenarios = _xor_models()
    profile = adaptivity_budget_profile(
        rows,
        scenarios,
        candidate_order=("U", "V", "direct"),
        acquisition_costs={"U": 1, "V": 1, "direct": 1},
        budgets=(0, 1, 2, 3),
        target_columns=("target",),
        support_reference="synthetic XOR no-routing budget profile",
    )
    assert profile.positive_gap_budgets_any_scenario == ()
    assert profile.positive_gap_budgets_all_scenarios == ()
    assert all(
        abs(gap) < 1e-12
        for row in profile.rows
        for gap in row.adaptivity_gap_bits.values()
    )


def test_budget_profile_rejects_duplicate_or_negative_budgets():
    rows, scenarios = routing_witness()
    kwargs = dict(
        accepted_rows=rows,
        scenarios=scenarios,
        candidate_order=("context", "assay0", "assay1"),
        acquisition_costs={"context": 1, "assay0": 1, "assay1": 1},
        target_columns=("target",),
        support_reference="synthetic budget validation",
    )
    for budgets in ((1, 1), (-1, 0), ()):
        try:
            adaptivity_budget_profile(budgets=budgets, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid budget sequence should fail")
