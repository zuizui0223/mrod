from dataclasses import replace
from itertools import product
from math import log2
import pytest
from causal_model.joint_budgeted_design import JointCalibrationScenario,plan_joint_observation_budget


def setup():
    worlds=tuple(product((0,1),repeat=2))
    rows=[{'target':a^b} for a,b in worlds]
    events=tuple(product(('0','1'),repeat=3))
    matrix=tuple(tuple((.8 if int(z)==a^b else .2) if (int(x),int(y))==(a,b) else 0
                       for x,y,z in events) for a,b in worlds)
    models=(JointCalibrationScenario('uniform',(1,1,1,1),events,matrix,'synthetic joint law'),
            JointCalibrationScenario('rare',(9,1,1,9),events,matrix,'synthetic prior sensitivity'))
    return rows,models


def plan(budget=2,models=None,**kwargs):
    rows,default=setup()
    args=dict(candidate_order=('U','V','direct'),acquisition_costs={'U':1,'V':1,'direct':1},
              budget=budget,target_columns=['target'],support_reference='synthetic four worlds')
    args.update(kwargs)
    return plan_joint_observation_budget(rows,models or default,**args)


def scores(r):
    return {s.candidate_names:s for s in r.bundle_scores}


def test_budget_one_prefers_direct_but_budget_two_prefers_zero_singleton_pair():
    assert plan(1).uniformly_best_bundles==(('direct',),)
    r=plan(2)
    assert r.uniformly_best_bundles==(('U','V'),)
    assert r.minimax_regret_bundles==(('U','V'),)
    assert r.minimum_worst_regret_bits==0
    assert not r.adaptive_policy_optimality_claimed
    ss=scores(r)
    assert ss[('U',)].information_by_scenario['uniform']==pytest.approx(0,abs=1e-12)
    assert ss[('V',)].information_by_scenario['uniform']==pytest.approx(0,abs=1e-12)
    assert ss[('U','V')].information_by_scenario['uniform']==pytest.approx(1)
    assert ss[('direct',)].information_by_scenario['uniform']==pytest.approx(1+.2*log2(.2)+.8*log2(.8))


def test_same_marginals_can_have_different_pair_information():
    rows=[{'target':0},{'target':1}]
    events=tuple(product(('0','1'),repeat=2))
    xor=JointCalibrationScenario('xor',(1,1),events,((.5,0,0,.5),(0,.5,.5,0)),'synthetic dependent')
    independent=replace(xor,name='independent',probabilities=((.25,)*4,)*2)
    r=plan_joint_observation_budget(rows,[xor,independent],candidate_order=['A','B'],
        acquisition_costs={'A':1,'B':1},budget=2,target_columns=['target'],support_reference='synthetic two targets')
    ss=scores(r)
    assert ss[('A',)].information_by_scenario=={'xor':0.0,'independent':0.0}
    assert ss[('B',)].information_by_scenario=={'xor':0.0,'independent':0.0}
    assert ss[('A','B')].information_by_scenario=={'xor':1.0,'independent':0.0}


def test_budget_zero_has_only_empty_bundle():
    r=plan(0)
    assert r.uniformly_best_bundles==((),)
    assert len(r.bundle_scores)==1


def test_unequal_acquisition_costs_change_feasible_bundles():
    r=plan(2,acquisition_costs={'U':2,'V':2,'direct':1})
    assert ('U','V') not in scores(r)
    assert r.uniformly_best_bundles==(('direct',),)


def test_missing_joint_law_is_not_zero_information_or_a_full_winner():
    _,models=setup()
    r=plan(models=(models[0],replace(models[1],probabilities=None)))
    assert not r.complete_vocabulary
    assert r.minimax_regret_bundles==()
    assert r.uniformly_best_bundles==()
    assert scores(r)[('U','V')].information_by_scenario['rare'] is None


def test_joint_event_row_order_is_not_a_semantic_change():
    _,models=setup()
    flipped=tuple(replace(m,joint_outcomes=m.joint_outcomes[::-1],
                          probabilities=tuple(row[::-1] for row in m.probabilities)) for m in models)
    a,b=plan(),plan(models=flipped)
    assert a.minimax_regret_bundles==b.minimax_regret_bundles
    for key in scores(a):
        assert scores(a)[key].information_by_scenario==pytest.approx(scores(b)[key].information_by_scenario)


def test_extra_unaffordable_candidates_do_not_create_an_oracle_outside_budget():
    r=plan(1,acquisition_costs={'U':1,'V':1,'direct':5})
    assert ('direct',) not in scores(r)
    assert r.minimum_worst_regret_bits==pytest.approx(0,abs=1e-12)


@pytest.mark.parametrize('override',[{'candidate_order':'U'}, {'candidate_order':[]},
    {'candidate_order':['U','U','direct']},{'acquisition_costs':{'U':1}},
    {'acquisition_costs':{'U':0,'V':1,'direct':1}}, {'budget':True}, {'budget':-1},
    {'comparison_tolerance_bits':float('nan')}, {'target_columns':'target'}])
def test_bad_public_schema_rejected(override):
    with pytest.raises(ValueError): plan(**override)


@pytest.mark.parametrize('mutation', ['duplicate_event','short_event','bad_probability','short_matrix','missing_target'])
def test_malformed_joint_model_is_an_error(mutation):
    rows,models=setup(); m=models[0]
    if mutation=='duplicate_event': m=replace(m,joint_outcomes=m.joint_outcomes[:-1]+m.joint_outcomes[:1])
    if mutation=='short_event': m=replace(m,joint_outcomes=tuple(ev[:2] for ev in m.joint_outcomes))
    if mutation=='bad_probability': m=replace(m,probabilities=((.5,)*8,)*4)
    if mutation=='short_matrix': m=replace(m,probabilities=m.probabilities[:2])
    if mutation=='missing_target': rows[0]={}
    with pytest.raises(ValueError):
        plan_joint_observation_budget(rows,[m],candidate_order=['U','V','direct'],
            acquisition_costs={'U':1,'V':1,'direct':1},budget=2,target_columns=['target'],support_reference='synthetic')


def test_shared_joint_event_support_is_required_across_scenarios():
    _,models=setup()
    other=replace(models[1],joint_outcomes=tuple(('x',)+ev[1:] for ev in models[1].joint_outcomes[:4]),
                  probabilities=tuple((.25,)*4 for _ in range(4)))
    with pytest.raises(ValueError): plan(models=(models[0],other))


def test_duplicate_scenario_names_rejected():
    _,models=setup()
    with pytest.raises(ValueError,match='unique'):
        plan(models=(models[0],replace(models[1],name=models[0].name)))
