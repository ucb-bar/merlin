"""Whole-program nonapplicability cannot replace any numerical/roster obligation."""
from copy import deepcopy

import pytest
from merlin.targetgen.capsule_grade import _name_set_sha256
from merlin_experiments.phase1.feedback import public_scalability


@pytest.fixture
def candidate():
    return public_scalability


def documents():
    score = {
        'gradeable': True, 'n_capsules': 1, 'n_passed': 1,
        'cohort_admission': {'n_admitted_capsules': 1, 'admitted_name_set_sha256': _name_set_sha256(['model'])},
        'per_capsule': [{'capsule': 'model', 'status': 'pass'}],
        'numeric_all_exact': True, 'structural_evidence_all_pass': True,
        'tier_reached': {'L0': 0, 'L1': 0, 'L2': 1},
    }
    observed = {'model': {
        'capsule': 'model', 'status': 'pass', 'tiers': {
            'L0': {'status': 'skipped', 'not_applicable': True,
                   'reason': 'command-buffer reference cannot interpret scalar regions of a whole-program ELF'},
            'L1': {'status': 'skipped', 'not_applicable': True,
                   'reason': 'command-buffer simulate cannot execute scalar regions of a whole-program ELF'},
            'L2': {'status': 'pass', 'engine': 'spike', 'fidelity': 'functional_model', 'derived_from_rtl': False},
        },
        'numeric': {'status': 'pass', 'missing_outputs': [], 'mismatch_count': 0,
                    'golden_source': 'independent_tensor',
                    'per_output': {'Y': {'status': 'pass', 'n_elements': 12}}},
    }}
    return score, observed


@pytest.mark.parametrize('mutation', ['none', 'wrong_reason', 'not_run', 'missing_output', 'missing_numeric', 'wrong_engine', 'wrong_counts', 'no_applicability', 'numeric_fail', 'numeric_unknown', 'output_fail', 'empty_outputs', 'empty_extent', 'non_integer_extent', 'no_golden_source', 'mismatch'])
def test_native_nonapplicability_requires_complete_independent_execution(candidate, mutation):
    score, observed = documents()
    row = observed['model']
    if mutation == 'wrong_reason': row['tiers']['L0']['reason'] = 'too expensive'
    if mutation == 'not_run': row['tiers']['L2']['status'] = 'not_run'
    if mutation == 'missing_output': row['numeric']['missing_outputs'] = ['Y']
    if mutation == 'missing_numeric': row.pop('numeric')
    if mutation == 'wrong_engine': row['tiers']['L2']['engine'] = 'gsim'
    if mutation == 'wrong_counts': score['tier_reached']['L0'] = 1
    if mutation == 'no_applicability': row['tiers']['L1']['not_applicable'] = False
    if mutation == 'numeric_fail': row['numeric']['status']='fail'
    if mutation == 'numeric_unknown': row['numeric']['status']='unknown'
    if mutation == 'output_fail': row['numeric']['per_output']['Y']['status']='fail'
    if mutation == 'empty_outputs': row['numeric']['per_output']={}
    if mutation == 'empty_extent': row['numeric']['per_output']['Y']['n_elements']=0
    if mutation == 'non_integer_extent': row['numeric']['per_output']['Y']['n_elements']=True
    if mutation == 'no_golden_source': row['numeric'].pop('golden_source')
    if mutation == 'mismatch': row['numeric']['mismatch_count']=1
    assert candidate.complete(score, observed, names={'model'}) is (mutation == 'none')


def test_public_projection_keeps_numeric_receipt_private(candidate):
    score, observed = documents()
    projected=candidate._observed_result(observed['model'])
    assert projected['numeric']==observed['model']['numeric']
    projected['numeric']['private_output_sentinel']=[716291]
    public = candidate.public_result({'score': score, 'observed': {'model':projected}})
    assert 'numeric' not in public['cases'][0]
    assert public['cases'][0]['tiers']['L0'] == 'skipped'


@pytest.mark.parametrize('mutation',['none','missing_observed','missing_scored','duplicate_scored','extra_observed','renamed','partial_count','wrong_cohort_hash','unmeasured','inflated_skipped','unrun_l2'])
def test_mixed_structural_and_whole_program_roster_remains_complete(candidate,mutation):
    score,observed=documents()
    observed['kernel']={'capsule':'kernel','status':'pass','tiers':{
        'L0':{'status':'pass'},'L1':{'status':'pass'},
        'L2':{'status':'pass','engine':'spike','fidelity':'functional_model','derived_from_rtl':False}}}
    score.update(n_capsules=2,n_passed=2,
        cohort_admission={'n_admitted_capsules':2,'admitted_name_set_sha256':_name_set_sha256(['model','kernel'])},
        per_capsule=[{'capsule':n,'status':'pass'}for n in ['model','kernel']],
        tier_reached={'L0':1,'L1':1,'L2':2})
    if mutation=='missing_observed':observed.pop('kernel')
    if mutation=='missing_scored':score['per_capsule'].pop()
    if mutation=='duplicate_scored':score['per_capsule'][1]['capsule']='model'
    if mutation=='extra_observed':observed['other']=deepcopy(observed['kernel'])
    if mutation=='renamed':score['per_capsule'][1]['capsule']='other'
    if mutation=='partial_count':score['n_passed']=1
    if mutation=='wrong_cohort_hash':score['cohort_admission']['admitted_name_set_sha256']='0'*64
    if mutation=='unmeasured':score['n_not_measured']=1
    if mutation=='inflated_skipped':score['tier_reached']['L0']=2
    if mutation=='unrun_l2':observed['model']['tiers']['L2']['status']='not_run'
    assert candidate.complete(score,observed,names={'model','kernel'})is(mutation=='none')
