"""Synthetic custody mutations for compact existing-performance projections."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

SOURCE = Path(__file__).parents[1] / 'src/merlin_experiments/phase1/construction_actions.py'
spec = importlib.util.spec_from_file_location('standalone_construction_actions', SOURCE)
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)


def response():
    pins = dict(zip(('model_sha256','comparison_domain_sha256','scope_sha256','elf_sha256',
                     'capsule_sha256','selection_sha256','facts_sha256'), (c*64 for c in '1234567')))
    scoped = dict(**{k:pins[k] for k in ('model_sha256','scope_sha256','elf_sha256')},
                  source_kind='modeled',status='modeled',activity_status='modeled',estimated_cycles_same_window=100,
                  resources=[dict(resource_id=1,name='load',capacity=2,busy_unit_cycles=100,active_cycles=80),
                             dict(resource_id=2,name='execute',capacity=1,busy_unit_cycles=70,active_cycles=70)],
                  overlaps=[dict(left=1,right=2,active_cycles=30)],traffic=[],unknown=[],suggestions=[])
    signals = dict(**{k:pins[k] for k in ('model_sha256','scope_sha256','comparison_domain_sha256')},
                   estimated_whole_soc_cycles=100,scoped_counts_status='observed',selected_window=0,
                   scoped_counts={'target.commands':6,'target.fences':2,'cpu.integer':17},
                   scoped_resource_feedback=copy.deepcopy(scoped))
    perf = dict(**pins,status='PROVISIONAL',complete_event_accounting=True,estimated_whole_soc_cycles=100,
                timing_qualified=False,timing_mode='untuned',scope='complete call',scope_excludes=['initialization'],
                structural_roofline={'status':'partial','structural_compute_floor_cycles':20},
                structural_comparison_binding={k:v for k,v in pins.items() if k!='elf_sha256'},
                scoped_resource_feedback=scoped,feedback_brief={'performance_signals':signals},
                feature_counts={'target.commands':999999})
    return dict(selfcheck_request_id='test-only',submission_sha256='a'*64,n_capsules=1,
                per_capsule=[dict(capsule='test-only',numerical_execution_passed=True,
                                  structural_qualification={'status':'pass'},performance=perf)])


def case(value):
    return A.performance_projection(value)['cases'][0]


def test_exact_roi_uses_scoped_counts_and_labels_model_activity():
    c=case(response())
    assert c['status']=='PROVISIONAL' and c['estimated_whole_soc_cycles']==100
    assert c['scoped_counts']['target.commands']==6 # Never whole-trace999999.
    assert c['structural_floor_to_model_ratio']==.2 and 'NOT physical' in c['ratio_kind']
    assert c['top_modeled_resources'][0]['name']=='execute'
    assert c['top_modeled_resources'][1]['capacity_normalized_activity']==.5
    assert c['overlaps'][0]['fraction_of_window']==.3


@pytest.mark.parametrize('field',['model_sha256','elf_sha256','scope_sha256','estimated_cycles_same_window'])
def test_resource_identity_or_window_mutation_is_unknown(field):
    r=response();r['per_capsule'][0]['performance']['scoped_resource_feedback'][field]='f'*64
    c=case(r)
    assert c['status']=='UNKNOWN' and c['estimated_whole_soc_cycles'] is None
    assert c['scoped_counts']=={} and c['top_modeled_resources']==[]


def test_missing_or_wrong_brief_never_falls_back_to_feature_counts():
    r=response();r['per_capsule'][0]['performance']['feedback_brief']['performance_signals']['scope_sha256']='b'*64
    c=case(r)
    assert c['estimated_whole_soc_cycles']==100 and c['scoped_counts_status']=='UNKNOWN'
    assert c['scoped_counts']=={} and 'scoped_command_and_CPU_counts'in c['unknown']
    del r['per_capsule'][0]['performance']['feedback_brief']
    assert case(r)['scoped_counts']=={}


def test_missing_resource_capture_or_invalid_capacity_does_not_fabricate_zeros():
    r=response();s=r['per_capsule'][0]['performance']['scoped_resource_feedback'];s['resources'][0]['busy_unit_cycles']=10000
    c=case(r);assert [x['name']for x in c['top_modeled_resources']]==['execute']
    s['activity_status']='UNKNOWN';c=case(r)
    assert c['resources_status']=='UNKNOWN' and c['top_modeled_resources']==[]


def test_rankings_refuse_different_model_domain_scope_and_incomplete_numerics():
    a=A.performance_projection(response());a['action_id']='first'
    b=copy.deepcopy(a);b['action_id']='second';b['submission_sha256']='b'*64;b['cases'][0]['estimated_whole_soc_cycles']=90
    p=A.performance_portfolio([a,b]);assert len(p['matched_rankings'])==1
    assert [x['rank']for x in p['matched_rankings'][0]['candidates']]==[1,2]
    for field in ['model_sha256','comparison_domain_sha256','scope_sha256']:
        c=copy.deepcopy(b);c['cases'][0][field]='f'*64
        assert len(A.performance_portfolio([a,c])['matched_rankings'])==2
    b['cases'][0]['numerical_screen']='unknown'
    assert len(A.performance_portfolio([a,b])['matched_rankings'][0]['candidates'])==1
    assert p['capsule_passes'] is None and p['activation_or_promotion'] is False


def test_completion_writes_perf_stream_and_atomic_portfolio_without_extra_checks(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path/'agent_selfcheck.py').write_text('import json\nprint(json.dumps('+repr(response())+'))\n')
    plan={'schema':A.SCHEMA,'actions':[dict(id='one',kind='selfcheck',boundary='integration',capability='test',argv=['--capsules','test-only'])]}
    out=tmp_path/'out';result=A.execute(plan,output=out,timeout_s=10,plan_sha256='test')
    stream=[json.loads(x)for x in (out/'performance.jsonl').read_text().splitlines()]
    portfolio=json.loads((out/'performance-portfolio.json').read_text())
    assert len(stream)==1 and stream[0]['cases'][0]['estimated_whole_soc_cycles']==100
    assert len(portfolio['latest'])==1 and not(out/'.performance-portfolio.json.pending').exists()
    assert result['performance_feedback']['completed_requests']==1 and result['capsule_passes'] is None
