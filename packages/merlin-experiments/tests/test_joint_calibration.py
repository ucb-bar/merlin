"""Local qualification and sparse native calibration must preserve held blindness."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1 import joint_calibration as C
from test_joint_calibration_eligibility import valid_grade

GATES=dict(C.GATES)


def cases():
    return [dict(id=f'c{i}', split='train' if i<4 else 'held',
                 source_group_sha256=str(i//2), capsule={'sha256':str(i//2)},
                 generated_root='unused', compiler_sha256=str(i%2)) for i in range(8)]


def test_accurate_but_wrong_variant_order_refuses():
    rows=cases();labels={r['id']:100+(i%2)*20 for i,r in enumerate(rows)}
    incumbent={k:v*1.3 for k,v in labels.items()}
    candidate={r['id']:115-(i%2)*5 for i,r in enumerate(rows)}
    result=C.local_validation(rows,labels,incumbent,candidate,gates=GATES)
    assert result['p95']<.30 and not result['admitted']
    assert result['rank_agreement']==0


def test_no_material_separation_cannot_grant_ranking():
    rows=cases();labels={r['id']:100 for r in rows}
    result=C.local_validation(rows,labels,labels,labels,gates=GATES)
    assert not result['admitted'] and result['rank_agreement'] is None


def test_local_improvement_never_grants_broad_qualification():
    rows=cases();labels={r['id']:100+i*10 for i,r in enumerate(rows)}
    result=C.local_validation(rows,labels,{k:v*1.4 for k,v in labels.items()},labels,gates=GATES)
    assert result['admitted'] and not result['timing_qualified'] and not result['broad_qualification']


def test_sparse_selector_keeps_source_groups_disjoint():
    selected=C.select_variant_cases(cases())
    assert selected['status']=='selected'
    train={r['source_group_sha256'] for r in selected['cases'] if r['split']=='train'}
    held={r['source_group_sha256'] for r in selected['cases'] if r['split']=='held'}
    assert not train&held
    assert C.select_variant_cases(cases()[:6])['status']=='waiting_for_independent_variants'


@pytest.mark.parametrize('identifiable',[False,True])
def test_native_probe_before_physical_fit_and_held_validation(tmp_path,monkeypatch,identifiable):
    root=tmp_path/'round'
    model=dict(prices={'compute':{'latency_cycles':1}},domain={'scope_sha256':'s','memory_regime_sha256':'m'})
    spec=dict(model='initial',model_sha256='initial',review_id='initial')
    selection=dict(runtime={'path':'initial'},facts={})
    rows=cases()
    for i,r in enumerate(rows):
        r['generated_root']=str(tmp_path/f'generated{i}')
        r['result']={'path':f'grade{i}'}
    plan=dict(schema=C.SCHEMA,selection={'path':'selection'},runtime={'path':'initial'},cases=rows,
              parameters=[dict(path=['prices','compute','latency_cycles'],values=[1,2])],gates=GATES)
    realread=C.PERF._read
    def read(pin):
        if pin['path']=='plan':return plan
        if pin['path']=='selection':return selection
        if pin['path']=='initial':return spec
        if pin['path'].startswith('grade'):return valid_grade(source=str(int(pin['path'][5:])%2))
        return realread(pin)
    monkeypatch.setattr(C.PERF,'_read',read)
    def runtime(pin,**kwargs):
        if pin['path']=='initial':return SimpleNamespace(model_json=json.dumps(model))
        spec=realread(pin)
        return SimpleNamespace(model_json=Path(spec['model']).read_text())
    monkeypatch.setattr(C.PERF,'_runtime',runtime)
    events=[]
    def observe(case,*,runtime,context,directory):
        price=json.loads(runtime.model_json)['prices']['compute']['latency_cycles']
        index=int(case['id'][1:]);feature=index+1 if identifiable else 1
        events.append(('observe',case['id'],price))
        report={'estimated_cycles':100+price*feature*10,'feature_counts':{'compute':feature}}
        return Path('fake.elf'),{},report
    monkeypatch.setattr(C,'_observe',observe)
    def labels(cases,*,labels,**kwargs):
        events.append(('physical',[c['id'] for c in cases]))
        labels.update({c['id']:100+2*(int(c['id'][1:])+1)*10 for c in cases})
    monkeypatch.setattr(C,'_collect_labels',labels)
    monkeypatch.setattr(C,'_capsule',lambda p:dict(name=p['sha256'],kind='layer',label='public',
        required_oracle_tiers=['L0','L1','L2','L3']))
    result=C.run_round({'path':'plan'},context=None,backend=None,output=root)
    value=json.loads(Path(result['path']).read_text())
    if not identifiable:
        assert value['status']=='not_identifiable'
        assert not any(e[0]=='physical' for e in events)
    else:
        assert value['timing_epochs'][-1]['qualification']=='limited_calibrated_provisional'
        measured=[e[1] for e in events if e[0]=='physical']
        assert measured==[['c0','c1','c2','c3'],['c4','c5','c6','c7']]
        # Only the training-selected candidate gets held replays.
        assert all(('observe',f'c{i}',2) in events for i in range(4,8))
        # Polling the completed round performs no new execution.
        count=len(events)
        assert C.run_round({'path':'plan'},context=None,backend=None,output=root)==result
        assert len(events)==count


def test_supervisor_idle_accepts_pin_metadata_without_allocating_runtime(tmp_path, monkeypatch):
    from merlin_experiments.phase1 import joint_calibration_service as S
    selection = C._write(tmp_path / 'selection.json', {'runtime': {}, 'members': {}})
    profile = C._write(tmp_path / 'profile.json', {'members': {}, 'performance_selection': selection})
    monkeypatch.setattr(S.joint_control, 'selected', lambda: {'policy': {'native_calibration': profile, 'gsim': {'witness_capsules': []}}})
    result = S.step(profile_pin={k: profile[k] for k in ('path', 'sha256')},
                    state_path=tmp_path / 'run' / 'joint_campaign.json', context=None, output=tmp_path / 'service')
    assert result['status'] == 'waiting_for_independent_variants'
    assert result['registered_variants'] == 0
    assert not list((tmp_path / 'service').glob('round-*'))


def test_supervisor_refuses_unselected_profile(tmp_path, monkeypatch):
    from merlin_experiments.phase1 import joint_calibration_service as S
    selection = C._write(tmp_path / 'selection.json', {'runtime': {}, 'members': {}})
    profile = C._write(tmp_path / 'profile.json', {'members': {}, 'performance_selection': selection})
    monkeypatch.setattr(S.joint_control, 'selected', lambda: {'policy': {'native_calibration': dict(profile, sha256='other')}})
    with pytest.raises(ValueError, match='immutable campaign policy'):
        S.step(profile_pin=profile, state_path=tmp_path / 'state', context=None, output=tmp_path / 'service')


@pytest.mark.parametrize('consumed,reserved', [(8,0),(6,2)])
def test_bootstrap_rolls_into_bounded_refinement(tmp_path, monkeypatch, consumed, reserved):
    ledger=tmp_path/'ledger.json'
    ledger.write_text(json.dumps({'admissions':[{'stage':'bootstrap'}]*consumed,
        'authorizations':[{'stage':'bootstrap','remaining_capsules':['other']*reserved}]}))
    monkeypatch.setattr(C.joint_control, 'gsim_ledger', lambda: ledger)
    monkeypatch.setattr(C.joint_control, 'selected', lambda: {'policy': {}})
    calls=[]
    monkeypatch.setattr(C.joint_measurements,'authorize',lambda names, **kwargs: calls.append((names,kwargs)))
    C._authorize_if_needed('current', {'stage':'bootstrap','question':'independent service price'})
    assert calls == [(['current'], {'question':'independent service price','stage':'refinement'})]


def test_one_bad_case_and_missing_cases_cannot_promote_strict_round():
    rows=cases();labels={r['id']:100+i*20 for i,r in enumerate(rows)}
    incumbent={k:v*1.4 for k,v in labels.items()}
    candidate=dict(labels);candidate['c7']*=1.05
    result=C.local_validation(rows,labels,incumbent,candidate,gates=GATES)
    assert not result['admitted'] and result['maximum_case_error'] > .04
    del candidate['c7']
    result=C.local_validation(rows,labels,incumbent,candidate,gates=GATES)
    assert not result['admitted'] and result['declared_held_cases']==4
    assert result['evaluated_held_cases']==3 and result['refusals']


@pytest.mark.parametrize('bad', [float('nan'),float('inf'),True,0,-1])
def test_invalid_cycle_observations_remain_outstanding(bad):
    rows=cases();labels={r['id']:100+i*20 for i,r in enumerate(rows)}
    candidate=dict(labels);candidate['c7']=bad
    result=C.local_validation(rows,labels,labels,candidate,gates=GATES)
    assert not result['admitted'] and not result['all_declared_cases_evaluated']
