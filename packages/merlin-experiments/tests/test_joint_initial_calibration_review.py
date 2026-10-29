"""Prospective scope review is not historical neutrality or fit authority."""
import copy
import json
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1 import joint_initial_calibration_review as R
from merlin_experiments.phase1 import joint_calibration_admission as A
from merlin_experiments.phase1 import joint_calibration as C
from merlin_experiments.phase1 import joint_control


@pytest.fixture
def owner(tmp_path,monkeypatch):
    write=lambda name,d:C._write(tmp_path/name,d)
    domain=dict(device_sha256='device',memory_regime_sha256='memory',label_engine_sha256='physical')
    model=dict(target={'prices':{'compute':{'latency_cycles':1}},'hardware':{'fixed':8}},
               domain=dict(domain,feature_engine_sha256='new'),diagnostics={'scoped_resources':{'enabled':True}})
    oldmodel=copy.deepcopy(model);oldmodel['domain']['feature_engine_sha256']='old'
    mp=write('model.json',model);oldmp=write('oldmodel.json',oldmodel)
    source=write('sources.json',{'sha256':'new'});facts=write('facts.json',{})
    align=write('alignment.json',{'mappings':[]})
    spec=dict(model=mp['path'],model_sha256=mp['sha256'],source_manifest=source['path'],
              facts_alignment=align,observer='selected-current-observer')
    oldspec=dict(spec,model=oldmp['path'],model_sha256=oldmp['sha256'])
    runtime=write('runtime.json',spec);oldruntime=write('oldruntime.json',oldspec)
    members={'public':{'capsule_sha256':'original-public-source'}}
    selection=write('selection.json',{'schema':R.P.SCHEMA,'facts':facts,'runtime':runtime,'members':members})
    previous=write('previous-selection.json',dict(R.P._read(selection),runtime=oldruntime))
    parameters=[dict(path=['target','prices','compute','latency_cycles'],values=[1,2],
                     mechanism='original service overhead',reviewed_scope='public native sequences only')]
    profile=write('profile.json',dict(performance_selection=previous,members={'public':{'source_group_sha256':'group'}},
                                    parameters=parameters))
    denial=write('denial.json',dict(schema='merlin.native_calibration_admission_review.v1',deny=[]))
    audit=write('audit.json',dict(initial_selection=selection,kernel_runtime=runtime,facts=facts))
    monkeypatch.setattr(joint_control,'selected',lambda:None)
    def freeze(pin,**kwargs):
        s=R.P._read(pin)
        return SimpleNamespace(model_json=json.dumps(R.P._read({'path':s['model'],'sha256':s['model_sha256']})),verify=lambda:None)
    monkeypatch.setattr(R.P,'_runtime',freeze)
    review=R.prepare(selection_pin=selection,parameter_profile_pin=profile,previous_denial_pin=denial,
        source_review_pin=audit,review_id='reviewed-new-source-initial',output=tmp_path/'initial-review.json')
    monkeypatch.setenv(R.ENV,json.dumps(review));monkeypatch.setenv(A.ENV,json.dumps(denial))
    plan=dict(runtime=runtime,selection=selection,parameters=parameters)
    return SimpleNamespace(root=tmp_path,write=write,selection=selection,profile=profile,
        runtime=runtime,review=review,denial=denial,plan=plan,model=model,spec=spec,
        audit=audit,parameters=parameters)


def test_fresh_initial_scope_admits_normal_round_preconditions_not_a_fit(owner):
    f=owner
    assert A.assert_allowed(f.plan) is None
    review=R.P._read(f.review)
    assert review['strict_gates']==C.GATES
    assert review['measurement_authorized'] is False and review['qualification_issued'] is False
    assert not (f.root/'probe-0').exists()


@pytest.mark.parametrize('damage',['selection','parameters','audit','denial','gates','continuation','campaign','runtime'])
def test_initial_review_cannot_cover_changed_ownership_or_scope(owner,monkeypatch,damage):
    f=owner;plan=copy.deepcopy(f.plan)
    if damage=='selection':
        d=R.P._read(f.selection);d['members']['private']={};plan['selection']=f.write('changed-selection.json',d)
    elif damage=='parameters':plan['parameters'][0]['values']=[1,3]
    elif damage=='audit':
        path=f.root/'audit.json';path.chmod(0o644);path.write_text('{}')
    elif damage=='denial':monkeypatch.setenv(A.ENV,json.dumps(f.write('other-denial.json',R.P._read(f.denial))))
    elif damage=='gates':
        r=R.P._read(f.review);r['strict_gates']['max_case']=.3
        monkeypatch.setenv(R.ENV,json.dumps(f.write('weak-review.json',r)))
    elif damage=='continuation':monkeypatch.setattr(joint_control,'selected',lambda:{'epoch':{'previous':'old'},'performance_selection':f.selection})
    elif damage=='campaign':monkeypatch.setattr(joint_control,'selected',lambda:{'performance_selection':f.profile})
    else:
        spec=dict(f.spec,observer='unreviewed-observer');plan['runtime']=f.write('changed-runtime.json',spec)
    with pytest.raises(ValueError):A.assert_allowed(plan)


def test_existing_matching_denial_is_not_removed_by_initial_review(owner,monkeypatch):
    f=owner
    evidence=f.write('structural-evidence.json',{'public':'unresolved structural cost'})
    denial=f.write('new-matching-denial.json',dict(schema='merlin.native_calibration_admission_review.v1',
        deny=[dict(model_sha256=f.spec['model_sha256'],price_paths=[p['path'] for p in f.parameters],
                   reason='specific source regime remains unsupported',evidence=[evidence])]))
    review=R.prepare(selection_pin=f.selection,parameter_profile_pin=f.profile,previous_denial_pin=denial,
        source_review_pin=f.audit,review_id='same-source-with-denial',output=f.root/'denied-review.json')
    monkeypatch.setenv(R.ENV,json.dumps(review));monkeypatch.setenv(A.ENV,json.dumps(denial))
    with pytest.raises(ValueError,match='blocked_structural_model_admission'):A.assert_allowed(f.plan)


def test_absent_new_review_preserves_historical_ancestry_gate(owner,monkeypatch):
    monkeypatch.delenv(R.ENV)
    with pytest.raises(ValueError,match='exact reviewed runtime ancestry'):A.assert_allowed(owner.plan)


@pytest.mark.parametrize('damage',['price','physical-domain','group','fact-price'])
def test_prepare_rejects_unreviewed_prices_domain_groups_and_rtl_facts(owner,damage):
    f=owner;selection=R.P._read(f.selection);spec=dict(f.spec);model=copy.deepcopy(f.model)
    if damage=='price':model['target']['prices']['compute']['latency_cycles']=2
    elif damage=='physical-domain':model['domain']['device_sha256']='other-device'
    elif damage=='group':selection['members']['public']['capsule_sha256']='other-source'
    else:spec['facts_alignment']=f.write('static-price-mapping.json',{'mappings':[{'model_path':f.parameters[0]['path']}]})
    mp=f.write('new-model.json',model);spec.update(model=mp['path'],model_sha256=mp['sha256'])
    rp=f.write('new-runtime.json',spec);selection['runtime']=rp;sp=f.write('new-selection.json',selection)
    audit=f.write('new-audit.json',dict(initial_selection=sp,kernel_runtime=rp,facts=selection['facts']))
    with pytest.raises(ValueError):
        R.prepare(selection_pin=sp,parameter_profile_pin=f.profile,previous_denial_pin=f.denial,
            source_review_pin=audit,review_id='cannot-admit-this',output=f.root/'bad-review.json')
    assert not (f.root/'bad-review.json').exists()
