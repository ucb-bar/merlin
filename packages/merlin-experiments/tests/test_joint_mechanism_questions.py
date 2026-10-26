"""Question intake/review never itself grants a fit or launches a simulator."""
import json
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1 import joint_mechanism_questions as Q
from merlin_experiments.phase1 import joint_diagnostic_collection as D
from merlin_experiments.phase1 import joint_calibration as C


@pytest.fixture
def owner(tmp_path,monkeypatch):
    selection=C._write(tmp_path/'selection.json',{'members':{}})
    profile=C._write(tmp_path/'profile.json',{'performance_selection':selection,'members':{'public':{}}})
    state=dict(started_at_epoch=1,policy_sha256='policy',performance_selection=selection,
        policy={'native_calibration':profile,'gsim':{'witness_capsules':['public']}})
    path=tmp_path/'campaign.json';path.write_text(json.dumps(state))
    env={Q.joint_control.ENV:str(path),Q.PERF.ENV:json.dumps(selection)}
    monkeypatch.setenv(Q.joint_control.ENV,str(path))
    monkeypatch.setattr(Q.joint_control,'selected',lambda:state)
    proposal=dict(schema=Q.SCHEMA,question='Is the source bank wait covered?',
        discriminator='Compare the pinned packet-ready and publication events.',
        capsules=['public'],purpose='unresolved_cost')
    return SimpleNamespace(root=tmp_path,state=state,path=path,env=env,proposal=proposal,
                           selection=selection,profile=profile)


def test_legacy_untouched_and_proposal_does_not_grant_authority(owner):
    f=owner;before=f.path.read_bytes()
    result=Q.submit(f.proposal,request_id='one',environment=f.env)
    assert result['status']=='queued_for_trusted_diagnostic_review'
    assert result['dispatch_authorized'] is False and result['fit_eligible'] is False
    assert result['numeric_assessment']=='not_run' and f.path.read_bytes()==before
    assert not (f.root/'joint_gsim_budget.json').exists()
    assert Q.submit(f.proposal,request_id='retry',environment=f.env)['question_id']==result['question_id']


@pytest.mark.parametrize('change',['outside','runtime','purpose','missing','budget'])
def test_untrusted_fields_or_outside_roster_denied(owner,change):
    f=owner;p=dict(f.proposal)
    if change=='outside':p['capsules']=['private']
    elif change=='runtime':p['runtime']={'path':'chosen-by-author'}
    elif change=='purpose':p['purpose']='fit'
    elif change=='missing':p['discriminator']=''
    else:p['capsules']=['public']*5
    with pytest.raises(ValueError):Q.submit(p,request_id='one',environment=f.env)
    assert not list(f.root.glob('mechanism_questions/*/proposal.json'))


def test_waits_for_completed_products_then_uses_existing_diagnostic_owner(owner,monkeypatch):
    f=owner;Q.submit(f.proposal,request_id='one',environment=f.env)
    calls=[]
    assert Q.step(state_path=f.path,context=None)['status']=='waiting_for_original_numerical_trace_products'
    C._write(f.root/'calibration_registry/entry/entry.json',{'original':'native'})
    monkeypatch.setattr(D,'eligible',lambda *a,**k:('public',{}))
    def run(**kw):
        request=Q.PERF._read(kw['request_pin']);calls.append(request)
        assert request['fit_eligible'] is False and request['max_observations']==1
        assert request['profile']==f.profile and request['performance_selection']==f.selection
        return dict(status='reviewed_diagnostic_limit_reached',fit_eligible=False)
    monkeypatch.setattr(D,'step',run)
    assert Q.step(state_path=f.path,context=None)['status']=='reviewed_diagnostic_limit_reached'
    assert Q.step(state_path=f.path,context=None)['status']=='no_pending_mechanism_questions'
    assert len(calls)==1


def test_stale_question_denied_without_physical_work(owner,monkeypatch):
    f=owner;Q.submit(f.proposal,request_id='one',environment=f.env)
    f.state['policy_sha256']='changed'
    monkeypatch.setattr(D,'step',lambda **k:pytest.fail('no diagnostic for stale question'))
    result=Q.step(state_path=f.path,context=None)
    assert result['status']=='mechanism_question_denied' and result['fit_eligible'] is False
