import hashlib
import json
from pathlib import Path

import pytest
from merlin_experiments.phase1.feedback import instrument_retirement as R, promotion as P
from test_joint_physical_correctness import setup

OLD='b'*64
NEW='a'*64

def pin(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}

def fixture(tmp_path,monkeypatch):
    ws,_=setup(tmp_path,monkeypatch);old=tmp_path/'old';old.mkdir();run=tmp_path/'run';run.mkdir()
    state={}
    for name in ['inherited','claimed','changed','current']:
        P._record(state,name,'L3',{'status':'pending','instrument':OLD,'digest':name,'execution_digest':hashlib.sha256(name.encode()).hexdigest()})
    prior=old/'tier_state.json';prior.write_text(json.dumps(state))
    summary=old/'joint_campaign_summary.json';summary.write_text(json.dumps({'completion_reason':'library_epoch'}))
    epoch={'previous_run':str(old),'tier_ledger':pin(prior),'pins':{'joint_campaign_summary.json':pin(summary)['sha256']}}
    ep=tmp_path/'epoch.json';ep.write_text(json.dumps(epoch))
    (run/'joint_campaign.json').write_text(json.dumps({'epoch':epoch}))
    (run/'environment.yaml').write_text(json.dumps({'workspace_path':str(ws)}))
    state['changed']['L3']['digest']='edited'
    state['current']['L3']['instrument']=NEW
    # _record keeps mirror and exact slot on the same object in this fixture.
    P._save_tier_state(ws,state)
    identity=state['claimed']['L3']['execution_digest']
    (ws/'.qa_channel/simreq_claimed.json').write_text(json.dumps({'identity':identity,'capsules':'claimed'}))
    return ws,run,ep,state


def call(ws,run,ep,**kwargs):
    return R.reconcile_inherited_pending(ws,context=None,current_run=run,epoch_pin=pin(ep),
        expected_ledger_sha256=pin(ws/'qa/tier_state.json')['sha256'],**kwargs)


def test_only_exact_unclaimed_inherited_pending_is_archived(tmp_path,monkeypatch):
    ws,run,ep,state=fixture(tmp_path,monkeypatch);before=(ws/'qa/tier_state.json').read_bytes()
    preview=call(ws,run,ep)
    assert [x['capsule'] for x in preview['retired']]==['inherited']
    assert (ws/'qa/tier_state.json').read_bytes()==before
    result=call(ws,run,ep,apply=True);after=P._tier_state(ws)
    assert not P._slots_ro(after,'inherited','L3')
    for name in ['claimed','changed','current']:assert after[name]==state[name]
    archive=json.loads(Path(result['archive']['path']).read_text())
    assert archive['previous_ledger_json'].encode()==before
    assert archive['inheritance_proof']['current_requests']


def test_pin_mutation_and_wrong_selected_epoch_refuse(tmp_path,monkeypatch):
    ws,run,ep,_=fixture(tmp_path,monkeypatch);before=(ws/'qa/tier_state.json').read_bytes()
    old=json.loads(ep.read_text())['tier_ledger']['path'];Path(old).write_text('{}')
    with pytest.raises(ValueError,match='provenance pin changed'):call(ws,run,ep,apply=True)
    assert (ws/'qa/tier_state.json').read_bytes()==before
    (run/'joint_campaign.json').write_text('{}')
    with pytest.raises(ValueError,match='not selected'):call(ws,run,ep,apply=True)


def test_unreadable_request_and_ledger_race_refuse(tmp_path,monkeypatch):
    ws,run,ep,_=fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='changed since operator review'):
        R.reconcile_inherited_pending(ws,context=None,current_run=run,epoch_pin=pin(ep),expected_ledger_sha256='0'*64,apply=True)
    (ws/'.qa_channel/simreq_unknown.json').write_text('{')
    with pytest.raises(ValueError):call(ws,run,ep,apply=True)
    assert P._slots_ro(P._tier_state(ws),'inherited','L3')
