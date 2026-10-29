"""A changed judge reopens physical obligations without discarding old evidence."""
import hashlib
import io
import json
from copy import deepcopy

import pytest
from merlin_experiments.phase1.feedback import instrument_retirement as R, promotion as P
from test_joint_physical_correctness import setup

EXECUTION = 'c' * 64
OLD = 'b' * 64
NEW = 'a' * 64


def seed(ws, *, status='pass', instrument=OLD):
    digest, components, _ = P.submission_digests(ws)
    row = {'status': status, 'digest': digest, 'components': components,
           'instrument': instrument, 'execution_digest': EXECUTION}
    state = {};P._record(state, 'A', 'L3', row);P._save_tier_state(ws, state)
    return deepcopy(state)


def promote(ws, row):
    return P.promote(ws, ws/'.qa_channel', {'per_capsule': [
        {'capsule': 'A', 'pass': True, 'execution_digest': EXECUTION}]},
        'L2', 'L3', {'A'}, io.StringIO(), context=None, closed_loop_rows={'A': row})


@pytest.mark.parametrize('status', ['pass', 'fail', 'joined'])
def test_terminal_old_instrument_archived_then_real_native_request_queued(tmp_path, monkeypatch, status):
    ws, row = setup(tmp_path, monkeypatch)
    original = seed(ws, status=status)
    # Existing old instrument's queue filename cannot suppress the replacement.
    (ws/'.qa_channel'/f'simreq_promo0_{EXECUTION[:16]}_A.json').write_text('{}')
    assert promote(ws, row) == ['A']
    state = P._tier_state(ws)
    assert state['A']['L3']['status'] == 'pending'
    assert state['A']['L3']['instrument'] == NEW
    assert state['A']['L2']['instrument'] == NEW
    archive = json.loads(next((ws/'qa/tier_instrument_history').glob('*.json')).read_text())
    assert archive['previous_state'] == original
    assert archive['retired'][0]['entry'] == original['A']['L3']
    requests = [json.loads(p.read_text()) for p in (ws/'.qa_channel').glob('simreq*.json')]
    request = next(r for r in requests if r.get('promoted'))
    assert request['sim'] == 'gsim' and request['tiers'] == 'L3'
    assert request['identity'] == EXECUTION and request['submission_snapshot']
    assert promote(ws, row) == []
    report = json.loads((ws/'qa/joint_physical_correctness.json').read_text())
    assert report['passed'] == 0 and not report['complete']


@pytest.mark.parametrize(('status', 'instrument'), [('pending', OLD), ('pass', NEW), ('fail', NEW), ('joined', NEW)])
def test_existing_pending_or_current_judge_record_is_not_replaced(tmp_path, monkeypatch, status, instrument):
    ws, row = setup(tmp_path, monkeypatch);original = seed(ws, status=status, instrument=instrument)
    assert promote(ws, row) == []
    assert P._tier_state(ws)['A']['L3'] == original['A']['L3']
    assert not list((ws/'.qa_channel').glob('simreq*.json'))
    assert not (ws/'qa/tier_instrument_history').exists()


def test_operator_preview_and_locked_apply_are_byte_pinned_and_keep_pending(tmp_path, monkeypatch):
    ws, _ = setup(tmp_path, monkeypatch);seed(ws)
    state=P._tier_state(ws);P._record(state,'B','L3',{'status':'pending','instrument':OLD,'digest':'old'})
    P._save_tier_state(ws,state);before=(ws/'qa/tier_state.json').read_bytes();pin=hashlib.sha256(before).hexdigest()
    preview=R.reconcile(ws,context=None,expected_ledger_sha256=pin)
    assert len(preview['retired'])==1 and not preview['applied']
    assert (ws/'qa/tier_state.json').read_bytes()==before
    applied=R.reconcile(ws,context=None,expected_ledger_sha256=pin,apply=True)
    after=P._tier_state(ws)
    assert after['B']==state['B'] and not P._slots_ro(after,'A','L3')
    archive=json.loads(open(applied['archive']['path']).read())
    assert archive['previous_state']==state
    assert archive['previous_ledger_json'].encode()==before
    assert archive['previous_ledger_sha256']==pin
    with pytest.raises(ValueError,match='changed since operator review'):
        R.reconcile(ws,context=None,expected_ledger_sha256=pin,apply=True)


def test_missing_selected_instrument_does_not_retire_or_queue(tmp_path,monkeypatch):
    ws,row=setup(tmp_path,monkeypatch);before=seed(ws)
    monkeypatch.setattr(P,'cert_instrument',lambda *a,**kw:None)
    assert promote(ws,row)==[]
    assert P._tier_state(ws)['A']['L3']==before['A']['L3']
    assert not (ws/'qa/tier_instrument_history').exists()
