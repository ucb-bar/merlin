"""Protected original tier boundaries survive source edits without ELF aliasing."""
from copy import deepcopy
import json

import pytest

from merlin_experiments.phase1.feedback import promotion as P, tier_boundary_join as J
from merlin_experiments.phase1.feedback import joint_physical_correctness as Q
from test_tier_boundary_join import observations, ledger, completion


def changed_source(left, right):
    current = deepcopy(left)
    current['common'].update(compiler_sha256='9'*64, submission_digest='b'*16)
    current['result_sha256'] = 'a'*64
    loop = dict(status='pass', digest='b'*16, execution_digest=current['execution_digest'], tier_boundary=current)
    cert = dict(status='pass', digest=left['common']['submission_digest'], instrument='a'*64,
        execution_digest=right['execution_digest'], tier_boundary=right, boundary_join=J.join(left, right))
    return loop, cert


def test_projection_preserves_original_source_and_physical_executable(tmp_path, monkeypatch):
    monkeypatch.setattr(Q, 'roster', lambda: {'A'})
    left, right = observations(); loop, cert = changed_source(left, right)
    state = {}; P._record(state, 'A', 'L2', loop); P._record(state, 'A', 'L3', cert)
    # Later queued work on the unchanged L2 ELF must not hide the reusable pass.
    P._record(state, 'A', 'L3', dict(status='pending', digest='b'*16, instrument='a'*64,
        execution_digest=left['execution_digest']))
    before = deepcopy(state)
    Q.project(tmp_path, state, source='b'*16, instrument='a'*64)
    report = json.loads((tmp_path/'qa/joint_physical_correctness.json').read_text())
    row = report['obligations']['A']
    assert report['passed'] == 1 and row['physical_executed'] is None
    assert row['certificate_source_sha256'] == left['common']['submission_digest']
    assert row['reused_execution_identity'] == right['execution_digest']
    assert row['boundary_reuse']['current_source_sha256'] == '9'*64
    assert row['boundary_reuse']['certificate_source_sha256'] == left['common']['compiler_sha256']
    assert state == before  # Projection never writes a certificate/cache alias.


@pytest.mark.parametrize('field', ['source_ir_sha256', 'raw_command_buffer_sha256',
    'effective_command_buffer_sha256', 'lowered_ir_sha256', 'kernel_object_sha256',
    'toolchain_shas', 'required_tiers', 'capsule', 'target'])
def test_changed_current_boundary_refuses(field):
    left, right = observations(); loop, cert = changed_source(left, right)
    loop['tier_boundary']['common'][field] = 'changed'
    assert J.reusable_certificate(cert, loop, name='A', source='b'*16, instrument='a'*64) is None


@pytest.mark.parametrize('damage', ['instrument', 'actual_l3', 'current_l2', 'harness', 'proof', 'source', 'no_boundary'])
def test_unbound_reuse_refuses(damage):
    left, right = observations(); loop, cert = changed_source(left, right)
    if damage == 'instrument': cert['instrument'] = 'b'*64
    elif damage == 'actual_l3': cert['execution_digest'] = left['execution_digest']
    elif damage == 'current_l2': loop['execution_digest'] = 'f'*64
    elif damage == 'harness': loop['tier_boundary']['harness_sha256'] = 'f'*64
    elif damage == 'proof': cert['boundary_join']['scope'] = 'forged'
    elif damage == 'source': loop['tier_boundary']['common']['submission_digest'] = 'other'
    else: loop.pop('tier_boundary')
    assert J.reusable_certificate(cert, loop, name='A', source='b'*16, instrument='a'*64) is None


def test_historical_recovery_does_not_overwrite_new_l2_source(tmp_path, monkeypatch):
    monkeypatch.setenv('TMPDIR', str(tmp_path))
    left, right = observations(); ws = ledger(tmp_path, left)
    journal = completion(ws, left, right)
    loop, _ = changed_source(left, right)
    state = P._tier_state(ws); P._record(state, 'A', 'L2', loop); P._save_tier_state(ws, state)
    before_l2 = deepcopy(P._slots_ro(state, 'A', 'L2'))
    with pytest.raises(ValueError):
        J._apply_verified(ws, J.join(left, right), instrument='a'*64, completion=journal)
    assert J._apply_historical_verified(ws, J.join(left, right), instrument='a'*64, completion=journal) == ['A=pass']
    state = P._tier_state(ws)
    assert P._slots_ro(state, 'A', 'L2') == before_l2
    assert state['A']['L2'] == loop
    cert = P._slots_ro(state, 'A', 'L3')[right['execution_digest']]
    assert cert['digest'] == left['common']['submission_digest']
    assert cert['boundary_join']['L2'] == left
    assert J.reusable_certificate(cert, loop, name='A', source='b'*16, instrument='a'*64)


def test_historical_recovery_rejects_replaced_pending_source(tmp_path, monkeypatch):
    monkeypatch.setenv('TMPDIR', str(tmp_path))
    left, right = observations(); ws = ledger(tmp_path, left)
    journal = completion(ws, left, right)
    state = P._tier_state(ws)
    P._slots(state, 'A', 'L3')[left['execution_digest']]['digest'] = 'b'*16
    P._save_tier_state(ws, state); before = P._tier_state(ws)
    with pytest.raises(ValueError):
        J._apply_historical_verified(ws, J.join(left, right), instrument='a'*64, completion=journal)
    assert P._tier_state(ws) == before
