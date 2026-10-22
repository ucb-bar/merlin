"""Cross-engine source obligations never alias the two native executables."""
from copy import deepcopy
import json

import pytest

from merlin_experiments.phase1.feedback import promotion as P, tier_boundary_join as J
from test_candidate_execution_products import retained  # real neutral producer custody fixture


def observations():
    common = {'capsule': 'A', 'compiler_sha256': 'c'*64, 'source_ir_sha256': 'd'*64,
        'submission_digest': 'd'*16,
        'effective_command_buffer_sha256': 'e'*64, 'kernel_object_sha256': 'f'*64,
        'target': 'fixture', 'required_tiers': ['L0','L1','L2','L3'],
        'toolchain_shas': {'fixture_rtl':'a'*40}, 'raw_command_buffer_sha256':'a'*64,
        'lowered_ir_sha256':'b'*64}
    left = dict(schema='merlin.original_tier_boundary.v1', common=common,
        tier='L2', engine='spike', execution_digest='1'*64, elf_sha256='2'*64,
        harness_sha256='3'*64, result_sha256='4'*64)
    right = dict(deepcopy(left), tier='L3', engine='gsim', execution_digest='5'*64,
        elf_sha256='6'*64, harness_sha256='7'*64, result_sha256='8'*64)
    return left, right


def ledger(tmp_path, left):
    ws = tmp_path/'workspace'
    state = {}
    for tier, status in [('L2', 'pass'), ('L3', 'pending')]:
        P._record(state, 'A', tier, dict(status=status, digest=left['common']['submission_digest'],
            execution_digest=left['execution_digest'], instrument='a'*64,
            tier_boundary=left))
    P._save_tier_state(ws, state)
    return ws


def test_native_recorder_credits_actual_l3_key_without_spike_elf_alias(tmp_path):
    left, right = observations(); ws = ledger(tmp_path, left)
    row = dict(capsule='A', **{'pass': True}, barrier_tier='L3', barrier_status='pass',
        execution_digest=right['execution_digest'], tier_boundary=right)
    assert P.record_cert(ws, {'per_capsule':[row]}, 'L3', identity=left['execution_digest'],
        source_identity_verified=True) == ['A=pass']
    state = P._tier_state(ws)
    certs = P._slots_ro(state, 'A', 'L3')
    assert certs[left['execution_digest']]['status'] == 'joined'
    assert certs[right['execution_digest']]['status'] == 'pass'
    assert state['A']['L3']['execution_digest'] == right['execution_digest']
    assert state['A']['L2']['execution_digest'] == left['execution_digest']
    assert certs[right['execution_digest']]['boundary_join']['L2'] == left
    assert P.record_cert(ws, {'per_capsule':[row]}, 'L3', identity=left['execution_digest'],
        source_identity_verified=True) == []


@pytest.mark.parametrize('field', ['capsule','compiler_sha256','source_ir_sha256',
    'effective_command_buffer_sha256','kernel_object_sha256'])
def test_different_source_input_kernel_or_capsule_refuses(field):
    left, right = observations(); right['common'][field] = 'changed'
    with pytest.raises(ValueError, match='boundary changed'):
        J.join(left,right)


@pytest.mark.parametrize('damage', ['no_custody','wrong_execution','same_engine','missing_boundary'])
def test_unproven_cross_engine_result_leaves_pending(tmp_path, damage):
    left, right = observations(); ws=ledger(tmp_path,left)
    row=dict(capsule='A', **{'pass':True}, barrier_status='pass',
        execution_digest=right['execution_digest'],tier_boundary=right)
    if damage=='wrong_execution':row['execution_digest']='9'*64
    if damage=='same_engine':right['engine']='spike'
    if damage=='missing_boundary':row.pop('tier_boundary')
    assert not P.record_cert(ws,{'per_capsule':[row]},'L3',identity=left['execution_digest'],
        source_identity_verified=damage!='no_custody')
    assert P._tier_state(ws)['A']['L3']['status']=='pending'


def prepare_observation(retained):
    root, grade, _ = retained
    grade.update(capsule='A', toolchain_shas={'fixture_rtl':'a'*40},
        numeric={'status':'pass'}, trace_check={'status':'pass','violations':[],
            'trace_requirement_status':'pass','unavailable_requirements':[]})
    grade['tiers'].update(L0={'status':'pass'},L1={'status':'pass'})
    grade['tiers']['L2']['engine']='fixture'
    path=root.parent/'capsule_result.json';path.write_text(json.dumps(grade))
    (root.parent/'run_manifest.yaml').write_text(
        'target: fixture\nbenchmark: A\nmetadata:\n  required_oracle_tiers: [L0,L1,L2,L3]\n')
    return path,root,grade


def test_observation_uses_actual_producer_and_rejects_mutated_elf(retained):
    path,root,grade=prepare_observation(retained)
    observed=J.observe(path,tier='L2')
    assert observed['common']['compiler_sha256']==grade['compiler_source']['candidate_sha256']
    record=grade['compiler_source']['executed_programs'][0]
    (root/record['elf']['path']).write_bytes(b'changed ELF')
    with pytest.raises(ValueError):J.observe(path,tier='L2')


def test_observation_rejects_mutated_source_snapshot(retained):
    path,root,grade=prepare_observation(retained)
    (root/grade['compiler_source']['snapshot']/'compiler.py').write_text('changed compiler')
    with pytest.raises(ValueError,match='source snapshot changed'):J.observe(path,tier='L2')


@pytest.mark.parametrize('damage', ['pending_source', 'loop_source', 'loop_boundary', 'missing_required_tier'])
def test_cross_engine_join_requires_actual_recorded_source_boundary(tmp_path, damage):
    left, right = observations(); ws = ledger(tmp_path, left)
    state = P._tier_state(ws)
    if damage == 'pending_source':
        P._slots(state, 'A', 'L3')[left['execution_digest']]['digest'] = 'other'
    elif damage == 'loop_source':
        P._slots(state, 'A', 'L2')[left['execution_digest']]['digest'] = 'other'
    elif damage == 'loop_boundary':
        P._slots(state, 'A', 'L2')[left['execution_digest']]['tier_boundary'] = None
    else:
        for observation in (left, right):
            observation['common']['required_tiers'] = ['L0', 'L1', 'L2']
        for tier in ('L2', 'L3'):
            P._slots(state, 'A', tier)[left['execution_digest']]['tier_boundary'] = left
    P._save_tier_state(ws, state)
    before = P._tier_state(ws)
    row = dict(capsule='A', **{'pass': True}, barrier_tier='L3', barrier_status='pass',
        execution_digest=right['execution_digest'], tier_boundary=right)
    assert P.record_cert(ws, {'per_capsule': [row]}, 'L3', identity=left['execution_digest'],
        source_identity_verified=True) == []
    assert P._tier_state(ws) == before


def completion(ws, left, right):
    from merlin_experiments.phase1.feedback import completions
    return completions.persist(ws, 'fixture_job', {'per_capsule': [{
        'capsule': 'A', 'pass': True, 'barrier_tier': 'L3', 'barrier_status': 'pass',
        'execution_digest': right['execution_digest'],
    }]}, identity=left['execution_digest'], source_digest=left['common']['submission_digest'],
        source_identity_verified=True, cert_tier='L3', instrument='a'*64)


@pytest.mark.parametrize('damage', [None, 'instrument', 'source', 'public_copy', 'cert_execution', 'pending_source'])
def test_retained_recovery_reopens_original_private_judge(tmp_path, monkeypatch, damage):
    monkeypatch.setenv('TMPDIR', str(tmp_path))
    left, right = observations(); ws = ledger(tmp_path, left)
    path = completion(ws, left, right)
    if damage == 'public_copy':
        public = ws / path.name
        public.write_bytes(path.read_bytes()); path = public
    elif damage == 'pending_source':
        state = P._tier_state(ws)
        for tier in ('L2', 'L3'):
            P._slots(state, 'A', tier)[left['execution_digest']]['digest'] = 'other'
        P._save_tier_state(ws, state)
    elif damage:
        record = json.loads(path.read_text())
        if damage == 'instrument': record['instrument'] = 'b'*64
        elif damage == 'source': record['source_digest'] = 'other'
        else:
            from merlin_experiments.phase1.feedback import completions
            import hashlib
            record['verdict']['per_capsule'][0]['execution_digest'] = '9'*64
            record['verdict_sha256'] = hashlib.sha256(completions._encoded(record['verdict'])).hexdigest()
        path.write_text(json.dumps(record))
    before = P._tier_state(ws)
    if damage:
        with pytest.raises(ValueError):
            J._apply_verified(ws, J.join(left, right), instrument='a'*64, completion=path)
        assert P._tier_state(ws) == before
    else:
        assert J._apply_verified(ws, J.join(left, right), instrument='a'*64, completion=path) == ['A=pass']
        assert P._slots_ro(P._tier_state(ws), 'A', 'L3')[left['execution_digest']]['status'] == 'joined'


def test_reopened_source_digest_uses_original_ledger_stream(tmp_path):
    ws = tmp_path / 'workspace'; source = ws / 'submission'; source.mkdir(parents=True)
    (source / 'compiler.py').write_text('source bytes')
    (source / '__pycache__').mkdir(); (source / '__pycache__' / 'ignored.pyc').write_bytes(b'ignored')
    assert P.submission_tree_digest(source) == P.submission_digests(ws)[0]


@pytest.mark.parametrize('damage', ['cached_tier', 'engine_label'])
def test_observation_requires_original_measured_engine_role(retained, damage):
    path, root, grade = prepare_observation(retained)
    if damage == 'cached_tier': grade['tiers']['L2']['measured_now'] = False
    else: grade['tiers']['L2']['engine'] = 'different-engine'
    path.write_text(json.dumps(grade))
    with pytest.raises(ValueError): J.observe(path, tier='L2')


def whole_program_observation(retained):
    path, root, grade = prepare_observation(retained)
    for tier, reason in {
        'L0': 'command-buffer reference cannot interpret scalar regions of a whole-program ELF',
        'L1': 'command-buffer simulate cannot execute scalar regions of a whole-program ELF',
    }.items():
        grade['tiers'][tier] = dict(status='skipped', not_applicable=True, reason=reason)
    grade['numeric'].update(per_output={'Y0': dict(status='pass', n_elements=16)},
        missing_outputs=[], mismatch_count=0, golden_source='native independent capsule reference')
    return path, root, grade


def test_whole_program_na_observation_preserves_actual_product_custody(retained):
    path, root, grade = whole_program_observation(retained)
    path.write_text(json.dumps(grade))
    assert J.observe(path, tier='L2')['common']['compiler_sha256'] == grade['compiler_source']['candidate_sha256']
    record = grade['compiler_source']['executed_programs'][0]
    (root / record['elf']['path']).write_bytes(b'changed executable')
    with pytest.raises(ValueError):
        J.observe(path, tier='L2')


@pytest.mark.parametrize('damage', ['arbitrary_reason', 'one_tier_only', 'missing_output',
    'empty_output', 'missing_golden', 'numeric_mismatch', 'candidate_native_role'])
def test_whole_program_na_does_not_admit_unproved_skips(retained, damage):
    path, root, grade = whole_program_observation(retained)
    if damage == 'arbitrary_reason': grade['tiers']['L0']['reason'] = 'not implemented'
    if damage == 'one_tier_only': grade['tiers']['L1'] = {'status':'pass'}
    if damage == 'missing_output': grade['numeric']['missing_outputs'] = ['Y1']
    if damage == 'empty_output': grade['numeric']['per_output']['Y0']['n_elements'] = 0
    if damage == 'missing_golden': grade['numeric'].pop('golden_source')
    if damage == 'numeric_mismatch': grade['numeric']['mismatch_count'] = 1
    if damage == 'candidate_native_role': grade['candidate_native_execution'] = {'status':'recorded'}
    path.write_text(json.dumps(grade))
    with pytest.raises(ValueError, match='complete numerical and structural checks'):
        J.observe(path, tier='L2')
