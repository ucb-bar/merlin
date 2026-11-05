"""The existing host selection accepts complete collections of bounded packets."""
from __future__ import annotations

import copy
import json

import pytest
from merlin_experiments.phase1.feedback import reference_experiments as reference
from test_reference_experiment_feedback import _fixture, _repin, _write

from merlin.verify.lean_contract import document_digest


def _shards(tmp_path, *, total=2, per_packet=1, requests=True):
    children, rows = [], []
    for start in range(0, total, per_packet):
        folder = tmp_path / f'shard-{len(children):04d}'
        folder.mkdir()
        child = _fixture(folder)
        template = child['packet']['tasks'][0]
        tasks = []
        end = min(start + per_packet, total)
        for number in range(start, end):
            task = copy.deepcopy(template)
            task['task_id'] = f'task-{number}'
            task['summary'] = f'Check selected mechanism {number}.'
            if requests:
                task['requests'][0]['request_id'] = f'request-{number}'
            else:
                task.update(requests=[], status='blocked')
            tasks.append(task)
        child['packet']['tasks'] = tasks
        child['packet']['coverage'] = dict(selected_start=start, selected_end=end, total_diagnoses=total,
                                            next_offset=end if end < total else None)
        _repin(child)
        children.append(child)
        rows.append(dict(packet=str(child['packet_path']), selection=str(child['path']),
                         selection_sha256=child['sha']))
    index = dict(schema=reference.SHARDS_SCHEMA, complete=True, tasks=total, shards=rows)
    path = tmp_path / 'packet-shards.json'
    return dict(path=path, sha=_write(path, index), index=index, children=children,
                compiler=children[0]['compiler'])


def _repin_index(fixture):
    for row, child in zip(fixture['index']['shards'], fixture['children'], strict=True):
        _repin(child)
        row.update(selection_sha256=child['sha'])
    fixture['sha'] = _write(fixture['path'], fixture['index'])


def test_index_is_inert_without_explicit_selection(tmp_path, monkeypatch):
    fixture = _shards(tmp_path)
    monkeypatch.setenv('MERLIN_REFERENCE_EXPERIMENTS', str(fixture['path']))
    assert reference.project(dict(reference_packet=str(fixture['path']))) is None
    assert reference.pending_requests(fixture['compiler']) == ()


def test_all_task_and_original_packet_identities_are_retained(tmp_path):
    fixture = _shards(tmp_path, total=35)
    with reference.invocation(fixture['path'], fixture['sha']):
        packet, context = reference._load(reference._current())
        assert [t['task_id'] for t in packet['tasks']] == [f'task-{i}' for i in range(35)]
        assert len(packet['source_packet_identities']) == 35
        assert context == fixture['children'][0]['packet']['context']
        for child in fixture['children']:
            task = child['packet']['tasks'][0]
            assert packet['task_packet_identities'][task['task_id']] == child['packet']['identity']
        requests = reference.pending_requests(fixture['compiler'])
        assert len(requests) == 32
        assert requests[31]['request_id'] == 'request-31'
        assert requests[31]['packet_identity'] == fixture['children'][31]['packet']['identity']
        found = reference.project(current_compiler_sha256=fixture['compiler'])
        assert found['packet_identity'] == fixture['children'][0]['packet']['identity']
        assert reference.pending_requests('f' * 64) == ()


def test_later_shard_observation_uses_original_packet_and_exact_index_journal(tmp_path):
    fixture = _shards(tmp_path)
    with reference.invocation(fixture['path'], fixture['sha']):
        packet, _ = reference._load(reference._current())
        planned = reference.pending_requests(fixture['compiler'])[1]
        request = dict(reference_experiment_request_id=planned['request_id'], submission='submission',
                       capsules='declared-case', sim='spike', tiers='L2')
        result = reference.observe_result(request, dict(submission_sha256=fixture['compiler'], sim='spike',
            variant=dict(applied=True, tuning_overrides=dict(retain=True), sha256='f' * 64),
            per_capsule=[dict(capsule='declared-case', numerical_execution_passed=True, status='passed')]),
            base_compiler_sha256=fixture['compiler'])
        assert result['packet_identity'] == fixture['children'][1]['packet']['identity']
        assert result['context_identity'] == packet['execution_context_identity']
        assert result['task_complete'] is False and result['mechanism_status'] == 'unknown'
    with reference.invocation(fixture['path'], fixture['sha']):
        assert [r['request_id'] for r in reference.pending_requests(fixture['compiler'])] == ['request-0']
        observed = reference.pending_requests(fixture['compiler'], include_observed=True)[1]
        assert observed['owner_observed'] is True
        journal = next(tmp_path.glob('.reference-experiments-*.json'))
        document = json.loads(journal.read_text())
        assert document['selection_sha256'] == fixture['sha']
        assert document['packet_identity'] == packet['identity']
        assert document['outcomes']['request-1']['packet_identity'] != packet['identity']


@pytest.mark.parametrize('mutation', [
    'incomplete', 'missing_last', 'wrong_total', 'coverage', 'selection_drift', 'packet_drift',
    'duplicate_task', 'duplicate_request', 'duplicate_selector', 'context', 'evaluator', 'visibility',
    'packet_pointer', 'oversize_packet', 'typed_context',
])
def test_shard_preflight_rejects_missing_mixed_changed_or_repeated_inputs(tmp_path, mutation):
    fixture = _shards(tmp_path)
    child = fixture['children'][1]
    if mutation == 'incomplete':
        fixture['index']['complete'] = False
    elif mutation == 'missing_last':
        fixture['index']['shards'].pop()
        fixture['children'].pop()
    elif mutation == 'wrong_total':
        fixture['index']['tasks'] = 3
    elif mutation == 'coverage':
        child['packet']['coverage']['selected_start'] = 0
    elif mutation == 'duplicate_task':
        child['packet']['tasks'][0]['task_id'] = 'task-0'
    elif mutation == 'duplicate_request':
        child['packet']['tasks'][0]['requests'][0]['request_id'] = 'request-0'
    elif mutation == 'duplicate_selector':
        fixture['index']['shards'][1] = copy.deepcopy(fixture['index']['shards'][0])
    elif mutation in ('context', 'evaluator', 'visibility', 'typed_context'):
        context = child['packet']['context']
        if mutation == 'context':
            context['source_sha256'] = 'f' * 64
        elif mutation == 'evaluator':
            context['evaluator']['identity'] = 'different-owner'
        elif mutation == 'visibility':
            context['visibility'] = 'public'
        else:
            context['tuning']['current']['retain'] = 0  # False and0 must remain distinct contexts.
        child['packet']['execution_context_identity'] = document_digest(context)
        child['packet']['tasks'][0]['requests'][0]['context_identity'] = document_digest(context)
    elif mutation == 'packet_pointer':
        fixture['index']['shards'][1]['packet'] = str(fixture['children'][0]['packet_path'])
    elif mutation == 'oversize_packet':
        child['packet']['tasks'] = [copy.deepcopy(child['packet']['tasks'][0]) for _ in range(17)]
        for i, task in enumerate(child['packet']['tasks']):
            task['task_id'] = f'extra-{i}'
            task['requests'] = []
    if mutation not in ('selection_drift', 'packet_drift', 'duplicate_selector'):
        _repin_index(fixture)
    elif mutation == 'duplicate_selector':
        fixture['sha'] = _write(fixture['path'], fixture['index'])
    elif mutation == 'selection_drift':
        child['path'].write_text('{}')
    else:
        child['packet_path'].write_text('{}')
    with pytest.raises((OSError, ValueError)):
        with reference.invocation(fixture['path'], fixture['sha']):
            pytest.fail('invalid index reached author work')
    assert reference.child_arguments() == ()


def test_changed_later_shard_suppresses_all_advice_and_is_rechecked_at_exit(tmp_path):
    fixture = _shards(tmp_path)
    with pytest.raises(ValueError):
        with reference.invocation(fixture['path'], fixture['sha']):
            fixture['children'][1]['packet_path'].write_text('{}')
            found = reference.project(current_compiler_sha256='f' * 64)
            assert found['status'] == 'unknown'
            assert 'prior_tasks' not in found
            assert reference.pending_requests(fixture['compiler']) == ()


def test_real_sized_index_has_complete_host_roster_and_bounded_author_guidance(tmp_path):
    fixture = _shards(tmp_path, total=635, per_packet=16, requests=False)
    workspace, run = tmp_path / 'author', tmp_path / 'run'
    workspace.mkdir()
    run.mkdir()
    (workspace / 'TASK.md').write_text('Ordinary task.\n')
    with reference.invocation(fixture['path'], fixture['sha']):
        packet, _ = reference._load(reference._current())
        assert len(packet['tasks']) == 635
        assert len(packet['task_packet_identities']) == 635
        found = reference.project(current_compiler_sha256='f' * 64)
        assert len(found['prior_tasks']) == 4
        assert found['total_prior_tasks'] == 635 and found['omitted_prior_tasks'] == 631
        assert found['required_selfchecks'] == []
        reference.stage_guidance(workspace, run)
        raw = (workspace / 'REFERENCE_EXPERIMENTS.json').read_bytes()
        staged = json.loads(raw)
        assert len(raw) < 64 << 10
        assert len(staged['tasks']) == 16
        assert staged['total_prior_tasks'] == 635 and staged['omitted_prior_tasks'] == 619
        assert staged['source_packet_count'] == 40
        assert staged['task_roster_identity'] == document_digest([t['task_id'] for t in packet['tasks']])
        assert raw == (run / 'REFERENCE_EXPERIMENTS.json').read_bytes()
        assert '"requests"' not in raw.decode() and '"settings"' not in raw.decode()
        assert str(tmp_path) not in raw.decode()


def test_duplicate_original_packet_rejected_even_with_distinct_selection_paths(tmp_path):
    fixture = _shards(tmp_path)
    first, second = fixture['children']
    second['selection'] = copy.deepcopy(first['selection'])
    fixture['index']['shards'][1].update(packet=str(first['packet_path']),
        selection_sha256=_write(second['path'], second['selection']))
    fixture['sha'] = _write(fixture['path'], fixture['index'])
    with pytest.raises(ValueError, match='duplicate_reference_shard_packet'):
        with reference.invocation(fixture['path'], fixture['sha']):
            pytest.fail('duplicate packet reached author work')


def test_each_shards_original_context_pins_are_reopened(tmp_path):
    fixture = _shards(tmp_path)
    leaf = tmp_path / 'owner-evidence.bin'
    leaf.write_bytes(b'original')
    import hashlib
    child = fixture['children'][1]
    child['selection']['context_pins'] = [dict(path=str(leaf), sha256=hashlib.sha256(leaf.read_bytes()).hexdigest())]
    _repin_index(fixture)
    with pytest.raises(ValueError, match='context_input_changed'):
        with reference.invocation(fixture['path'], fixture['sha']):
            leaf.write_bytes(b'changed')
            assert reference.project(current_compiler_sha256=fixture['compiler'])['status'] == 'unknown'


def test_index_identity_cannot_replace_original_packet_in_result_join(tmp_path):
    fixture = _shards(tmp_path)
    with reference.invocation(fixture['path'], fixture['sha']):
        packet, _ = reference._load(reference._current())
        planned = reference.pending_requests(fixture['compiler'])[1]
        metadata = {key: planned[key] for key in ('request_id', 'task_id', 'context_identity')}
        metadata['packet_identity'] = packet['identity']
        result = reference.observe_result(dict(tuning_experiment=dict(reference_experiment=metadata)),
            dict(submission_sha256=fixture['compiler']), base_compiler_sha256=fixture['compiler'])
        assert result is None
        assert not list(tmp_path.glob('.reference-experiments-*.json'))


def test_same_private_child_transport_selects_complete_index(tmp_path):
    fixture = _shards(tmp_path)
    with reference.invocation(fixture['path'], fixture['sha']):
        argv = reference.child_arguments()

    @reference.selected_child
    def child(argv):
        assert argv == ['ordinary']
        return reference.pending_requests(fixture['compiler'])

    assert [r['request_id'] for r in child([*argv, 'ordinary'])] == ['request-0', 'request-1']
    assert reference.child_arguments() == ()


def test_prior_projection_surfaces_later_construction_gaps_before_repeated_evidence(tmp_path):
    fixture = _shards(tmp_path, total=18, per_packet=16, requests=False)
    for task in fixture['children'][0]['packet']['tasks']:
        task.update(kind='missing_evidence', goals=['evidence.producer_capture_semantics'],
                    next_action=dict(route='evidence_or_legality', action='recover_producer_capture_semantics'))
    adapter, representation = fixture['children'][1]['packet']['tasks']
    adapter.update(kind='adapter_gap', goals=['schedule.consumed'],
        next_action=dict(route='adapter_or_preservation', action='bind_checked_schedule_consumer'))
    representation.update(kind='representation_gap', goals=['source.standalone_region'],
        next_action=dict(route='new_representation_or_transformation', action='outline_complete_source_region'))
    _repin_index(fixture)
    workspace, run = tmp_path / 'author', tmp_path / 'run'
    workspace.mkdir()
    run.mkdir()
    (workspace / 'TASK.md').write_text('Ordinary task.\n')
    with reference.invocation(fixture['path'], fixture['sha']):
        packet, _ = reference._load(reference._current())
        assert packet['tasks'][0]['task_id'] == 'task-0'
        found = reference.project(current_compiler_sha256='f' * 64)
        assert [t['task_id'] for t in found['prior_tasks'][:3]] == ['task-16', 'task-17', 'task-0']
        assert 'schedule.consumed' in found['pending_goals']
        assert found['total_prior_tasks'] == 18
        # The one-next-experiment and executable request order retain original semantics.
        assert reference.project(current_compiler_sha256=fixture['compiler'])['task_id'] == 'task-0'
        reference.stage_guidance(workspace, run)
        staged = json.loads((workspace / 'REFERENCE_EXPERIMENTS.json').read_text())
        assert [t['task_id'] for t in staged['tasks'][:3]] == ['task-16', 'task-17', 'task-0']
        assert staged['task_roster_identity'] == document_digest([t['task_id'] for t in packet['tasks']])
