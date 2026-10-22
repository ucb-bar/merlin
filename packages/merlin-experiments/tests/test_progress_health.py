"""Progress health fixtures never launch compilers, agents or simulators."""
import json

from merlin_experiments.tracking.progress_health import ProgressHealth


def put(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))


def sample(tmp_path, private=False):
    return ProgressHealth(tmp_path, private).sample({'campaign': {'run_dir': str(tmp_path / 'run'),
                                                               'workspace': str(tmp_path / 'workspace')}})


def codes(result):
    return {a['code'] for a in result['alerts']}


def test_all_gated_verdict_is_not_numerical_progress_and_exception_is_separate(tmp_path):
    put(tmp_path / 'workspace/qa/verdict.json', {'all_pass': True, 'n_execution_passed': 0,
                                               'per_capsule': [{'capsule': 'a', 'numeric': {'status': 'skipped'}}]})
    put(tmp_path / 'workspace/.qa_channel/resp_one.json', {'error': 'broker: self-check aborted (timed out)',
        'requested_at_unix_ns': 1000000000, 'grade_started_at_unix_ns': 3000000000,
        'graded_at_unix_ns': 6000000000})
    result = sample(tmp_path)
    assert {'NO_OBSERVED_NUMERICAL_EXECUTION', 'GRADE_EXCEPTIONS', 'REQUEST_TIMEOUTS'} <= codes(result)
    timeout = next(a for a in result['alerts'] if a['code'] == 'REQUEST_TIMEOUTS')
    assert timeout['observations'][0]['stage_status'] == 'UNKNOWN'
    assert result['numeric']['failed'] == 0
    assert result['requests'][0]['queue_wall_s'] == 2
    assert result['requests'][0]['started_to_completion_wall_s'] == 3
    assert result['requests'][0]['stage_status'] == 'UNKNOWN'


def test_same_snapshot_failure_group_and_default_privacy(tmp_path):
    rows = [{'capsule': name, 'numeric': {'status': 'skipped'},
             'failure': {'plane': 'target_to_command_buffer', 'category': 'structural_invariant_violation',
                         'detail': 'same parse failure'}} for name in ['SECRET1', 'SECRET2']]
    put(tmp_path / 'workspace/.qa_channel/resp_one.json', {'submission_sha256': 'private-source', 'per_capsule': rows})
    rows[1]['failure']['detail'] = 'same boundary failure with different source pin'
    put(tmp_path / 'workspace/.qa_channel/resp_one.json', {'submission_sha256': 'private-source', 'per_capsule': rows})
    result = sample(tmp_path)
    assert 'REPEATED_PRE_EXECUTION_FAILURE' in codes(result)
    assert 'private-source' not in json.dumps(result) and 'SECRET' not in json.dumps(result)
    private = sample(tmp_path, True)
    assert next(a for a in private['alerts'] if a['code'] == 'REPEATED_PRE_EXECUTION_FAILURE')['count'] == 2
    rows[1]['failure']['category'] = 'different'
    put(tmp_path / 'workspace/.qa_channel/resp_one.json', {'submission_sha256': 'private-source', 'per_capsule': rows})
    assert 'REPEATED_PRE_EXECUTION_FAILURE' not in codes(sample(tmp_path))


def test_observed_numeric_failure_counts_execution_and_response_is_snapshot_bound(tmp_path):
    put(tmp_path / 'workspace/qa/progress.json', {'graded_source_sha256': 'old', 'updated_at': '2026-10-22T00:00:00Z',
                                                'n_execution_passed': 0})
    put(tmp_path / 'workspace/.qa_channel/resp_one.json', {'submission_sha256': 'new',
        'graded_at_unix_ns': 1792627800000000000,
        'per_capsule': [{'capsule': 'a', 'numeric': {'status': 'fail'}}]})
    result = sample(tmp_path)
    assert 'NO_OBSERVED_NUMERICAL_EXECUTION' not in codes(result)
    assert result['numeric']['completed'] == 1 and result['numeric']['failed'] == 1
    assert next(a for a in result['alerts'] if a['code'] == 'PROGRESS_BEHIND_SELFCHECK')['same_snapshot'] is False


def test_missing_and_oversized_records_remain_unknown(tmp_path):
    result = sample(tmp_path)
    assert result['records_observed'] == 0 and result['alerts'] == []
    assert result['numeric']['completed'] is None
    put(tmp_path / 'workspace/qa/verdict.json', {'error': 'x' * 2000})
    result = ProgressHealth(tmp_path, max_bytes=100).sample({'campaign': {'run_dir': str(tmp_path / 'run'),
                                                                       'workspace': str(tmp_path / 'workspace')}})
    assert 'record exceeds remaining byte bound' in result['notices']
    assert result['records_observed'] == 0


def archive(tmp_path, token, source, capsules, selected):
    work = tmp_path / 'run/_qa_work'
    runs = work / ('fruns_' + token)
    put(work / 'background_snapshot_owners' / (token + '.json'), {
        'schema': 'merlin.background_snapshot_owner.v1', 'token': token,
        'runs_dir': str(runs), 'source': 'separate-owner-admission-source', 'state': 'complete'})
    put(runs / 'snapshot_progress.json', {
        'schema': 'merlin.phase1.grade_progress.v1', 'graded_source_sha256': source,
        'n_selected': selected, 'n_completed': len(capsules), 'state': 'superseded_snapshot',
        'per_capsule': [{'capsule': c, 'status': 'fail', 'numeric_status': 'skipped',
                         'failure_plane': 'command_buffer_schema', 'failure_category': 'protocol_violation',
                         'failure_detail': 'missing property'} for c in capsules]})


def test_archives_keep_owner_provenance_and_deduplicate_exact_sources(tmp_path):
    archive(tmp_path, '1', 'same', ['a', 'b'], 173)
    archive(tmp_path, '2', 'same', ['b', 'c'], 153)
    archive(tmp_path, '3', 'other', ['a'], 153)
    put(tmp_path / 'workspace/qa/progress.json', {
        'schema': 'merlin.phase1.grade_progress.v1', 'graded_source_sha256': 'same',
        'snapshot_current': True, 'n_completed': 0, 'n_selected': 153, 'per_capsule': []})
    result = sample(tmp_path, True)
    assert result['completed_case_records'] == 4
    assert result['numeric']['completed'] == 0
    groups = {g['source_sha256']: g for g in result['snapshot_observations']}
    assert groups['same']['completed_case_records'] == 3
    assert len(groups['same']['owners']) == 2
    assert [o['recorded_n_selected'] for o in groups['same']['owners']] == [173, 153]
    assert groups['other']['matches_live_progress_source'] is False
    assert groups['other']['live_progress_claims_current'] is None
    assert 'REPEATED_PRE_EXECUTION_FAILURE' in codes(result)
    public = sample(tmp_path)
    assert all(g['source_sha256'] is None for g in public['snapshot_observations'])
    assert 'separate-owner-admission-source' not in json.dumps(public)


def test_archive_without_matching_owner_is_not_attributed(tmp_path):
    archive(tmp_path, '1', 'same', ['a'], 153)
    put(tmp_path / 'run/_qa_work/background_snapshot_owners/1.json', {
        'schema': 'merlin.background_snapshot_owner.v1', 'token': 'different', 'runs_dir': 'elsewhere'})
    result = sample(tmp_path)
    assert result['completed_case_records'] is None
    assert result['snapshot_observations'] == []
