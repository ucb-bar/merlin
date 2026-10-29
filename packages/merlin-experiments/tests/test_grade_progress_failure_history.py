"""A new running batch must not erase the last actionable source-bound failure."""
import json
from types import SimpleNamespace

from merlin_experiments.phase1.feedback import grade_progress as G


def test_new_empty_batch_keeps_prior_failures_without_current_failure_claim(tmp_path):
    workspace = tmp_path / 'workspace'
    (workspace / 'qa').mkdir(parents=True)
    path = workspace / 'qa/progress.json'
    path.write_text(json.dumps(dict(graded_source_sha256='a' * 64, per_capsule=[
        dict(capsule=name, status='fail', numeric_status='skipped', failure_plane='spike',
             failure_detail="link failed: object: undefined reference to `required_completion'")
        for name in ('one', 'two')])))
    calls = []
    guard = SimpleNamespace(current=lambda: calls.append('current') or True)
    progress = G.SnapshotProgress(workspace, tmp_path / 'runs', 'b' * 64, guard)
    progress.selected(16)
    result = json.loads(path.read_text())
    assert result['n_completed'] == 0 and result['all_pass'] is None
    assert result['observed_failure_summary']['n_observed_failures'] == 0
    previous = result['previous_snapshot_failures']
    assert previous['graded_source_sha256'] == 'a' * 64
    assert previous['n_observed_failures'] == 2
    assert previous['groups'][0]['missing_symbols'] == ['required_completion']
    assert previous['groups'][0]['n_observed'] == 2
    assert not result['previous_failures_are_current_verdict']
    assert calls == ['current']
    # Another empty batch preserves the same bounded old failure summary.
    G.SnapshotProgress(workspace, tmp_path / 'next', 'c' * 64, guard).selected(16)
    assert json.loads(path.read_text())['previous_snapshot_failures'] == previous


def test_missing_or_malformed_progress_has_no_old_authority(tmp_path):
    assert G._previous_failure_summary(tmp_path / 'missing') is None
    path = tmp_path / 'malformed'
    path.write_text('not json')
    assert G._previous_failure_summary(path) is None


def test_success_and_unrun_rows_never_become_shared_link_failure():
    value = G.observed_failures([
        {'capsule': 'passed', 'status': 'pass'},
        {'capsule': 'unrun', 'status': 'gated'},
    ], 'source')
    assert value['n_observed_failures'] == 0 and value['groups'] == []
    assert value['numerical_failure_inferred'] is False
