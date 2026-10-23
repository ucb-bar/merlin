"""Two bounded owners overlap a draining grade without losing or misattributing feedback."""
import json
import threading
import time
from types import SimpleNamespace

from merlin.targetgen.capsule_scheduling import run_pipelined
from merlin_experiments.phase1.feedback import dual_snapshots
from merlin_experiments.phase1.feedback.grade_progress import SnapshotProgress


def _guard():
    return SimpleNamespace(current=lambda: True, reason=lambda: None)


def test_new_snapshot_runs_while_old_drains_and_late_progress_cannot_clobber(tmp_path):
    ws, run = tmp_path / 'workspace', tmp_path / 'run'
    (ws / "submission").mkdir(parents=True)
    (ws / "submission/manifest.yaml").write_text("{}")
    stop, release_old = threading.Event(), threading.Event()
    old_started = threading.Event()
    source = 'a' * 64

    def first(tick):
        progress = SnapshotProgress(ws, run / 'old', source, _guard())
        progress.selected(1)

        def execute(cap, parallel):
            old_started.set()
            assert release_old.wait(3), 'new snapshot never ran beside the old drain'
            return {'capsule': 'old', 'status': 'pass'}

        run_pipelined([{'name': 'old'}], execute, max_workers=2)
        progress.finish({'snapshot_current': True})
        return {'all_pass': False}

    def second(tick):
        assert old_started.is_set()
        progress = SnapshotProgress(ws, run / 'new', source, _guard())
        progress.selected(2)
        progress.finish({'snapshot_current': True})
        stop.set()
        release_old.set()
        return {'all_pass': False}

    dual_snapshots.run(ws, run, stop, interval_s=0, first_grade=first, grade=second,
                       source_identity=lambda: source, poll_s=.002)
    assert json.loads((ws / 'qa/progress.json').read_text())['n_selected'] == 2
    assert json.loads((run / 'old/snapshot_progress.json').read_text())['n_selected'] == 1
    assert json.loads((run / 'new/snapshot_progress.json').read_text())['n_selected'] == 2
    owners = [json.loads(p.read_text()) for p in (run / '_qa_work/background_snapshot_owners').glob('*.json')]
    assert len(owners) == 2 and all(x['state'] == 'complete' for x in owners)
    assert len({x['token'] for x in owners}) == 2


def test_two_owner_limit_and_stop_joins_both_without_cancelling(tmp_path):
    ws, run = tmp_path / 'workspace', tmp_path / 'run'
    (ws / "submission").mkdir(parents=True)
    (ws / "submission/manifest.yaml").write_text("{}")
    stop, release, two_started = threading.Event(), threading.Event(), threading.Event()
    count, lock = [], threading.Lock()

    def grade(tick):
        def execute(cap, parallel):
            with lock:
                count.append(tick)
                if len(count) == 2:
                    two_started.set()
            assert release.wait(3)
            return {'capsule': str(tick), 'status': 'pass'}
        run_pipelined([{'name': str(tick)}], execute, max_workers=2)
        return {'all_pass': False}

    manager = threading.Thread(target=lambda: dual_snapshots.run(ws, run, stop, interval_s=0,
        first_grade=grade, grade=grade, source_identity=lambda: 'a' * 64, poll_s=.002))
    manager.start()
    assert two_started.wait(2)
    time.sleep(.03)
    assert len(count) == 2
    stop.set()
    time.sleep(.02)
    assert manager.is_alive(), 'stop abandoned a still-running grade'
    release.set()
    manager.join(2)
    assert not manager.is_alive()
    owners = list((run / '_qa_work/background_snapshot_owners').glob('*.json'))
    assert len(owners) == 2 and all(json.loads(p.read_text())['state'] == 'complete' for p in owners)


def test_stale_live_verdict_is_ignored_but_owned_archive_is_retained(tmp_path, monkeypatch):
    from merlin_experiments.phase1.feedback import snapshot_publication as publication
    from merlin_experiments.phase1.feedback.loop_grading import write_verdict
    from merlin_experiments.phase1 import joint_control

    monkeypatch.setattr(joint_control, 'stage_evidence', lambda row: row)
    ws = tmp_path / 'ws'
    publication.claim(ws, 'new')
    path = ws / 'qa/verdict.json'
    path.write_text(json.dumps({'current': 'new'}))
    with publication.scope(ws, 'old'):
        write_verdict(tmp_path / 'history/old.json', {'current': 'old'})
        write_verdict(path, {'current': 'old'})
    assert json.loads(path.read_text()) == {'current': 'new'}
    assert json.loads((tmp_path / 'history/old.json').read_text())['current'] == 'old'


def test_lifecycle_selects_dual_owners_only_for_focused_continuous_grading(tmp_path, monkeypatch):
    from merlin_experiments.phase1.feedback import lifecycle, joint_test_scope

    monkeypatch.setattr(joint_test_scope, 'focused', lambda: True)
    observed = []

    def run(ws, directory, stop, **kwargs):
        observed.append(kwargs['interval_s'])
        kwargs['first_grade'](123)
        kwargs['grade'](124)

    monkeypatch.setattr(dual_snapshots, 'run', run)
    handle = lifecycle.start_background(tmp_path / 'ws', tmp_path / 'run',
        lifecycle.GradeCadence(1200, False, 120), interval_grades=True,
        source_identity=lambda: 'source',
        fast_grade_callback=lambda *a: observed.append(('first', a[2])),
        grade_callback=lambda *a, **kw: observed.append(('later', a[2], kw['previous_scratch_key'])))
    lifecycle.stop_background(handle)
    assert observed == [120, ('first', 123), ('later', 124, None)]


def test_completed_failure_archive_survives_empty_live_progress(tmp_path):
    ws, run = tmp_path / 'ws', tmp_path / 'run'
    (ws / 'qa').mkdir(parents=True)
    (ws / 'qa/progress.json').write_text('{"per_capsule": []}')
    history = run / 'qa_history'
    history.mkdir(parents=True)
    (history / 'verdict_fast_1.json').write_text(json.dumps({
        'graded_source_sha256': 'a' * 64,
        'per_capsule': [{'capsule': 'seen', 'failure_detail': 'link failed: undefined reference to `done_symbol\''}]}))
    progress = SnapshotProgress(ws, run / '_qa_work/fruns_2', 'b' * 64, _guard())
    progress.selected(3)
    row = json.loads((ws / 'qa/progress.json').read_text())
    assert row['n_completed'] == 0
    assert row['previous_snapshot_failures']['graded_source_sha256'] == 'a' * 64
    assert row['previous_snapshot_failures']['n_observed_failures'] == 1
    assert row['previous_failures_are_current_verdict'] is False


def _cheap_covered(source):
    return {'snapshot_current': True, 'all_pass': None, 'graded_request_sha256': source,
            'joint_test_selection': {'curriculum': {
                'schema': 'merlin.joint_correctness_curriculum.v1', 'pending': [],
                'original_development_required': 1, 'original_unresolved': 0,
                'exact_correctness_reused': ['one']}}}


def test_unchanged_cheap_complete_source_stops_even_while_final_all_pass_unknown(tmp_path):
    ws, run = tmp_path / 'ws', tmp_path / 'run'
    (ws / "submission").mkdir(parents=True)
    (ws / "submission/manifest.yaml").write_text("{}")
    stop = threading.Event()
    calls = []
    source = 'a' * 64
    timer = None

    def grade(tick):
        nonlocal timer
        calls.append(tick)
        timer = threading.Timer(.05, stop.set)
        timer.start()
        return _cheap_covered(source)

    dual_snapshots.run(ws, run, stop, interval_s=0, first_grade=grade, grade=grade,
                       source_identity=lambda: source, poll_s=.002)
    timer.join()
    assert len(calls) == 1


def test_missing_curriculum_does_not_stop_and_launch_identity_is_not_result_identity(tmp_path):
    ws, run = tmp_path / 'ws', tmp_path / 'run'
    (ws / "submission").mkdir(parents=True)
    (ws / "submission/manifest.yaml").write_text("{}")
    stop = threading.Event()
    calls = []

    def grade(tick):
        calls.append(tick)
        if len(calls) == 2:
            stop.set()
        return {'all_pass': True, 'snapshot_current': True, 'graded_request_sha256': 'a' * 64}

    dual_snapshots.run(ws, run, stop, interval_s=0, first_grade=grade, grade=grade,
                       source_identity=lambda: 'a' * 64, poll_s=.002)
    assert len(calls) == 2
    assert dual_snapshots._completed_development_source(_cheap_covered('a' * 64), 'b' * 64) is None
    assert dual_snapshots._completed_development_source(_cheap_covered('b' * 64), 'b' * 64) == 'b' * 64
    invalid = _cheap_covered('a' * 64)
    invalid['joint_test_selection']['curriculum'].pop('pending')
    assert dual_snapshots._completed_development_source(invalid, 'a' * 64) is None


def test_missing_manifest_defers_without_identity_scan_failure_or_pass_then_admits_first(tmp_path):
    ws, run = tmp_path / 'ws', tmp_path / 'run'
    ws.mkdir()
    stop = threading.Event()
    calls, finished, identities = [], [], []

    def identity():
        identities.append(True)
        return 'a' * 64

    def first(tick):
        calls.append('first')
        return {'all_pass': False}

    manager = threading.Thread(target=lambda: dual_snapshots.run(
        ws, run, stop, interval_s=60, first_grade=first,
        grade=lambda tick: calls.append('later'), source_identity=identity,
        on_finished=lambda *args: finished.append(args), poll_s=.002))
    manager.start()
    path = run / 'background_snapshot_readiness.json'
    deadline = time.monotonic() + 2
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(.002)
    try:
        record = json.loads(path.read_text())
        assert record['status'] == 'waiting_for_manifest'
        assert record['all_pass'] is None and not record['grade_attempted']
        before = path.stat().st_mtime_ns
        time.sleep(.02)
        assert path.stat().st_mtime_ns == before  # no repeated record churn
        assert calls == finished == identities == []
        assert not (run / 'grade_health.json').exists()
        assert not list((run / '_qa_work/background_snapshot_owners').glob('*.json'))
        (ws / 'submission').mkdir()
        (ws / 'submission/manifest.yaml').write_text('{}')
        deadline = time.monotonic() + 2
        while not finished and time.monotonic() < deadline:
            time.sleep(.002)
        assert calls == ['first'] and len(finished) == 1
        assert finished[0][1]['all_pass'] is False and finished[0][2] is None
    finally:
        stop.set()
        manager.join(2)
    assert not manager.is_alive()


def test_manifest_disappearing_before_fast_grade_is_deferred_and_first_retry_preserved(tmp_path):
    from merlin_experiments.phase1.feedback.background_admission import SubmissionNotReady

    ws, run = tmp_path / 'ws', tmp_path / 'run'
    (ws / 'submission').mkdir(parents=True)
    (ws / 'submission/manifest.yaml').write_text('{}')
    stop = threading.Event()
    calls, finished = [], []

    def first(tick):
        calls.append(tick)
        if len(calls) == 1:
            raise SubmissionNotReady('no submission/manifest.yaml to grade yet')
        return {'all_pass': False}

    def completed(*args):
        finished.append(args)
        stop.set()

    dual_snapshots.run(ws, run, stop, interval_s=60, first_grade=first,
                       grade=lambda tick: (_ for _ in ()).throw(AssertionError('lost initial grade')),
                       source_identity=lambda: 'a' * 64, on_finished=completed, poll_s=.002)
    assert len(calls) == 2 and len(finished) == 1
    records = [json.loads(p.read_text()) for p in (run / '_qa_work/background_snapshot_owners').glob('*.json')]
    assert sorted(row['state'] for row in records) == ['complete', 'deferred']
    deferred = next(row for row in records if row['state'] == 'deferred')
    assert deferred['error'] is None and 'manifest' in deferred['deferred_reason']


def test_real_failure_retains_bounded_reason_and_reaches_health_callback(tmp_path):
    ws, run = tmp_path / 'ws', tmp_path / 'run'
    (ws / 'submission').mkdir(parents=True)
    (ws / 'submission/manifest.yaml').write_text('{}')
    stop = threading.Event()
    finished = []

    def grade(tick):
        raise ValueError('missing selected tool\n' + 'x' * 700)

    def completed(*args):
        finished.append(args)
        stop.set()

    dual_snapshots.run(ws, run, stop, interval_s=60, first_grade=grade, grade=grade,
                       source_identity=lambda: 'a' * 64, on_finished=completed, poll_s=.002)
    error = finished[0][2]
    assert error.startswith('ValueError: missing selected tool ')
    assert len(error) <= 500 and '\n' not in error
    records = list((run / '_qa_work/background_snapshot_owners').glob('*.json'))
    assert len(records) == 1 and json.loads(records[0].read_text())['error'] == error


def test_fast_grade_missing_manifest_is_explicit_readiness_not_generic_failure(tmp_path):
    import pytest

    from merlin_experiments.phase1.feedback.background_admission import SubmissionNotReady
    from merlin_experiments.phase1.feedback.loop_grading import fast_grade

    with pytest.raises(SubmissionNotReady, match='manifest.yaml'):
        fast_grade(tmp_path / 'workspace', tmp_path / 'run', 0, 1, inputs=None)
    assert not (tmp_path / 'run').exists()
