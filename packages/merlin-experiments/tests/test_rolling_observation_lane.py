"""Advisory large-case timing is its own lane: a grade's verdict never waits for it."""
import json
import threading
import time
from types import SimpleNamespace

import pytest

from merlin_experiments.phase1.feedback import completed_grade as CG
from merlin_experiments.phase1.feedback import joint_performance as P
from merlin_experiments.phase1.feedback import qa
from merlin_experiments.phase1.feedback.result_details import hydrate
from merlin_experiments.phase1.feedback import rolling_scalability as R
from merlin_experiments.phase1.feedback.visible_model_gate import publish
from merlin.common import cpu_pool
from merlin.targetgen import capsule_common, capsule_runner


@pytest.fixture
def scene(tmp_path, monkeypatch):
    monkeypatch.setattr(P, 'deadline', lambda timeout: 1e12)
    monkeypatch.setattr(P, 'selected', lambda: ({'sha256': 'selection'}, {}))
    monkeypatch.setattr(capsule_runner, 'suite_for', lambda target: 'suite')
    monkeypatch.setattr(capsule_common, 'load_capsule', lambda p: {'name': p.name})
    monkeypatch.setattr(cpu_pool, 'compute_capacity', lambda *a, **k: 4)
    digests = {}  # case -> executable digest; a changed one is a different executable
    monkeypatch.setattr(qa, '_execution_digest_from_result', lambda path: digests.get(path.parent.name, 'elf-a'))
    gates, calls = {}, []

    def attach(row, root, **kw):
        name = root.parent.name
        calls.append((root.parent.parent.parent.parent.name, name))
        gate = gates.get(name)
        if gate is not None:
            assert gate.wait(120)  # generous: a loaded CI host imports slowly
        row['performance'] = {'status': 'PROVISIONAL', 'estimated_whole_soc_cycles': 7, 'timed_in': str(root)}

    monkeypatch.setattr(P, 'attach', attach)
    cases = {name: tmp_path / 'frozen' / name for name in ('small', 'large')}
    selection = SimpleNamespace(cases=cases, context=SimpleNamespace(target='test'), verify=lambda: None)
    lane = R.ObservationLane()

    def grade(token, names=('small', 'large')):
        runs = tmp_path / 'run/_qa_work' / f'fruns_{token}'
        for name in names:
            path = runs / 'runs/suite' / name / 'capsule_result.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({'capsule': name, 'numeric': {'status': 'pass'}, 'status': 'pass',
                                        'tiers': {'L2': {'status': 'pass', 'engine': 'spike'}}}))
        feedback = R.PerformanceFeedback(selection, compiler=tmp_path / 'candidate', runs_root=runs,
                                         timeout=60, lane=lane)
        for name in names:
            feedback.completed({'capsule': name})
        return feedback, runs

    return SimpleNamespace(grade=grade, gates=gates, calls=calls, lane=lane, digests=digests,
                           selection=selection, tmp=tmp_path)


def _verdict(source, names=('small', 'large')):
    return {'per_capsule': [{'capsule': n, 'status': 'pass', 'tiers': {'L2': 'pass'}} for n in names],
            'graded_source_sha256': source, 'graded_at': f'at-{source}', 'snapshot_current': True,
            'n_passed': len(names), 'all_pass': None}


def _publish(ws, run, token, verdict, feedback):
    """What fast_grade does after its numerical rows are complete."""
    def write(path, document, expect=None):
        return publish(path, document, expect=expect)

    feedback.project(verdict, wait=False)
    archive = run / 'qa_history' / f'verdict_fast_{token}.json'
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_text(json.dumps(verdict))
    CG.publish(ws, verdict, token, write)
    feedback.follow_up(lambda projections: CG.follow_up(ws, archive, verdict, token, write, projections))
    return archive


def _ws(tmp_path):
    ws = tmp_path / 'ws'
    (ws / 'qa').mkdir(parents=True)
    return ws


def _rows(path):
    return {row['capsule']: row for row in hydrate(json.loads(path.read_text()), document_path=path)['per_capsule']}


def test_a_slow_observation_never_delays_the_verdict_and_lands_on_its_own_grade(scene):
    s = scene
    s.gates['large'] = threading.Event()
    ws = _ws(s.tmp)
    feedback, runs = s.grade(100)  # returns while the large case is still being timed
    verdict = _verdict('src-100')
    archive = _publish(ws, s.tmp / 'run', 100, verdict, feedback)
    live = _rows(ws / 'qa/verdict.json')
    assert 'performance' not in live['large'] and live['large']['status'] == 'pass'
    before = archive.stat().st_mtime_ns
    s.gates['large'].set()
    s.lane.drain()
    for document in (ws / 'qa/verdict.json', ws / 'qa' / CG.FILENAME, archive):
        rows = _rows(document)
        assert rows['large']['performance']['status'] == 'PROVISIONAL', document
        assert rows['small']['performance']['status'] == 'PROVISIONAL', document
        assert rows['large']['status'] == 'pass' and rows['large']['tiers'] == {'L2': 'pass'}
    assert json.loads((ws / 'qa/verdict.json').read_text())['n_passed'] == 2
    assert archive.stat().st_mtime_ns == before  # readers order the archive by modification time
    record = json.loads((runs / 'rolling_scalability_performance/large.json').read_text())
    assert record['qualification_authority'] is False


def test_a_follow_up_whose_receipt_changed_is_dropped_without_touching_counts(scene, monkeypatch, capsys):
    s = scene
    s.gates['large'] = threading.Event()
    ws = _ws(s.tmp)
    feedback, runs = s.grade(100)
    _publish(ws, s.tmp / 'run', 100, _verdict('src-100'), feedback)
    landed = R.PerformanceFeedback.landed

    def change_then_land(self, name, **kw):  # the receipt changes after its observation was checked
        if name == 'large':
            (runs / 'runs/suite/large/capsule_result.json').write_text(
                json.dumps({'capsule': 'large', 'numeric': {'status': 'fail'}}))
        return landed(self, name, **kw)

    monkeypatch.setattr(R.PerformanceFeedback, 'landed', change_then_land)
    s.gates['large'].set()
    s.lane.drain()
    for document in (ws / 'qa' / CG.FILENAME, ws / 'qa/verdict.json'):
        rows = _rows(document)
        assert 'performance' not in rows['large'] and rows['large']['status'] == 'pass'
        assert json.loads(document.read_text())['n_passed'] == 2
    assert 'follow-up refused: ValueError: rolling numerical result changed' in capsys.readouterr().out


def test_receipt_changed_during_observation_is_dropped(scene, monkeypatch, capsys):
    s = scene
    ws = _ws(s.tmp)
    s.gates['large'] = threading.Event()
    feedback, runs = s.grade(100)
    _publish(ws, s.tmp / 'run', 100, _verdict('src-100'), feedback)
    deadline = time.monotonic() + 120
    while ('fruns_100', 'large') not in s.calls and time.monotonic() < deadline:
        time.sleep(0.005)  # the observation has pinned its receipt and is timing the case
    (runs / 'runs/suite/large/capsule_result.json').write_text(json.dumps({'capsule': 'large'}))
    s.gates['large'].set()
    s.lane.drain()
    assert 'performance' not in _rows(ws / 'qa' / CG.FILENAME)['large']
    assert 'receipt changed during observation' in capsys.readouterr().out


def test_a_newer_grade_supersedes_an_in_flight_observation_of_the_same_case(scene, capsys):
    s = scene
    ws = _ws(s.tmp)
    s.gates['large'] = threading.Event()
    old, _ = s.grade(100)
    _publish(ws, s.tmp / 'run', 100, _verdict('src-100'), old)
    s.digests['large'] = 'elf-b'  # the newer grade compiled different bytes for this case
    new, _ = s.grade(200, names=('large',))
    s.gates['large'].set()
    s.lane.drain()
    _publish(ws, s.tmp / 'run', 200, _verdict('src-200', ('large',)), new)
    s.lane.drain()
    rows = _rows(ws / 'qa' / CG.FILENAME)
    assert rows['large'][CG.ORIGIN]['token'] == '200'
    assert rows['large']['performance']['timed_in'].endswith('fruns_200/runs/suite/large/generated')
    assert 'large' not in old.records
    assert 'dropped: superseded' in capsys.readouterr().out
    assert len(s.lane.latest) == 2 and not s.lane.busy_runs()


def test_a_follow_up_never_rewrites_a_row_a_newer_grade_owns(scene):
    s = scene
    ws = _ws(s.tmp)
    s.gates['small'] = threading.Event()
    old, _ = s.grade(100, names=('small',))
    _publish(ws, s.tmp / 'run', 100, _verdict('src-100', ('small',)), old)
    CG.publish(ws, _verdict('src-200', ('small',)), 200, lambda path, doc, expect=None: publish(path, doc))
    s.gates['small'].set()
    s.lane.drain()
    for document in (ws / 'qa' / CG.FILENAME, ws / 'qa/verdict.json'):
        row = _rows(document)['small']
        assert 'performance' not in row and json.loads(document.read_text())['graded_source_sha256'] == 'src-200'


def test_an_unchanged_executable_reuses_the_prior_observation(scene):
    s = scene
    ws = _ws(s.tmp)
    first, _ = s.grade(100)
    s.lane.drain()
    _publish(ws, s.tmp / 'run', 100, _verdict('src-100'), first)
    second, runs = s.grade(200)  # same executables: no second observation
    s.lane.drain()
    _publish(ws, s.tmp / 'run', 200, _verdict('src-200'), second)
    assert sorted(name for _, name in s.calls) == ['large', 'small']
    record = json.loads((runs / 'rolling_scalability_performance/large.json').read_text())
    assert record['reused_observation']['record']['path'].endswith('fruns_100/rolling_scalability_performance/large.json')
    assert record['numerical_result']['path'].endswith('fruns_200/runs/suite/large/capsule_result.json')
    assert _rows(ws / 'qa' / CG.FILENAME)['large']['performance']['status'] == 'PROVISIONAL'
    s.digests['large'] = 'elf-c'
    s.grade(300)
    s.lane.drain()
    assert sorted(name for _, name in s.calls) == ['large', 'large', 'small']


def test_an_equal_in_flight_observation_is_followed_not_repeated(scene):
    s = scene
    s.gates['large'] = threading.Event()
    first, _ = s.grade(100, names=('large',))
    second, _ = s.grade(200, names=('large',))
    assert len(s.calls) <= 1
    s.gates['large'].set()
    s.lane.drain()
    assert [name for _, name in s.calls] == ['large']
    assert 'large' in first.records and 'large' in second.records


def test_observations_lease_below_the_grade_class(scene, monkeypatch):
    s = scene
    seen = []
    monkeypatch.setattr(P, 'attach', lambda row, root, **kw: seen.append(cpu_pool.lease_priority()))
    with cpu_pool.cpu_priority(cpu_pool.GRADE):
        s.grade(100, names=('small',))
    s.lane.drain()
    assert seen == [cpu_pool.PREPARATION]


def test_retention_keeps_a_grade_whose_timing_is_still_observed(scene, monkeypatch):
    from merlin_experiments.phase1.feedback import scratch_retention

    s = scene
    run = s.tmp / 'run'
    monkeypatch.setattr(R, '_LANES', {str(run.resolve()): s.lane})
    s.gates['large'] = threading.Event()
    s.grade(100, names=('large',))
    owners = run / '_qa_work/background_snapshot_owners'
    owners.mkdir(parents=True)
    (owners / '100.json').write_text(json.dumps({'token': '100', 'state': 'complete',
                                                 'runs_dir': str(run / '_qa_work/fruns_100')}))
    slimmed = []
    monkeypatch.setattr(scratch_retention, 'slim_once', lambda root, **kw: slimmed.append(root) or 0)
    scratch_retention.prune_superseded_grades(run, keep_newest=0)
    assert slimmed == []
    s.gates['large'].set()
    s.lane.drain()
    scratch_retention.prune_superseded_grades(run, keep_newest=0)
    assert slimmed == [run / '_qa_work/fruns_100']
