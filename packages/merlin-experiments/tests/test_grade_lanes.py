"""Duration lanes: a grade's long cases drain without holding the next grade or its short results."""
import json
import threading

from merlin.common import cpu_pool
from merlin.targetgen.capsule_scheduling import observe_admission, rolling_admission, run_pipelined
from merlin_experiments.phase1.feedback import dual_snapshots, grade_lanes


def test_cases_split_by_estimate_and_by_their_own_elapsed_time():
    now = 1000.0
    boundary = 120.0  # the grade cadence
    assert not grade_lanes.is_long(30.0, now, boundary, now=now)  # priced short
    assert grade_lanes.is_long(900.0, now, boundary, now=now)  # priced long
    assert not grade_lanes.is_long(None, now, boundary, now=now)  # unpriced starts short once
    assert grade_lanes.is_long(None, now - 121, boundary, now=now)  # ... and is reclassified by elapsed time
    assert grade_lanes.is_long(30.0, now - 121, boundary, now=now)  # a mispriced short case too
    assert not grade_lanes.is_long(900.0, now, 0, now=now)  # no cadence, no lanes
    tracker = grade_lanes.InFlight('run')
    tracker.update({'big': 900.0, 'small': 1.0})
    assert tracker.long_cases(boundary) == {'big'} and not tracker.long_lane(boundary)
    tracker.update({'big': 900.0})
    assert tracker.long_lane(boundary)
    tracker.update({})
    assert not tracker.long_lane(boundary)  # nothing running is not a lane


def test_the_scheduler_reports_running_cases_priced_like_admission():
    seen = []
    with observe_admission(lambda reason: None, running=seen.append), \
            rolling_admission(admission_costs={'priced': 50.0}):
        run_pipelined([{'name': 'priced'}, {'name': 'unpriced'}], lambda cap, n: {'capsule': cap['name']},
                      max_workers=1)
    assert seen[0] == {'priced': 50.0} and seen[-1] == {}
    assert {'unpriced': None} in seen


def test_a_long_case_running_elsewhere_is_withheld_only_while_it_is_long():
    older, newer, other_run = grade_lanes.InFlight('run'), grade_lanes.InFlight('run'), grade_lanes.InFlight('x')
    older.update({'model': 3600.0, 'gemm': 1.0})
    other_run.update({'conv': 3600.0})
    with grade_lanes.scope(120, newer):
        assert grade_lanes.withheld({'name': 'model'}) == grade_lanes.WITHHELD
        assert grade_lanes.withheld({'name': 'gemm'}) is None  # short work is cheap to repeat on new bytes
        assert grade_lanes.withheld({'name': 'conv'}) is None  # another run's grades are not this run's
        assert grade_lanes.relevance() is grade_lanes.withheld
    with grade_lanes.scope(120, older):
        assert grade_lanes.withheld({'name': 'model'}) is None  # never withheld from its own grade
    older.update({})
    with grade_lanes.scope(120, newer):
        assert grade_lanes.withheld({'name': 'model'}) is None  # finished: the next grade runs it again


def test_an_older_grade_skips_only_cases_a_newer_grade_already_started():
    older, newer = grade_lanes.InFlight('run', order=100), grade_lanes.InFlight('run', order=200)
    elsewhere = grade_lanes.InFlight('x', order=300)
    newer.update({'gemm': 1.0})
    newer.update({})  # finished: still started by the newer snapshot
    elsewhere.update({'conv': 1.0})
    with grade_lanes.scope(120, older):
        assert grade_lanes.withheld({'name': 'gemm'}) == grade_lanes.NEWER  # redundant, not obsolete
        assert grade_lanes.withheld({'name': 'conv'}) is None  # another run's grade is not this run's
        assert grade_lanes.withheld({'name': 'model'}) is None  # nobody newer started it: run it here
    older.update({'model': 1.0})
    with grade_lanes.scope(120, newer):
        assert grade_lanes.withheld({'name': 'model'}) is None  # the newest snapshot never defers to an older


def test_dedupe_lets_the_older_grade_finish_what_the_newer_has_not_reached():
    older, newer = grade_lanes.InFlight('dedupe', order=1), grade_lanes.InFlight('dedupe', order=2)
    newer.update({'a': 1.0, 'b': 1.0})
    ran = []
    with grade_lanes.scope(120, older), observe_admission(lambda reason: None):
        from merlin.targetgen.capsule_scheduling import admission_relevance

        with admission_relevance(superseded=grade_lanes.relevance()):
            rows = run_pipelined([{'name': n} for n in 'abc'],
                                 lambda cap, n: ran.append(cap['name']) or {'capsule': cap['name'], 'status': 'pass'},
                                 max_workers=2)
    assert ran == ['c']
    gated = {row['capsule']: row for row in rows if row.get('status') == 'gated'}
    assert set(gated) == {'a', 'b'} and gated['a']['failure']['detail'] == grade_lanes.NEWER


def test_a_long_lane_frees_the_owner_and_the_next_grade_runs_its_short_cases(tmp_path):
    ws, run = tmp_path / 'ws', tmp_path / 'run'
    (ws / 'submission').mkdir(parents=True)
    (ws / 'submission/manifest.yaml').write_text('{}')
    stop, release = threading.Event(), threading.Event()
    lock, grades, results, priorities = threading.Lock(), [], {}, []
    second_done = threading.Event()

    def grade(tick):
        with lock:
            grades.append(tick)
            first = len(grades) == 1

        def execute(cap, parallel):
            if cap['name'] == 'model':
                assert release.wait(120)
                priorities.append(cpu_pool.lease_priority())
            return {'capsule': cap['name'], 'status': 'pass'}

        with rolling_admission(admission_costs={'model': 3600.0, 'gemm': 1.0}):
            rows = run_pipelined([{'name': 'model'}, {'name': 'gemm'}], execute, max_workers=2)
        results[tick] = {row['capsule']: row for row in rows}
        if not first:
            second_done.set()
        return {'all_pass': False}

    manager = threading.Thread(target=lambda: dual_snapshots.run(
        ws, run, stop, interval_s=0.05, first_grade=grade, grade=grade,
        source_identity=lambda: 'a' * 64, poll_s=.002))
    manager.start()
    try:
        # Two owners was the limit; the first grade still holds its long case, yet later grades run.
        assert second_done.wait(60), 'the next grade waited behind a long case'
        first = grades[0]
        assert first not in results
        owners = {path.stem: json.loads(path.read_text())
                  for path in (run / '_qa_work/background_snapshot_owners').glob('*.json')
                  if not path.name.endswith('.next.json')}
        assert owners[str(first)]['state'] == 'long_lane'
        assert owners[str(first)]['lease_priority'] == cpu_pool.PREPARATION
        later = results[grades[1]]
        assert later['gemm']['status'] == 'pass'
        assert later['model']['status'] == 'gated'  # never a second concurrent run of the long case
        assert later['model']['background_deferred'] == grade_lanes.WITHHELD
    finally:
        release.set()
        stop.set()
        manager.join(10)
    assert not manager.is_alive()
    assert results[first]['model']['status'] == 'pass'  # started long work always finishes
    assert priorities == [cpu_pool.PREPARATION]

