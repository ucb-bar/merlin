"""Startup observations preserve calls and retain the last blocked boundary."""
import json
import threading

import pytest
from merlin_experiments.phase1.telemetry import startup


def records(root):
    return [json.loads(line) for line in (root / 'rounds/author_startup.jsonl').read_text().splitlines()]


def test_nested_calls_keep_arguments_results_and_round_turn_identity(tmp_path):
    marker, secret = object(), 'private credential payload'
    calls = []

    def inner(arg, *, value):
        calls.append((arg, value))
        return marker

    @startup.observed
    def run(run_dir, rnd):
        startup.turn(2)
        return startup.call('sandbox_assembly', lambda: startup.call('runtime_verify', inner, marker, value=secret))

    assert run(tmp_path, 7) is marker
    assert calls == [(marker, secret)]
    rows = records(tmp_path)
    assert [(r['stage'], r['event']) for r in rows] == [
        ('sandbox_assembly', 'start'), ('runtime_verify', 'start'),
        ('runtime_verify', 'finish'), ('sandbox_assembly', 'finish')]
    assert all((r['round'], r['turn']) == (7, 2) for r in rows)
    assert rows[1]['parent_stage_id'] == rows[0]['stage_id']
    assert rows[2]['stage_id'] == rows[1]['stage_id']
    assert rows[3]['wall_s'] >= rows[2]['wall_s'] >= 0
    assert all(r['monotonic_s'] > 0 and r['at'].endswith('+00:00') for r in rows)
    assert secret not in json.dumps(rows)


def test_error_keeps_original_exception_and_hides_message(tmp_path):
    failure = RuntimeError('credential and private argv')

    @startup.observed
    def run(run_dir, rnd):
        def fail():
            raise failure
        return startup.call('final_answer_masks', fail)

    with pytest.raises(RuntimeError) as caught:
        run(tmp_path, 0)
    assert caught.value is failure
    rows = records(tmp_path)
    assert [r['event'] for r in rows] == ['start', 'error']
    assert rows[-1]['exception_type'] == 'RuntimeError'
    assert str(failure) not in json.dumps(rows)
    assert startup.call('unselected', lambda: 17) == 17
    assert len(records(tmp_path)) == 2


def test_blocked_callback_has_a_retained_start_before_release(tmp_path):
    entered, release = threading.Event(), threading.Event()

    @startup.observed
    def run(run_dir, rnd):
        startup.turn(1)
        def blocked():
            entered.set()
            assert release.wait(5)
        startup.call('snapshot_overlay', blocked)

    thread = threading.Thread(target=run, args=(tmp_path, 3))
    thread.start()
    try:
        assert entered.wait(5)
        rows = records(tmp_path)
        assert len(rows) == 1
        assert (rows[0]['event'], rows[0]['stage'], rows[0]['turn']) == ('start', 'snapshot_overlay', 1)
        assert 'wall_s' not in rows[0]
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    assert records(tmp_path)[-1]['event'] == 'finish'


def test_unavailable_diagnostics_do_not_change_callback(tmp_path):
    # A non-directory run root causes only the optional diagnostic writer to fail.
    root = tmp_path / 'file'
    root.write_text('unavailable')

    @startup.observed
    def run(run_dir, rnd):
        return startup.call('provider_popen', lambda: 'original result')

    assert run(root, 0) == 'original result'


def test_nested_observer_restores_outer_round_and_turn(tmp_path):
    @startup.observed
    def child(run_dir, rnd):
        startup.turn(5)
        startup.call('runtime_verify', lambda: None)

    @startup.observed
    def parent(run_dir, rnd):
        startup.turn(2)
        child(run_dir, 4)
        startup.call('provider_popen', lambda: None)

    parent(tmp_path, 1)
    assert [(r['round'], r['turn']) for r in records(tmp_path) if r['event'] == 'start'] == [(4, 5), (1, 2)]
