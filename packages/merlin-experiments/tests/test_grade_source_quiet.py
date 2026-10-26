"""A due background grade waits for a quiet source, bounded, without a hot loop of digests."""
import math
import os
import threading
import time

from merlin_experiments.phase1.feedback import dual_snapshots, source_quiet
from merlin_experiments.phase1.feedback.source_quiet import QuietSource


class Clock:
    def __init__(self, wall_offset=1_000_000.0):
        self.now, self.offset = 0.0, wall_offset

    def __call__(self):
        return self.now

    def wall(self):
        return self.now + self.offset


def _gate(identity, clock, *, changed_at=None, quiet=30, cap=120, poll=1.0):
    last_change = None if changed_at is None else (lambda: changed_at[0] + clock.offset)
    return QuietSource(identity, quiet_s=quiet, max_delay_s=cap, poll_s=poll,
                       last_change=last_change, clock=clock, wall=clock.wall)


def _poll_until_ready(gate, clock, *, limit=400.0, step=1.0):
    while clock.now <= limit:
        ready, source = gate.ready()
        if ready:
            return clock.now, source
        clock.now += step
    return None, None


def test_quiet_source_launches_after_the_quiet_period():
    clock = Clock()
    changed_at = [0.0]  # the author's last edit: just now
    gate = _gate(lambda: 'a' * 64, clock, changed_at=changed_at)
    at, source = _poll_until_ready(gate, clock)
    assert source == 'a' * 64
    assert 30 <= at < 30 + source_quiet.MAX_CHECK_S


def test_an_edit_while_waiting_restarts_the_quiet_period():
    clock, changed_at, current = Clock(), [0.0], ['a']
    gate = _gate(lambda: current[0], clock, changed_at=changed_at)
    assert gate.ready()[0] is False
    clock.now = 20.0
    current[0], changed_at[0] = 'b', 18.0
    at, source = _poll_until_ready(gate, clock)
    assert source == 'b' and 48 <= at < 48 + source_quiet.MAX_CHECK_S


def test_idle_source_launches_at_once_including_the_first_grade():
    clock = Clock()
    clock.now = 500.0
    gate = _gate(lambda: 'a' * 64, clock, changed_at=[100.0])  # last edit long before the grade fell due
    assert gate.ready() == (True, 'a' * 64)
    # The next due grade of a source changed long ago (new identity, old metadata) does not wait either.
    gate.launched()
    clock.now = 900.0
    gate.source_identity = lambda: 'b' * 64
    assert gate.ready() == (True, 'b' * 64)


def test_continuous_edits_launch_at_the_cap():
    clock, counter = Clock(), [0]

    def identity():
        counter[0] += 1
        return str(counter[0])

    gate = _gate(identity, clock, changed_at=None)  # no metadata: every new identity is an edit now
    clock.now = 10.0
    at, source = _poll_until_ready(gate, clock)
    assert source is not None
    assert 10 + 120 <= at < 10 + 120 + source_quiet.MAX_CHECK_S
    # The next grade's cap counts from its own due time.
    gate.launched()
    clock.now = 400.0
    at, _ = _poll_until_ready(gate, clock, limit=800)
    assert 400 + 120 <= at < 400 + 120 + source_quiet.MAX_CHECK_S


def test_identity_is_sampled_at_a_bounded_rate_while_waiting():
    clock, calls = Clock(), []

    def identity():
        calls.append(clock.now)
        return str(len(calls))  # always changing: the gate waits the whole cap

    gate = _gate(identity, clock)
    at, _ = _poll_until_ready(gate, clock, step=0.25)  # polled four times a second
    assert at is not None
    assert len(calls) <= math.ceil(at / source_quiet.MAX_CHECK_S) + 1
    assert all(b - a >= source_quiet.MAX_CHECK_S for a, b in zip(calls, calls[1:]))


def test_zero_disables_the_debounce(monkeypatch):
    monkeypatch.setenv('MERLIN_GRADE_QUIET_S', '0')
    calls = []
    gate = source_quiet.for_workspace('/nonexistent', lambda: calls.append(1) or 'a', poll_s=1.0)
    assert not gate.enabled
    assert gate.ready() == (True, 'a') and gate.ready() == (True, 'a') and len(calls) == 2


def test_the_debounce_is_off_unless_an_operator_sets_a_quiet_period(monkeypatch):
    # A replay of two recorded runs made a 30 s quiet period a net loss (more idle time, later
    # feedback) for an author editing in bursts, so an unset knob launches due grades at once.
    monkeypatch.delenv('MERLIN_GRADE_QUIET_S', raising=False)
    gate = source_quiet.for_workspace('/nonexistent', lambda: 'a', poll_s=1.0)
    assert not gate.enabled and gate.ready() == (True, 'a')


def test_knobs_parse_non_negative_seconds_and_keep_defaults_otherwise(monkeypatch):
    monkeypatch.setenv('MERLIN_GRADE_QUIET_S', '12.5')
    monkeypatch.setenv('MERLIN_GRADE_QUIET_MAX_DELAY_S', '60')
    assert (source_quiet.quiet_s(), source_quiet.max_delay_s()) == (12.5, 60.0)
    for bad in ('-1', 'nan', 'inf', 'soon'):
        monkeypatch.setenv('MERLIN_GRADE_QUIET_S', bad)
        assert source_quiet.quiet_s() == source_quiet.QUIET_DEFAULT_S
    monkeypatch.delenv('MERLIN_GRADE_QUIET_S')
    monkeypatch.delenv('MERLIN_GRADE_QUIET_MAX_DELAY_S')
    assert (source_quiet.quiet_s(), source_quiet.max_delay_s()) == (0.0, 120.0)


def test_latest_change_skips_build_state_and_dates_deletions(tmp_path):
    root = tmp_path / 'submission'
    (root / 'pkg').mkdir(parents=True)
    (root / 'pkg/a.py').write_text('a')
    (root / 'build').mkdir()
    old = time.time() - 1000
    for path in (root / 'pkg/a.py', root / 'pkg', root / 'build', root):
        os.utime(path, (old, old))
    authored = source_quiet.latest_change_epoch(root)
    assert authored is not None
    (root / 'build/out.o').write_text('x')  # build state the identity ignores
    os.utime(root / 'build', (old, old))
    assert source_quiet.latest_change_epoch(root) == authored
    time.sleep(.05)  # past the filesystem's timestamp granularity
    (root / 'pkg/a.py').unlink()  # a deletion moves its parent directory
    assert source_quiet.latest_change_epoch(root) > authored
    assert source_quiet.latest_change_epoch(tmp_path / 'missing') is None


def _workspace(tmp_path):
    ws, run = tmp_path / 'ws', tmp_path / 'run'
    (ws / 'submission').mkdir(parents=True)
    (ws / 'submission/manifest.yaml').write_text('{}')
    return ws, run


def _run_until_first_launch(ws, run, identity, *, timeout=5.0):
    stop, launched = threading.Event(), []

    def grade(tick):
        launched.append(time.monotonic())
        stop.set()
        return {'all_pass': False}

    manager = threading.Thread(target=lambda: dual_snapshots.run(
        ws, run, stop, interval_s=0, first_grade=grade, grade=grade, source_identity=identity,
        poll_s=.01))
    started = time.monotonic()
    manager.start()
    manager.join(timeout)
    stop.set()
    manager.join(2)
    assert not manager.is_alive() and launched
    return launched[0] - started


def test_owner_loop_waits_for_a_freshly_written_source_to_settle(tmp_path, monkeypatch):
    monkeypatch.setenv('MERLIN_GRADE_QUIET_S', '0.4')
    monkeypatch.setenv('MERLIN_GRADE_QUIET_MAX_DELAY_S', '30')
    ws, run = _workspace(tmp_path)  # just written: the author is mid-edit
    assert _run_until_first_launch(ws, run, lambda: 'a' * 64) >= 0.3


def test_owner_loop_launches_a_long_idle_first_grade_at_once(tmp_path, monkeypatch):
    monkeypatch.setenv('MERLIN_GRADE_QUIET_S', '20')
    ws, run = _workspace(tmp_path)
    # ctime cannot be set back from a test; the stat walk itself is covered above.
    monkeypatch.setattr(source_quiet, 'latest_change_epoch', lambda root: time.time() - 600)
    assert _run_until_first_launch(ws, run, lambda: 'a' * 64) < 5


def test_owner_loop_grades_a_continuously_edited_source_at_the_cap(tmp_path, monkeypatch):
    monkeypatch.setenv('MERLIN_GRADE_QUIET_S', '20')
    monkeypatch.setenv('MERLIN_GRADE_QUIET_MAX_DELAY_S', '0.5')
    ws, run = _workspace(tmp_path)
    counter = [0]

    def identity():
        counter[0] += 1
        return f'{counter[0]:064x}'

    assert 0.4 <= _run_until_first_launch(ws, run, identity) < 4


def test_owner_loop_without_debounce_launches_at_once(tmp_path, monkeypatch):
    monkeypatch.setenv('MERLIN_GRADE_QUIET_S', '0')
    ws, run = _workspace(tmp_path)
    assert _run_until_first_launch(ws, run, lambda: 'a' * 64) < 2


def test_gate_reports_seconds_waited_only_while_a_sample_has_not_settled():
    clock, changed_at = Clock(), [0.0]
    gate = _gate(lambda: 'a', clock, changed_at=changed_at)
    assert gate.ready() == (False, 'a') and gate.waiting_s == 0
    clock.now = 10.0
    assert gate.ready() == (False, 'a') and gate.waiting_s == 10
    clock.now = 12.0  # within the check interval: no sample, the last observation stands
    assert gate.ready() == (False, None) and gate.waiting_s == 10
    clock.now = 30.0
    assert gate.ready() == (True, 'a') and gate.waiting_s is None
    gate.launched()
    assert gate.waiting_s is None


def test_a_waiting_grade_is_visible_to_the_operator_and_logged_once(tmp_path, monkeypatch, capsys):
    import json

    monkeypatch.setenv('MERLIN_GRADE_QUIET_S', '0.6')
    monkeypatch.setenv('MERLIN_GRADE_QUIET_MAX_DELAY_S', '30')
    ws, run = _workspace(tmp_path)  # just written: the author is mid-edit
    path = run / 'background_snapshot_readiness.json'
    stop, launched, seen = threading.Event(), threading.Event(), []
    writes = []
    real_replace = type(path).replace

    def replace(self, target):
        if type(path)(target) == path:
            writes.append(json.loads(self.read_text()))
        return real_replace(self, target)

    monkeypatch.setattr(type(path), 'replace', replace)

    def grade(tick):
        seen.append(json.loads(path.read_text()))
        launched.set()
        stop.set()
        return {'all_pass': False}

    manager = threading.Thread(target=lambda: dual_snapshots.run(
        ws, run, stop, interval_s=0, first_grade=grade, grade=grade, source_identity=lambda: 'a' * 64,
        poll_s=.01))
    manager.start()
    assert launched.wait(5)
    manager.join(2)
    waiting = [w for w in writes if w['status'] == 'waiting_for_quiet_source']
    assert waiting and waiting[0]['grade_attempted'] is False and waiting[0]['all_pass'] is None
    assert waiting[0]['quiet_s'] == 0.6 and waiting[0]['max_delay_s'] == 30
    assert all(type(w['waited_s']) is float and 0 <= w['waited_s'] < 0.6 for w in waiting)
    # One record per identity sample (every 0.2 s here), never one per 10 ms poll.
    assert len(waiting) <= 5
    assert seen[0]['status'] == 'manifest_present'  # the settled grade replaced the waiting record
    assert capsys.readouterr().out.count('waiting for the source to stay unchanged') == 1
