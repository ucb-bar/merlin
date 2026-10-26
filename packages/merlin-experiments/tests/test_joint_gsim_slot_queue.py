"""A busy joint gSIM correctness slot is a queue, not a refusal.

Measured on a fresh four-hour run: 35 author gSIM self-checks were refused with "concurrency or worker
budget exhausted" while 9% of the correctness budget was spent, and 28 refused promotions left their
certificate records pending for bytes nothing would ever run again.
"""
import json
import os
import subprocess
import sys
import threading
import time

import pytest

from merlin_experiments.phase1 import joint_control as J
from merlin_experiments.phase1 import joint_gsim_queue as Q


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    names = [f'case{i}' for i in range(4)]
    policy = tmp_path / 'policy.json'
    policy.write_text(json.dumps({'version': 1, 'correctness_curriculum': {'original_capsules': names},
                                  'gsim_correctness': {'capsules': names, 'max_jobs': 1}}))
    state = tmp_path / 'joint_campaign.json'
    state.write_text(json.dumps({'policy': J.policy(policy), 'started_at_epoch': 1000}))
    monkeypatch.setenv(J.ENV, str(state))
    monkeypatch.delenv(Q.CLASS_ENV, raising=False)
    clock = {'now': 1100.0}
    monkeypatch.setattr(J.time, 'time', lambda: clock['now'])
    return names, state, clock


def _ledger():
    import fcntl

    with J.correctness_ledger().open() as handle:  # waiting threads rewrite it under the same lock
        fcntl.flock(handle, fcntl.LOCK_SH)
        return json.loads(handle.read())


def _until(predicate, limit=10.0):
    end = time.monotonic() + limit
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _waiter(capsule, **kwargs):
    outcome = {}

    def run():
        try:
            with J.physical_correctness(capsule, poll_s=0.01, **kwargs) as budget:
                outcome['budget'] = budget
        except BaseException as exc:  # noqa: BLE001 -- the test reads it
            outcome['error'] = exc
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, outcome


def test_author_waits_for_a_busy_slot_and_runs_when_it_frees(campaign, tmp_path):
    names, _, _ = campaign
    progress, queue = tmp_path / 'progress.json', {}
    with J.physical_correctness(names[0]):
        thread, outcome = _waiter(names[1], progress=progress, queue=queue)
        assert _until(lambda: len(_ledger()['waiting']) == 1 and Q.waiting(progress) is not None)
        assert Q.waiting(progress)['note'] == 'queued for a gSIM slot (0 ahead)'
        assert thread.is_alive() and 'error' not in outcome
    thread.join(10)
    assert outcome == {'budget': 1200}
    assert queue['state'] == 'admitted' and queue['ahead_at_enqueue'] == 0
    assert queue['note'].startswith('queued for a gSIM slot (0 ahead)')
    assert json.loads(progress.read_text())['status'] == Q.ADMITTED_STATUS
    ledger = _ledger()
    assert ledger['waiting'] == [] and ledger['active'] == []
    assert ledger['admissions'][-1]['capsule'] == names[1] and 'queued_s' in ledger['admissions'][-1]


def test_waiters_are_served_in_arrival_order(campaign):
    names, _, _ = campaign
    with J.physical_correctness(names[0]):
        first, _ = _waiter(names[1])
        assert _until(lambda: len(_ledger()['waiting']) == 1)
        second, _ = _waiter(names[2])
        assert _until(lambda: len(_ledger()['waiting']) == 2)
        ahead = {row['capsule']: Q.ahead(_ledger(), row['token']) for row in _ledger()['waiting']}
        assert ahead == {names[1]: 0, names[2]: 1}
    first.join(10)
    second.join(10)
    assert not first.is_alive() and not second.is_alive()
    assert [row['capsule'] for row in _ledger()['admissions']] == names[:3]


def test_authors_are_served_before_waiting_background_work():
    data = {'active': [], 'waiting': []}
    Q.enqueue(data, 'bg', capsule='b', request_class=Q.BACKGROUND, now=1)
    Q.enqueue(data, 'a1', capsule='x', request_class=Q.AUTHOR, now=2)
    Q.enqueue(data, 'a2', capsule='y', request_class=Q.AUTHOR, now=3)
    assert [Q.ahead(data, t) for t in ('a1', 'a2', 'bg')] == [0, 1, 2]
    assert Q.admissible(data, 'a1', 1) and not Q.admissible(data, 'bg', 2)
    assert Q.background_slots(data, 5) == 0  # an author is waiting
    Q.leave(data, 'a1')
    Q.leave(data, 'a2')
    assert Q.background_slots(data, 5) == 5 and Q.background_slots({'active': [{}] * 5}, 5) == 0
    # A launched child that has not taken its slot yet is busy; once registered it is counted once.
    assert Q.background_slots({'active': [{'request': 'j1'}]}, 3, starting=['j1', 'j2']) == 1
    Q.enqueue(data, 'bg', capsule='b', request_class=Q.BACKGROUND, now=9)  # idempotent: keeps its place
    assert len(data['waiting']) == 1


def test_background_work_does_not_wait_and_is_told_it_may_retry(campaign, monkeypatch):
    names, _, _ = campaign
    monkeypatch.setenv(Q.CLASS_ENV, Q.BACKGROUND)
    with J.physical_correctness(names[0]):
        with pytest.raises(Q.SlotBusy, match='concurrency full') as caught:
            with J.physical_correctness(names[1]):
                pytest.fail('background admitted past the cap')
        assert caught.value.retryable is True
        assert _ledger()['waiting'] == []

        @J.selfcheck_policy
        def run(argv):
            pytest.fail('a refused request reached the simulator')
        assert run(['--sim', 'gsim', '--capsules', names[1]]) == 2


def test_selfcheck_policy_marks_only_slot_busy_refusals_retryable(campaign, monkeypatch, tmp_path, capsys):
    names, _, _ = campaign
    monkeypatch.setenv(Q.CLASS_ENV, Q.BACKGROUND)
    out = tmp_path / 'out.json'

    @J.selfcheck_policy
    def run(argv):
        return 0
    with J.physical_correctness(names[0]):
        assert run(['--sim', 'gsim', '--capsules', names[1], '--out', str(out)]) == 2
    busy = json.loads(out.read_text())
    assert busy['joint_policy_refused'] is True and busy['joint_policy_retryable'] is True
    assert run(['--sim', 'gsim', '--capsules', 'unlisted', '--out', str(out)]) == 2
    assert 'joint_policy_retryable' not in json.loads(out.read_text())


def test_budget_exhaustion_is_still_refused_without_waiting(campaign):
    names, _, _ = campaign
    with J.physical_correctness(names[0]):
        pass
    data = _ledger()
    data['reserved_s'] = J.selected()['policy']['gsim_correctness']['worker_budget_s']
    J.correctness_ledger().write_text(json.dumps(data))
    started = time.monotonic()
    with pytest.raises(ValueError, match='worker budget exhausted'):
        with J.physical_correctness(names[1]):
            pytest.fail('exhausted budget admitted')
    assert time.monotonic() - started < 5 and _ledger()['waiting'] == []


def test_a_wait_never_outlives_the_campaign_clock(campaign):
    names, _, clock = campaign
    with J.physical_correctness(names[0]):
        thread, outcome = _waiter(names[1])
        assert _until(lambda: len(_ledger()['waiting']) == 1)
        clock['now'] = 1000 + J.selected()['policy']['duration_s']
        thread.join(10)
        assert isinstance(outcome.get('error'), ValueError) and 'deadline' in str(outcome['error'])
        assert _ledger()['waiting'] == []


def test_a_wait_ends_at_an_epoch_stop(campaign):
    names, state, _ = campaign
    with J.physical_correctness(names[0]):
        thread, outcome = _waiter(names[1])
        assert _until(lambda: len(_ledger()['waiting']) == 1)
        (state.parent / 'joint_epoch_stop.json').write_text(json.dumps(
            {'schema': 'merlin.joint_epoch_stop.v1', 'reason': 'reviewed checkpoint'}))
        thread.join(10)
        assert 'stop' in str(outcome.get('error'))


def test_a_dead_waiter_gives_up_its_place():
    child = subprocess.Popen([sys.executable, '-c', 'pass'])
    child.wait()
    data = {'active': [{'pid': child.pid, 'token': 't'}],
            'waiting': [{'pid': child.pid, 'token': 'w', 'class': Q.AUTHOR, 'seq': 0},
                        {'pid': os.getpid(), 'token': 'me', 'class': Q.AUTHOR, 'seq': 1}]}
    Q.settle(data)
    assert data['active'] == [] and [row['token'] for row in data['waiting']] == ['me']


def test_selfcheck_policy_states_the_wait_on_the_verdict(campaign, tmp_path):
    names, _, _ = campaign
    out, progress = tmp_path / 'verdict.json', tmp_path / 'progress.json'
    seen = []

    @J.selfcheck_policy
    def run(argv):
        seen.append(argv)
        out.write_text(json.dumps({'all_pass': True}))
        return 0
    result = {}
    holder = J.physical_correctness(names[0])
    holder.__enter__()
    thread = threading.Thread(target=lambda: result.update(rc=run(
        ['--sim', 'gsim', '--capsules', names[1], '--out', str(out), '--progress-out', str(progress)])))
    thread.start()
    try:
        assert _until(lambda: Q.waiting(progress) is not None)
    finally:
        holder.__exit__(None, None, None)
    thread.join(10)
    assert result == {'rc': 0}
    assert '--progress-out' in seen[0] and str(progress) in seen[0]
    verdict = json.loads(out.read_text())
    assert verdict['all_pass'] is True
    assert verdict['gsim_slot_queue']['note'].startswith('queued for a gSIM slot (0 ahead)')


def test_shim_reads_the_same_waiting_status():
    from merlin_experiments.phase1.tools import selfcheck as shim

    assert shim._GSIM_QUEUE_STATUS == Q.WAITING_STATUS
    assert shim._gsim_queue({'status': Q.WAITING_STATUS, 'gsim_slot_queue': {'ahead': 3}}) == {'ahead': 3}
    assert shim._gsim_queue_note({'ahead': 3}) == 'queued for a gSIM slot (3 ahead)'
    assert shim._gsim_queue({'status': 'grading', 'n_finished': 1}) is None


def test_waiting_handlers_do_not_hold_a_selfcheck_lane(tmp_path):
    from merlin_experiments.phase1.brokers import selfcheck as broker

    Q.publish(tmp_path / 'progress_a.json', Q.WAITING_STATUS, {'ahead': 0})
    (tmp_path / 'progress_b.json').write_text(json.dumps({'n_finished': 1}))
    active = {'req_a.json': {'diagnostic': False, 'variant': False},
              'req_b.json': {'diagnostic': False, 'variant': False}}
    assert broker._lane_holders(tmp_path, active) == [active['req_b.json']]
    assert broker._lane_admits(broker._lane_holders(tmp_path, active), diagnostic=False, variant=False,
                               ordinary_capacity=2, variant_capacity=1)


def test_simjob_keeps_promotions_queued_while_authors_wait(campaign):
    from merlin_experiments.phase1.brokers import simjob

    names, _, _ = campaign
    promoted = {'promoted': True, 'capsules': names[1]}

    def launchable(request, sim='gsim', running=None):
        return simjob._joint_gsim_launchable(request, sim, {}, running or {})
    gate = {}
    assert simjob._joint_gsim_launchable(promoted, 'gsim', gate, {})
    assert not simjob._joint_gsim_launchable(promoted, 'gsim', gate, {})  # one slot, spent by this pass
    assert not launchable(promoted, running={'j': {'correctness_gsim': True}})  # launched, not started yet
    with J.physical_correctness(names[0]):
        assert not launchable(promoted)  # every slot busy
        assert launchable({'capsules': names[1]})  # author: its child waits in the ledger
        assert launchable(promoted, 'spike')
    ledger = json.loads(J.correctness_ledger().read_text())
    ledger['waiting'] = [{'pid': os.getpid(), 'token': 'a', 'class': Q.AUTHOR, 'seq': 0}]
    J.correctness_ledger().write_text(json.dumps(ledger))
    assert not launchable(promoted)  # a waiting author goes first
    assert simjob._queue_wait_allowance({'capsules': names[1]}, 'gsim') > 0
    assert simjob._queue_wait_allowance(promoted, 'gsim') == 0


def test_a_slot_names_the_broker_request_that_took_it(campaign, monkeypatch):
    names, _, _ = campaign
    monkeypatch.setenv(Q.REQUEST_ENV, 'promo0_x')
    with J.physical_correctness(names[0]):
        assert _ledger()['active'][0]['request'] == 'promo0_x'
        assert J.correctness_background_slots(['promo0_x']) == 0
    assert J.correctness_background_slots(['promo0_x']) == 0  # still starting from the broker's view
    assert J.correctness_background_slots([]) == 1


def test_simjob_requeues_a_promotion_refused_for_a_busy_slot(tmp_path):
    from merlin_experiments.phase1.brokers import simjob

    snapshot = tmp_path / 'snapshot'
    snapshot.mkdir()
    (tmp_path / 'simrun_j').write_text('running')
    job = {'promoted': True, 'slots': (), 'snapshot_root': snapshot, 'aliases': ['k'], 'log': None}
    claimed, queued = {'j', 'k', 'other'}, {}
    refusal = {'all_pass': False, 'error': 'busy', 'joint_policy_refused': True, 'joint_policy_retryable': True}
    assert simjob._requeue_refused(tmp_path, 'j', job, refusal, claimed, queued)
    assert claimed == {'other'} and set(queued) == {'j', 'k'}
    assert not snapshot.exists() and not (tmp_path / 'simrun_j').exists()
    assert not list(tmp_path.glob('simresp_*'))
    for out, promoted in ((dict(refusal, joint_policy_retryable=False), True), (refusal, False),
                          ({'all_pass': True}, True)):
        assert not simjob._requeue_refused(tmp_path, 'j', dict(job, promoted=promoted), out, set(), {})


def test_the_simjob_reap_requeues_before_publishing():
    from pathlib import Path

    from merlin_experiments.phase1.brokers import simjob

    source = Path(simjob.__file__).read_text()
    reap = source[source.index('if rc == 124:'):]
    assert reap.index('_requeue_refused(') < reap.index('resp.write_text(')
    assert 'if not out.get("joint_policy_refused")' in reap  # a refusal is never a cached completion
    launch = source[source.index('if not _joint_gsim_launchable(r, sim, gsim_gate, running):'):]
    assert launch.index('continue') < launch.index('_reserve_veril_workers(')
