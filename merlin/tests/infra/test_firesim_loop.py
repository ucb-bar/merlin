"""The loop policing itself: the four guards the shared FPGA queue does not provide.

Every test here runs with the queue MOCKED. Nothing in this file may reach hardware: the board is
shared, and a test suite that submits is a test suite that cannot be run while someone is measuring.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

# This target-specific queue governor is example-owned, not part of the installed core.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from examples.gemmini.phase2 import firesim_loop as FL  # noqa: E402
from merlin.perf.firesim_batch import BatchMember
from merlin.perf.firesim_checkpoint import CheckpointError, QueueHost, SubmissionEvidence

_PROGRAM = "a" * 64
_WEIGHTS = "b" * 64
_ME = "loop-user"
_OTHER = "another-user"


def _status(*rows: str, daemon: str = "ALIVE") -> str:
    header = f"[firesim-queue] daemon: {daemon} (heartbeat 1s ago)\n" + (
        "   id user         prio state      phase        wall_s  rc cmd\n"
    )
    return header + "".join(row if row.endswith("\n") else row + "\n" for row in rows)


def _row(job_id: int, user: str, state: str, *, priority: int = 5, cmd: str = "runworkload-full workload=w") -> str:
    return f"{job_id:>5} {user:<12} {priority:>4} {state:<10} {'-':<11} {12:>8} {'-':>3} {cmd}"


def _ledger(tmp_path: Path, run_id: str = "run1") -> FL.BudgetLedger:
    return FL.BudgetLedger(tmp_path / "spend.jsonl", run_id=run_id)


def _tripwire(tmp_path: Path, limit: int = 3) -> FL.FailureTripwire:
    return FL.FailureTripwire(tmp_path / "tripwire.json", limit=limit)


def _governor(tmp_path, *, budget=None, limit=3, now=1_000_000.0, run_id="run1") -> FL.LoopGovernor:
    return FL.LoopGovernor(
        budget=budget or FL.FpgaBudget(seconds_per_day=10_000.0, seconds_per_run=5_000.0),
        ledger=_ledger(tmp_path, run_id),
        tripwire=_tripwire(tmp_path, limit),
        user=_ME,
        clock=lambda: now,
    )


# ----------------------------------------------------------------- the measured cost model
def test_cost_model_reproduces_the_observed_job():
    """31 windows in 515 s is the largest batch actually run; the model must land on it."""
    assert FL.estimate_job_seconds(31) == pytest.approx(515.5, abs=1.0)
    assert FL.estimate_job_seconds(1) == pytest.approx(FL.FIXED_JOB_SECONDS + FL.SECONDS_PER_WINDOW)


def test_batching_a_generation_is_an_order_of_magnitude_cheaper_than_one_job_per_candidate():
    serial, batched = FL.serial_vs_batched_seconds(31)
    assert serial > 9 * batched
    # And the generation prices its order-effect control, which link_batch appends.
    assert FL.generation_window_count(31) == 32
    assert FL.affordable_window_count(515.5) == 31


# ----------------------------------------------------------------- reading the live queue
def test_queue_status_is_parsed_structurally_including_wrapped_commands():
    text = _status(
        _row(700, _OTHER, "RUNNING"),
        _row(701, _ME, "QUEUED", priority=0, cmd="/bin/sh -c 'set -e"),
        "        unset CONDA_PREFIX CONDA_DEFAULT_ENV",
        "        exec make'",
    )
    status = FL.parse_queue_status(text)
    assert status.daemon_alive is True
    assert [job.job_id for job in status.jobs] == [700, 701]
    assert status.jobs[1].priority == 0 and status.jobs[1].state == "QUEUED"
    # the wrapped command lines belong to the job above them, not to a phantom job
    assert "unset CONDA_PREFIX" in status.jobs[1].command


def test_a_status_line_that_cannot_be_classified_is_refused_not_skipped():
    """A dropped line is a queued job the contention guard would never see."""
    with pytest.raises(FL.QueueStatusUnreadable):
        FL.parse_queue_status("[firesim-queue] daemon: ALIVE\nwhat is this line\n")


def test_an_unknown_state_spelling_stays_a_job_row():
    status = FL.parse_queue_status(_status(_row(702, _OTHER, "PREEMPTED")))
    assert status.jobs[0].state == "PREEMPTED"


def test_read_queue_status_asks_the_queue_for_status_and_nothing_else():
    seen: list[list[str]] = []

    def runner(argv, **_kwargs):
        seen.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, _status(), "")

    FL.read_queue_status("/opt/queue/bin/firesim-queue", runner=runner)
    assert seen == [["/opt/queue/bin/firesim-queue", "status"]]


# ----------------------------------------------------------------- guard 1: the budget
def test_the_budget_refuses_a_job_that_would_pass_the_daily_limit(tmp_path):
    governor = _governor(tmp_path, budget=FL.FpgaBudget(seconds_per_day=600.0, seconds_per_run=600.0))
    idle = FL.parse_queue_status(_status())
    # 40 windows is 619 s of board time and does not fit the day at all
    assert governor.authorize(window_count=40, priority=0, status=idle).action == FL.ACTION_HALT
    # one 20-window job (389 s) fits; the second does not
    assert governor.authorize(window_count=20, priority=0, status=idle).allowed
    reservation = governor.reserve(20)
    verdict = governor.authorize(window_count=20, priority=0, status=idle)
    assert verdict.action == FL.ACTION_HALT and verdict.refusal == "budget_exhausted_day"
    with pytest.raises(FL.BudgetExhausted):
        governor.require(window_count=20, priority=0, status=idle)
    assert reservation.startswith("run1-")


def test_the_run_budget_binds_even_when_the_day_has_room(tmp_path):
    governor = _governor(tmp_path, budget=FL.FpgaBudget(seconds_per_day=100_000.0, seconds_per_run=600.0))
    idle = FL.parse_queue_status(_status())
    governor.reserve(20)
    verdict = governor.authorize(window_count=20, priority=0, status=idle)
    assert verdict.refusal == "budget_exhausted_run"


def test_the_ledger_survives_a_restart_so_a_new_process_cannot_re_grant_the_day(tmp_path):
    first = _ledger(tmp_path, "run1")
    first.reserve(400.0, windows=21, now=1_000.0)
    # a different run, a fresh object, the same day
    second = _ledger(tmp_path, "run2")
    assert second.spent_since(0.0) == pytest.approx(400.0)
    assert second.spent_this_run() == 0.0


def test_spend_is_reconciled_from_the_estimate_to_the_observed_wall(tmp_path):
    ledger = _ledger(tmp_path)
    reservation = ledger.reserve(FL.estimate_job_seconds(31), windows=32, now=1_000.0)
    ledger.settle(reservation, 515.0, now=1_100.0, job_id=734)
    assert ledger.spent_since(0.0) == pytest.approx(515.0)
    assert ledger.spent_this_run() == pytest.approx(515.0)
    with pytest.raises(ValueError):
        ledger.settle("run1-99", 10.0, now=1_200.0)


def test_spend_older_than_a_day_stops_counting_against_the_daily_budget(tmp_path):
    ledger = _ledger(tmp_path)
    ledger.reserve(9_000.0, windows=31, now=1_000.0)
    governor = FL.LoopGovernor(
        budget=FL.FpgaBudget(seconds_per_day=10_000.0, seconds_per_run=10_000.0),
        ledger=ledger,
        tripwire=_tripwire(tmp_path),
        user=_ME,
        clock=lambda: 1_000.0 + 2 * 86_400.0,
    )
    assert governor.authorize(window_count=31, priority=0, status=FL.parse_queue_status(_status())).allowed


# ----------------------------------------------------------------- guard 2: priority 0
def test_a_non_zero_priority_is_refused(tmp_path):
    governor = _governor(tmp_path)
    idle = FL.parse_queue_status(_status())
    for priority in (5, 10, 20):
        verdict = governor.authorize(window_count=4, priority=priority, status=idle)
        assert verdict.action == FL.ACTION_HALT and verdict.refusal == "priority_not_zero"
    with pytest.raises(FL.PriorityRefused):
        governor.require(window_count=4, priority=FL.QUEUE_DEFAULT_PRIORITY, status=idle)
    assert FL.LOOP_PRIORITY == 0 and FL.LOOP_PRIORITY < FL.QUEUE_DEFAULT_PRIORITY


# ----------------------------------------------------------------- guard 3: yield to other users
def test_another_users_queued_job_makes_the_loop_yield(tmp_path):
    governor = _governor(tmp_path)
    status = FL.parse_queue_status(_status(_row(800, _OTHER, "QUEUED")))
    verdict = governor.authorize(window_count=4, priority=0, status=status)
    assert verdict.action == FL.ACTION_YIELD and verdict.refusal == "queue_contended"
    assert verdict.detail["users"] == [_OTHER]
    with pytest.raises(FL.QueueContended):
        governor.require(window_count=4, priority=0, status=status)


def test_our_own_queued_job_is_not_someone_to_yield_to(tmp_path):
    governor = _governor(tmp_path)
    status = FL.parse_queue_status(_status(_row(801, _ME, "QUEUED")))
    assert governor.authorize(window_count=4, priority=0, status=status).allowed


def test_a_dead_daemon_is_a_yield(tmp_path):
    governor = _governor(tmp_path)
    status = FL.parse_queue_status(_status(daemon="DOWN"))
    assert governor.authorize(window_count=4, priority=0, status=status).refusal == "queue_daemon_down"


# ----------------------------------------------------------------- guard 4: repeated failures
def test_consecutive_failures_halt_the_loop_and_only_a_diagnosis_clears_it(tmp_path):
    governor = _governor(tmp_path, limit=3)
    idle = FL.parse_queue_status(_status())
    for _ in range(3):
        governor.tripwire.record_failure("queue job FAILED after 21 s")
    verdict = governor.authorize(window_count=4, priority=0, status=idle)
    assert verdict.action == FL.ACTION_HALT and verdict.refusal == "repeated_failures"
    with pytest.raises(FL.RepeatedFailureHalt):
        governor.require(window_count=4, priority=0, status=idle)
    # a restarted loop sees the same halt: the counter is on disk, not in this object
    assert _tripwire(tmp_path, 3).state().halted is True
    with pytest.raises(ValueError):
        governor.tripwire.clear(diagnosis="  ")
    governor.tripwire.clear(diagnosis="the staged ELF was linked for the wrong hwdb entry")
    assert governor.authorize(window_count=4, priority=0, status=idle).allowed


def test_a_success_resets_the_consecutive_counter(tmp_path):
    tripwire = _tripwire(tmp_path, limit=3)
    tripwire.record_failure("a")
    tripwire.record_failure("b")
    assert tripwire.state().consecutive == 2 and not tripwire.state().halted
    assert tripwire.record_success().consecutive == 0
    assert tripwire.record_failure("c").consecutive == 1


def test_fifteen_blind_retries_cannot_happen(tmp_path):
    """Jobs 707-721 were fifteen consecutive FAILED jobs; the tripwire stops at its limit."""
    governor = _governor(tmp_path, limit=3)
    idle = FL.parse_queue_status(_status())
    allowed = 0
    for attempt in range(15):
        if not governor.authorize(window_count=4, priority=0, status=idle).allowed:
            break
        allowed += 1
        governor.settle_failure(governor.reserve(4), 21.0, reason=f"attempt {attempt} FAILED")
    assert allowed == 3


# ----------------------------------------------------------------- the simulator's place
def test_the_preflight_carries_no_cycles_at_all():
    preflight = FL.CorrectnessPreflight(program_sha256=_PROGRAM, completed=True, labels=("w0",))
    assert preflight.passed
    assert not hasattr(preflight, "cycles")
    assert "cycles" not in preflight.to_dict()["labels"]
    assert preflight.to_dict()["cycles_withheld"] == FL.SIMULATOR_RANKING_BIAS


def test_a_simulated_measurement_may_not_be_ranked():
    hardware = [
        FL.CandidateMeasurement("c1", 200, FL.SOURCE_HARDWARE),
        FL.CandidateMeasurement("c0", 100, FL.SOURCE_HARDWARE),
    ]
    assert [item.label for item in FL.rank_candidates(hardware)] == ["c0", "c1"]
    with pytest.raises(FL.SimulatorRankingRefused):
        FL.rank_candidates([*hardware, FL.CandidateMeasurement("c2", 50, FL.SOURCE_SIMULATOR)])


def test_the_preflight_reads_digests_and_records_a_missing_one_as_unknown():
    run = {
        "completed": True,
        "records": [
            {"label": "w0", "cycles": 10, "digest": "7"},
            {"label": "w1", "cycles": 11, "digest": "9"},
            {"label": "w2", "cycles": 12},
        ],
    }
    preflight = FL.CorrectnessPreflight.from_simulator_run(
        run, program_sha256=_PROGRAM, expected_digests={"w0": 7, "w1": 8, "w2": 9, "w3": 4}
    )
    assert not preflight.passed
    failures = dict(preflight.failures)
    assert "does not match the oracle" in failures["w1"]
    assert failures["w2"].startswith("UNKNOWN(")
    assert "no window record" in failures["w3"]
    with pytest.raises(ValueError):
        FL.CorrectnessPreflight.from_simulator_run(run, program_sha256=_PROGRAM, expected_digests={})


# ----------------------------------------------------------------- the submission environment
def test_the_identity_variables_must_be_dropped():
    with pytest.raises(ValueError):
        FL.assert_identity_dropped(None)
    with pytest.raises(ValueError):
        FL.assert_identity_dropped({"PATH": "/usr/bin", "HOME": "/home/someone"})
    FL.assert_identity_dropped({"PATH": "/usr/bin"})
    assert FL.SUBMISSION_COMMAND_PREFIX == ("env", "-u", "HOME", "-u", "USER", "-u", "LOGNAME")
    assert FL.loop_client_environment(base={"PATH": "/usr/bin", "USER": "x", "HOME": "/h"}) == {"PATH": "/usr/bin"}


# ----------------------------------------------------------------- the generation entry point
@pytest.fixture
def host(tmp_path) -> QueueHost:
    executable = tmp_path / "firesim-queue"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)
    state, chipyard = tmp_path / "queue-state", tmp_path / "chipyard"
    state.mkdir()
    chipyard.mkdir()
    return QueueHost(queue_executable=executable, queue_state_root=state, chipyard=chipyard)


def _request(tmp_path, host, *, candidates: int = 3, generation_id: str = "gen1") -> FL.GenerationRequest:
    from merlin.perf.firesim_checkpoint import sha256_file

    elf = tmp_path / "generation.elf"
    elf.write_bytes(b"\x7fELF generation")
    policy = tmp_path / "policy.json"
    policy.write_text("{}", encoding="utf-8")
    labels = tuple(f"cand{index}" for index in range(candidates))
    members = tuple(
        BatchMember(
            label=label,
            program_sha256=_PROGRAM,
            weights_sha256=_WEIGHTS,
            observed_window_seconds=11.5,
        )
        for label in labels
    )
    preflight = FL.CorrectnessPreflight(
        program_sha256=sha256_file(elf), completed=True, labels=labels + (f"{labels[0]}__order_control",)
    )
    return FL.GenerationRequest(
        generation_id=generation_id,
        members=members,
        elf=elf,
        validation_policy=policy,
        workload="synthetic-workload",
        bootbinary="synthetic.elf",
        hw_config="synthetic_design",
        host=host,
        preflight=preflight,
    )


class FakeQueue:
    """The queue client and the submit step, with no FPGA behind either."""

    def __init__(self, status_text: str, *, wall_s: float = 500.0, fail: str = ""):
        self.status_text, self.wall_s, self.fail = status_text, wall_s, fail
        self.submissions: list[tuple[str, ...]] = []
        self.status_calls = 0

    def runner(self, argv, **_kwargs):
        assert list(argv)[1:] == ["status"], "the loop may only ever RUN a status query itself"
        self.status_calls += 1
        return subprocess.CompletedProcess(argv, 0, self.status_text, "")

    def submit(self, submission, evidence_dir, *, env=None, runner=None, **_kwargs):
        assert env is not None and "HOME" not in env
        self.submissions.append(submission.argv())
        if self.fail:
            raise CheckpointError(self.fail)
        Path(evidence_dir).mkdir(parents=True, exist_ok=True)
        return SubmissionEvidence(
            job_id=900,
            submission_json=Path(evidence_dir) / "submission.json",
            client_log=Path(evidence_dir) / "queue-client.log",
            daemon_log=Path(evidence_dir) / "daemon.log",
            uart_log=Path(evidence_dir) / "uartlog",
            wall_s=self.wall_s,
            stage_from_sha256=_PROGRAM,
        )


def test_a_generation_is_one_job_of_n_windows_submitted_at_priority_zero(tmp_path, host):
    queue = FakeQueue(_status(_row(899, _ME, "DONE")))
    governor = _governor(tmp_path)
    outcome = FL.measure_generation(
        _request(tmp_path, host),
        governor,
        evidence_dir=tmp_path / "evidence",
        env={"PATH": "/usr/bin"},
        runner=queue.runner,
        submitter=queue.submit,
    )
    assert queue.status_calls == 1, "the queue is polled before every submission"
    assert len(queue.submissions) == 1, "a generation is ONE job, never one job per candidate"
    argv = queue.submissions[0]
    assert argv[1] == "runworkload-full"
    assert argv[argv.index("--priority") + 1] == "0"
    # three candidates plus the order-effect control, in one bootbinary
    assert outcome.windows == ("cand0", "cand1", "cand2", "cand0__order_control")
    assert outcome.job_id == 900
    assert outcome.estimated_seconds == pytest.approx(FL.estimate_job_seconds(4))
    assert governor.ledger.spent_this_run() == pytest.approx(500.0), "charged the OBSERVED wall"
    assert governor.tripwire.state().consecutive == 0


def test_the_loop_does_not_submit_while_another_user_is_queued(tmp_path, host):
    queue = FakeQueue(_status(_row(898, _OTHER, "QUEUED")))
    with pytest.raises(FL.QueueContended):
        FL.measure_generation(
            _request(tmp_path, host),
            _governor(tmp_path),
            evidence_dir=tmp_path / "evidence",
            env={"PATH": "/usr/bin"},
            runner=queue.runner,
            submitter=queue.submit,
        )
    assert queue.submissions == []


def test_the_loop_does_not_submit_past_its_budget(tmp_path, host):
    queue = FakeQueue(_status())
    governor = _governor(tmp_path, budget=FL.FpgaBudget(seconds_per_day=100.0, seconds_per_run=100.0))
    with pytest.raises(FL.BudgetExhausted):
        FL.measure_generation(
            _request(tmp_path, host),
            governor,
            evidence_dir=tmp_path / "evidence",
            env={"PATH": "/usr/bin"},
            runner=queue.runner,
            submitter=queue.submit,
        )
    assert queue.submissions == []
    assert governor.ledger.spent_this_run() == 0.0, "a refused job reserves nothing"


def test_a_failed_job_is_charged_and_counted(tmp_path, host):
    queue = FakeQueue(_status(), fail="queue job 901 did not finish DONE (state FAILED)")
    governor = _governor(tmp_path, limit=2)
    for _ in range(2):
        with pytest.raises(FL.GenerationFailed) as raised:
            FL.measure_generation(
                _request(tmp_path, host),
                governor,
                evidence_dir=tmp_path / "evidence",
                env={"PATH": "/usr/bin"},
                runner=queue.runner,
                submitter=queue.submit,
            )
    assert raised.value.halted is True and raised.value.consecutive == 2
    # a failed job still flashed the bitstream, so it is charged
    assert governor.ledger.spent_this_run() > 0.0
    with pytest.raises(FL.RepeatedFailureHalt):
        FL.measure_generation(
            _request(tmp_path, host),
            governor,
            evidence_dir=tmp_path / "evidence",
            env={"PATH": "/usr/bin"},
            runner=queue.runner,
            submitter=queue.submit,
        )
    assert len(queue.submissions) == 2, "the third attempt never reached the queue"


def test_an_environment_that_still_carries_identity_is_refused(tmp_path, host):
    queue = FakeQueue(_status())
    with pytest.raises(ValueError):
        FL.measure_generation(
            _request(tmp_path, host),
            _governor(tmp_path),
            evidence_dir=tmp_path / "evidence",
            env={"PATH": "/usr/bin", "HOME": "/home/someone"},
            runner=queue.runner,
            submitter=queue.submit,
        )
    assert queue.submissions == []


def test_a_generation_may_not_be_submitted_without_a_passing_preflight_of_its_own_bytes(tmp_path, host):
    request = _request(tmp_path, host)
    with pytest.raises(ValueError):  # a preflight of a different program
        FL.GenerationRequest(
            **{
                **request.__dict__,
                "preflight": FL.CorrectnessPreflight(
                    program_sha256=_PROGRAM, completed=True, labels=request.preflight.labels
                ),
            }
        )
    with pytest.raises(ValueError):  # a preflight that did not pass
        FL.GenerationRequest(
            **{
                **request.__dict__,
                "preflight": FL.CorrectnessPreflight(
                    program_sha256=request.preflight.program_sha256,
                    completed=True,
                    labels=request.preflight.labels,
                    failures=(("cand1", "digest 3 does not match the oracle 4"),),
                ),
            }
        )


def test_a_reservation_cannot_be_settled_twice(tmp_path):
    ledger = _ledger(tmp_path)
    reservation = ledger.reserve(100.0, windows=4, now=1_000.0)
    ledger.settle(reservation, 90.0, now=1_100.0)
    with pytest.raises(ValueError):
        ledger.settle(reservation, 90.0, now=1_200.0)
    assert ledger.spent_this_run() == pytest.approx(90.0)
