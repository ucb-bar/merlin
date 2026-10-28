"""A check runs in an owned worker beside the compile and stops it, with evidence, if it fails.

The fake checks live in a temporary module that the worker imports by ``module:function``; every
test drives the real worker process, queue, cancellation and interrupt paths.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from merlin.verify.speculative_gate import GateFailure, SpeculativeGate

FAKE_CHECKS = """
import json, os, time
from pathlib import Path

from merlin.verify.gate_errors import GateRejected, GateUnsupported


def accept(job):
    return {"detail": "ok"}


def reject(job):
    raise GateRejected("Lean did not accept the pair: theorem scalar_preserves failed")


def unsupported(job):
    raise GateUnsupported("outside the modelled scalar domain: scf.if")


def crash(job):
    raise RuntimeError("worker blew up")


def slow(job):
    stamp = Path(job["directory"]) / "stamp.json"
    start = time.monotonic()
    time.sleep(float(job["params"].get("seconds", 0.5)))
    stamp.write_text(json.dumps({"start": start, "end": time.monotonic()}))
    return {}


def forever(job):
    Path(job["directory"], "pid").write_text(str(os.getpid()))
    time.sleep(120)


def forever_with_child(job):
    import subprocess

    child = subprocess.Popen(["sleep", "300"])  # stands in for the checker process a worker runs
    Path(job["directory"], "pids").write_text(json.dumps([os.getpid(), child.pid]))
    time.sleep(120)


def reject_after(job):
    time.sleep(float(job["params"].get("seconds", 0.5)))
    raise GateRejected("rejected after a delay")
"""


@pytest.fixture
def gate_factory(tmp_path):
    (tmp_path / "fake_checks.py").write_text(FAKE_CHECKS)
    made = []

    def make(**kwargs):
        gate = SpeculativeGate(tmp_path / "gate", worker_path=[tmp_path], **kwargs)
        made.append(gate)
        return gate

    yield make
    for gate in made:
        gate.close()


def test_accepted_check_lets_join_return(gate_factory):
    gate = gate_factory()
    gate.submit("fake_checks:accept", before="a", after="a", label="identity")
    gate.join()
    assert [v["verdict"] for v in gate.verdicts] == ["accepted"]


def test_rejected_check_raises_with_the_inputs_and_the_reason(gate_factory):
    gate = gate_factory()
    gate.submit("fake_checks:reject", before="%0 = old", after="%0 = new", label="cse-dce")
    with pytest.raises(GateFailure) as raised:
        gate.join()
    report = raised.value.report
    assert report["verdict"] == "not_established"
    assert report["check"] == "fake_checks:reject"
    assert report["label"] == "cse-dce"
    assert "theorem scalar_preserves failed" in report["reason"]
    assert Path(report["before"]).read_text() == "%0 = old"
    assert Path(report["after"]).read_text() == "%0 = new"
    assert report["first_difference"]["line"] == 1


def test_failure_is_printed_for_the_person_running_the_compile(gate_factory, capsys):
    gate = gate_factory()
    gate.submit("fake_checks:reject", before="x", after="y", label="cse-dce")
    with pytest.raises(GateFailure):
        gate.join()
    err = capsys.readouterr().err
    assert "fake_checks:reject" in err and "not_established" in err
    assert "theorem scalar_preserves failed" in err
    assert "before:" in err and "after:" in err


def test_unsupported_input_is_a_failure_not_a_pass(gate_factory):
    gate = gate_factory()
    gate.submit("fake_checks:unsupported", before="a", after="b")
    with pytest.raises(GateFailure) as raised:
        gate.join()
    assert raised.value.report["verdict"] == "unsupported"


def test_a_crashing_worker_is_unavailable_never_accepted(gate_factory):
    gate = gate_factory()
    gate.submit("fake_checks:crash", before="a", after="b")
    with pytest.raises(GateFailure) as raised:
        gate.join()
    assert raised.value.report["verdict"] == "unavailable"
    assert "worker blew up" in raised.value.report["reason"]


def test_an_unknown_check_is_unavailable(gate_factory):
    gate = gate_factory()
    gate.submit("fake_checks:missing", before="a", after="b")
    with pytest.raises(GateFailure) as raised:
        gate.join()
    assert raised.value.report["verdict"] == "unavailable"


def test_poll_reports_a_failure_that_finished_while_the_compile_ran(gate_factory):
    gate = gate_factory()
    gate.submit("fake_checks:reject", before="a", after="b")
    deadline = time.monotonic() + 20
    with pytest.raises(GateFailure):
        while time.monotonic() < deadline:
            gate.poll()
            time.sleep(0.05)


def test_jobs_run_one_at_a_time(gate_factory):
    gate = gate_factory()
    first = gate.submit("fake_checks:slow", before="a", after="a", params={"seconds": 0.6})
    second = gate.submit("fake_checks:slow", before="b", after="b", params={"seconds": 0.1})
    gate.join()
    one = json.loads((first.directory / "stamp.json").read_text())
    two = json.loads((second.directory / "stamp.json").read_text())
    assert two["start"] >= one["end"]


def test_workers_run_checks_at_once(gate_factory):
    gate = gate_factory(workers=2)
    first = gate.submit("fake_checks:slow", before="a", after="a", params={"seconds": 2.0})
    second = gate.submit("fake_checks:slow", before="b", after="b", params={"seconds": 2.0})
    gate.join()
    one = json.loads((first.directory / "stamp.json").read_text())
    two = json.loads((second.directory / "stamp.json").read_text())
    assert two["start"] < one["end"] and one["start"] < two["end"]
    assert len(gate.verdicts) == 2


def test_close_kills_every_running_worker(gate_factory):
    gate = gate_factory(workers=2)
    jobs = [gate.submit("fake_checks:forever", before=x, after=x) for x in "ab"]
    deadline = time.monotonic() + 20
    while not all((j.directory / "pid").exists() for j in jobs) and time.monotonic() < deadline:
        time.sleep(0.05)
    pids = [int((j.directory / "pid").read_text()) for j in jobs]
    gate.close()
    for pid in pids:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


def test_a_gate_needs_a_worker(tmp_path):
    with pytest.raises(ValueError, match="at least one worker"):
        SpeculativeGate(tmp_path / "gate", workers=0)


def test_close_kills_the_running_worker_promptly(gate_factory):
    gate = gate_factory()
    job = gate.submit("fake_checks:forever", before="a", after="a")
    pid_file = job.directory / "pid"
    deadline = time.monotonic() + 20
    while not pid_file.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    pid = int(pid_file.read_text())
    started = time.monotonic()
    gate.close()
    assert time.monotonic() - started < 10
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


@pytest.mark.parametrize("delay", [0.0, 0.05, 0.2, 0.5])
def test_a_cancelled_check_reports_nothing(gate_factory, capsys, delay):
    gate = gate_factory()
    gate.submit("fake_checks:forever", before="a", after="a")
    time.sleep(delay)
    gate.close()
    gate.poll()
    assert gate.verdicts == []
    assert "speculative check failed" not in capsys.readouterr().err


def test_a_job_over_its_budget_is_unavailable(gate_factory):
    gate = gate_factory(job_timeout=1.0)
    gate.submit("fake_checks:forever", before="a", after="a")
    with pytest.raises(GateFailure) as raised:
        gate.join()
    assert raised.value.report["verdict"] == "unavailable"
    assert "budget" in raised.value.report["reason"]


def test_no_new_work_is_accepted_after_a_failure(gate_factory):
    gate = gate_factory()
    gate.submit("fake_checks:reject", before="a", after="b")
    with pytest.raises(GateFailure):
        gate.join()
    with pytest.raises(GateFailure):
        gate.submit("fake_checks:accept", before="a", after="a")


def test_a_failure_stops_work_that_is_still_running_in_the_main_thread(gate_factory):
    gate = gate_factory(interrupt=True)
    started = time.monotonic()
    with pytest.raises(GateFailure):
        with gate.guard():
            gate.submit("fake_checks:reject_after", before="a", after="b", params={"seconds": 0.5})
            time.sleep(60)  # stands in for the rest of the compile
    assert time.monotonic() - started < 30


def test_guard_leaves_an_unrelated_interrupt_alone(gate_factory):
    gate = gate_factory(interrupt=True)
    with pytest.raises(KeyboardInterrupt):
        with gate.guard():
            raise KeyboardInterrupt


def test_guard_joins_pending_checks_before_returning(gate_factory):
    gate = gate_factory(interrupt=True)
    with pytest.raises(GateFailure):
        with gate.guard():
            gate.submit("fake_checks:reject_after", before="a", after="b", params={"seconds": 0.3})


def test_a_worker_that_cannot_start_is_a_failure_not_a_silent_pass(gate_factory, monkeypatch):
    from merlin.verify import speculative_gate

    def cannot_start(*args, **kwargs):
        raise OSError("no exec for the worker")

    monkeypatch.setattr(speculative_gate, "start_owned", cannot_start)
    gate = gate_factory()
    gate.submit("fake_checks:accept", before="a", after="a")
    with pytest.raises(GateFailure) as raised:
        gate.join(timeout=30)
    assert raised.value.report["verdict"] == "unavailable"
    assert "no exec for the worker" in raised.value.report["reason"]


def test_join_never_succeeds_without_a_verdict_for_every_submission(gate_factory, monkeypatch):
    gate = gate_factory()
    # A scheduler that drops the job without recording anything must not look like success.
    monkeypatch.setattr(gate, "_execute", lambda job: None)
    gate.submit("fake_checks:accept", before="a", after="a")
    with pytest.raises(GateFailure) as raised:
        gate.join(timeout=30)
    assert raised.value.report["verdict"] == "unavailable"
    assert "no verdict" in raised.value.report["reason"]


def test_an_interrupt_never_outlives_the_guard_it_was_meant_for(gate_factory, monkeypatch):
    original = SpeculativeGate._interrupt_main

    def late():
        time.sleep(0.4)  # the guard's owner has already seen the failure and moved on
        original()

    monkeypatch.setattr(SpeculativeGate, "_interrupt_main", staticmethod(late))
    gate = gate_factory(interrupt=True)
    with pytest.raises(GateFailure):
        with gate.guard():
            gate.submit("fake_checks:reject", before="a", after="b")
            gate.join()
    try:
        time.sleep(1.0)
    except KeyboardInterrupt:
        pytest.fail("the gate's interrupt arrived after its guard had returned")


def test_a_guard_in_a_worker_thread_never_interrupts_the_main_thread(gate_factory):
    import threading

    gate = gate_factory(interrupt=True)
    outcome = {}

    def lower_in_a_worker():
        try:
            with gate.guard():
                gate.submit("fake_checks:reject_after", before="a", after="b", params={"seconds": 0.2})
                time.sleep(2.0)  # the worker's compile carries on; it is not interruptible from here
        except GateFailure as failure:
            outcome["failure"] = failure

    worker = threading.Thread(target=lower_in_a_worker)
    try:
        worker.start()
        worker.join(timeout=60)
        time.sleep(0.5)
    except KeyboardInterrupt:
        pytest.fail("a failure in a worker thread's guard interrupted the main thread")
    assert "failure" in outcome


COMPILE = """
import sys, time
from pathlib import Path
from merlin.verify.speculative_gate import SpeculativeGate
gate = SpeculativeGate(Path(sys.argv[1]), worker_path=[Path(sys.argv[2])])
gate.submit("fake_checks:forever_with_child", before="a", after="a")
time.sleep(120)
"""


@pytest.mark.skipif(not shutil.which("setpriv"), reason="parent-death signals need util-linux setpriv")
@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGKILL])
def test_a_killed_compile_leaves_no_check_running(tmp_path, sig):
    """A compile stopped by a signal runs no cleanup; its worker and the worker's checker processes
    must still go with it rather than run on for the rest of a long check."""
    (tmp_path / "fake_checks.py").write_text(FAKE_CHECKS)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([*sys.path])}
    compile_ = subprocess.Popen([sys.executable, "-c", COMPILE, str(tmp_path / "gate"), str(tmp_path)], env=env)
    deadline = time.monotonic() + 30
    found = []
    while not found and time.monotonic() < deadline:
        found = list((tmp_path / "gate").glob("*/pids"))
        time.sleep(0.05)
    assert found, "the check never started"
    time.sleep(0.2)
    pids = json.loads(found[0].read_text())
    compile_.send_signal(sig)
    compile_.wait(timeout=10)
    deadline = time.monotonic() + 10
    alive = pids
    while alive and time.monotonic() < deadline:
        alive = [pid for pid in pids if _alive(pid)]
        time.sleep(0.05)
    assert not alive, f"left running after the compile died: {alive}"


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    stat = Path(f"/proc/{pid}/stat")
    return not (stat.is_file() and stat.read_text().split()[2] == "Z")  # a zombie has exited
