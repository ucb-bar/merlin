"""A full disk drops log lines and retries work; it never ends the runner's graders or brokers."""

from __future__ import annotations

import errno
import io
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from merlin_experiments.phase1.brokers import selfcheck as S
from merlin_experiments.phase1.context import InvocationContext
from merlin_experiments.phase1.feedback import lifecycle as L
from merlin_experiments.phase1.telemetry.resilient_output import ResilientStream


def _until(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert predicate(), "condition did not hold in time"


class _FullDisk(io.StringIO):
    full = True

    def write(self, text):
        if self.full:
            raise OSError(errno.ENOSPC, "No space left on device")
        return super().write(text)

    def flush(self):
        if self.full:
            raise OSError(errno.ENOSPC, "No space left on device")


def test_output_drops_writes_while_full_and_reports_them_once_it_recovers():
    raw = _FullDisk()
    stream = ResilientStream(raw)
    print("lost one", file=stream, flush=True)
    stream.write("lost two")
    raw.full = False
    stream.write("kept\n")
    stream.write("next\n")
    # print() writes its text and its newline separately: three writes were lost.
    assert raw.getvalue() == "[output] 3 writes dropped: No space left on device\nkept\nnext\n"
    assert stream.dropped == 0


def _broker_config(root):
    context = InvocationContext(root, root / "descriptor.yaml", root, "fixture", root, root, root, ())
    return L.BrokerConfig(context, (), root / "timing.json")


def _fake_broker(monkeypatch, tmp_path, program):
    """One fake broker on the self-check channel; every launch (first or restart) runs ``program``."""
    monkeypatch.setattr(L._TR, "COMMON_BROKERS", (
        SimpleNamespace(module="merlin_experiments.phase1.brokers.fake", log="broker.log", channel=".qa_channel"),
    ))
    monkeypatch.setattr(L, "BROKER_SUPERVISE_S", 0.02)
    monkeypatch.setattr(L, "BROKER_RESTART_BACKOFF_S", 0.02)
    actual_popen, launches = L.subprocess.Popen, []

    def spawn(argv, **kwargs):
        launches.append(argv)
        return actual_popen([sys.executable, "-c", program, str(tmp_path / "workspace"), str(len(launches))], **kwargs)

    monkeypatch.setattr(L.subprocess, "Popen", spawn)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace, launches


def test_a_broker_that_dies_is_restarted_with_its_argv_and_stops_cleanly(tmp_path, monkeypatch, capsys):
    program = (
        "import pathlib, sys, time\n"
        "channel = pathlib.Path(sys.argv[1]) / '.qa_channel'\n"
        "print('launch', sys.argv[2], flush=True)\n"
        "if sys.argv[2] == '1': sys.exit(3)\n"
        "while not (channel / 'STOP').exists(): time.sleep(0.01)\n"
    )
    workspace, launches = _fake_broker(monkeypatch, tmp_path, program)
    brokers = L.start_brokers(workspace, _broker_config(tmp_path))
    first = brokers[0]
    try:
        _until(lambda: len(launches) == 2 and brokers[0] is not first and brokers[0].poll() is None)
        assert launches[0] == launches[1]
    finally:
        L.stop_brokers(workspace, brokers)
    assert brokers[0].returncode == 0
    time.sleep(0.1)
    assert len(launches) == 2, "no broker may start after the brokers were stopped"
    log = (workspace / ".qa_channel/broker.log").read_text()
    assert log.splitlines() == ["launch 1", "launch 2"], "a restart appends to the broker's log"
    out = capsys.readouterr().out
    assert "fake exited with 3" in out and "restarted (1/" in out


def test_restarts_end_at_the_limit_and_never_follow_stop(tmp_path, monkeypatch, capsys):
    workspace, launches = _fake_broker(monkeypatch, tmp_path, "import sys; sys.exit(1)\n")
    monkeypatch.setattr(L, "BROKER_RESTART_LIMIT", 2)
    monkeypatch.setattr(L, "BROKER_RESTART_MAX_BACKOFF_S", 0.02)
    brokers = L.start_brokers(workspace, _broker_config(tmp_path))
    try:
        seen = []
        _until(lambda: seen.append(capsys.readouterr().out) or "will not be restarted again" in "".join(seen))
        assert len(launches) == 3
    finally:
        L.stop_brokers(workspace, brokers)

    workspace2 = tmp_path / "second"
    workspace2.mkdir()
    brokers = L.start_brokers(workspace2, _broker_config(tmp_path))
    (workspace2 / ".qa_channel/STOP").write_text("stop")  # an exit after STOP was requested
    before = len(launches)
    try:
        time.sleep(0.2)
        assert len(launches) == before
    finally:
        L.stop_brokers(workspace2, brokers)


def test_the_in_turn_grader_resumes_after_an_unexpected_failure(tmp_path, monkeypatch, capsys):
    from merlin_experiments.phase1.feedback import dual_snapshots, joint_test_scope

    monkeypatch.setattr(L, "GRADER_RESUME_S", 0.01)
    monkeypatch.setattr(joint_test_scope, "focused", lambda: True)
    calls = []

    def run(ws, run_dir, stop, **kwargs):
        calls.append(kwargs["first_required"])
        if len(calls) == 1:
            raise OSError(errno.ENOSPC, "No space left on device")
        stop.wait(10)

    monkeypatch.setattr(dual_snapshots, "run", run)
    (tmp_path / "run").mkdir()
    handle = L.start_background(
        tmp_path / "workspace",
        tmp_path / "run",
        L.GradeCadence(1800, False, 900),
        interval_grades=True,
        grade_callback=lambda *args, **kwargs: {},
        fast_grade_callback=lambda *args: {},
    )
    try:
        _until(lambda: len(calls) == 2)
        assert handle[0].is_alive()
    finally:
        L.stop_background(handle)
    assert not handle[0].is_alive()
    assert "[in-turn grader] resuming in" in capsys.readouterr().out
    health = json.loads((tmp_path / "run/grade_health.json").read_text())
    assert "No space left on device" in health["error"]


def test_a_failed_broker_pass_answers_its_request_or_requeues_it(tmp_path, monkeypatch):
    channel = tmp_path / ".qa_channel"
    channel.mkdir()
    request = channel / "req_7_abc.json"
    request.write_text("{}")
    seen = {request.name}
    S._settle_failed_request(channel, request, seen, OSError(errno.ENOSPC, "No space left on device"))
    response = json.loads((channel / "resp_7_abc.json").read_text())
    assert response["all_pass"] is False and "OSError" in response["error"]
    assert (channel / "done_7_abc").read_text() == "err"

    other = channel / "req_8_def.json"
    other.write_text("{}")
    seen = {other.name}

    def full(path: Path, text: str) -> None:
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(S, "_atomic_write", full)
    S._settle_failed_request(channel, other, seen, OSError(errno.ENOSPC, "No space left on device"))
    assert other.name not in seen, "an unanswerable request goes back to the queue"
    assert S._pending_requests(channel, seen) == [other]
