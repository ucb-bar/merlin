"""Owned subprocess cleanup preserves capture contracts and unrelated jobs."""

import io
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from merlin.common import owned_process
from merlin.verify.lean import differential


def _running(pid):
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()[0]
    except FileNotFoundError:
        return False
    return state != "Z"


def _tree(tmp_path, *, leader_exits=False):
    checker = (
        "import os,signal,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"Path({str(tmp_path / 'checker.pid')!r}).write_text(str(os.getpid())); time.sleep(3600)"
    )
    leader = (
        "import os,subprocess,sys,time; from pathlib import Path; "
        f"Path({str(tmp_path / 'leader.pid')!r}).write_text(str(os.getpid())); "
        f"child=subprocess.Popen([sys.executable, '-c', {checker!r}]); "
        f"ready=Path({str(tmp_path / 'checker.pid')!r}); "
        "exec('while not ready.exists(): time.sleep(0.01)'); "
        "print('before-timeout',flush=True); print('error-before-timeout',file=sys.stderr,flush=True); "
        + ("sys.exit(0)" if leader_exits else "time.sleep(3600)")
    )
    return [sys.executable, "-c", leader]


def _assert_tree_stopped(tmp_path):
    for name in ("leader", "checker"):
        pid = int((tmp_path / f"{name}.pid").read_text())
        deadline = time.monotonic() + 1
        while _running(pid) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not _running(pid), name


def test_normal_capture_cwd_environment_and_input_match_subprocess_run(tmp_path):
    command = [
        sys.executable,
        "-c",
        "import os,sys; print(os.getcwd()); print(os.environ['OWNED_TEST']); "
        "print(sys.stdin.read()); print('snowman:\u2603',file=sys.stderr)",
    ]
    kwargs = {
        "cwd": tmp_path,
        "env": {**os.environ, "OWNED_TEST": "configured"},
        "input": "payload",
        "capture_output": True,
        "text": True,
        "timeout": 5,
    }
    expected = subprocess.run(command, check=False, **kwargs)
    actual = owned_process.run_owned(command, check=False, **kwargs)
    assert (actual.args, actual.returncode, actual.stdout, actual.stderr) == (
        expected.args,
        expected.returncode,
        expected.stdout,
        expected.stderr,
    )


def test_success_has_unique_session_before_command_execution():
    result = owned_process.run_owned(
        [sys.executable, "-c", "import os,json; print(json.dumps([os.getpid(),os.getpgrp(),os.getsid(0)]))"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    pid, group, session = json.loads(result.stdout)
    assert pid == group == session
    assert group != os.getpgrp()


@pytest.mark.parametrize("text", [False, True])
def test_checked_failure_preserves_standard_error_object(text):
    command = [sys.executable, "-c", "import sys; print('stdout'); print('stderr',file=sys.stderr); sys.exit(7)"]
    with pytest.raises(subprocess.CalledProcessError) as old:
        subprocess.run(command, capture_output=True, text=text, timeout=5, check=True)
    with pytest.raises(subprocess.CalledProcessError) as new:
        owned_process.run_owned(command, capture_output=True, text=text, timeout=5, check=True)
    assert (new.value.cmd, new.value.returncode, new.value.output, new.value.stderr) == (
        old.value.cmd,
        old.value.returncode,
        old.value.output,
        old.value.stderr,
    )


@pytest.mark.parametrize("leader_exits", [False, True])
def test_timeout_kills_term_ignoring_descendant_and_preserves_timeout_output(tmp_path, leader_exits):
    sibling = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    command = _tree(tmp_path, leader_exits=leader_exits)
    try:
        with pytest.raises(subprocess.TimeoutExpired) as failure:
            owned_process.run_owned(command, capture_output=True, text=True, timeout=1)
        assert failure.value.cmd == command
        assert failure.value.timeout == 1
        assert failure.value.output == b"before-timeout\n"
        assert failure.value.stderr == b"error-before-timeout\n"
        _assert_tree_stopped(tmp_path)
        assert sibling.poll() is None
    finally:
        sibling.terminate()
        sibling.wait(timeout=5)


def test_keyboard_interrupt_preserves_exception_and_cleans_real_descendants(tmp_path, monkeypatch):
    command = _tree(tmp_path)
    original = subprocess.Popen.communicate
    interruption = KeyboardInterrupt("deliberate cancellation")

    def interrupt(process, *args, **kwargs):
        if process.args == command:
            deadline = time.monotonic() + 5
            while not (tmp_path / "checker.pid").exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert (tmp_path / "checker.pid").exists()
            assert kwargs["timeout"] == 120
            raise interruption
        return original(process, *args, **kwargs)

    monkeypatch.setattr(subprocess.Popen, "communicate", interrupt)
    with pytest.raises(KeyboardInterrupt) as failure:
        owned_process.run_owned(command, capture_output=True, text=True, timeout=120)
    assert failure.value is interruption
    _assert_tree_stopped(tmp_path)


def test_startup_failure_propagates_without_signalling_any_group(tmp_path, monkeypatch):
    signalled = []
    monkeypatch.setattr(owned_process.os, "killpg", lambda *args: signalled.append(args))
    with pytest.raises(FileNotFoundError) as failure:
        owned_process.run_owned([str(tmp_path / "missing-tool")], capture_output=True, text=True, timeout=120)
    assert failure.value.filename == str(tmp_path / "missing-tool")
    assert signalled == []


@pytest.mark.parametrize(
    "conflict",
    [
        {"input": "x", "stdin": subprocess.PIPE},
        {"capture_output": True, "stdout": subprocess.PIPE},
        {"start_new_session": False},
        {"process_group": 0},
        {"terminate_grace": -1},
        {"kill_grace": -1},
        {"kill_grace": float("nan")},
        {"bufsize": 4096},
        {"terminate_grace": float("inf")},
    ],
)
def test_conflicting_or_invalid_ownership_options_refuse_before_execution(conflict):
    with pytest.raises(ValueError):
        owned_process.run_owned(["missing-tool"], **conflict)


def test_differential_error_tail_and_success_match_previous_contract():
    command = [sys.executable, "-c", "import sys; print('o'*2100); print('e'*4100,file=sys.stderr); sys.exit(9)"]
    baseline = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
    expected = f"{command[0]} exited {baseline.returncode}:\n{baseline.stdout[-2000:]}\n{baseline.stderr[-4000:]}"
    with pytest.raises(RuntimeError) as failure:
        differential._run(command)
    assert str(failure.value) == expected
    success = differential._run([sys.executable, "-c", "print('same stdout')"])
    assert success.returncode == 0 and success.stdout == "same stdout\n" and success.stderr == ""


def test_differential_keeps_exact_command_cwd_timeout_and_capture_options(tmp_path, monkeypatch):
    observed = []
    command = ["selected-tool", "exact-argument"]

    def run(argv, **kwargs):
        observed.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, "output", "")

    monkeypatch.setattr(differential, "run_owned", run)
    differential._run(command, cwd=tmp_path)
    assert observed == [
        (command, {"cwd": tmp_path, "capture_output": True, "text": True, "timeout": 120, "check": False})
    ]


def test_unreapable_child_never_causes_an_unbounded_cleanup_wait(monkeypatch):
    interruption = KeyboardInterrupt("cancel a simulated kernel-uninterruptible child")
    waits, signals = [], []

    class Unreapable:
        pid = 123456789
        returncode = None
        stdin = stdout = stderr = None

        def communicate(self, *, input, timeout):
            assert timeout == 120
            raise interruption

        def wait(self, *, timeout):
            waits.append(timeout)
            raise subprocess.TimeoutExpired("simulated-kernel-wait", timeout)

    monkeypatch.setattr(owned_process, "start_owned", lambda *args, **kwargs: Unreapable())
    monkeypatch.setattr(owned_process.os, "killpg", lambda *args: signals.append(args))
    with pytest.raises(KeyboardInterrupt) as failure:
        owned_process.run_owned(["simulated-tool"], timeout=120)
    assert failure.value is interruption
    assert waits == [0.2, 1.0]
    assert signals == [(123456789, signal.SIGTERM), (123456789, signal.SIGKILL)]
    note = "\n".join(interruption.__notes__)
    assert "123456789" in note and "'direct_child_reaped': False" in note
    assert "'descendant_exit_confirmed': False" in note


def test_owned_pipes_are_unbuffered_and_text_input_is_write_through():
    process = owned_process.start_owned(
        [sys.executable, "-c", "import sys; print(sys.stdin.read())"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert isinstance(process.stdin.buffer, io.FileIO)
        assert process.stdin.write_through
        assert isinstance(process.stdout.buffer, io.FileIO)
        assert isinstance(process.stderr.buffer, io.FileIO)
        assert process.communicate(input="same input", timeout=5) == ("same input\n", "")
    finally:
        owned_process.stop_owned(process)
        owned_process.close_owned_pipes(process)


def test_interruption_during_term_grace_still_requests_owned_kill(monkeypatch):
    signals, waits = [], []
    interruption = KeyboardInterrupt("second cancellation during cleanup")

    class Interrupted:
        pid = 123456789
        returncode = None

        def wait(self, *, timeout):
            waits.append(timeout)
            if len(waits) == 1:
                raise interruption
            self.returncode = -9

    monkeypatch.setattr(owned_process.os, "killpg", lambda *args: signals.append(args))
    with pytest.raises(KeyboardInterrupt) as error:
        owned_process.stop_owned(Interrupted())
    assert error.value is interruption
    assert waits == [0.2, 1.0]
    assert signals == [(123456789, signal.SIGTERM), (123456789, signal.SIGKILL)]
