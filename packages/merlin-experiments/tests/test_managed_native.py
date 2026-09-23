"""Real local process ownership through the production managed supervisor."""

from __future__ import annotations

import ctypes
import dataclasses
import errno
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from merlin_experiments.execution import _protocol as protocol
from merlin_experiments.execution.chia_native import CleanupIncomplete, Invitation, Session, cleanup, run, setup

from merlin.common.paths import python_import_roots


def test_close_attempts_every_invocation_and_remains_retryable(service, monkeypatch):
    session = Session(service[0])
    first, second = session.reserve(), session.reserve()
    cancel = session.cancel
    receipt = session.receipt
    attempted = []
    observed = []

    def fail_first(invitation, **kwargs):
        attempted.append(invitation.invocation)
        if invitation == first:
            raise ConnectionError("synthetic transport interruption")
        return cancel(invitation, **kwargs)

    def observe(invitation, **kwargs):
        observed.append(invitation.invocation)
        if invitation == first:
            raise ConnectionError("synthetic observation interruption")
        return receipt(invitation, **kwargs)

    monkeypatch.setattr(session, "cancel", fail_first)
    monkeypatch.setattr(session, "receipt", observe)
    with pytest.raises(CleanupIncomplete, match="remains unresolved"):
        session.close()
    assert attempted == observed == [first.invocation, second.invocation]
    assert not session._closed
    assert receipt(second)["cleanup_complete"]
    monkeypatch.setattr(session, "cancel", cancel)
    monkeypatch.setattr(session, "receipt", receipt)
    session.close()
    assert session._closed
    assert receipt(first)["cleanup_complete"]


@pytest.fixture
def service(tmp_path):
    try:
        descriptor = protocol.self_pidfd()
    except (RuntimeError, OSError) as exc:
        pytest.skip(f"Linux pidfds unavailable: {exc}")
    os.close(descriptor)
    directory = tmp_path / "managed"
    directory.mkdir(mode=0o700)
    endpoint = directory / "service.sock"
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join(map(str, python_import_roots()))}
    process = subprocess.Popen(
        [sys.executable, "-m", "merlin_experiments.execution.native_supervisor", "--endpoint", str(endpoint)],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 10
        while not endpoint.exists():
            if process.poll() is not None or time.monotonic() >= deadline:
                pytest.fail("supervisor did not start: " + process.communicate(timeout=1)[1])
            time.sleep(0.02)
        yield endpoint, environment, process.pid
    finally:
        if process.poll() is None:
            process.terminate()
        _, error = process.communicate(timeout=12)
        assert process.returncode == 0, error


def _worker(service, invitation, command, *, cwd, environment=None):
    script = (
        "import json,sys; from merlin_experiments.execution.chia_native import Invitation,setup,run; "
        "data=json.loads(sys.stdin.readline()); setup(Invitation(**data['invitation'])); "
        "raise SystemExit(run(data['command'],cwd=data['cwd'],env=data['environment']))"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", script],
        env=service[1],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    request = json.dumps(
        {
            "invitation": dataclasses.asdict(invitation),
            "command": command,
            "cwd": str(cwd),
            "environment": environment,
        }
    )
    process.stdin.write(request + "\n")
    process.stdin.flush()
    return process


def test_command_preserves_exit_stdio_cwd_environment_and_reaps_guardian(service, tmp_path):
    with Session(service[0]) as session:
        invitation = session.reserve()
        worker = _worker(
            service,
            invitation,
            [
                sys.executable,
                "-c",
                "import os,sys; print(os.getcwd()); print(os.environ['VALUE'],file=sys.stderr); sys.exit(7)",
            ],
            cwd=tmp_path,
            environment={"VALUE": "native-value"},
        )
        stdout, stderr = worker.communicate(timeout=15)
        assert worker.returncode == 7, stderr
        assert stdout.strip() == str(tmp_path)
        assert stderr.strip() == "native-value"
        receipt = session.receipt(invitation)
        assert receipt["cleanup_complete"] and receipt["guardian_reaped"]
        assert receipt["guardian"]["returncode"] == 7
        stored = service[0].parent / "receipts" / session.supervisor / (invitation.invocation + ".json")
        assert json.loads(stored.read_text()) == receipt
        assert invitation.session not in stored.read_text()


def test_worker_loss_cleans_detached_native_child(service, tmp_path):
    pidfile = tmp_path / "native.pid"
    code = (
        "import os,signal,time; from pathlib import Path; "
        "os.setsid(); signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        f"Path({str(pidfile)!r}).write_text(str(os.getpid())); time.sleep(60)"
    )
    with Session(service[0]) as session:
        invitation = session.reserve()
        worker = _worker(service, invitation, [sys.executable, "-c", code], cwd=tmp_path)
        worker_fd = protocol.pidfd_open(worker.pid)
        try:
            deadline = time.monotonic() + 10
            while not pidfile.exists():
                assert worker.poll() is None
                assert time.monotonic() < deadline
                time.sleep(0.02)
            native_fd = protocol.pidfd_open(int(pidfile.read_text()))
            try:
                protocol.send_signal(worker_fd, signal.SIGKILL)
                worker.communicate(timeout=10)
                receipt = session.receipt(invitation)
                assert receipt["cleanup_complete"] and receipt["guardian_reaped"]
                assert protocol.exited(native_fd)
                assert receipt["guardian"]["native_cleanup"]["reaped_children"] >= 1
            finally:
                os.close(native_fd)
        finally:
            if worker.poll() is None:
                protocol.send_signal(worker_fd, signal.SIGKILL)
                worker.communicate(timeout=10)
            os.close(worker_fd)


def test_no_hook_cannot_launch_native_command(tmp_path):
    with pytest.raises(RuntimeError, match="requires a managed"):
        run([sys.executable, "-c", "raise AssertionError('must not execute')"], cwd=tmp_path)


def test_cancel_before_attachment_is_no_launch_and_foreign_session_refuses(service):
    with Session(service[0]) as session:
        invitation = session.reserve()
        session.cancel(invitation)
        receipt = session.receipt(invitation)
        assert receipt["cleanup_complete"] and receipt["guardian_created"] is False
        with pytest.raises(RuntimeError, match="did not become ready"):
            setup(invitation)
        cleanup(invitation)
        forged = dataclasses.replace(invitation, session="wrong-session")
        with pytest.raises(RuntimeError, match="did not become ready"):
            setup(forged)


def test_wrong_host_refuses_before_launch(service):
    with protocol.connect(service[0]) as connection:
        descriptor = protocol.self_pidfd()
        try:
            protocol.send(connection, {"operation": "session", "host": {}}, [descriptor])
            reply, attachments = protocol.receive(connection)
            protocol.close_fds(attachments)
            assert reply == {"error": "PermissionError"}
        finally:
            os.close(descriptor)


def test_setup_body_failure_reaps_original_guardian_without_native_start(service):
    with Session(service[0]) as session:
        invitation = session.reserve()
        setup(invitation)
        # Observe only threads/children of the explicitly owned service, while
        # READY holds its single guardian alive awaiting START.
        children = set()
        for path in Path(f"/proc/{service[2]}/task").glob("*/children"):
            children.update(map(int, path.read_text().split()))
        assert len(children) == 1
        guardian_fd = protocol.pidfd_open(children.pop())
        try:
            cleanup(invitation)
            receipt = session.receipt(invitation)
            assert receipt["cleanup_complete"] and receipt["guardian_reaped"]
            assert receipt["guardian"]["native_started"] is False
            assert protocol.exited(guardian_fd)
            info = dict(
                line.split(":", 1) for line in Path(f"/proc/self/fdinfo/{guardian_fd}").read_text().splitlines()
            )
            assert info["Pid"].strip() == "-1"  # Original identity is no longer a waitable zombie.
        finally:
            os.close(guardian_fd)


@pytest.mark.parametrize("use_python_api", [False, True])
def test_pidfd_capability_paths_have_close_on_exec_identity(monkeypatch, use_python_api):
    open_function = protocol._libc_function("pidfd_open", [ctypes.c_int, ctypes.c_uint])
    signal_function = protocol._libc_function(
        "pidfd_send_signal", [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint]
    )
    calls = []

    def stdlib_open(pid):
        calls.append("open")
        return open_function(pid, 0)

    def stdlib_signal(fd, signum):
        calls.append("signal")
        assert signal_function(fd, signum, None, 0) == 0

    if use_python_api:
        monkeypatch.setattr(os, "pidfd_open", stdlib_open, raising=False)
        monkeypatch.setattr(signal, "pidfd_send_signal", stdlib_signal, raising=False)
    else:
        monkeypatch.delattr(os, "pidfd_open", raising=False)
        monkeypatch.delattr(signal, "pidfd_send_signal", raising=False)
    descriptor = protocol.self_pidfd()
    try:
        assert not os.get_inheritable(descriptor)
        assert not protocol.exited(descriptor)
        assert calls == (["open", "signal"] if use_python_api else [])
    finally:
        os.close(descriptor)


@pytest.mark.parametrize("unsupported", ["symbol", "kernel"])
def test_unsupported_worker_refuses_before_guardian_admission(service, monkeypatch, unsupported):
    with Session(service[0]) as session:
        invitation = session.reserve()
        with monkeypatch.context() as patch:
            patch.delattr(os, "pidfd_open", raising=False)
            if unsupported == "symbol":
                patch.setattr(protocol.ctypes, "CDLL", lambda *_args, **_kwargs: object())
                expected = RuntimeError
            else:

                def unavailable(*_args):
                    ctypes.set_errno(errno.ENOSYS)
                    return -1

                patch.setattr(protocol, "_libc_function", lambda *_: unavailable)
                expected = OSError
            with pytest.raises(expected, match="pidfd_open"):
                setup(invitation)
        session.cancel(invitation)
        receipt = session.receipt(invitation)
        assert receipt["cleanup_complete"] and receipt["guardian_created"] is False


def test_driver_death_cancels_live_worker_and_persists_reaped_receipt(service, tmp_path):
    driver = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import dataclasses,json,time; from merlin_experiments.execution.chia_native import Session; "
                f"session=Session({str(service[0])!r}); "
                "print(json.dumps(dataclasses.asdict(session.reserve())),flush=True); time.sleep(60)"
            ),
        ],
        env=service[1],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    driver_fd = protocol.pidfd_open(driver.pid)
    worker = None
    native_fd = None
    try:
        invitation = Invitation(**json.loads(driver.stdout.readline()))
        pidfile = tmp_path / "driver-loss-native.pid"
        command = [
            sys.executable,
            "-c",
            (
                "import os,time; from pathlib import Path; "
                f"Path({str(pidfile)!r}).write_text(str(os.getpid())); time.sleep(60)"
            ),
        ]
        worker = _worker(service, invitation, command, cwd=tmp_path)
        deadline = time.monotonic() + 10
        while not pidfile.exists():
            assert worker.poll() is None and time.monotonic() < deadline
            time.sleep(0.02)
        native_fd = protocol.pidfd_open(int(pidfile.read_text()))
        protocol.send_signal(driver_fd, signal.SIGKILL)
        driver.communicate(timeout=10)
        worker.communicate(timeout=12)
        stored = service[0].parent / "receipts" / invitation.supervisor / (invitation.invocation + ".json")
        receipt = json.loads(stored.read_text())
        assert receipt["cleanup_complete"] and receipt["guardian_reaped"]
        assert protocol.exited(native_fd)
    finally:
        if driver.poll() is None:
            protocol.send_signal(driver_fd, signal.SIGKILL)
            driver.communicate(timeout=10)
        if worker is not None and worker.poll() is None:
            worker.kill()
            worker.communicate(timeout=12)
        if native_fd is not None:
            os.close(native_fd)
        os.close(driver_fd)


def test_wrong_worker_pidfd_refuses_before_launch(service):
    with Session(service[0]) as session:
        invitation = session.reserve()
        descriptor = os.open("/dev/null", os.O_RDONLY)
        try:
            with protocol.connect(service[0]) as connection:
                protocol.send(
                    connection,
                    {
                        "operation": "attach",
                        "session": invitation.session,
                        "invocation": invitation.invocation,
                        "host": protocol.host_identity(),
                    },
                    [descriptor],
                )
                reply, attachments = protocol.receive(connection)
                protocol.close_fds(attachments)
                assert reply == {"error": "PermissionError"}
        finally:
            os.close(descriptor)


def _wait_file(path, process):
    deadline = time.monotonic() + 10
    while not path.exists():
        assert process.poll() is None and time.monotonic() < deadline
        time.sleep(0.02)


def test_success_reaps_detached_grandchild_and_preserves_an_unrelated_peer(service, tmp_path):
    marker = tmp_path / "grandchild.pid"
    release = tmp_path / "release-leader"
    detached = (
        "import os,signal,time; from pathlib import Path; "
        "os.setsid(); signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        f"Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(60)"
    )
    leader = (
        "import subprocess,sys,time; from pathlib import Path; "
        f"subprocess.Popen([sys.executable,'-c',{detached!r}]); "
        f"path=Path({str(release)!r})\n"
        "while not path.exists(): time.sleep(.02)\n"
    )
    peer = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(60)"])
    peer_fd = protocol.pidfd_open(peer.pid)
    worker = None
    child_fd = None
    try:
        with Session(service[0]) as session:
            invitation = session.reserve()
            worker = _worker(service, invitation, [sys.executable, "-c", leader], cwd=tmp_path)
            _wait_file(marker, worker)
            child_fd = protocol.pidfd_open(int(marker.read_text()))
            release.touch()
            _, error = worker.communicate(timeout=12)
            assert worker.returncode == 0, error
            receipt = session.receipt(invitation)
            assert receipt["cleanup_complete"] and receipt["guardian_reaped"]
            assert receipt["guardian"]["returncode"] == 0
            assert receipt["guardian"]["native_cleanup"]["reaped_children"] >= 1
            assert protocol.exited(child_fd)
            assert not protocol.exited(peer_fd)
    finally:
        if worker is not None and worker.poll() is None:
            worker.kill()
            worker.communicate(timeout=12)
        protocol.send_signal(peer_fd, signal.SIGTERM)
        peer.wait(timeout=5)
        os.close(peer_fd)
        if child_fd is not None:
            os.close(child_fd)


def test_abandoned_ready_setup_expires_without_launch(service):
    with Session(service[0]) as session:
        invitation = session.reserve()
        setup(invitation)
        try:
            receipt = session.receipt(invitation, timeout=13)
            assert receipt["cleanup_complete"] and receipt["guardian_reaped"]
            assert receipt["guardian"]["native_started"] is False
            assert receipt["guardian"]["reason"] == "admission_expired"
        finally:
            cleanup(invitation)


def test_pending_receipt_deadline_is_cleanup_incomplete(service):
    with Session(service[0]) as session:
        invitation = session.reserve()
        with pytest.raises(CleanupIncomplete, match="receipt unavailable"):
            session.receipt(invitation, timeout=0.001)


@pytest.mark.parametrize("error", [TimeoutError("transport deadline"), ConnectionError("transport broken")])
def test_receipt_normalizes_only_transport_timeouts(service, monkeypatch, error):
    from merlin_experiments.execution import chia_native

    with Session(service[0]) as session:
        invitation = session.reserve()
        with monkeypatch.context() as patch:

            def fail(*args, **kwargs):
                raise error

            patch.setattr(chia_native, "_exchange", fail)
            expected = CleanupIncomplete if isinstance(error, TimeoutError) else ConnectionError
            with pytest.raises(expected) as caught:
                session.receipt(invitation)
            if isinstance(error, TimeoutError):
                assert caught.value.__cause__ is error
            else:
                assert caught.value is error


@pytest.mark.parametrize("stop_mode", ["cancel", "service_sigterm"])
def test_explicit_stop_reaps_active_leader_and_detached_child(service, tmp_path, stop_mode):
    leader_pid, child_pid = tmp_path / "leader.pid", tmp_path / "child.pid"
    child = (
        "import os,signal,time; from pathlib import Path; "
        "os.setsid(); signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        f"Path({str(child_pid)!r}).write_text(str(os.getpid())); time.sleep(60)"
    )
    leader = (
        "import os,signal,subprocess,sys,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        f"subprocess.Popen([sys.executable,'-c',{child!r}]); "
        f"Path({str(leader_pid)!r}).write_text(str(os.getpid())); time.sleep(60)"
    )
    session = Session(service[0])
    invitation = session.reserve()
    worker = _worker(service, invitation, [sys.executable, "-c", leader], cwd=tmp_path)
    identities = []
    try:
        _wait_file(leader_pid, worker)
        _wait_file(child_pid, worker)
        identities = [protocol.pidfd_open(int(path.read_text())) for path in (leader_pid, child_pid)]
        if stop_mode == "cancel":
            session.cancel(invitation)
        else:
            supervisor_fd = protocol.pidfd_open(service[2])
            try:
                protocol.send_signal(supervisor_fd, signal.SIGTERM)
            finally:
                os.close(supervisor_fd)
        worker.communicate(timeout=12)
        stored = service[0].parent / "receipts" / session.supervisor / (invitation.invocation + ".json")
        receipt = json.loads(stored.read_text())
        assert receipt["cleanup_complete"] and receipt["guardian_reaped"]
        assert receipt["guardian"]["native_started"]
        assert receipt["guardian"]["native_cleanup"]["reaped_children"] >= 2
        assert all(protocol.exited(fd) for fd in identities)
        for fd in identities:
            fields = dict(line.split(":", 1) for line in Path(f"/proc/self/fdinfo/{fd}").read_text().splitlines())
            assert fields["Pid"].strip() == "-1", "native identity must be reaped, not merely exited"
        if stop_mode == "cancel":
            assert session.receipt(invitation) == receipt
            session.close()
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.communicate(timeout=12)
        protocol.close_fds(identities)


@pytest.mark.parametrize("malformed", ["argv", "descriptors", "oversized"])
def test_invalid_start_never_launches_and_guardian_is_reaped(service, tmp_path, malformed):
    session = Session(service[0])
    invitation = session.reserve()
    with protocol.connect(service[0]) as connection:
        identity = protocol.self_pidfd()
        try:
            protocol.send(
                connection,
                {
                    "operation": "attach",
                    "session": invitation.session,
                    "invocation": invitation.invocation,
                    "host": protocol.host_identity(),
                },
                [identity],
            )
            ready, attachments = protocol.receive(connection)
            protocol.close_fds(attachments)
            assert ready["state"] == "ready"
        finally:
            os.close(identity)
        command = {
            "operation": "start",
            "argv": [sys.executable, "-c", "raise AssertionError('must not execute')"],
            "cwd": str(tmp_path),
            "env": {},
        }
        if malformed == "argv":
            command["argv"] = 42
        if malformed == "oversized":
            command["env"] = {"OVERSIZED": "x" * protocol.MAX_PACKET}
            with pytest.raises(ValueError, match="exceeds limit"):
                protocol.send(connection, command, [0, 1, 2])
        else:
            protocol.send(connection, command, [] if malformed == "descriptors" else [0, 1, 2])
            finished, attachments = protocol.receive(connection)
            protocol.close_fds(attachments)
            assert finished["state"] == "finished"
    receipt = session.receipt(invitation)
    assert receipt["guardian_created"] and receipt["guardian_reaped"]
    if malformed == "oversized":
        assert receipt["cleanup_complete"]
        assert receipt["guardian"]["native_started"] is False
        session.close()
    else:
        assert not receipt["cleanup_complete"], "invalid protocol must not become a successful lifecycle"
        if receipt["guardian"]:
            assert receipt["guardian"]["native_started"] is False
            assert receipt["guardian"]["native_cleanup"]["complete"]
        with pytest.raises(CleanupIncomplete):
            session.close()
