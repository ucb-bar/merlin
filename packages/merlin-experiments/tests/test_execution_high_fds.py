"""Real host descriptors and callback cleanup; no target execution evidence."""

import fcntl
import json
import os
import resource
import signal
import socket
import time
from contextlib import contextmanager

import pytest
from merlin_experiments.execution import _protocol as protocol
from merlin_experiments.phase2.contracts import StageGateError
from merlin_experiments.phase2.supervised_feedback import bounded_feedback

from merlin.common.owned_process import _readable


@contextmanager
def high_fd_pressure():
    """Keep new control sockets/pidfds above select's 1023 descriptor limit."""
    if resource.getrlimit(resource.RLIMIT_NOFILE)[0] < 1200:
        pytest.skip("high-descriptor control requires at least 1200 open descriptors")
    descriptors = []
    try:
        while not descriptors or descriptors[-1] < 1050:
            descriptors.append(os.open(os.devnull, os.O_RDONLY | os.O_CLOEXEC))
        yield
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def require_high_descriptor():
    if resource.getrlimit(resource.RLIMIT_NOFILE)[0] <= 2048:
        pytest.skip("high-descriptor identity control requires descriptor 2048")


def test_readable_reuses_original_selector_and_preserves_high_socket_identity():
    assert protocol.readable is _readable
    require_high_descriptor()
    left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    high = socket.socket(fileno=fcntl.fcntl(left.fileno(), fcntl.F_DUPFD_CLOEXEC, 2048))
    try:
        assert high.fileno() >= 2048
        assert protocol.readable([high], 0) == []
        right.send(b"complete atomic packet")
        assert protocol.readable([high], 1) == [high]
        assert high.recv(128) == b"complete atomic packet"
        assert protocol.readable([high], 0) == []
        right.close()
        assert protocol.readable([high], 1) == [high]  # EOF must remain observable.
    finally:
        high.close()
        left.close()
        right.close()


def test_exited_checks_the_actual_high_pidfd_without_pid_reuse():
    require_high_descriptor()
    read_end, write_end = os.pipe2(os.O_CLOEXEC)
    child = os.fork()
    if child == 0:
        os.close(write_end)
        os.read(read_end, 1)
        os._exit(0)
    os.close(read_end)
    descriptor = protocol.pidfd_open(child)
    high = fcntl.fcntl(descriptor, fcntl.F_DUPFD_CLOEXEC, 2048)
    try:
        assert not protocol.exited(high)
        os.close(write_end)
        write_end = None
        assert protocol.readable([high], 2) == [high]
        assert protocol.exited(high)
        assert os.waitpid(child, 0) == (child, 0)
        child = None
    finally:
        if write_end is not None:
            os.close(write_end)
        if child is not None:
            protocol.send_signal(high, signal.SIGKILL)
            os.waitpid(child, 0)
        os.close(high)
        os.close(descriptor)


def test_closed_descriptor_is_a_refusal_instead_of_no_exit():
    descriptor = os.open(os.devnull, os.O_RDONLY)
    os.close(descriptor)
    with pytest.raises((OSError, ValueError)):
        protocol.exited(descriptor)


def neutral_callback(*, stalled):
    if stalled:
        time.sleep(30)
    return {"neutral": "complete host callback"}


@pytest.mark.parametrize("stalled", [False, True])
def test_real_high_fd_guardian_preserves_result_deadline_and_reaped_lease(tmp_path, stalled):
    with high_fd_pressure():
        if stalled:
            with pytest.raises(StageGateError, match="deadline"):
                bounded_feedback(
                    neutral_callback,
                    timeout_s=0.15,
                    output=tmp_path / "workers",
                    kwargs={"stalled": True},
                    lease_path=tmp_path / "host.lease",
                )
        else:
            assert bounded_feedback(
                neutral_callback,
                timeout_s=3,
                output=tmp_path / "workers",
                kwargs={"stalled": False},
                lease_path=tmp_path / "host.lease",
            ) == {"neutral": "complete host callback"}
    [receipt_path] = (tmp_path / "workers").glob("component_worker_*/lifecycle.json")
    receipt = json.loads(receipt_path.read_text())
    assert receipt["guardian"]["reaped"] and receipt["worker"]["reaped"]
    assert receipt["cleanup"]["status"] == "COMPLETE"
    assert receipt["lease_release"]["status"] == "RELEASED"
    assert receipt["status"] == ("refused" if stalled else "returned")
    assert receipt["deadline_exceeded"] is stalled
    assert str(protocol.readable.__code__.co_filename) in {row[0] for row in receipt["sources"]}


def test_native_guardian_high_fd_cancel_before_start_keeps_no_launch(tmp_path):
    from merlin_experiments.execution.native_guardian import supervise

    # Only the original host guardian runs: cancellation precedes any command.
    with high_fd_pressure():
        control, child_control = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        owner = protocol.self_pidfd()
        child = os.fork()
        if child == 0:
            control.close()
            receipt = supervise(child_control, owner, owner)
            (tmp_path / "native-guardian.json").write_text(json.dumps(receipt))
            os._exit(0)
        child_control.close()
        child_fd = protocol.pidfd_open(child)
        try:
            assert protocol.readable([control], 3) == [control]
            ready, fds = protocol.receive(control)
            assert not fds and ready["state"] == "ready"
            protocol.send(control, {"operation": "cancel"})
            assert protocol.readable([child_fd], 3) == [child_fd]
            assert os.waitpid(child, 0) == (child, 0)
            child = None
        finally:
            if child is not None:
                protocol.send_signal(child_fd, signal.SIGKILL)
                os.waitpid(child, 0)
            os.close(child_fd)
            os.close(owner)
            control.close()
    receipt = json.loads((tmp_path / "native-guardian.json").read_text())
    assert receipt["native_started"] is False
    assert receipt["reason"] == "cancelled" and receipt["error"] is None
    assert receipt["native_cleanup"]["complete"] is True
