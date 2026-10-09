"""Owned native processes prove invocation closure, not target/OS timing roles."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from merlin_experiments.execution import _protocol as protocol
from merlin_experiments.execution.owned_children import become_child_subreaper
from merlin_experiments.phase2.contracts import StageGateError
from merlin_experiments.phase2.feedback_guardian import has_parent_feedback_lease
from merlin_experiments.phase2.portfolio_launch import acquire_host_resource_lease
from merlin_experiments.phase2.supervised_feedback import bounded_feedback


def wait_for(path, seconds=4):
    deadline = time.monotonic() + seconds
    while not path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(str(path))
        time.sleep(0.01)


def publish_descendant(root, document):
    pending = root / "descendant.pending"
    pending.write_text(json.dumps(document), encoding="utf-8")
    pending.replace(root / "descendant.json")


def paused_descendant_publication(root):
    original = Path.write_text

    def pause_after_create(path, text, *args, **kwargs):
        with path.open("w", encoding="utf-8") as stream:
            (root / "publication_created").touch()
            wait_for(root / "publication_release")
            return stream.write(text)

    Path.write_text = pause_after_create
    try:
        publish_descendant(root, {"pid": os.getpid(), "guardian": os.getppid()})
    finally:
        Path.write_text = original


def descendant_publication_reader(root):
    (root / "reader_started").touch()
    wait_for(root / "descendant.json")
    print(json.dumps(json.loads((root / "descendant.json").read_text())))


def double_fork(*, root, lease_path, ignore_term=False):
    assert has_parent_feedback_lease(lease_path)
    assert acquire_host_resource_lease(root / "worker-attempt", lease_path=lease_path) is None
    lease_fds = []
    for path in Path("/proc/self/fd").iterdir():
        try:
            if os.readlink(path) == str(lease_path):
                lease_fds.append(path)
        except FileNotFoundError:
            pass
    assert not lease_fds  # Neither private socket nor lease descriptor is delegated.
    guardian = os.getppid()
    first = os.fork()
    if first == 0:
        second = os.fork()
        if second:
            os._exit(0)
        os.setsid()
        if ignore_term:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        assert not has_parent_feedback_lease(lease_path)
        publish_descendant(root, {"pid": os.getpid(), "guardian": guardian})
        while True:
            time.sleep(1)
    os.waitpid(first, 0)
    wait_for(root / "descendant.json")
    # The test driver obtains a stable pidfd before allowing publication.
    wait_for(root / "publish")
    return {"published": True}


def coordinator_case(root):
    result = bounded_feedback(
        double_fork,
        timeout_s=3,
        output=root / "workers",
        kwargs={"root": root, "lease_path": root / "host.lease"},
        lease_path=root / "host.lease",
    )
    print(json.dumps(result))


def run_case(name, root):
    script = "import pathlib,runpy,sys; runpy.run_path(sys.argv[1])[sys.argv[2]](pathlib.Path(sys.argv[3]))"
    return subprocess.Popen(
        [sys.executable, "-c", script, str(Path(__file__).resolve()), name, str(root)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def test_descendant_identity_is_visible_only_after_complete_publication(tmp_path):
    writer = run_case("paused_descendant_publication", tmp_path)
    reader = run_case("descendant_publication_reader", tmp_path)
    try:
        wait_for(tmp_path / "publication_created")
        wait_for(tmp_path / "reader_started")
        # The actual reader cannot see the empty create-before-write window.
        assert not (tmp_path / "descendant.json").exists()
        assert reader.poll() is None
        (tmp_path / "publication_release").touch()
        _, writer_error = writer.communicate(timeout=4)
        output, reader_error = reader.communicate(timeout=4)
        assert writer.returncode == 0, writer_error
        assert reader.returncode == 0, reader_error
        assert json.loads(output) == {"pid": writer.pid, "guardian": os.getpid()}
    finally:
        for process in (writer, reader):
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=4)


def test_owned_double_fork_is_reaped_before_parent_lease_release(tmp_path):
    process = run_case("coordinator_case", tmp_path)
    fd = None
    try:
        wait_for(tmp_path / "descendant.json")
        child = json.loads((tmp_path / "descendant.json").read_text())
        fd = protocol.pidfd_open(child["pid"])
        assert not protocol.exited(fd)
        assert acquire_host_resource_lease(tmp_path / "driver-attempt", lease_path=tmp_path / "host.lease") is None
        (tmp_path / "publish").touch()
        stdout, stderr = process.communicate(timeout=6)
        assert process.returncode == 0, stderr
        assert json.loads(stdout) == {"published": True}
        assert protocol.exited(fd)
        receipts = list((tmp_path / "workers").glob("component_worker_*/lifecycle.json"))
        receipt = json.loads(receipts[0].read_text())
        assert receipt["cleanup"]["status"] == "COMPLETE" and receipt["guardian"]["reaped"]
        assert receipt["lease_release"]["status"] == "RELEASED"
        guard = json.loads((receipts[0].parent / "guardian_cleanup_0.json").read_text())
        assert guard["cleanup"]["complete"] and guard["cleanup"]["reaped_children"] >= 2
        lease = acquire_host_resource_lease(tmp_path / "reacquired", lease_path=tmp_path / "host.lease")
        assert lease is not None
        lease.close()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=6)
        if fd is not None:
            os.close(fd)


def coordinator_loss_case(root):
    bounded_feedback(
        double_fork,
        timeout_s=10,
        output=root / "workers",
        kwargs={"root": root, "lease_path": root / "host.lease", "ignore_term": True},
        lease_path=root / "host.lease",
    )


def owned_loss_harness(root):
    # Only this disposable harness changes subreaper state and adopts its guardian.
    become_child_subreaper()
    process = run_case("coordinator_loss_case", root)
    child_fd = guardian_fd = None
    try:
        wait_for(root / "descendant.json")
        child = json.loads((root / "descendant.json").read_text())
        child_fd = protocol.pidfd_open(child["pid"])
        guardian = child["guardian"]
        guardian_fd = protocol.pidfd_open(guardian)
        process.kill()
        process.wait(timeout=2)
        # The ignored TERM keeps native work live until guardian escalation.
        held = acquire_host_resource_lease(root / "loss-attempt", lease_path=root / "host.lease")
        assert held is None and not protocol.exited(child_fd)
        deadline = time.monotonic() + 5
        while not protocol.exited(guardian_fd):
            if time.monotonic() >= deadline:
                raise TimeoutError("independent guardian did not close after coordinator loss")
            time.sleep(0.01)
        os.waitpid(guardian, 0)
        assert protocol.exited(child_fd)
        cleanup_path = next((root / "workers").glob("component_worker_*/guardian_cleanup_0.json"))
        cleanup = json.loads(cleanup_path.read_text())
        assert cleanup["cleanup"]["complete"] and cleanup["cleanup"]["reaped_children"] >= 2
        lease = acquire_host_resource_lease(root / "loss-reacquired", lease_path=root / "host.lease")
        assert lease is not None
        lease.close()
        print(
            json.dumps(
                {
                    "guardian_reaped": True,
                    "descendants_reaped": True,
                    "lease_held_until_cleanup": True,
                    "target_roles": "UNKNOWN",
                    "namespace_isolation": "UNKNOWN",
                }
            )
        )
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=6)
        for fd in (child_fd, guardian_fd):
            if fd is not None:
                os.close(fd)


def test_independent_guardian_holds_custodial_lease_after_coordinator_loss(tmp_path):
    process = run_case("owned_loss_harness", tmp_path)
    stdout, stderr = process.communicate(timeout=9)
    assert process.returncode == 0, stderr
    assert json.loads(stdout)["lease_held_until_cleanup"] is True


def test_busy_parent_lease_refuses_before_callback_or_guardian(tmp_path):
    lease = acquire_host_resource_lease(tmp_path, lease_path=tmp_path / "host.lease")
    assert lease is not None
    try:
        with pytest.raises(StageGateError, match="parent resource lease is busy"):
            bounded_feedback(
                lambda: pytest.fail("busy resource launched work"),
                timeout_s=1,
                output=tmp_path / "workers",
                kwargs={},
                lease_path=tmp_path / "host.lease",
            )
        assert not list((tmp_path / "workers").glob("**/guardian_cleanup_*.json"))
    finally:
        lease.close()


def guardian_loss_coordinator(root):
    try:
        coordinator_loss_case(root)
    except StageGateError as error:
        (root / "refused").write_text(str(error))
        wait_for(root / "owner_exit")
    else:
        raise AssertionError("lost guardian authorized result")


def owned_guardian_loss_harness(root):
    become_child_subreaper()
    process = run_case("guardian_loss_coordinator", root)
    fds = []
    try:
        wait_for(root / "descendant.json")
        child = json.loads((root / "descendant.json").read_text())
        guardian_fd = protocol.pidfd_open(child["guardian"])
        fds.append(guardian_fd)
        protocol.send_signal(guardian_fd, signal.SIGKILL)
        wait_for(root / "refused")
        lifecycle_path = next((root / "workers").glob("component_worker_*/lifecycle.json"))
        lifecycle = json.loads(lifecycle_path.read_text())
        assert lifecycle["status"] == "refused"
        assert lifecycle["cleanup"]["status"] == lifecycle["lease_release"]["status"] == "UNKNOWN"
        assert acquire_host_resource_lease(root / "quarantine-attempt", lease_path=root / "host.lease") is None
        # Only this disposable test reaper adopts the lost guardian's children.
        # Manual cleanup cannot issue the normal controller's missing authority.
        children = [
            int(pid)
            for pid in Path(f"/proc/self/task/{os.getpid()}/children").read_text().split()
            if int(pid) != process.pid
        ]
        assert len(children) >= 2
        for pid in children:
            fd = protocol.pidfd_open(pid)
            fds.append(fd)
            protocol.send_signal(fd, signal.SIGKILL)
        for pid in children:
            os.waitpid(pid, 0)
        (root / "owner_exit").touch()
        stdout, stderr = process.communicate(timeout=3)
        assert process.returncode == 0, stderr + stdout
        print(
            json.dumps(
                {
                    "missing_cleanup_refused": True,
                    "live_parent_lease_retained": True,
                    "manual_test_children_reaped": True,
                    "target_roles": "UNKNOWN",
                }
            )
        )
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=3)
        from merlin_experiments.execution.owned_children import reap_owned_children

        assert reap_owned_children(None, 4)["complete"]
        for fd in fds:
            os.close(fd)


def test_actual_guardian_loss_retains_lease_and_refuses_instead_of_claiming_cleanup(tmp_path):
    process = run_case("owned_guardian_loss_harness", tmp_path)
    stdout, stderr = process.communicate(timeout=10)
    assert process.returncode == 0, stderr
    assert json.loads(stdout)["live_parent_lease_retained"] is True
