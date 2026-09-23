"""Caller-owned task cancellation never borrows ownership of cluster peers or OS descendants."""

from __future__ import annotations

import json
import os
import socket
import sys
import time
import types
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pytest

from merlin.benchharness.chia_bridge import ChiaRun


@pytest.fixture
def transport(tmp_path, monkeypatch):
    class Ref:
        def __init__(self, name, value=None, *, ready=False):
            self.name, self.value, self.ready = name, value, ready

        def hex(self):
            return self.name

    class Cancelled(Exception):
        pass

    state = {"cancelled": [], "waits": [], "acknowledge": True, "get_interrupt": False, "normal_gets": []}
    ray = types.ModuleType("ray")
    ray.ObjectRef = Ref

    def get(ref, *, timeout=None):
        if not ref.ready:
            raise TimeoutError("not terminal")
        if isinstance(ref.value, BaseException):
            raise ref.value
        return ref.value

    def cancel(ref, *, force, recursive):
        assert force is False and recursive is True
        state["cancelled"].append(ref.name)
        if state["acknowledge"]:
            ref.ready, ref.value = True, Cancelled()

    def wait(refs, *, num_returns, timeout):
        state["waits"].append(timeout)
        ready = [ref for ref in refs if ref.ready]
        return ready, [ref for ref in refs if ref not in ready]

    ray.get, ray.cancel, ray.wait = get, cancel, wait
    function = types.ModuleType("chia.base.ChiaFunction")

    def normal_get(ref):
        state["normal_gets"].append(ref.name)
        if state["get_interrupt"]:
            raise KeyboardInterrupt("operator interrupted")
        return get(ref)

    function.get = normal_get
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setitem(sys.modules, "chia.base.ChiaFunction", function)
    run = ChiaRun(handle=SimpleNamespace(run_dir=tmp_path), metrics=None, profile_path=tmp_path / "profile")
    return Ref, state, run


def receipt(run):
    return json.loads((run.run_dir / "chia/tasks.json").read_text())


def test_normal_result_uses_chia_public_get_and_preserves_child_code(transport):
    from merlin.benchharness.chia_tasks import chia_tasks

    Ref, state, run = transport
    with chia_tasks(run) as tasks:
        ref = tasks.submit(lambda: Ref("owned", {"returncode": 17}, ready=True))
        assert tasks.get(ref) == {"returncode": 17}
    assert state["normal_gets"] == ["owned"]
    assert state["cancelled"] == []
    assert receipt(run)["tasks"][0]["returncode"] == 17
    assert receipt(run)["native_descendants"] == "not_verified"


def test_partial_dispatch_cancels_owned_ref_only_and_records_unknown_dispatch(transport):
    from merlin.benchharness.chia_tasks import chia_tasks

    Ref, state, run = transport
    peer = Ref("borrowed-peer")

    def failed_dispatch():
        raise RuntimeError("dispatch failed")

    with pytest.raises(RuntimeError, match="dispatch failed") as error:
        with chia_tasks(run) as tasks:
            tasks.submit(lambda: Ref("owned"))
            tasks.submit(failed_dispatch)
    assert state["cancelled"] == ["owned"] and not peer.ready
    assert receipt(run)["tasks"][1]["state"] == "dispatch_unknown"
    assert receipt(run)["cleanup_complete"] is False
    assert any("cleanup" in note for note in error.value.__notes__)


def test_get_interruption_is_preserved_after_cancellation(transport):
    from merlin.benchharness.chia_tasks import chia_tasks

    Ref, state, run = transport
    state["get_interrupt"] = True
    with pytest.raises(KeyboardInterrupt, match="operator interrupted"):
        with chia_tasks(run) as tasks:
            tasks.get(tasks.submit(lambda: Ref("owned")))
    assert state["cancelled"] == ["owned"]
    assert receipt(run)["cleanup_complete"] is True


def test_nonacknowledgement_is_bounded_and_not_reported_as_stopped(transport):
    from merlin.benchharness.chia_tasks import TaskCleanupIncomplete, chia_tasks

    Ref, state, run = transport
    state["acknowledge"] = False
    started = time.monotonic()
    with pytest.raises(TaskCleanupIncomplete, match="not acknowledged"):
        with chia_tasks(run, teardown_timeout_s=0.01) as tasks:
            tasks.submit(lambda: Ref("first"))
            tasks.submit(lambda: Ref("second"))
    assert time.monotonic() - started < 1
    assert state["cancelled"] == ["first", "second"]
    assert sum(state["waits"]) <= 0.01
    assert receipt(run)["cleanup_complete"] is False
    assert {row["state"] for row in receipt(run)["tasks"]} == {"cancel_unacknowledged"}


def test_task_exception_does_not_discard_other_results(transport):
    from merlin.benchharness.chia_tasks import chia_tasks

    Ref, state, run = transport
    with chia_tasks(run) as tasks:
        bad = tasks.submit(lambda: Ref("bad", ValueError("worker failed"), ready=True))
        good = tasks.submit(lambda: Ref("good", {"returncode": 0}, ready=True))
        with pytest.raises(ValueError, match="worker failed"):
            tasks.get(bad)
        assert tasks.get(good) == {"returncode": 0}
    assert state["cancelled"] == []
    assert [row["state"] for row in receipt(run)["tasks"]] == ["failed", "returned"]


def test_receipt_write_failure_does_not_replace_dispatch_failure(transport, monkeypatch):
    from merlin.benchharness.chia_tasks import chia_tasks

    _, _, run = transport

    def broken_replace(*args):
        raise OSError("receipt filesystem unavailable")

    def dispatch():
        monkeypatch.setattr(os, "replace", broken_replace)
        raise ValueError("original dispatcher failed")

    with pytest.raises(ValueError, match="original dispatcher failed") as error:
        with chia_tasks(run) as tasks:
            tasks.submit(dispatch)
    assert any("receipt" in note for note in error.value.__notes__)


def test_unsupported_wrapped_reference_is_not_unwrapped_privately(transport):
    from merlin.benchharness.chia_tasks import chia_tasks

    Ref, state, run = transport
    with pytest.raises(TypeError, match="raw public Ray ObjectRef"):
        with chia_tasks(run) as tasks:
            tasks.submit(lambda: SimpleNamespace(ref=Ref("opaque")))
    assert state["cancelled"] == []
    assert receipt(run)["tasks"][0]["state"] == "dispatch_unknown"


def test_duplicate_reference_cannot_replace_its_original_owner(transport):
    from merlin.benchharness.chia_tasks import chia_tasks

    Ref, state, run = transport
    ref = Ref("owned")
    with pytest.raises(ValueError, match="already owned"):
        with chia_tasks(run) as tasks:
            tasks.submit(lambda: ref)
            tasks.submit(lambda: ref)
    assert state["cancelled"] == ["owned"]
    assert receipt(run)["tasks"][1]["state"] == "dispatch_unknown"


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_invalid_teardown_budget_refuses_before_dispatch(transport, timeout):
    from merlin.benchharness.chia_tasks import chia_tasks

    _, state, run = transport
    with pytest.raises(ValueError, match="positive finite"):
        with chia_tasks(run, teardown_timeout_s=timeout):
            pytest.fail("invalid budget entered task ownership scope")
    assert state["cancelled"] == []


@pytest.mark.skipif(
    os.environ.get("MERLIN_TEST_CHIA_RAY") != "1", reason="opt-in isolated Ray cancellation qualification"
)
def test_real_ray_cancels_only_owned_task_in_borrowed_cluster(tmp_path, monkeypatch):
    if sys.platform != "linux" or {name for _, name in socket.if_nameindex()} != {"lo"}:
        pytest.fail("Ray qualification requires a loopback-only network namespace/container (--network=none)")
    import ray
    from chia.base.ChiaFunction import ChiaFunction

    from merlin.benchharness.chia_bridge import chia_run
    from merlin.benchharness.chia_tasks import chia_tasks

    if ray.is_initialized():
        pytest.fail("requires its own isolated Ray process")
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path))
    monkeypatch.delenv("CHIA_AET_SINK", raising=False)

    @ChiaFunction(num_cpus=1, max_retries=0)
    def owned(start):
        Path(start).write_text("started")
        while True:
            time.sleep(0.02)

    @ray.remote(num_cpus=0, max_retries=0)
    def peer(release):
        while not Path(release).exists():
            time.sleep(0.02)
        return "peer-finished"

    with TemporaryDirectory(prefix="mctask-", dir="/tmp") as ray_temp:
        monkeypatch.setenv("RAY_TMPDIR", ray_temp)
        try:
            ray.init(address="local", num_cpus=1, include_dashboard=False, object_store_memory=80 * 1024 * 1024)
            release = tmp_path / "peer-release"
            peer_ref = peer.remote(str(release))
            with pytest.raises(RuntimeError, match="operator stop"):
                with chia_run(suite="task-contract", method="fixture", target="fixture") as run:
                    with chia_tasks(run, teardown_timeout_s=10) as tasks:
                        started = tmp_path / "owned-started"
                        tasks.submit(owned.chia_remote, str(started))
                        deadline = time.monotonic() + 15
                        while not started.exists():
                            if time.monotonic() >= deadline:
                                raise AssertionError("owned task did not start")
                            time.sleep(0.02)
                        raise RuntimeError("operator stop")
            assert receipt(run)["cleanup_complete"] is True
            assert ray.is_initialized()
            assert ray.wait([peer_ref], timeout=0)[0] == []
            release.write_text("go")
            assert ray.get(peer_ref, timeout=10) == "peer-finished"
        finally:
            ray.shutdown()
