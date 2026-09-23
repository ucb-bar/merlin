"""Independent native-group evidence and cleanup; synthetic work, no Ray required."""

from __future__ import annotations

import json
import subprocess
import sys
import time
import types
from types import SimpleNamespace

import pytest
import test_managed_native as native_tests
from merlin_experiments.execution.chia_group import NativeTaskGroup, local_options
from merlin_experiments.execution.chia_native import CleanupIncomplete, Session

service = native_tests.service


def _run(tmp_path):
    state = {"failed": False}
    return SimpleNamespace(run_dir=tmp_path, mark_failed=lambda: state.update(failed=True)), state


class Ref:
    def __init__(self, process):
        self.process = process

    def hex(self):
        return str(self.process.pid)


@pytest.mark.parametrize("cancel", [False, True])
def test_real_native_group_success_or_interrupted_owner(service, tmp_path, cancel):
    run, state = _run(tmp_path)
    session = Session(service[0])
    marker = tmp_path / "native-started"
    command = [
        sys.executable,
        "-c",
        f"from pathlib import Path; import time; Path({str(marker)!r}).touch(); time.sleep({60 if cancel else 0})",
    ]
    owner = SimpleNamespace(submit=lambda launch, *args, **kwargs: launch(*args, **kwargs))
    processes = []

    def launch(*args, **kwargs):
        process = native_tests._worker(service, kwargs["_chia_setup_args"][0], command, cwd=tmp_path)
        processes.append(process)
        return Ref(process)

    primary = ValueError("synthetic driver interruption")
    peer = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        try:
            with NativeTaskGroup(run, session) as group:
                ref = group.submit(owner, launch, run_id="case")
                deadline = time.monotonic() + 10
                while not marker.exists():
                    assert time.monotonic() < deadline
                    time.sleep(0.02)
                if cancel:
                    raise primary
                _, error = ref.process.communicate(timeout=10)
                assert ref.process.returncode == 0, error
                group.returned(ref, {"returncode": 0})
        except ValueError as error:
            assert cancel and error is primary
        assert peer.poll() is None
        document = json.loads((tmp_path / "chia/native.json").read_text())
        receipt = document["tasks"][0]["receipt"]
        assert document["cleanup_complete"] and receipt["cleanup_complete"]
        assert receipt["guardian_reaped"] and receipt["guardian"]["native_started"]
        assert state["failed"] == cancel
    finally:
        session.close()
        for process in processes:
            process.communicate(timeout=10)
        peer.terminate()
        peer.wait(timeout=5)


def test_reserved_dispatch_failure_preserves_primary_and_no_launch(service, tmp_path):
    run, state = _run(tmp_path)
    primary = ValueError("dispatcher lost acknowledgement")

    def fail(*args, **kwargs):
        raise primary

    with Session(service[0]) as session:
        with pytest.raises(ValueError) as caught:
            with NativeTaskGroup(run, session) as group:
                group.submit(SimpleNamespace(submit=fail), None, run_id="unacknowledged")
        assert caught.value is primary
    document = json.loads((tmp_path / "chia/native.json").read_text())
    assert state["failed"] and document["cleanup_complete"]
    assert not document["tasks"][0]["receipt"]["guardian_created"]


def test_bypass_return_cannot_masquerade_as_native_success(service, tmp_path):
    run, state = _run(tmp_path)
    with Session(service[0]) as session:
        with pytest.raises(CleanupIncomplete):
            with NativeTaskGroup(run, session) as group:
                ref = group.submit(
                    SimpleNamespace(submit=lambda *args, **kwargs: SimpleNamespace(hex=lambda: "bypass")),
                    None,
                    run_id="bypass",
                )
                group.returned(ref, {"returncode": 0})
    document = json.loads((tmp_path / "chia/native.json").read_text())
    assert state["failed"] and not document["cleanup_complete"]
    assert not document["tasks"][0]["receipt"]["guardian_created"]


def test_all_receipts_attempted_and_primary_survives_accounting_failure(service, tmp_path, monkeypatch):
    run, _ = _run(tmp_path)
    primary = ValueError("original task failure")

    def bad_mark():
        raise OSError("accounting failure")

    run.mark_failed = bad_mark
    session = Session(service[0])
    actual = session.receipt
    observed = []

    def receipt(invitation, **kwargs):
        observed.append(invitation.invocation)
        if invitation == session._invocations[0]:
            raise OSError("first receipt unavailable")
        return actual(invitation, **kwargs)

    try:
        with pytest.raises(ValueError) as caught:
            with NativeTaskGroup(run, session) as group:
                owner = SimpleNamespace(submit=lambda *args, **kwargs: SimpleNamespace(hex=lambda: "ref"))
                group.submit(owner, None, run_id="first")
                group.submit(owner, None, run_id="second")
                monkeypatch.setattr(session, "receipt", receipt)
                raise primary
        assert caught.value is primary and primary.__notes__
        assert all(invitation.invocation in observed for invitation in session._invocations)
        document = json.loads((tmp_path / "chia/native.json").read_text())
        assert document["tasks"][1]["receipt"]["cleanup_complete"]
        assert any(error["operation"] == "mark_failed" for error in document["cleanup_errors"])
    finally:
        monkeypatch.setattr(session, "receipt", actual)
        session.close()


def test_persistence_failure_keeps_document_incomplete(service, tmp_path, monkeypatch):
    run, state = _run(tmp_path)
    with Session(service[0]) as session:
        group = NativeTaskGroup(run, session)
        monkeypatch.setattr(group, "_write", lambda: (_ for _ in ()).throw(OSError("disk failure")))
        with pytest.raises(CleanupIncomplete):
            with group:
                pass
        assert not group.document["cleanup_complete"] and state["failed"]


@pytest.mark.parametrize("local_cpu", [0, 1])
def test_only_driver_node_resources_authorize_dispatch(monkeypatch, local_cpu):
    ray = types.ModuleType("ray")
    ray.get_runtime_context = lambda: SimpleNamespace(get_node_id=lambda: "driver")
    ray.nodes = lambda: [
        {"Alive": True, "NodeID": "driver", "Resources": {"CPU": local_cpu, "verilator": 1}},
        {"Alive": True, "NodeID": "unmanaged", "Resources": {"CPU": 64, "verilator": 64}},
    ]
    strategies = types.ModuleType("ray.util.scheduling_strategies")
    strategies.NodeAffinitySchedulingStrategy = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setitem(sys.modules, "ray.util.scheduling_strategies", strategies)
    if not local_cpu:
        with pytest.raises(RuntimeError, match="driver's own Ray node"):
            local_options({"verilator": 1})
    else:
        result = local_options({"verilator": 1})
        assert result["scheduling_strategy"].node_id == "driver"
        assert result["scheduling_strategy"].soft is False
