"""A capsule owner publishes a verified handoff without starting model workers."""

import importlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.common.paths import repo_root


@pytest.fixture
def world(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(repo_root() / "examples/gemmini/phase1/launch"))
    module = importlib.import_module("capsule_supervision")
    clock = [1000.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(module.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    monkeypatch.setattr(module.signal, "setitimer", lambda *a: None)
    monkeypatch.setattr(module.signal, "getitimer", lambda *a: (0, 0))
    process = SimpleNamespace(pid=123456, poll=lambda: 0)

    class Processes:
        known = {process.pid: (7, "native")}

        def __init__(self):
            self.starts = []
            self.closed = False

        def start(self, argv, path, **kwargs):
            self.starts.append((argv, path, kwargs))
            return {"process": process}

        def owned(self):
            return {}

        def close(self):
            self.closed = True

    processes = Processes()
    receipt = {"receipt": {"path": str(tmp_path / "certification.json"), "sha256": "a" * 64}}
    calls = []
    monkeypatch.setattr(module, "verify_handoff", lambda *args: calls.append(args) or receipt)
    launch = SimpleNamespace(
        document={
            "validation_scope": "capsules",
            "budgets": {"supervisor_backstop_s": 10800},
            "external_validation": {"budget_s": 7200, "start_after": "capsule_owner_terminal",
                                    "submission": "verified_frozen_capsule_handoff"},
            "inputs": {"repo": str(tmp_path), "descriptor": str(tmp_path / "descriptor")},
            "launcher": {},
        },
        variables={"RUN": str(tmp_path / "run"), "SUPERVISOR_PSS_MAX_BYTES": "100",
                   "SUPERVISOR_HOST_HEADROOM_BYTES": "10"},
        path=Path, expand=lambda value: value, manager_cpus=[6, 7], argv=["controller"],
    )
    return SimpleNamespace(module=module, launch=launch, clock=clock, process=process,
                           processes=processes, receipt=receipt, calls=calls, prep=tmp_path)


def run(w, **kwargs):
    return w.module.run(w.launch, w.prep, 0, memory=kwargs.pop("memory", lambda _: (1, 100)),
                        processes_factory=lambda: w.processes, env={}, poll_s=100, **kwargs)


def test_only_terminal_successful_native_verifies_handoff_and_prep_is_outside_clock(world):
    w = world
    assert run(w) == 0
    state = json.loads((w.prep / "status.json").read_text())
    assert state["capsule_complete"] is True
    assert state["external_full_models_complete"] is False and state["phase1_complete"] is False
    assert state["preparation_wall_s"] == 1000 and state["capsule_budget_s"] == 10800
    assert state["elapsed_s"] == 0
    assert [item[2]["role"] for item in w.processes.starts] == ["native"]
    assert len(w.calls) == 1 and w.processes.closed
    assert state["capsule_certification"] == w.receipt
    assert state["external_validation"]["status"] == "not_started"


def test_native_failure_never_verifies_or_claims_capsule_completion(world):
    w = world
    w.process.poll = lambda: 1
    assert run(w) == 1
    assert not w.calls and w.processes.closed
    assert json.loads((w.prep / "status.json").read_text())["capsule_complete"] is False


def test_verifier_refusal_closes_owned_tree_and_is_not_a_pass(world, monkeypatch):
    w = world
    def refuse(*args):
        raise ValueError("actual frozen grade changed")
    monkeypatch.setattr(w.module, "verify_handoff", refuse)
    with pytest.raises(ValueError, match="frozen grade changed"):
        run(w)
    state = json.loads((w.prep / "status.json").read_text())
    assert state["capsule_complete"] is False and w.processes.closed


@pytest.mark.parametrize("reason", ["deadline", "memory", "headroom"])
def test_boundaries_stop_native_and_do_not_start_external_workers(world, reason):
    w = world
    w.process.poll = lambda: None
    if reason == "deadline":
        w.launch.document["budgets"]["supervisor_backstop_s"] = 1
    memory = {"deadline": (1, 100), "memory": (101, 100), "headroom": (1, 9)}[reason]
    assert run(w, memory=lambda _: memory) == 2
    assert not w.calls and w.processes.closed
    state = json.loads((w.prep / "status.json").read_text())
    assert state["capsule_complete"] is False
    assert state["termination_reason"] == ("capsule_deadline" if reason == "deadline" else "capsule_memory_guard")


def test_deadline_also_interrupts_slow_strict_receipt_reopening(world, monkeypatch):
    w = world
    def interrupted(*args):
        raise w.module.CapsuleStopped("capsule_deadline")
    monkeypatch.setattr(w.module, "verify_handoff", interrupted)
    assert run(w) == 2 and w.processes.closed
    assert json.loads((w.prep / "status.json").read_text())["capsule_complete"] is False


def test_an_existing_shorter_owner_alarm_is_not_extended(world, monkeypatch):
    calls = []
    monkeypatch.setattr(world.module.signal, "getitimer", lambda *args: (5, 0))
    monkeypatch.setattr(world.module.signal, "setitimer", lambda *args: calls.append(args))
    assert run(world) == 0
    assert calls[0][1] == 5
    assert calls[-1][1] == 5


def test_actual_controller_process_is_reaped_before_external_owner_can_start(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(repo_root() / "examples/gemmini/phase1/launch"))
    module = importlib.import_module("capsule_supervision")
    supervisor = importlib.import_module("launch_supervisor")
    monkeypatch.setattr(module, "verify_handoff", lambda *args: {"receipt": {"path": "fixture", "sha256": "a" * 64}})
    launch = SimpleNamespace(
        document={"validation_scope": "capsules", "budgets": {"supervisor_backstop_s": 10},
                  "external_validation": {"budget_s": 10, "start_after": "capsule_owner_terminal",
                                          "submission": "verified_frozen_capsule_handoff"},
                  "inputs": {"repo": str(tmp_path), "descriptor": str(tmp_path / "descriptor")}, "launcher": {}},
        variables={"RUN": str(tmp_path / "run"), "SUPERVISOR_PSS_MAX_BYTES": "1000000000",
                   "SUPERVISOR_HOST_HEADROOM_BYTES": "1"}, path=Path, expand=lambda value: value,
        manager_cpus=[min(os.sched_getaffinity(0))], argv=[sys.executable, "-c", "print('controller completed')"],
    )
    processes = supervisor.Processes()
    assert module.run(launch, tmp_path, time.monotonic() - 100, memory=lambda _: (0, 1000000000),
                      processes_factory=lambda: processes, env=dict(os.environ), poll_s=0.01) == 0
    assert (tmp_path / "phase1-native.log").read_text().strip() == "controller completed"
    assert len(processes.children) == 1 and processes.children[0]["process"].returncode == 0
    assert not processes.owned()
