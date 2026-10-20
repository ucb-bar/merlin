"""Original pass owners across neutral broker threads; no target qualification."""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import agent_workspace as AW
from merlin_experiments.phase2 import broker as B
from merlin_experiments.phase2 import broker_file_channel as F
from merlin_experiments.phase2 import broker_policy as BP
from merlin_experiments.phase2 import corpus_feedback as CF
from test_agent_sandbox_inputs import case as sandbox_case
from test_broker_file_channel import reply, submit

from merlin.common.cpu_pool import shared_compute_pool
from merlin.common.digest import sha256_file
from merlin.llvmlower.reviewed_pass_capabilities import (
    PassCapabilityBinding,
    ReviewedPassCapabilities,
    ReviewedPassCapability,
)
from merlin.targetgen.compiler_pass_selection import (
    CompilerPassSelection,
    active_compiler_passes,
    scoped_compiler_passes,
)


class NeutralOriginalOwner:
    def __init__(self, evidence):
        self.evidence = evidence
        self.identity = sha256_file(evidence)

    def revalidate(self):
        if sha256_file(self.evidence) != self.identity:
            raise ValueError("neutral original evidence changed")


@pytest.fixture
def selected(tmp_path):
    evidence = tmp_path / "neutral-evidence"
    evidence.write_text("source-bound host threading control\n")
    owner = NeutralOriginalOwner(evidence)
    binding = PassCapabilityBinding("hardware", "software", "integer", "runtime", "library")
    with shared_compute_pool(timeout_s=30):
        registry = ReviewedPassCapabilities((ReviewedPassCapability("neutral_control", binding, owner),))
        selected = CompilerPassSelection(registry, binding, mechanisms=("neutral_control",))
    return selected, evidence


def make_selected_broker(root, selected, monkeypatch, *, workflow_selection=None):
    candidate = root / "candidate"
    candidate.mkdir(parents=True)
    receipt = root / "control/receipts.jsonl"
    target = SimpleNamespace()
    workflow = BP.select_workflow(
        BP.CORPUS_FEEDBACK_V1, candidate=candidate, target_experiment=target, receipt_path=receipt
    )
    workflow._compiler_pass_selection = selected if workflow_selection is None else workflow_selection

    def inspect(_):
        assert active_compiler_passes() is selected
        return SimpleNamespace(to_dict=lambda: {"neutral_original_pass": selected.identity})

    monkeypatch.setattr(CF, "inspect_compiler_package", inspect)
    policy = AW.AgentSandboxPolicy(
        (), (), "available_not_an_isolation_claim", True, True, True, compiler_pass_selection=selected
    )
    return B.Broker(
        policy, target, candidate,
        (B.BrokerAction(BP.INVENTORY_ACTION, ("inventory",), (), "neutral inventory", False),),
        receipt, deadline=time.monotonic() + 15, workflow=workflow, max_calls=2, max_tool_seconds=5,
    )


def test_thread_without_explicit_pass_selection_keeps_original_escape_refusal(selected, tmp_path, monkeypatch):
    selection, _ = selected
    broker = make_selected_broker(tmp_path / "legacy-missing-transport", selection, monkeypatch)
    del broker._compiler_pass_selection  # Exact pre-fix constructor omission.
    with scoped_compiler_passes(selection), ThreadPoolExecutor(max_workers=1) as workers:
        future = workers.submit(broker.execute, {"action": BP.INVENTORY_ACTION})
        with pytest.raises(RuntimeError, match="escaped their original scoped owner"):
            future.result(timeout=2)
    assert not broker.calls and broker.receipt_path.read_bytes() == b""


def test_original_selected_pass_crosses_file_thread_and_joins_receipt(selected, tmp_path, monkeypatch):
    selection, _ = selected
    broker = make_selected_broker(tmp_path / "explicit", selection, monkeypatch)
    with scoped_compiler_passes(selection):
        with broker.serving_files(workspace=broker.candidate.parent, control_dir=broker.receipt_path.parent) as channel:
            submit(channel, broker.token)
            response = reply(broker)
            assert response["returncode"] == 0
            assert json.loads(response["stdout"]) == {"neutral_original_pass": selection.identity}
    rows = [json.loads(line) for line in broker.receipt_path.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["state"] == "complete"
    CF.verify_broker_action_receipts(
        broker.receipt_path, tuple(broker.actions.values()),
        {"broker_invocations": [{"action": BP.INVENTORY_ACTION,
                                  "bindings_sha256": rows[0]["bindings_command_sha256"]}]},
    )
    assert active_compiler_passes() is None


def test_both_original_policy_constructors_capture_same_passes(selected, tmp_path, monkeypatch):
    selection, _ = selected
    original = sandbox_case.__wrapped__(tmp_path, monkeypatch)
    with scoped_compiler_passes(selection):
        inside = AW.inner_execution_policy(
            original.target, original.candidate, original.snapshot, inputs=original.inputs
        )
        outside = AW.outer_codex_policy(
            original.candidate, original.snapshot, (), original.target, inputs=original.inputs
        )
        assert inside.compiler_pass_selection is outside.compiler_pass_selection is selection
        assert B.inner_command(inside, original.target, original.candidate, ["/usr/bin/true"], 3)


def test_changed_original_pass_evidence_refuses_before_action(selected, tmp_path, monkeypatch):
    selection, evidence = selected
    broker = make_selected_broker(tmp_path / "changed", selection, monkeypatch)
    with scoped_compiler_passes(selection):
        with broker.serving_files(workspace=broker.candidate.parent, control_dir=broker.receipt_path.parent) as channel:
            original = evidence.read_bytes()
            evidence.write_text("changed original evidence\n")
            try:
                submit(channel, broker.token)
                assert reply(broker)["returncode"] == 125
                assert not broker.calls and broker.receipt_path.read_bytes() == b""
            finally:
                evidence.write_bytes(original)


def test_policy_cannot_swap_original_pass_owner(selected, tmp_path, monkeypatch):
    selection, _ = selected
    other = CompilerPassSelection(selection.registry, replace(selection.binding, software="foreign"),
                                  mechanisms=("neutral_control",))
    root = tmp_path / "wrong-owner"
    with pytest.raises(B.StageGateError, match="different compiler passes"):
        make_selected_broker(root, selection, monkeypatch, workflow_selection=other)
    assert not (root / "control").exists()
    policy = AW.AgentSandboxPolicy((), (), "available_not_an_isolation_claim", True, True, True,
                                  compiler_pass_selection=selection)
    with scoped_compiler_passes(other), pytest.raises(B.StageGateError, match="captured owner"):
        policy.verify_execution()
    with pytest.raises(B.StageGateError, match="original compiler pass selection"):
        replace(policy, compiler_pass_selection=object())


@pytest.mark.parametrize("reply_failure", [False, True])
def test_unexpected_dispatch_failure_replies_then_refuses_and_drains(selected, tmp_path, monkeypatch, reply_failure):
    selection, _ = selected
    broker = make_selected_broker(tmp_path / "unexpected", selection, monkeypatch)
    closed = []
    broker.workflow.evaluation_pool = SimpleNamespace(close=lambda: closed.append(True))

    def failure(_):
        raise RuntimeError("private source detail must stay host-only")

    monkeypatch.setattr(broker, "execute", failure)
    attempted_reply = threading.Event()
    if reply_failure:
        def failed_reply(*_):
            attempted_reply.set()
            raise OSError("reply destination unavailable")

        monkeypatch.setattr(F.IO, "_write", failed_reply)
    with pytest.raises(B.StageGateError, match="transport failed") as raised:
        with broker.serving_files(workspace=broker.candidate.parent, control_dir=broker.receipt_path.parent) as channel:
            submit(channel, broker.token)
            if reply_failure:
                assert attempted_reply.wait(2)
            else:
                response = reply(broker)
                assert response["returncode"] == 125
                assert "RuntimeError" in response["stderr"] and "private source" not in response["stderr"]
    assert isinstance(raised.value.__cause__, RuntimeError)
    if reply_failure:
        assert "failure reply also failed: OSError" in raised.value.__cause__.__notes__[0]
    assert closed == [True] and not broker.calls and broker.receipt_path.read_bytes() == b""
    assert not any(t.name.startswith("original-file-broker") for t in threading.enumerate())
