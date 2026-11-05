"""Broker ownership and request transport, using synthetic compilation only.

The substituted core owner supplies no compiler, numerical or runtime verdict.
Complete native execution is exercised separately by the core integration test.
"""

import json
import os
import sys
import time
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import broker as B
from merlin_experiments.phase2 import broker_policy as BP
from merlin_experiments.phase2 import source_choices_workflow as S


@pytest.fixture
def selection(tmp_path, monkeypatch):
    class Owner:
        def __init__(self):
            self.source = tmp_path / "public_source.mlir"
            self.output_root = tmp_path / "control" / "private_products"
            self.max_request_bytes = 4096
            self.seen = []
            self.verify_hook = self.record_hook = None

        def revalidate(self):
            if self.verify_hook is not None:
                self.verify_hook()

        def compile(self, document):
            self.seen.append(document)

            def record():
                if self.record_hook is not None:
                    self.record_hook()
                return {"schema": "synthetic_transport_only", "choice_id": document["choice_id"]}

            return SimpleNamespace(record=record)

    monkeypatch.setitem(sys.modules, "merlin.llvmlower.source_choices", SimpleNamespace(SourceDenseChoices=Owner))
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    owner = Owner()
    owner.output_root.mkdir(parents=True)
    receipt = tmp_path / "control" / "receipts.jsonl"
    target = SimpleNamespace()
    workflow = BP.select_workflow(
        BP.SOURCE_CHOICES_V1,
        candidate=candidate,
        target_experiment=target,
        receipt_path=receipt,
        source_choices=owner,
    )
    return SimpleNamespace(owner=owner, candidate=candidate, receipt=receipt, target=target, workflow=workflow)


def broker(selection, actions=None):
    return B.Broker(
        SimpleNamespace(argv=(), network="available_not_an_isolation_claim", clear_environment=True),
        selection.target,
        selection.candidate,
        actions or selection.workflow.build_registry(),
        selection.receipt,
        deadline=time.monotonic() + 120,
        max_calls=4,
        max_tool_seconds=30,
        workflow=selection.workflow,
    )


def invoke(selection, document=None, *, raw=None, active=None):
    path = selection.candidate / "request.json"
    payload = raw if raw is not None else json.dumps(document).encode()
    path.write_bytes(payload)
    return (active or broker(selection)).execute({"action": S.ACTION, "bindings": {"request": path.name}})


def test_fixed_host_registry_needs_no_candidate_manifest(selection):
    actions = BP.action_registry(BP.SOURCE_CHOICES_V1, selection.candidate, selection.target)
    assert actions == (S.action(),) == selection.workflow.build_registry()
    reply = invoke(selection, {"choice_id": "declared"})
    assert reply["returncode"] == 0
    assert selection.owner.seen == [{"choice_id": "declared"}]
    retained = selection.receipt.parent / "source_choice_requests" / "0000.json"
    assert json.loads(retained.read_bytes()) == selection.owner.seen[0]
    receipt = json.loads(selection.receipt.read_bytes())
    assert receipt["workflow_id"] == BP.SOURCE_CHOICES_V1 and receipt["state"] == "complete"
    assert str(selection.receipt.parent) not in reply["stdout"]


@pytest.mark.parametrize("kind", ["extra_action", "changed_command", "unavailable"])
def test_sealed_registry_cannot_add_candidate_commands(selection, kind):
    if kind == "extra_action":
        actions = (S.action(), B.BrokerAction("candidate-run", ("arbitrary",), (), "no", False))
    else:
        original = S.action()
        actions = (
            B.BrokerAction(
                original.name,
                ("arbitrary", "{request}") if kind == "changed_command" else original.argv_template,
                original.placeholders,
                original.purpose,
                original.required,
                "not installed" if kind == "unavailable" else None,
            ),
        )
    with pytest.raises(B.StageGateError, match="exact fixed"):
        broker(selection, actions)
    assert not selection.owner.seen


def test_saved_descriptor_cannot_replace_live_owner(selection):
    with pytest.raises(B.StageGateError, match="exact live"):
        BP.select_workflow(
            BP.SOURCE_CHOICES_V1,
            candidate=selection.candidate,
            target_experiment=selection.target,
            receipt_path=selection.receipt,
            source_choices={"source_sha256": "a" * 64},
        )


@pytest.mark.parametrize("overlap", ["candidate_outputs", "candidate_control", "ancestor_control"])
def test_original_private_products_cannot_overlap_candidate(selection, overlap):
    receipt = selection.receipt
    if overlap == "candidate_outputs":
        selection.owner.output_root = selection.candidate / "private_products"
        receipt = selection.candidate.parent / "receipts.jsonl"
    elif overlap == "candidate_control":
        receipt = selection.candidate / "control" / "receipts.jsonl"
        selection.owner.output_root = receipt.parent / "products"
    else:
        receipt = selection.candidate.parent / "receipts.jsonl"
    with pytest.raises(B.StageGateError, match="private source and output"):
        S.SourceChoicesPolicy(
            candidate=selection.candidate,
            target_experiment=selection.target,
            receipt_path=receipt,
            source_choices=selection.owner,
        )


def test_nested_request_refuses_before_opening_any_path(selection, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("nested request must never reach a host file open")

    with monkeypatch.context() as scope:
        scope.setattr(S.os, "open", forbidden)
        with pytest.raises(B.StageGateError, match="direct candidate-owned"):
            S._request_bytes(selection.candidate, "nested/request.json", 4096, selection.workflow._candidate_directory)


def test_request_symlink_swap_cannot_open_a_private_file(selection, monkeypatch):
    path = selection.candidate / "request.json"
    path.write_text("{}")
    private = selection.receipt.parent / "private.json"
    private.write_text('{"private":"not candidate input"}')
    original_open = os.open
    opened = []

    def swap(value, flags, *args, **kwargs):
        opened.append((str(value), kwargs.get("dir_fd")))
        if value == path.name:
            path.unlink()
            path.symlink_to(private)
        return original_open(value, flags, *args, **kwargs)

    monkeypatch.setattr(S.os, "open", swap)
    with pytest.raises(OSError):
        S._request_bytes(selection.candidate, path.name, 4096, selection.workflow._candidate_directory)
    assert opened[-1][1] is not None


@pytest.mark.parametrize("raw", [b'{"choice_id":"x","choice_id":"y"}', b"[1,", b" " * 4097])
def test_invalid_json_and_byte_budget_never_compile(selection, raw):
    assert invoke(selection, raw=raw)["returncode"] == 125
    assert not selection.owner.seen


@pytest.mark.parametrize("when", ["verify", "record"])
def test_live_callback_cannot_replace_original_workflow_owner(selection, when):
    active = broker(selection)

    def mutate():
        selection.workflow.source_choices = object()

    if when == "verify":
        selection.owner.verify_hook = mutate
    else:
        selection.owner.record_hook = mutate
    assert invoke(selection, {"choice_id": "declared"}, active=active)["returncode"] == 125
    assert len(selection.owner.seen) == (0 if when == "verify" else 1)


def test_new_directory_at_same_candidate_path_cannot_inherit_selection(selection):
    active = broker(selection)
    selection.candidate.rename(selection.candidate.with_name("old_candidate"))
    selection.candidate.mkdir()
    assert invoke(selection, {"choice_id": "declared"}, active=active)["returncode"] == 125
    assert not selection.owner.seen


def test_result_validation_is_inside_admission_deadline(selection, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(S.time, "monotonic", lambda: clock[0])
    selection.owner.record_hook = lambda: clock.__setitem__(0, clock[0] + 31)
    reply = invoke(selection, {"choice_id": "declared"})
    assert reply["returncode"] == 125
    assert json.loads(selection.receipt.read_text())["returncode"] == 125


def test_original_diagnostic_stays_private_while_refusal_is_receipted(selection):
    active = broker(selection)
    marker = "original-private-source-detail"

    def refuse():
        raise ValueError(marker)

    selection.owner.verify_hook = refuse
    reply = invoke(selection, {"choice_id": "declared"}, active=active)
    assert reply["returncode"] == 125
    assert marker not in reply["stdout"] + reply["stderr"]
    (diagnostic,) = (selection.receipt.parent / "host_refusals").glob("*.txt")
    assert marker in diagnostic.read_text()
    assert json.loads(selection.receipt.read_text())["state"] == "complete"


def test_changed_receipt_selection_does_not_mask_original_refusal(selection):
    active = broker(selection)
    selection.workflow.receipt_path = None
    reply = invoke(selection, {"choice_id": "declared"}, active=active)
    assert reply["returncode"] == 125 and not selection.owner.seen
    assert json.loads(selection.receipt.read_text())["state"] == "complete"
