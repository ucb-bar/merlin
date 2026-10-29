"""SDK-free fail-closed buffer gate controls. No synthetic proof acceptance."""

import copy
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.verify import buffer_reuse_gate as gate
from merlin.verify.buffer_reuse_intake import ROOT


def forbidden(*args, **kwargs):
    pytest.fail("native callback or consumer invoked while proof unavailable")


@pytest.mark.parametrize(
    "input_value", [None, {"status": "checked"}, {"qualified": True, "allocator": "total_private_unobservable"}]
)
def test_unfinished_route_returns_unavailable_without_native(tmp_path, monkeypatch, input_value):
    monkeypatch.setattr(gate, "_run", forbidden)
    result = gate.check(input_value, lean=Path("/forbidden"), destination=tmp_path / "new")
    assert result.status == "unavailable" and result.completion is None
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize("saved", [{"status": "checked"}, {"status": "verified", "statements": {ROOT: "fake"}}])
def test_saved_receipt_never_activates_replay(tmp_path, monkeypatch, saved):
    path = tmp_path / "saved.json"
    path.write_text(json.dumps(saved))
    monkeypatch.setattr(gate, "_run", forbidden)
    result = gate.replay(path, lean=Path("/forbidden"), destination=tmp_path / "new")
    assert result.status == "unavailable" and result.completion is None


@pytest.mark.parametrize("kind", ["none", "dict", "constructed", "copied"])
def test_missing_or_forged_completion_never_delivers(kind):
    token = gate.Completion("fake", "/saved.json")
    selected = {"none": None, "dict": {"status": "checked"}, "constructed": token, "copied": copy.copy(token)}[kind]
    with pytest.raises(gate.Unavailable):
        gate.deliver(selected, None, forbidden)


def test_no_caller_proof_callback_parameter(tmp_path):
    with pytest.raises(TypeError):
        gate.check(None, lean=Path("/forbidden"), destination=tmp_path, checker=forbidden)


def test_even_synthetic_green_audit_is_not_a_live_completion(tmp_path, monkeypatch):
    candidate = SimpleNamespace(roots=(ROOT,))
    audit = {
        "violations": [],
        "assurance": "lean_kernel_std3_no_native_reflection",
        "theorems_audited": 1,
        "axioms_used": ["propext"],
        "imports": [{"module": "RawChainEntries", "olean": "/fake.olean"}],
    }
    gate._audit(audit, {ROOT: "SYNTHETIC TEST TYPE"}, candidate, {"RawChainEntries"})
    monkeypatch.setattr(gate, "_run", forbidden)
    assert gate.check(candidate, lean=Path("/forbidden"), destination=tmp_path / "new").completion is None
    assert not gate._LIVE


@pytest.mark.parametrize("mutation", ["wrong_root", "bad_type", "axiom", "missing_import"])
def test_malformed_audit_cannot_issue_authority(mutation):
    audit = {
        "violations": [],
        "assurance": "lean_kernel_std3_no_native_reflection",
        "theorems_audited": 1,
        "axioms_used": ["propext"],
        "imports": [{"module": "RawChainEntries", "olean": "/fake.olean"}],
    }
    statements = {ROOT: "SYNTHETIC TEST TYPE"}
    if mutation == "wrong_root":
        statements = {"other": "fake"}
    elif mutation == "bad_type":
        statements[ROOT] = True
    elif mutation == "axiom":
        audit["axioms_used"] = ["sorryAx"]
    else:
        audit["imports"] = []
    with pytest.raises(gate.Unavailable):
        gate._audit(audit, statements, SimpleNamespace(roots=(ROOT,)), {"RawChainEntries"})


def test_timeout_preserves_first_partial_output(tmp_path, monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 1, output=b"first partial", stderr=b"first diagnostic")

    monkeypatch.setattr(gate, "run_owned", timeout)
    with pytest.raises(gate.Unavailable):
        gate._run(["/fake/lean"], tmp_path, {}, tmp_path, "query", 1)
    assert (tmp_path / "query.stdout").read_text() == "first partial"
    assert (tmp_path / "query.stderr").read_text() == "first diagnostic"


def test_signal_is_unavailable_not_semantic_refusal(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "run_owned", lambda *a, **k: subprocess.CompletedProcess(a[0], -11, "first raw", ""))
    with pytest.raises(gate.Unavailable, match="signal 11"):
        gate._run(["/fake/lean"], tmp_path, {}, tmp_path, "query", 1)
    assert (tmp_path / "query.stdout").read_text() == "first raw"


def test_completion_is_not_serializable():
    import pickle

    with pytest.raises(TypeError, match="cannot be serialized"):
        pickle.dumps(gate.Completion("fake", "/saved"))
