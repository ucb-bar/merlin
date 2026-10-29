"""Caller work roots cannot install numerical approval or proof resources."""

import json

import pytest

from merlin.verify import bf16_policy_gate as gate
from merlin.verify.gate_errors import GateRejected, GateUnavailable


@pytest.fixture(autouse=True)
def clean_owner_environment(monkeypatch):
    for name in ("MERLIN_REPO_ROOT", "MERLIN_WORK_DIR", "MERLIN_OUT_ROOT"):
        monkeypatch.delenv(name, raising=False)


def test_governance_owner_is_the_installed_implementation():
    root = gate.installed_root()
    assert root == gate.checkout_root().resolve()
    assert (root / "src/merlin/verify/bf16_policy_gate.py").is_file()


@pytest.mark.parametrize("name", ["MERLIN_REPO_ROOT", "MERLIN_WORK_DIR", "MERLIN_OUT_ROOT"])
def test_caller_tree_refuses_before_governance_or_tools(tmp_path, monkeypatch, name):
    # Even a caller-installed apparent approval must never be consulted. The
    # refusal is about the owner, independently of this document's contents.
    admission = tmp_path / gate.ADMISSION
    admission.parent.mkdir(parents=True)
    admission.write_text(json.dumps({"decision": "approved", "reviewer": "caller"}))
    monkeypatch.setenv(name, str(tmp_path))

    def forbidden(*args, **kwargs):
        pytest.fail("caller-selected governance or checker was opened")

    monkeypatch.setattr(gate, "require_admission", forbidden)
    monkeypatch.setattr(gate, "_tools", forbidden)
    monkeypatch.setattr(gate, "_ref", forbidden)
    with pytest.raises(GateRejected, match=name):
        gate.check_policy({"params": {"sdk_bin": "/forbidden", "compiler_python": "/forbidden", "index_bits": 64}})


def test_matching_owner_environment_is_not_false_approval(monkeypatch):
    root = gate.installed_root()
    monkeypatch.setenv("MERLIN_REPO_ROOT", str(root))
    monkeypatch.setenv("MERLIN_WORK_DIR", str(root))
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(root / "out"))
    assert gate.installed_root() == root
    with pytest.raises(GateRejected, match="human_review_pending"):
        gate.check_policy({"params": {"sdk_bin": "/unused", "compiler_python": "/unused", "index_bits": 64}})


def test_missing_installed_source_owner_is_unavailable(monkeypatch):
    monkeypatch.setattr(gate, "checkout_root", lambda: None)
    with pytest.raises(GateUnavailable, match="installed source owner"):
        gate.installed_root()


def test_direct_admission_reader_cannot_select_another_owner(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("caller-selected review record opened")

    monkeypatch.setattr(gate, "_json", forbidden)
    with pytest.raises(GateRejected, match="installed source owner"):
        gate.require_admission(tmp_path)
