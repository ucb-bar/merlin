"""Policy receipts bind actual compiler handoffs and cannot turn timeouts into success."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from test_lean_w8a8_chain import identity_suffix, report
from test_lean_w8a8_module import source
from test_lean_w8a8_receipts import stage_at

from merlin.verify.lean import policy
from merlin.verify.lean.check import _file_ref


def handoff_at(root, stage):
    subject = root / "compiler-input.mlir"
    subject.write_bytes((stage / "after.mlir").read_bytes())
    path = root / "handoff.json"
    path.write_text(json.dumps({"schema": "merlin.lean.handoff.v1", **_file_ref(subject)}))
    return path, subject


def test_policy_chain_binds_actual_handoff_bytes(tmp_path, lean_tools):
    first = stage_at(tmp_path, source())
    last = identity_suffix(tmp_path, first)
    handoff, subject = handoff_at(tmp_path, last)
    receipt = policy.check_w8a8_chain_draft(
        [first, last], tmp_path / "result", m2m_python=lean_tools.m2m_python, scope="all", handoff=handoff
    )
    assert receipt["verdict"] == "abstracted", receipt
    checked = report(receipt)
    assert checked["qualification_passed"] and checked["suffix_match"]
    assert checked["handoff"]["record"] == _file_ref(handoff)
    assert checked["handoff"]["subject"] == _file_ref(subject)
    assert checked["handoff"]["subject"]["sha256"] == receipt["after"]["sha256"]
    assert checked["checker_timeout_seconds"] == 300


@pytest.mark.parametrize("change", ["different_program", "edited_subject", "wrong_schema"])
def test_handoff_mismatch_refused_before_checking(tmp_path, lean_tools, change):
    first = stage_at(tmp_path, source())
    handoff, subject = handoff_at(tmp_path, first)
    if change in {"different_program", "edited_subject"}:
        subject.write_bytes((first / "before.mlir").read_bytes())
        if change == "different_program":
            handoff.write_text(json.dumps({"schema": "merlin.lean.handoff.v1", **_file_ref(subject)}))
    else:
        bound = json.loads(handoff.read_text())
        bound["schema"] = "unknown"
        handoff.write_text(json.dumps(bound))
    receipt = policy.check_w8a8_chain_draft(
        [first], tmp_path / "result", m2m_python=lean_tools.m2m_python, scope="all", handoff=handoff
    )
    assert receipt["verdict"] == "unmeasured"
    assert "handoff" in receipt["reason"]


@pytest.mark.parametrize("change", ["record", "subject"])
def test_handoff_change_during_checker_refused(tmp_path, lean_tools, monkeypatch, change):
    first = stage_at(tmp_path, source())
    handoff, subject = handoff_at(tmp_path, first)
    run = subprocess.run

    def mutate(command, **kwargs):
        result = run(command, **kwargs)
        if command[1] == "check-w8a8-chain-draft":
            target = handoff if change == "record" else subject
            with target.open("a") as stream:
                stream.write("\n")
        return result

    monkeypatch.setattr(policy.subprocess, "run", mutate)
    receipt = policy.check_w8a8_chain_draft(
        [first], tmp_path / "result", m2m_python=lean_tools.m2m_python, scope="all", handoff=handoff
    )
    assert receipt["verdict"] == "unmeasured"
    assert "changed during verification" in receipt["reason"]


@pytest.mark.parametrize("chain", [False, True])
def test_timeout_invalidates_an_old_receipt(tmp_path, lean_tools, monkeypatch, chain):
    first = stage_at(tmp_path, source())
    destination = tmp_path / "result" if chain else first
    destination.mkdir(exist_ok=True)
    (destination / "receipt.json").write_text('{"verdict": "verified"}')
    run = subprocess.run

    def expire(command, **kwargs):
        if command[1] in {"check-w8a8-draft", "check-w8a8-chain-draft"}:
            assert kwargs["timeout"] == 1
            raise subprocess.TimeoutExpired(command, 1)
        return run(command, **kwargs)

    monkeypatch.setattr(policy.subprocess, "run", expire)
    receipt = (
        policy.check_w8a8_chain_draft(
            [first], destination, m2m_python=lean_tools.m2m_python, scope="all", timeout_seconds=1
        )
        if chain
        else policy.check_w8a8_draft(first, m2m_python=lean_tools.m2m_python, scope="all", timeout_seconds=1)
    )
    assert receipt["verdict"] == "unmeasured" and receipt["schema"] == "merlin.lean.attempt.v1"
    assert "checker timed out after 1 seconds; no claim established" in receipt["reason"]
    assert json.loads((destination / "receipt.json").read_text()) == receipt


@pytest.mark.parametrize("timeout", [0, -1, True, 1.5])
def test_invalid_timeout_is_not_a_proof_domain(tmp_path, timeout):
    receipt = policy.check_w8a8_chain_draft(
        [], tmp_path / "result", m2m_python=Path("missing"), timeout_seconds=timeout
    )
    assert receipt["verdict"] == "unmeasured"
    assert "positive integer number of seconds" in receipt["reason"]
