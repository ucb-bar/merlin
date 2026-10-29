"""Adversarial protocol and runtime checks for actual bounded-policy receipt attempts."""

from __future__ import annotations

import json
import os
import shutil
import subprocess

import pytest
from test_lean_w8a8_bounded import contract, embedding_source
from test_lean_w8a8_bounded_receipts import evidence, invoke
from test_lean_w8a8_chain import identity_suffix
from test_lean_w8a8_handoff import handoff_at
from test_lean_w8a8_receipts import stage_at

from merlin.verify.lean import bounded_policy, interpreted

pytestmark = pytest.mark.usefixtures("lean_oracle")


@pytest.mark.parametrize(
    "change",
    [
        "exit_zero",
        "verdict",
        "review",
        "primitive_laws",
        "checker",
        "intervals",
        "functions",
        "integer_flag",
        "source_initialized",
        "suffix_order",
        "suffix_match",
        "unknown_field",
    ],
)
def test_untrusted_driver_output_cannot_approve_a_different_claim(tmp_path, lean_tools, monkeypatch, change):
    first = stage_at(tmp_path, embedding_source())
    last = identity_suffix(tmp_path, first)
    run = interpreted.Runtime.run

    def forged(self, arguments, **kwargs):
        proc = run(self, arguments, **kwargs)
        value = json.loads(proc.stdout)
        code = proc.returncode
        if change == "exit_zero":
            code = 0
        elif change == "verdict":
            value["verdict"] = "verified"
        elif change == "review":
            value["policy_reviewed"] = True
        elif change == "primitive_laws":
            value["primitive_model_laws_verified"] = True
        elif change == "checker":
            value["checker"] += "_different"
        elif change == "intervals":
            value["functions"][0]["input_intervals"][0]["upper"] = 8
        elif change == "functions":
            value["functions"] = []
        elif change == "integer_flag":
            value["boundary_qualification_passed"] = 1
        elif change == "source_initialized":
            value["functions"][0]["source_initialized"] = False
        elif change == "suffix_order":
            value["suffix_boundaries"][0]["index"] = 1
        elif change == "suffix_match":
            value["suffix_match"] = False
        else:
            value["unproved_policy_approved"] = True
        return subprocess.CompletedProcess(proc.args, code, json.dumps(value), proc.stderr)

    monkeypatch.setattr(interpreted.Runtime, "run", forged)
    receipt, _ = invoke([first, last], contract(7), tmp_path / "bounded", lean_tools)
    assert receipt["verdict"] == "unmeasured" and receipt["schema"] == "merlin.lean.attempt.v1", receipt
    assert "cert" not in receipt


@pytest.mark.parametrize("change", ["artifact", "inventory"])
def test_actual_interpreter_artifact_changes_refuse_publication(tmp_path, lean_tools, monkeypatch, change):
    first = stage_at(tmp_path, embedding_source())
    library = lean_tools.lean_exe.parent.parent / "lib/lean"
    private = tmp_path / "proof-lib"
    driver = private / "MerlinVerify/Driver/W8A8Bounded.olean"
    # Lean resolves the namespace directory as a unit. Execute an owned complete copy
    # so tampering cannot affect any other test, agent or shared build artifact.
    shutil.copytree(library / "MerlinVerify", private / "MerlinVerify")
    output = interpreted._lake_output
    run = interpreted.Runtime.run

    def search(*args):
        result = output(*args)
        return str(private) + os.pathsep + result if args == ("printenv", "LEAN_PATH") else result

    def changed(self, arguments, **kwargs):
        proc = run(self, arguments, **kwargs)
        if change == "artifact":
            driver.write_bytes(b"changed private artifact")
        else:
            injected = private / "MerlinVerify/Check/Shadow.olean"
            injected.parent.mkdir(parents=True, exist_ok=True)
            injected.write_bytes(b"new unimported module")
        return proc

    monkeypatch.setattr(interpreted, "_lake_output", search)
    monkeypatch.setattr(interpreted.Runtime, "run", changed)
    receipt, _ = invoke([first], contract(7), tmp_path / "bounded", lean_tools)
    assert receipt["verdict"] == "unmeasured" and "changed" in receipt["reason"], receipt
    assert "cert" not in receipt


@pytest.mark.parametrize("change", ["record", "subject", "none"])
def test_receipt_binds_the_actual_compiler_handoff(tmp_path, lean_tools, monkeypatch, change):
    first = stage_at(tmp_path, embedding_source())
    handoff, subject = handoff_at(tmp_path, first)
    run = interpreted.Runtime.run

    def changed(self, arguments, **kwargs):
        proc = run(self, arguments, **kwargs)
        if change != "none":
            path = handoff if change == "record" else subject
            path.write_text(path.read_text() + "\n")
        return proc

    monkeypatch.setattr(interpreted.Runtime, "run", changed)
    receipt, _ = invoke([first], contract(7), tmp_path / "bounded", lean_tools, handoff=handoff)
    if change == "none":
        report = evidence(receipt)
        assert report["handoff"]["subject"]["sha256"] == receipt["after"]["sha256"]
    else:
        assert receipt["verdict"] == "unmeasured" and "changed" in receipt["reason"], receipt


def test_timeout_invalidates_an_old_receipt(tmp_path, lean_tools, monkeypatch):
    first = stage_at(tmp_path, embedding_source())
    destination = tmp_path / "bounded"
    destination.mkdir()
    (destination / "receipt.json").write_text('{"verdict":"verified"}')

    def expired(self, arguments, **kwargs):
        raise subprocess.TimeoutExpired(arguments, kwargs["timeout_seconds"])

    monkeypatch.setattr(interpreted.Runtime, "run", expired)
    receipt, _ = invoke([first], contract(7), destination, lean_tools)
    assert receipt["verdict"] == "unmeasured" and "cert" not in receipt
    assert json.loads((destination / "receipt.json").read_text()) == receipt


def test_cli_runs_the_actual_separate_entry_point(tmp_path, lean_tools, capsys):
    first = stage_at(tmp_path, embedding_source())
    contracts = tmp_path / "contracts.json"
    contracts.write_text(json.dumps(contract(7)))
    code = bounded_policy.main(
        [
            str(first),
            "--destination",
            str(tmp_path / "bounded"),
            "--input-contracts",
            str(contracts),
            "--m2m-python",
            str(lean_tools.m2m_python),
            "--scope",
            "all",
        ]
    )
    assert code == 1
    report = evidence(json.loads(capsys.readouterr().out))
    assert report["conditional_numerical"]["conditional_initialized_policy_refinement"]


@pytest.mark.parametrize("change", ["stale_cache", "fresh_unapproved"])
def test_axiom_admission_uses_the_actual_interpreted_artifacts(tmp_path, lean_tools, monkeypatch, change):
    from merlin.verify.lean import bounded_numerics, check

    first = stage_at(tmp_path, embedding_source())
    if change == "stale_cache":
        key = (f"{bounded_numerics.MODULE}.check_sound", check._sha(lean_tools.lean_exe))
        monkeypatch.setitem(check._AXIOMS, key, ["unapproved.cached.assumption"])
    else:
        monkeypatch.setattr(interpreted.Runtime, "axioms", lambda *args: ["unapproved.fresh.assumption"])
    receipt, _ = invoke([first], contract(7), tmp_path / "bounded", lean_tools)
    if change == "stale_cache":
        report = evidence(receipt)
        assert "unapproved.cached.assumption" not in report["conditional_numerical"]["axioms"]
    else:
        assert receipt["verdict"] == "unmeasured" and "unapproved axioms" in receipt["reason"]


def test_contract_file_cannot_be_overwritten_by_the_destination(tmp_path, lean_tools):
    destination = tmp_path / "bounded"
    destination.mkdir()
    contracts = destination / "receipt.json"
    contracts.write_text(json.dumps(contract(7)))
    original = contracts.read_bytes()
    with pytest.raises(ValueError, match="destination receipt file"):
        bounded_policy.check_w8a8_bounded_chain_draft(
            [], destination, input_contracts=contracts, m2m_python=lean_tools.m2m_python
        )
    assert contracts.read_bytes() == original
