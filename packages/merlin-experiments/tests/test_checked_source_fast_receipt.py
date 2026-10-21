"""Original private fast receipts reopen witnesses; neutral fixtures issue no target authority."""

import copy
import hashlib
import json

import pytest
import yaml
from merlin_experiments.phase2 import archive_native_screen as ARCHIVE
from merlin_experiments.phase2.contracts import StageGateError
from test_checked_source_model_diagnostic_execution import execution, fast_selection, finish  # noqa: F401

from merlin.targetgen import checked_source_model_diagnostic_execution as E

_PINNED_NATIVE_FILE = E.NATIVE._pinned_native_file


@pytest.fixture
def receipt(execution, monkeypatch):  # noqa: F811
    x = fast_selection(execution)
    monkeypatch.setattr(E.NATIVE, "_pinned_native_file", _PINNED_NATIVE_FILE)

    def leaf(name, content):
        p = x.root / name
        p.write_text(content)
        return E.NATIVE._digest(p)

    declaration = dict(kind="model", label="dev", oracle_tier_ceiling="L2",
                       required_oracle_tiers=["L0", "L1", "L2", "L3"], interface_mlir="model.mlir")
    capsule = leaf("capsule.yaml", yaml.safe_dump(declaration))
    source = leaf("model.mlir", "original pinned source fixture")
    console = leaf("console.txt", "original complete-output fixture")
    native = x.result["candidate_native_execution"]
    native.update(elf=leaf("program.elf", "original ELF fixture"), tiers={"L2": {"console": console}}, source={
        "capsule_declaration": capsule, "interface": source,
        **{name: leaf(name + ".fixture", "original " + name)
           for name in ("inputs", "weights", "weight_manifest", "golden")},
    })
    x.result["candidate_emission"] = dict(capsule_declaration=capsule, source_interface=source,
        command_buffer=leaf("command_buffer.json", "{}"), lowered_mlir=leaf("lowered.mlir", "original LLVM fixture"))
    x.native.write_text(json.dumps(native))
    monkeypatch.setattr(E.NATIVE, "independent_frozen_source_eligibility",
                        lambda *_a, **_k: copy.deepcopy(x.result["candidate_source_eligibility"]))
    grade = finish(x)
    original = dict(functional_status="PASS", compiled={"elf_sha256": native["elf"]["sha256"]},
                    checked_source_diagnostic=x.selection.record(), original_grade=grade)
    directory = x.root / "private-observation"
    directory.mkdir()

    def publish():
        p = directory / "ordinary_observation.json"
        p.write_text(json.dumps(original))
        value = {k: v for k, v in original.items() if k not in {"original_grade", "checked_source_diagnostic"}}
        return {**value, "evidence_sha256": hashlib.sha256(p.read_bytes()).hexdigest()}

    return x, original, directory, publish


def reopen(case):
    x, _original, directory, publish = case
    return ARCHIVE.original_numerical_observation(directory, publish(), checked_source=(x.selection, x.passes))


def test_fast_private_receipt_reopens_actual_pinned_witnesses_and_original_audit(receipt, monkeypatch):
    x, original, directory, _publish = receipt
    calls = []

    def audit(selection, emission, certificate, native, **kwargs):
        calls.append((selection, emission, certificate, native, kwargs))
        return copy.deepcopy(x.tiers)

    monkeypatch.setattr(E, "_audit_execution", audit)
    reopened, _pin = reopen(receipt)
    assert reopened == original and len(calls) == 1
    assert calls[0][0] is x.selection and calls[0][1] == original["original_grade"]["candidate_emission"]
    assert (directory / "ordinary_observation.json").is_file() and x.issuer.calls == []


@pytest.mark.parametrize("witness", ["output", "ELF", "source", "IR", "native_receipt"])
def test_fast_private_receipt_cannot_reuse_pass_after_missing_original_witness(receipt, witness):
    x, original, _directory, _publish = receipt
    grade = original["original_grade"]
    pin = {
        "output": grade["candidate_native_execution"]["tiers"]["L2"]["console"],
        "ELF": grade["candidate_native_execution"]["elf"],
        "source": grade["candidate_emission"]["source_interface"],
        "IR": grade["candidate_emission"]["lowered_mlir"],
        "native_receipt": grade["numerical_native_receipt"],
    }[witness]
    from pathlib import Path

    Path(pin["path"]).unlink()
    with pytest.raises(StageGateError, match="source/ISA/full-output witnesses"):
        reopen(receipt)
    assert x.issuer.calls == []


@pytest.mark.parametrize("tier", ["L0", "L1", "L2"])
def test_fast_private_receipt_requires_current_source_isa_and_complete_output_audits(receipt, monkeypatch, tier):
    x, *_ = receipt
    x.tiers["tiers"][tier]["status"] = "unverified"
    with pytest.raises(StageGateError, match="source/ISA/full-output witnesses"):
        reopen(receipt)


def test_fast_private_receipt_recomputes_source_eligibility(receipt, monkeypatch):
    monkeypatch.setattr(E.NATIVE, "independent_frozen_source_eligibility", lambda *_a, **_k: {"n_eligible": 0})
    with pytest.raises(StageGateError, match="source/ISA/full-output witnesses"):
        reopen(receipt)


def test_new_saved_fast_status_cannot_be_reopened_under_old_selection(receipt):
    from merlin.targetgen import checked_source_model_diagnostic as S

    x, original, directory, publish = receipt
    old = S.select_checked_source_diagnostic(x.passes, target="fixture", rtl_facts=x.root / "facts.json",
                                             board_config="config")
    original["checked_source_diagnostic"] = old.record()
    with pytest.raises(StageGateError, match="live complete-source selection"):
        ARCHIVE.original_numerical_observation(directory, publish(), checked_source=(old, x.passes))
