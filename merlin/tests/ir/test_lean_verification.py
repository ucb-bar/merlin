"""Real-kernel integration is opt-in; unavailable is always tested independently."""

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from merlin.verify.lean_checker import (
    accepted_pipeline_receipt,
    accepted_receipt,
    verify_pipeline_change,
    verify_scalar_transform,
)
from merlin.verify.lean_contract import LeanToolchain, coverage_policy_record, digest
from merlin.verify.lean_export import UnsupportedLeanIR, equivalence_query, export_scalar

BEFORE = """builtin.module {
  func.func @kernel(%x: i8) -> i8 {
    %z = arith.constant 0 : i8
    %r = arith.addi %x, %z : i8
    func.return %r : i8
  }
}"""
AFTER = "builtin.module { func.func @kernel(%x: i8) -> i8 { func.return %x : i8 } }"


def test_missing_tool_and_forged_report_never_prove():
    result = verify_scalar_transform(BEFORE, AFTER, entry="kernel", toolchain=None)
    assert result.status == "unavailable"
    forged = replace(result, status="verified")
    assert not accepted_receipt(forged, BEFORE, AFTER)
    assert not hasattr(__import__("merlin.verify.lean_contract", fromlist=["issue"]), "issue")


@pytest.mark.parametrize("flag", ["nsw", "nuw", "nsw,nuw"])
def test_poison_arithmetic_refuses(flag):
    source = BEFORE.replace("arith.addi %x, %z :", f"arith.addi %x, %z overflow<{flag}> :")
    with pytest.raises(UnsupportedLeanIR, match="poison"):
        export_scalar(source, "kernel")


def test_unknown_flags_and_other_functions_refuse():
    source = (
        "builtin.module { func.func @kernel(%x: i8) -> i16 { "
        "%r = arith.extui %x {nneg} : i8 to i16 func.return %r : i16 } }"
    )
    with pytest.raises(UnsupportedLeanIR, match="unmodeled"):
        export_scalar(source, "kernel")
    with pytest.raises(UnsupportedLeanIR, match="exactly one"):
        export_scalar(BEFORE[:-1] + "func.func @other() { func.return } }", "kernel")


def test_explicit_oot_primitive_has_frozen_signature_and_bindings():
    text = (
        "builtin.module { func.func @kernel(%x: i8, %y: i8) -> i8 { "
        '%r = "fresh.sum"(%x, %y) : (i8, i8) -> i8 func.return %r : i8 } }'
    )
    registry = {
        "fresh.sum": {
            "effects": {"reads": [], "writes": []},
            "operand_names": ["a", "b"],
            "operand_types": [{"kind": "integer", "bits": 8}, {"kind": "integer", "bits": 8}],
            "result_types": [{"kind": "integer", "bits": 8}],
            "immediates": [],
            "semantics": {
                "primitive": "integer.add",
                "bindings": {"lhs": "b", "rhs": "a"},
                "parameters": {"width": 8, "overflow": "wrap"},
            },
        }
    }
    result = export_scalar(text, "kernel", registry)
    assert "Merlin.bvAdd 8 v1 v0" in result.term()
    wrong = json.loads(json.dumps(registry))
    wrong["fresh.sum"]["operand_types"][0]["bits"] = 16
    with pytest.raises(UnsupportedLeanIR, match="frozen instruction"):
        export_scalar(text, "kernel", wrong)
    wrong = json.loads(json.dumps(registry))
    wrong["fresh.sum"]["effects"] = {"status": "unknown"}
    with pytest.raises(UnsupportedLeanIR, match="pure primitive"):
        export_scalar(text, "kernel", wrong)


def test_complete_signature_and_tactics_are_not_candidate_policy():
    assert coverage_policy_record()["mode"] == "strict_within_coverage"
    left, right = export_scalar(BEFORE, "kernel"), export_scalar(AFTER, "kernel")
    with pytest.raises(ValueError, match="vocabulary"):
        equivalence_query(left, right, tactic="sorry")
    assert "#print axioms merlin_transform" in equivalence_query(left, right)


def test_malformed_ir_is_scoped_feedback():
    receipt = verify_scalar_transform("builtin.module { invalid", AFTER, entry="kernel", toolchain=None)
    assert receipt.status == "unsupported" and receipt.reason


def test_closed_import_membership_and_bytes(tmp_path):
    executable = tmp_path / "lean"
    executable.write_bytes(b"operator-selected executable")
    library = tmp_path / "lib"
    library.mkdir()
    product = library / "First.olean"
    product.write_bytes(b"compiled product")
    selected = LeanToolchain(
        executable, digest(executable.read_bytes()), (library,), ((product, digest(product.read_bytes())),), tmp_path
    )
    selected.verify()
    (library / "Transitive.olean.private").write_bytes(b"undeclared transitive")
    with pytest.raises(ValueError, match="membership"):
        selected.verify()
    (library / "Transitive.olean.private").unlink()
    product.write_bytes(b"changed product")
    with pytest.raises(ValueError, match="bytes changed"):
        selected.verify()


def test_dependency_hash_refuses_changes_during_read_and_does_not_cache(tmp_path, monkeypatch):
    from merlin.verify import lean_contract

    dependency = tmp_path / "product.olean"
    dependency.write_bytes(b"first product")
    original = lean_contract.hashlib.file_digest

    def mutate(stream, algorithm):
        observed = original(stream, algorithm)
        dependency.write_bytes(b"second product with a different size")
        return observed

    monkeypatch.setattr(lean_contract.hashlib, "file_digest", mutate)
    with pytest.raises(ValueError, match="changed while"):
        lean_contract.file_digest(dependency)
    assert not any(key[0] == str(dependency) for key in lean_contract._FILE_CACHE)
    monkeypatch.setattr(lean_contract.hashlib, "file_digest", original)
    assert lean_contract.file_digest(dependency) == digest(dependency.read_bytes())


def test_dependency_hash_refreshes_after_same_size_replacement(tmp_path):
    from merlin.verify.lean_contract import file_digest

    dependency = tmp_path / "product.olean"
    dependency.write_bytes(b"first product")
    first = file_digest(dependency)
    replacement = tmp_path / "replacement.olean"
    replacement.write_bytes(b"other product")
    replacement.replace(dependency)
    assert file_digest(dependency) == digest(b"other product") != first


@pytest.fixture
def real_toolchain(tmp_path):
    manifest = os.environ.get("MERLIN_LEAN_TOOLCHAIN_MANIFEST")
    if not manifest:
        pytest.skip("explicit real Lean dependency selection not installed")
    return LeanToolchain.from_manifest(Path(manifest), work_root=tmp_path)


def test_real_lean_checks_actual_scalar_pair_and_rejects_wrong_pair(real_toolchain):
    valid = verify_scalar_transform(BEFORE, AFTER, entry="kernel", toolchain=real_toolchain)
    assert valid.verified, valid.reason
    assert accepted_receipt(valid, BEFORE, AFTER)
    cached = verify_scalar_transform(BEFORE, AFTER, entry="kernel", toolchain=real_toolchain)
    assert cached.verified and cached.cache_hit
    assert accepted_receipt(cached, BEFORE, AFTER)
    assert not accepted_receipt(replace(valid), BEFORE, AFTER)
    wrong = AFTER.replace("func.return %x", "%one = arith.constant 1 : i8 %r = arith.addi %x, %one : i8 func.return %r")
    invalid = verify_scalar_transform(BEFORE, wrong, entry="kernel", toolchain=real_toolchain)
    assert not invalid.verified
    assert not accepted_receipt(valid, BEFORE, wrong)
    assert not set(valid.axioms) - {"propext", "Classical.choice", "Quot.sound"}


def test_real_shared_pipeline_has_live_formal_gate(real_toolchain):
    from merlin.transforms import PipelineOptions, apply_common_pipeline
    from merlin.transforms.certificates import parse_ir

    receipts = []

    def check(change):
        receipt = verify_pipeline_change(change, toolchain=real_toolchain)
        assert accepted_pipeline_receipt(receipt, change, toolchain=real_toolchain), receipt.reason
        receipts.append(receipt)
        return receipt

    result = apply_common_pipeline(parse_ir(BEFORE), PipelineOptions(64), verifier=check)
    assert receipts and not result.plan_only
    assert all(row.verified for row in receipts)


def test_real_lean_parameterized_large_loop_does_not_expand_elements(real_toolchain):
    from merlin.transforms import PipelineOptions, apply_common_pipeline
    from merlin.transforms.certificates import parse_ir

    source = parse_ir("""builtin.module {
      func.func @kernel(%x: i32) -> i32 {
        %l = arith.constant 0 : index
        %u = arith.constant 1000000000003 : index
        %s = arith.constant 1 : index
        %result = scf.for %i = %l to %u step %s iter_args(%acc = %x) -> (i32) {
          scf.yield %acc : i32
        }
        func.return %result : i32
      }
    }""")
    checked = []

    def check(change):
        receipt = verify_pipeline_change(change, toolchain=real_toolchain)
        assert accepted_pipeline_receipt(receipt, change, toolchain=real_toolchain), receipt.reason
        checked.append(receipt)
        return receipt

    result = apply_common_pipeline(source, PipelineOptions(64, (17,)), verifier=check)
    assert checked and all(r.domain == "ordered_loop_segmentation_v1" for r in checked)
    assert len(result.changes[0].after_ir) < 6000


def test_real_oot_integer_primitive_agrees_with_standard_ir(real_toolchain):
    text = (
        "builtin.module { func.func @kernel(%a: i8, %b: i8) -> i8 { "
        '%r = "fresh.sum"(%a, %b) : (i8, i8) -> i8 func.return %r : i8 } }'
    )
    source = (
        "builtin.module { func.func @kernel(%a: i8, %b: i8) -> i8 { %r = arith.addi %b, %a : i8 func.return %r : i8 } }"
    )
    registry = {
        "fresh.sum": {
            "effects": {"reads": [], "writes": []},
            "operand_names": ["x", "y"],
            "operand_types": [{"kind": "integer", "bits": 8}] * 2,
            "result_types": [{"kind": "integer", "bits": 8}],
            "immediates": [],
            "semantics": {
                "primitive": "integer.add",
                "bindings": {"lhs": "y", "rhs": "x"},
                "parameters": {"width": 8, "overflow": "wrap"},
            },
        }
    }
    result = verify_scalar_transform(source, text, entry="kernel", registry=registry, toolchain=real_toolchain)
    assert result.verified, result.reason


def test_real_unapproved_axiom_does_not_create_completion(real_toolchain):
    from merlin.verify.lean_checker import _check_generated

    unproved = verify_scalar_transform(BEFORE, AFTER, entry="kernel", toolchain=None)
    query = (
        "import Merlin.Foundation\naxiom injected : False\n"
        "theorem merlin_transform : False := injected\n#print axioms merlin_transform\n"
    )
    result = _check_generated(query, unproved, real_toolchain)
    assert not result.verified and "unapproved axiom" in result.reason
    assert not accepted_receipt(result, BEFORE, AFTER)


def test_real_operator_toolchain_preparation(real_toolchain, tmp_path):
    from merlin.verify.lean_cli import prepare_toolchain

    roots = tuple(root for root in real_toolchain.library_roots if not (root / "Merlin/Foundation.olean").exists())
    manifest = prepare_toolchain(
        executable=real_toolchain.executable,
        library_roots=roots,
        runtime_files=(),
        output=tmp_path / "prepared",
        work_root=tmp_path,
        dependency_revision=real_toolchain.dependency_revision,
        timeout_s=real_toolchain.timeout_s,
        memory_mb=real_toolchain.memory_mb,
    )
    selected = LeanToolchain.from_manifest(manifest, work_root=tmp_path)
    result = verify_scalar_transform(BEFORE, AFTER, entry="kernel", toolchain=selected)
    assert result.verified and accepted_receipt(result, BEFORE, AFTER), result.reason
