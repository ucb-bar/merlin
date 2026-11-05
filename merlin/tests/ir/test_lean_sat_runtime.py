"""Explicit SAT custody and honest native-reflection scope; no simulator runs."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from merlin.verify import lean_checker as CHECK
from merlin.verify.lean_contract import LeanToolchain, coverage_policy_record, file_digest
from merlin.verify.lean_sat_runtime import (
    NATIVE_ASSUMPTION,
    completion_assurance,
    prepared_sat_solver,
    proof_runtime_arguments,
)

BEFORE = """module { func.func @kernel(%x: i8, %y: i8) -> i8 {
  %p = arith.muli %x, %y : i8
  func.return %p : i8
} }"""
AFTER = BEFORE.replace("arith.muli %x, %y", "arith.muli %y, %x")


@pytest.fixture
def selected(tmp_path):
    binary = tmp_path / "bin" / "lean"
    binary.parent.mkdir()
    binary.write_text("fixture bytes; never executed")
    binary.chmod(0o755)
    library = tmp_path / "lib"
    library.mkdir()
    product = library / "Fixture.olean"
    product.write_bytes(b"fixture product; no semantic authority")
    solver = binary.parent / "cadical"
    solver.write_text("fixture solver; never executed")
    solver.chmod(0o755)
    toolchain = LeanToolchain(binary, file_digest(binary), (library,), ((product, file_digest(product)),), tmp_path)
    return toolchain, solver


def _with_solver(toolchain, solver, *, admitted=False):
    row = (solver, file_digest(solver))
    return replace(toolchain, files=(*toolchain.files, row), sat_solver=row, allow_native_reflection=admitted)


def test_unpinned_default_sibling_or_path_cannot_run_for_bv(selected, monkeypatch):
    toolchain, _ = selected
    monkeypatch.setattr(CHECK, "_run", lambda *_args, **_kwargs: pytest.fail("unselected solver reached execution"))
    receipt = CHECK.verify_scalar_transform(BEFORE, AFTER, entry="kernel", toolchain=toolchain, tactic="bv_decide")
    assert receipt.status == "unavailable" and not receipt.verified
    assert "explicitly selected pinned SAT" in receipt.reason


def test_native_permission_is_original_and_mandatory_before_execution(selected, monkeypatch):
    toolchain, solver = selected
    pinned = _with_solver(toolchain, solver)
    monkeypatch.setattr(CHECK, "_run", lambda *_args, **_kwargs: pytest.fail("unadmitted reflection reached execution"))
    receipt = CHECK.verify_scalar_transform(BEFORE, AFTER, entry="kernel", toolchain=pinned, tactic="bv_decide")
    assert receipt.status == "unavailable" and not receipt.verified
    assert "native-reflection admission" in receipt.reason
    assert replace(pinned, allow_native_reflection=True).identity() != pinned.identity()
    with pytest.raises(ValueError, match="boolean permission"):
        replace(pinned, allow_native_reflection=1).verify()


def test_missing_or_changed_solver_refuses_before_execution(selected, monkeypatch):
    toolchain, solver = selected
    pinned = _with_solver(toolchain, solver, admitted=True)
    monkeypatch.setattr(CHECK, "_run", lambda *_args, **_kwargs: pytest.fail("changed solver reached execution"))
    solver.write_text("different bytes")
    receipt = CHECK.verify_scalar_transform(BEFORE, AFTER, entry="kernel", toolchain=pinned, tactic="bv_decide")
    assert not receipt.verified and receipt.status == "error"
    solver.unlink()
    receipt = CHECK.verify_scalar_transform(BEFORE, AFTER, entry="kernel", toolchain=pinned, tactic="bv_decide")
    assert not receipt.verified and receipt.status == "unavailable"


def test_solver_must_be_in_closed_inventory_and_query_cannot_override_it(selected):
    toolchain, solver = selected
    with pytest.raises(ValueError, match="original file inventory"):
        replace(toolchain, sat_solver=(solver, file_digest(solver))).verify()
    admitted = _with_solver(toolchain, solver, admitted=True)
    assert proof_runtime_arguments("by bv_decide", admitted) == ("-Dsat.solver=" + str(solver),)
    with pytest.raises(ValueError, match="override"):
        proof_runtime_arguments('set_option sat.solver "foreign"\nby bv_decide', admitted)


def test_native_axioms_only_with_original_permission_and_actual_bv_mode(selected):
    toolchain, solver = selected
    admitted = _with_solver(toolchain, solver, admitted=True)
    method, assumptions = completion_assurance(
        "by bv_decide", admitted, ("Lean.ofReduceBool", "Lean.trustCompiler"), (), CHECK._ALLOWED_AXIOMS
    )
    assert method == "lean_native_reflection" and assumptions == (NATIVE_ASSUMPTION,)
    for query, selected_toolchain in (("by rfl", admitted), ("by bv_decide", _with_solver(toolchain, solver))):
        with pytest.raises(ValueError, match="unadmitted"):
            completion_assurance(query, selected_toolchain, ("Lean.trustCompiler",), (), CHECK._ALLOWED_AXIOMS)
    with pytest.raises(ValueError, match="unapproved axiom"):
        completion_assurance(
            "by bv_decide", admitted, ("Lean.trustCompiler", "injectedAxiom"), (), CHECK._ALLOWED_AXIOMS
        )
    assert completion_assurance("by rfl", admitted, (), (), CHECK._ALLOWED_AXIOMS) == ("lean_kernel", ())
    assert coverage_policy_record(version="3")["native_reflection"]["kernel_only"] is False


def test_preparation_selects_only_explicit_adjacent_reviewed_layout(selected):
    toolchain, solver = selected
    files = (*toolchain.files, (solver, file_digest(solver)))
    assert prepared_sat_solver(toolchain.executable, (), files) is None
    assert prepared_sat_solver(toolchain.executable, (solver,), files) == files[-1]
    assert prepared_sat_solver(toolchain.executable, (solver.parent.parent / "elsewhere/cadical",), files) is None
    assert prepared_sat_solver(toolchain.executable, (solver,), files, explicit=solver) == files[-1]


def test_manifest_roundtrip_preserves_permission_and_solver_identity(selected):
    toolchain, solver = selected
    toolchain = _with_solver(toolchain, solver, admitted=True)
    manifest = toolchain.work_root / "toolchain.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "merlin.lean_toolchain.v1",
                "executable": {"path": str(toolchain.executable), "sha256": toolchain.executable_sha256},
                "library_roots": list(map(str, toolchain.library_roots)),
                "files": [{"path": str(path), "sha256": sha} for path, sha in toolchain.files],
                "sat_solver": {"path": str(solver), "sha256": file_digest(solver)},
                "allow_native_reflection": True,
            }
        )
    )
    reread = LeanToolchain.from_manifest(manifest, work_root=toolchain.work_root)
    assert reread.identity() == toolchain.identity()


def test_real_relocated_pinned_solver_proves_and_labels_native_scope(tmp_path, monkeypatch):
    selected_manifest = os.environ.get("MERLIN_LEAN_SAT_TEST_MANIFEST")
    if not selected_manifest:
        pytest.skip("requires the explicitly selected original Lean library closure")
    document = json.loads(Path(selected_manifest).read_text())
    original = LeanToolchain.from_manifest(Path(selected_manifest), work_root=tmp_path)
    solver_source = original.executable.parent / "cadical"
    assert solver_source.is_file()
    relocated = tmp_path / "frozen" / "bin" / "cadical"
    relocated.parent.mkdir(parents=True)
    shutil.copyfile(solver_source, relocated)
    relocated.chmod(0o555)
    solver_row = {"path": str(relocated), "sha256": file_digest(relocated)}
    document["files"].append(solver_row)
    # Cadical's original reviewed RUNPATH is ../lib. Preserve only the explicitly
    # selected loader aliases as ordinary readonly files, with their own pins.
    runtime_lib = relocated.parent.parent / "lib"
    runtime_lib.mkdir()
    for name in ("libc++.so.1", "libc++abi.so.1", "libunwind.so.1"):
        staged = runtime_lib / name
        shutil.copyfile(solver_source.parent.parent / "lib" / name, staged)
        staged.chmod(0o444)
        document["files"].append({"path": str(staged), "sha256": file_digest(staged)})
    document["sat_solver"], document["allow_native_reflection"] = solver_row, True
    manifest = tmp_path / "relocated-toolchain.json"
    manifest.write_text(json.dumps(document))
    toolchain = LeanToolchain.from_manifest(manifest, work_root=tmp_path)
    ambient = tmp_path / "ambient"
    ambient.mkdir()
    marker = ambient / "executed"
    fallback = ambient / "cadical"
    fallback.write_text(f"#!/bin/sh\n: > '{marker}'\nexit 1\n")
    fallback.chmod(0o755)
    monkeypatch.setenv("PATH", str(ambient))
    records = []
    original_run = CHECK._run

    def observe(*args, **kwargs):
        result = original_run(*args, **kwargs)
        records.extend(json.loads(path.read_text()) for path in args[2].rglob("invocation.json"))
        return result

    monkeypatch.setattr(CHECK, "_run", observe)
    result = CHECK.verify_scalar_transform(BEFORE, AFTER, entry="kernel", toolchain=toolchain, tactic="bv_decide")
    assert result.verified and CHECK.accepted_receipt(result, BEFORE, AFTER), result.reason
    assert result.method == "lean_native_reflection" and NATIVE_ASSUMPTION in result.assumptions
    assert {"Lean.ofReduceBool", "Lean.trustCompiler"} <= set(result.axioms)
    assert not marker.exists()
    assert records and all("-Dsat.solver=" + str(relocated) in row["argv"] for row in records)
    assert all(row["stage"] == "lean_native_reflection" for row in records)
    wrong = AFTER.replace("arith.muli %y, %x", "arith.muli %y, %y")
    assert not CHECK.verify_scalar_transform(
        BEFORE, wrong, entry="kernel", toolchain=toolchain, tactic="bv_decide"
    ).verified
    assert CHECK.verify_scalar_transform(
        BEFORE, AFTER, entry="kernel", toolchain=toolchain, tactic="bv_decide"
    ).cache_hit
    relocated.chmod(0o755)
    relocated.write_bytes(b"changed original solver")
    drift = CHECK.verify_scalar_transform(BEFORE, AFTER, entry="kernel", toolchain=toolchain, tactic="bv_decide")
    assert not drift.verified and not drift.cache_hit


def test_generated_query_lexing_never_selects_comment_or_string_tools():
    from merlin.verify.lean_sat_runtime import _tokens

    source = '"bv_decide" -- set_option sat.solver ignored\n/- bv_decide /- nested -/ -/ simp'
    assert _tokens(source) == ("simp",)
    assert _tokens('set_option /- comment -/ sat.solver "forbidden"')[:2] == ("set_option", "sat.solver")
