"""Stale/coherently rehashed selections, retained roots and writer refusals."""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from merlin.tests.ir.test_lean_graph_loops_memory_scf import check
from merlin.tests.ir.test_lean_graph_loops_memory_scf import context as context
from merlin.verify import lean_graph_loops_memory_scf as validator
from merlin.verify.lean_graph_loops import _reference
from merlin.verify.lean_graph_loops_memory_scf import audit as _audit
from merlin.verify.lean_graph_loops_memory_scf_model import MEMORY_SCF_ROOTS


@pytest.mark.parametrize("field", ("contract", "manifest", "fixed_context"))
@pytest.mark.parametrize("rehash", (False, True))
def test_replacement_context_cannot_become_original(context, tmp_path, field, rehash):
    selection, _ = context
    path = tmp_path / (field + ".json")
    data = json.loads(getattr(selection, field).read_bytes())
    if field == "contract":
        data["allocator_mode"] = "finite_capacity"
    elif field == "fixed_context":
        data["suspect_in_actual_closure"] = True
    else:
        data["producer_revision"] = "0" * 40
    path.write_text(json.dumps(data))
    changes = {field: path}
    if rehash:
        changes[field + "_sha256"] = _reference(path)["sha256"]
    mutated = replace(selection, **changes)
    with pytest.raises(ValueError):
        mutated.dependencies()


@pytest.mark.parametrize("field", ("executable", "library", "build_record"))
def test_rehashed_proof_selection_cannot_replace_policy(context, tmp_path, field):
    selection, _ = context
    path = tmp_path / field
    if field != "library":
        path.write_bytes(getattr(selection.proof, field).read_bytes())
    else:
        path.mkdir()
    proof = replace(selection.proof, **{field: path})
    with pytest.raises(ValueError, match="replacement"):
        replace(selection, proof=proof).dependencies()


@pytest.mark.parametrize("kind", ("missing", "extra", "duplicate", "new_axiom", "interrupted"))
def test_exact_root_transitive_audit_refuses(kind):
    lines = [f"'{root}' does not depend on any axioms" for root in MEMORY_SCF_ROOTS]
    if kind == "missing":
        lines.pop()
    elif kind == "extra":
        lines.append("'unrelated' does not depend on any axioms")
    elif kind == "duplicate":
        lines.append(lines[0])
    elif kind == "new_axiom":
        lines[-1] = f"'{MEMORY_SCF_ROOTS[-1]}' depends on axioms: [newAssumption]"
    else:
        lines[-1] = f"'{MEMORY_SCF_ROOTS[-1]}' depends on axioms: [propext,\nerror: interrupted report"
    with pytest.raises(ValueError):
        _audit("\n".join(lines), MEMORY_SCF_ROOTS)


@pytest.mark.parametrize("kind", ("original_middle", "private_middle", "query", "model", "exports"))
def test_intervening_writer_never_earns_acceptance(context, tmp_path, monkeypatch, kind):
    selection, paths = context
    texts = [path.read_text() for path in paths["two_epochs"][:3]]
    real_run = subprocess.run

    def run(command, *args, **kwargs):
        result = real_run(command, *args, **kwargs)
        if command[0] == str(selection.proof.executable):
            target = {
                "original_middle": tmp_path / "middle.mlir",
                "private_middle": tmp_path / "proof/middle.mlir",
                "query": tmp_path / "proof/Obligation.lean",
                "model": tmp_path / "proof/model.json",
                "exports": tmp_path / "proof/actual-triple.json",
            }[kind]
            with target.open("a") as output:
                output.write("\n ")
        return result

    monkeypatch.setattr(validator.subprocess, "run", run)
    result = check(context, tmp_path, texts=texts)
    assert result["returncode"] == 0
    assert not result["lean_complete_composition_checked"]
    assert "changed" in result["refusal"]


def test_saved_positive_report_has_no_authority(context, tmp_path):
    _, paths = context
    texts = [path.read_text() for path in paths["two_epochs"][:3]]
    texts[2] = texts[2].replace("value = 1 : i32", "value = 2 : i32")
    (tmp_path / "old_saved_report.json").write_text(json.dumps(dict(lean_complete_composition_checked=True)))
    with pytest.raises(ValueError):
        check(context, tmp_path, texts=texts)


def test_actual_whole_program_has_nonzero_complete_outcome(context, tmp_path):
    import os

    from merlin.verify.lean_graph_loops_memory_scf_witness import WITNESS_ROOTS, witness_obligation

    selection, _ = context
    result = check(context, tmp_path / "bound")
    model = json.loads(Path(result["model"]["path"]).read_bytes())
    query = tmp_path / "Witness.lean"
    query.write_text(witness_obligation(model))
    dependencies = selection.dependencies()
    env = dict(
        os.environ,
        LEAN_PATH=str(selection.proof.library),
        LEAN_SYSROOT=str(selection.proof.executable.parent.parent),
        LEAN_NUM_THREADS="2",
    )
    checked = subprocess.run(
        [str(selection.proof.executable), str(query)], env=env, capture_output=True, text=True, check=False, timeout=120
    )
    (tmp_path / "stdout.log").write_text(checked.stdout)
    (tmp_path / "stderr.log").write_text(checked.stderr)
    assert checked.returncode == 0
    assert dependencies == selection.dependencies()
    report = _audit(checked.stdout, MEMORY_SCF_ROOTS + WITNESS_ROOTS)
    (tmp_path / "INHABITED.json").write_text(
        json.dumps(
            dict(
                root_audit=report,
                seeds=[17, 29],
                complete_public_load_observations=[17, 34, 29, 58],
                actual_full_instruction_sequence=True,
                common_abstract_interpretation=True,
                numeric_or_native_primitive_conformance_proved=False,
            ),
            indent=2,
        )
        + "\n"
    )
