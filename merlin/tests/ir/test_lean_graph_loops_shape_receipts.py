"""Original context and same-buffer bindings are gates, not saved-receipt trust."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from merlin.tests.ir.test_lean_graph_loops_shape import check, context_data
from merlin.verify import lean_graph_loops_shape as checker
from merlin.verify.lean_graph_loops import _reference
from merlin.verify.lean_graph_loops_memory_scf import audit
from merlin.verify.lean_graph_loops_shape_model import SHAPE_ROOTS


@pytest.fixture(scope="module")
def context():
    return context_data()


def rewritten(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")
    return _reference(path)


@pytest.mark.parametrize("kind", ("build_rehashed", "compiler_reselected", "library_reselected", "manifest_rehashed"))
def test_original_selection_rejects_coherent_local_replacement(context, tmp_path, kind):
    selection, _ = context
    if kind == "build_rehashed":
        data = json.loads(selection.proof.build_record.read_bytes())
        data["attempts"] = data["attempts"][:-1]
        path = tmp_path / "proof.json"
        row = rewritten(path, data)
        selected = dataclasses.replace(
            selection, proof=dataclasses.replace(selection.proof, build_record=path, build_sha256=row["sha256"])
        )
    elif kind == "compiler_reselected":
        selected = dataclasses.replace(selection, proof=dataclasses.replace(selection.proof, sha256="0" * 64))
    elif kind == "library_reselected":
        selected = dataclasses.replace(selection, proof=dataclasses.replace(selection.proof, library=tmp_path))
    else:
        data = json.loads(selection.manifest.read_bytes())
        data["producer_source"] = data["producer_source"][:-1]
        path = tmp_path / "manifest.json"
        row = rewritten(path, data)
        selected = dataclasses.replace(selection, manifest=path, manifest_sha256=row["sha256"])
    with pytest.raises(ValueError, match="original policy"):
        selected.dependencies()


@pytest.mark.parametrize(
    "kind",
    (
        "affected",
        "unknown_import",
        "changed_import",
        "wrong_original_context",
        "audit_axiom",
        "audit_missing_root",
        "audit_failed",
        "host_stale",
        "descriptor_policy",
    ),
)
def test_rehashed_evidence_cannot_grant_admission(context, tmp_path, kind):
    selection, _ = context
    contract = json.loads(selection.contract.read_bytes())
    if kind.startswith("audit_"):
        data = json.loads(Path(contract["root_audit"]["path"]).read_bytes())
        if kind == "audit_axiom":
            data["audit"][SHAPE_ROOTS[0]] = ["unreviewedNumericalLaw"]
        elif kind == "audit_missing_root":
            del data["audit"][SHAPE_ROOTS[0]]
        else:
            data["returncode"] = 1
        contract["root_audit"] = rewritten(tmp_path / "audit.json", data)
    elif kind == "host_stale":
        contract["host_sources"][0]["sha256"] = "0" * 64
    elif kind == "descriptor_policy":
        contract["descriptor_policy"] = "matching_metadata_grants_runtime_conformance"
    else:
        data = json.loads(Path(contract["unaffected_assessment"]["path"]).read_bytes())
        if kind == "affected":
            data["suspect_selected"] = True
        elif kind == "unknown_import":
            data["missing_original_bindings"] = [{"module": "Unknown"}]
        elif kind == "changed_import":
            data["changed"] = [{"module": "Unknown"}]
        else:
            data["original_context"]["sha256"] = "0" * 64
        contract["unaffected_assessment"] = rewritten(tmp_path / "assessment.json", data)
    path = tmp_path / "contract.json"
    row = rewritten(path, contract)
    selected = dataclasses.replace(selection, contract=path, contract_sha256=row["sha256"])
    with pytest.raises(ValueError):
        selected.dependencies()


def test_missing_actual_import_not_hidden_by_rehashed_rosters(context, tmp_path):
    selection, _ = context
    build = json.loads(selection.proof.build_record.read_bytes())
    removed = build["attempts"].pop()["module"]
    path = tmp_path / "build.json"
    row = rewritten(path, build)
    proof = dataclasses.replace(selection.proof, build_record=path, build_sha256=row["sha256"])
    contract = json.loads(selection.contract.read_bytes())
    contract["selected_context"]["proof"]["build_record"] = str(path)
    contract["selected_context"]["proof"]["build_sha256"] = row["sha256"]
    contract["proof_modules"].remove(removed)
    path = tmp_path / "contract.json"
    row = rewritten(path, contract)
    selected = dataclasses.replace(selection, proof=proof, contract=path, contract_sha256=row["sha256"])
    with pytest.raises(ValueError, match="actual inspected imports"):
        selected.dependencies()


@pytest.mark.parametrize("kind", ("original", "saved_buffer", "export", "model", "query"))
def test_same_buffers_checked_after_oracle(context, tmp_path, monkeypatch, kind):
    _, cases = context
    original, candidate, _ = cases["rank_two"]
    run = checker.subprocess.run

    def changed(command, *args, **kwargs):
        result = run(command, *args, **kwargs)
        if command[0] == str(context[0].proof.executable):
            path = {
                "original": tmp_path / "source.mlir",
                "saved_buffer": tmp_path / "proof/before.mlir",
                "export": tmp_path / "proof/actual-pair.json",
                "model": tmp_path / "proof/model.json",
                "query": tmp_path / "proof/Obligation.lean",
            }[kind]
            path.write_bytes(path.read_bytes() + b"\n ")
        return result

    monkeypatch.setattr(checker.subprocess, "run", changed)
    result = check(context, tmp_path, texts=(original.read_text(), candidate.read_text()))
    assert result["returncode"] == 0 and not result["lean_complete_shape_checked"]
    assert "changed" in result["refusal"]


@pytest.mark.parametrize("kind", ("missing", "duplicated", "extra", "new_axiom"))
def test_exact_ordered_axiom_audit(kind):
    lines = [f"'{root}' does not depend on any axioms" for root in SHAPE_ROOTS]
    if kind == "missing":
        lines.pop()
    elif kind == "duplicated":
        lines.append(lines[-1])
    elif kind == "extra":
        lines.append("'unrelated' does not depend on any axioms")
    else:
        lines[0] = f"'{SHAPE_ROOTS[0]}' depends on axioms: [sorryAx]"
    with pytest.raises(ValueError):
        audit("\n".join(lines), SHAPE_ROOTS)
