"""Original context, exact compiler buffers and inhabited semantic witnesses."""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
from pathlib import Path

import pytest

from merlin.tests.ir.test_lean_graph_loops_shape import parse
from merlin.tests.ir.test_lean_graph_loops_shape_epoch import case_paths, project, selected
from merlin.verify import lean_graph_loops_shape_epoch as checker
from merlin.verify.lean_graph_loops import _reference
from merlin.verify.lean_graph_loops_memory_scf import audit
from merlin.verify.lean_graph_loops_shape_epoch_cases import CASES
from merlin.verify.lean_graph_loops_shape_epoch_selection import ALLOCATOR_POLICY, DESCRIPTOR_POLICY, ENTRY_POLICY
from merlin.verify.lean_graph_loops_shape_epoch_witness import witness_obligation
from merlin.verify.lean_graph_loops_shape_model import SHAPE_ROOTS


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")
    return _reference(path)


@pytest.mark.parametrize("kind", ("compiler", "library", "parent", "build"))
def test_reselected_context_refused(tmp_path, kind):
    original = selected()
    if kind == "compiler":
        replacement = dataclasses.replace(original, proof=dataclasses.replace(original.proof, sha256="0" * 64))
    elif kind == "library":
        replacement = dataclasses.replace(original, proof=dataclasses.replace(original.proof, library=tmp_path))
    elif kind == "parent":
        replacement = dataclasses.replace(original, parent_sha256="0" * 64)
    else:
        replacement = dataclasses.replace(original, proof=dataclasses.replace(original.proof, build_sha256="0" * 64))
    with pytest.raises(ValueError, match="original contract"):
        replacement.dependencies()


@pytest.mark.parametrize(
    "kind",
    (
        "allocator",
        "descriptor",
        "entry",
        "host",
        "root_missing",
        "axiom",
        "audit_failed",
        "suspect",
        "missing_import",
        "source_replaced",
        "provider_reselected",
    ),
)
def test_coherently_rehashed_evidence_does_not_grant_admission(tmp_path, kind):
    original = selected()
    contract = json.loads(original.contract.read_bytes())
    if kind in ("allocator", "descriptor", "entry"):
        contract["policies"][kind] = "unchecked"
    elif kind == "host":
        contract["host_sources"][0]["sha256"] = "0" * 64
    elif kind == "provider_reselected":
        contract["producer_source"] = contract["producer_source"][:-1]
    elif kind in ("suspect", "missing_import"):
        data = json.loads(Path(contract["assessment"]["path"]).read_bytes())
        data["suspect_selected" if kind == "suspect" else "missing"] = True if kind == "suspect" else ["Unknown"]
        contract["assessment"] = save(tmp_path / "assessment.json", data)
    elif kind == "source_replaced":
        data = json.loads(original.proof.build_record.read_bytes())
        data["attempts"][-1]["source"]["sha256"] = "0" * 64
        path = tmp_path / "build.json"
        row = save(path, data)
        proof = dataclasses.replace(original.proof, build_record=path, build_sha256=row["sha256"])
        contract["selected"]["proof"]["build_record"] = str(path)
        contract["selected"]["proof"]["build_sha256"] = row["sha256"]
        original = dataclasses.replace(original, proof=proof)
    else:
        data = json.loads(Path(contract["root_audit"]["path"]).read_bytes())
        if kind == "root_missing":
            del data["audit"]["GraphShapeEpochObligation.complete_chain"]
        elif kind == "axiom":
            data["audit"]["GraphShapeEpochObligation.complete_chain"] = ["unreviewedBridge"]
        else:
            data["returncode"] = 1
        contract["root_audit"] = save(tmp_path / "audit.json", data)
    path = tmp_path / "contract.json"
    row = save(path, contract)
    replacement = dataclasses.replace(original, contract=path, contract_sha256=row["sha256"])
    with pytest.raises(ValueError):
        replacement.dependencies()


@pytest.mark.parametrize("kind", ("original", "saved_middle", "model"))
def test_post_oracle_binding_change_refuses(tmp_path, monkeypatch, kind):
    context = selected()
    paths = []
    for index, source in enumerate(case_paths(CASES[0])):
        path = tmp_path / f"input{index}.mlir"
        path.write_bytes(source.read_bytes())
        paths.append(path)
    run = checker.subprocess.run

    def changed(command, *args, **kwargs):
        result = run(command, *args, **kwargs)
        if command[0] == str(context.proof.executable):
            target = {
                "original": paths[0],
                "saved_middle": tmp_path / "proof/normalized.mlir",
                "model": tmp_path / "proof/model.json",
            }[kind]
            target.write_bytes(target.read_bytes() + b"\n ")
        return result

    monkeypatch.setattr(checker.subprocess, "run", changed)
    result = checker.verify_epoch_files(
        *paths,
        selection=context,
        output=tmp_path / "proof",
        index_bits=64,
        max_nodes=5000,
        allocator_policy=ALLOCATOR_POLICY,
        descriptor_policy=DESCRIPTOR_POLICY,
        entry_policy=ENTRY_POLICY,
    )
    assert result["returncode"] == 0 and not result["lean_complete_chain_checked"]
    assert "changed" in result["refusal"]


@pytest.mark.parametrize("case", (CASES[0], CASES[1], CASES[-1]), ids=lambda case: case.name)
def test_actual_nonzero_complete_lean_witness(tmp_path, case):
    context = selected()
    before = context.dependencies()
    paths = case_paths(case)
    originals = [_reference(path) for path in paths]
    model = project([parse(path.read_text()) for path in paths], tmp_path / "capture", case.index_bits)
    text, roots = witness_obligation(model)
    query = tmp_path / "Witness.lean"
    query.write_text(text)
    query_ref = _reference(query)
    proof = context.proof
    env = dict(
        os.environ, LEAN_PATH=str(proof.library), LEAN_SYSROOT=str(proof.executable.parent.parent), LEAN_NUM_THREADS="2"
    )
    command = [str(proof.executable), "-DautoImplicit=false", "-DrelaxedAutoImplicit=false", str(query)]
    run = subprocess.run(command, env=env, capture_output=True, text=True, check=False, timeout=600)
    (tmp_path / "stdout.log").write_text(run.stdout)
    (tmp_path / "stderr.log").write_text(run.stderr)
    assert context.dependencies() == before and query_ref == _reference(query)
    assert originals == [_reference(path) for path in paths]
    assert run.returncode == 0
    audited = audit(run.stdout, SHAPE_ROOTS + roots)
    (tmp_path / "WITNESS.json").write_text(
        json.dumps(
            dict(
                originals=originals,
                query=query_ref,
                returncode=run.returncode,
                audit=audited,
                declared_nat_addition_interpretation=True,
                numerical_native_fidelity_proved=False,
                all_initializer_cells_proved_without_volume_unrolling=True,
            ),
            indent=2,
        )
        + "\n"
    )
