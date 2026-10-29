"""Actual emitted resource geometry, including independent mutations and Lean checks."""

from __future__ import annotations

import builtins
import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from xdsl.dialects import builtin, func
from xdsl.irdl import IRDLOperation, irdl_op_definition, operand_def, traits_def
from xdsl.traits import MemoryWriteEffect

from merlin.common.paths import repo_root
from merlin.semantic_compiler.allocation_types import Reservation, StorageBank
from merlin.semantic_compiler.rules import AddressConstraint
from merlin.transforms.analyses import IRIndex
from merlin.transforms.certificates import ir_digest, parse_ir
from merlin.transforms.resource_allocation import (
    ATTRIBUTE,
    InstructionConstraints,
    ResourceBinding,
    allocate_instruction_resources,
    emit_resource_allocation,
)
from merlin.verify.lean_contract import LeanToolchain, file_digest, imported_product
from merlin.verify.lean_resource_geometry import (
    GeometryObligation,
    lean_resource_query,
    project_resource_geometry,
)


@irdl_op_definition
class _StateWrite(IRDLOperation):
    name = "public_probe.resource_state"
    value = operand_def(builtin.i8)
    traits = traits_def(MemoryWriteEffect())


def _case(*, state=False):
    module = parse_ir("""builtin.module {
      func.func @forward(%x: i8, %y: i8) -> (i8, i8) {
        %a = arith.addi %x, %y : i8
        %b = arith.muli %a, %y : i8
        %c = arith.subi %a, %x : i8
        return %b, %c : i8, i8
      }
    }""")
    entry = next(op for op in module.walk() if isinstance(op, func.FuncOp)).body.block
    ops = tuple(entry.ops)
    if state:
        event = _StateWrite.build(operands=[ops[0].results[0]])
        entry.insert_op_before(event, ops[-1])
    index = IRIndex(module)
    bindings = tuple(ResourceBinding(index.id(value), "arguments", 2, 2) for value in entry.args)
    bindings += tuple(ResourceBinding(index.id(value), "results", 2, 2) for op in entry.ops for value in op.results)
    constraints = (
        InstructionConstraints(
            index.operation_ids[ops[1]], validity=(AddressConstraint("eq_offset", "out", "in0", 2),)
        ),
        InstructionConstraints(
            index.operation_ids[ops[2]], validity=(AddressConstraint("eq_offset", "in0", "out", -4),)
        ),
    )
    if state:
        constraints += (
            InstructionConstraints(
                index.operation_ids[event],
                validity=(AddressConstraint("aligned", "in0", value=2),),
                input_read_offsets=(1,),
                completion_offset=3,
            ),
        )
    model = dict(
        entry="forward",
        bindings=bindings,
        banks=(StorageBank("arguments", "shared", 12, "slot", 2), StorageBank("results", "shared", 12, "slot", 2)),
        reservations=(Reservation("results", 4, 2),),
        instruction_constraints=constraints,
        fixed_inputs={index.id(entry.args[0]): 0, index.id(entry.args[1]): 2},
        fixed_outputs=(8, 10),
    )
    return module, model


def _in_place_case():
    module = parse_ir("""builtin.module {
      func.func @forward(%x: i8) -> i8 {
        %a = arith.addi %x, %x : i8
        %b = arith.muli %a, %x : i8
        return %b : i8
      }
    }""")
    entry = next(op for op in module.walk() if isinstance(op, func.FuncOp)).body.block
    index = IRIndex(module)
    bindings = (ResourceBinding(index.id(entry.args[0]), "boundary", 1),)
    bindings += tuple(ResourceBinding(index.id(op.results[0]), "local", 1) for op in entry.ops if op.results)
    model = dict(
        entry="forward",
        bindings=bindings,
        banks=(StorageBank("boundary", "inputs", 1, "slot"), StorageBank("local", "register", 1, "slot")),
        instruction_constraints=(
            InstructionConstraints(index.operation_ids[tuple(entry.ops)[1]], completion_offset=2, in_place_inputs=(0,)),
        ),
        fixed_outputs=(0,),
    )
    return module, model


def _emit(module, model):
    allocation = allocate_instruction_resources(module, **model)
    assert allocation.status == "feasible", allocation.reason
    return emit_resource_allocation(module, allocation, **model)


def _change_address(emitted, row_index, address):
    header = dict(emitted.attributes[ATTRIBUTE].data)
    rows = list(header["values"])
    row = dict(rows[row_index].data)
    row["address"] = builtin.IntegerAttr(address, builtin.i32)
    rows[row_index] = builtin.DictionaryAttr(row)
    header["values"] = builtin.ArrayAttr(rows)
    emitted.attributes[ATTRIBUTE] = builtin.DictionaryAttr(header)


def test_projection_reads_actual_assignments_and_contains_complete_geometry():
    module, model = _case(state=True)
    emitted = _emit(module, model)
    projection = project_resource_geometry(module, emitted, **model)
    assert projection.source_sha256 == ir_digest(module)
    assert projection.emitted_sha256 == ir_digest(emitted)
    assert projection.constraints_sha256 == emitted.attributes[ATTRIBUTE].data["constraints_sha256"].data
    kinds = {row.kind for row in projection.obligations}
    assert {
        "fits",
        "disjoint",
        "separated",
        "fixed_address",
        "aligned",
        "equal_offset",
        "read_within",
        "zero_extent_event",
        "serial_issue",
    } <= kinds
    assert any(row.kind == "equal_offset" and row.numbers == (6, 10, -4) for row in projection.obligations)
    assert any(row.kind == "read_within" and row.numbers == (1, 4, 1, 7, 8) for row in projection.obligations)
    assert any(row.kind == "disjoint" and row.subjects[-1] == "preserved_input" for row in projection.obligations)
    assert lean_resource_query(projection).endswith("#print axioms merlin_transform\n")
    assert project_resource_geometry(module, emitted, **model) == projection


def test_final_use_overwrite_is_a_distinct_proven_obligation():
    module, model = _in_place_case()
    projection = project_resource_geometry(module, _emit(module, model), **model)
    overwrites = [row for row in projection.obligations if row.kind == "final_use_overwrite"]
    assert len(overwrites) == 1
    assert overwrites[0].numbers == (0, 0, 1, 1, 1, 1, 0, 2, 4)


@pytest.mark.parametrize(
    "fault",
    ["reserved", "capacity", "alignment", "boundary", "source", "constraints", "state_read", "partial_overwrite"],
)
def test_actual_annotation_or_model_mutations_cannot_create_geometry_proof(fault):
    module, model = _case(state=True)
    if fault == "partial_overwrite":
        module, model = _in_place_case()
    emitted = _emit(module, model)
    if fault in {"reserved", "capacity", "alignment", "boundary"}:
        _change_address(
            emitted,
            0 if fault == "boundary" else 2,
            {"reserved": 4, "capacity": 12, "alignment": 7, "boundary": 2}[fault],
        )
    elif fault == "source":
        operation = next(op for op in emitted.walk() if op.name == "arith.addi")
        operation.operands = (operation.operands[0], operation.operands[0])
    elif fault == "constraints":
        model["banks"] = tuple(replace(bank, capacity=14) for bank in model["banks"])
    elif fault == "state_read":
        model["instruction_constraints"] = (
            *model["instruction_constraints"][:-1],
            replace(model["instruction_constraints"][-1], input_read_offsets=(4,)),
        )
    else:
        model["instruction_constraints"] = (replace(model["instruction_constraints"][0], in_place_inputs=()),)
    with pytest.raises(ValueError):
        project_resource_geometry(module, emitted, **model)


def test_projection_never_runs_the_solver_and_budget_refusal_is_explicit(monkeypatch):
    module, model = _case()
    emitted = _emit(module, model)
    original = builtins.__import__

    def deny_solver(name, *args, **kwargs):
        if name == "z3" or name.startswith("z3."):
            raise AssertionError("geometry projection ran the solver")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", deny_solver)
    assert project_resource_geometry(module, emitted, **model).obligations
    with pytest.raises(ValueError, match="obligation budget"):
        project_resource_geometry(module, emitted, **model, max_obligations=1)


@pytest.fixture(scope="module")
def real_resource_toolchain(tmp_path_factory):
    manifest = os.environ.get("MERLIN_LEAN_TOOLCHAIN_MANIFEST")
    if not manifest:
        pytest.skip("explicit independently selected Lean toolchain unavailable")
    work = tmp_path_factory.mktemp("resource-geometry")
    selected = LeanToolchain.from_manifest(Path(manifest), work_root=work)
    source = repo_root() / "src/merlin/verify/lean/Merlin/ResourceGeometry.lean"
    product = work / "lib/Merlin/ResourceGeometry.olean"
    product.parent.mkdir(parents=True)
    before = file_digest(source)
    environment = dict(
        os.environ, LEAN_PATH=os.pathsep.join(str(root) for root in selected.library_roots), TMPDIR=str(work)
    )
    result = subprocess.run(
        [
            str(selected.executable),
            "-j1",
            f"-M{selected.memory_mb}",
            f"--root={source.parent.parent}",
            "-o",
            str(product),
            str(source),
        ],
        env=environment,
        cwd=work,
        capture_output=True,
        text=True,
        timeout=selected.timeout_s,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert file_digest(source) == before
    additions = tuple(
        (path, file_digest(path)) for path in (work / "lib").rglob("*") if path.is_file() and imported_product(path)
    )
    extended = replace(
        selected, library_roots=(work / "lib", *selected.library_roots), files=(*selected.files, *additions)
    )
    extended.verify()
    return extended


def _run_lean(query, toolchain, directory):
    directory.mkdir()
    source = directory / "Check.lean"
    source.write_text(query)
    environment = dict(
        os.environ,
        LEAN_PATH=os.pathsep.join(str(root) for root in toolchain.library_roots),
        TMPDIR=str(toolchain.work_root),
    )
    toolchain.verify()
    result = subprocess.run(
        [str(toolchain.executable), "--json", "-j1", f"-M{toolchain.memory_mb}", str(source)],
        cwd=directory,
        env=environment,
        capture_output=True,
        text=True,
        timeout=toolchain.timeout_s,
    )
    toolchain.verify()
    return result


@pytest.mark.parametrize("case", ["shared_backing", "state_event", "in_place"])
def test_real_lean_checks_supported_actual_assignment_domain(case, real_resource_toolchain, tmp_path):
    module, model = _in_place_case() if case == "in_place" else _case(state=case == "state_event")
    projection = project_resource_geometry(module, _emit(module, model), **model)
    result = _run_lean(lean_resource_query(projection), real_resource_toolchain, tmp_path / "valid")
    assert result.returncode == 0, result.stdout + result.stderr
    messages = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    assert not any(row.get("severity") in {"warning", "error"} for row in messages)
    report = next(row["data"] for row in messages if "'merlin_transform'" in row.get("data", ""))
    assert "does not depend on any axioms" in report


def test_real_lean_rejects_wrong_geometry_after_a_valid_structural_projection(real_resource_toolchain, tmp_path):
    module, model = _case()
    projection = project_resource_geometry(module, _emit(module, model), **model)
    faults = (
        GeometryObligation("fits", ("capacity",), (6, 2, 7, 2)),
        GeometryObligation("fits", ("alignment",), (6, 2, 12, 4)),
        GeometryObligation("disjoint", ("reservation",), (0, 2, 1, 2)),
        GeometryObligation("separated", ("live_backing_alias",), (0, 2, 1, 4, 1, 2, 2, 5)),
        GeometryObligation("final_use_overwrite", ("extra_use",), (0, 0, 1, 1, 1, 2, 0, 2, 4)),
        GeometryObligation("read_within", ("state_read",), (0, 1, 4, 3, 5)),
        GeometryObligation("equal_offset", ("signed_offset",), (6, 10, -3)),
        GeometryObligation("fixed_address", ("boundary",), (4, 5)),
        GeometryObligation("zero_extent_event", ("state_storage",), (1,)),
        GeometryObligation("serial_issue", ("state_order",), (3, 3)),
    )
    wrong = replace(projection, obligations=faults)
    result = _run_lean(lean_resource_query(wrong), real_resource_toolchain, tmp_path / "wrong")
    assert result.returncode != 0
    messages = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    errors = [row for row in messages if row.get("severity") == "error"]
    assert len(errors) == len(faults)
    assert all("decide" in row["data"] for row in errors)


def test_nonzero_view_windows_produce_translated_existing_interval_obligations():
    module, model = _case(state=True)
    model = dict(model, banks=tuple(replace(b, start=8, capacity=b.capacity + 8) for b in model["banks"]),
                 reservations=tuple(replace(r, start=r.start + 8) for r in model["reservations"]),
                 fixed_inputs={name: address + 8 for name, address in model["fixed_inputs"].items()},
                 fixed_outputs=tuple(address + 8 for address in model["fixed_outputs"]))
    emitted = _emit(module, model)
    projection = project_resource_geometry(module, emitted, **model)
    translated = [r for r in projection.obligations if r.kind == "fits" and r.subjects[-1] == "window"]
    assert translated and any(r.numbers == (0, 2, 12, 1) for r in translated)
    assert all(r.numbers[0] >= 0 and r.numbers[0] + r.numbers[1] <= r.numbers[2] for r in translated)
    assert lean_resource_query(projection).endswith("#print axioms merlin_transform\n")
    _change_address(emitted, 0, 0)
    with pytest.raises(ValueError, match="range|fixed external|ABI"):
        project_resource_geometry(module, emitted, **model)
