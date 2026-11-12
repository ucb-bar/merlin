from __future__ import annotations

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
from merlin.verification.lean_contract import LeanToolchain, file_digest, imported_product
from merlin.verification.lean_resource_geometry import (
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
    module = parse_ir(
        "builtin.module {\n      func.func @forward(%x: i8, %y: i8) -> (i8, i8) {\n        %a = arith.addi %x, %y : i8\n        %b = arith.muli %a, %y : i8\n        %c = arith.subi %a, %x : i8\n        return %b, %c : i8, i8\n      }\n    }"
    )
    entry = next((op for op in module.walk() if isinstance(op, func.FuncOp))).body.block
    ops = tuple(entry.ops)
    if state:
        event = _StateWrite.build(operands=[ops[0].results[0]])
        entry.insert_op_before(event, ops[-1])
    index = IRIndex(module)
    bindings = tuple((ResourceBinding(index.id(value), "arguments", 2, 2) for value in entry.args))
    bindings += tuple((ResourceBinding(index.id(value), "results", 2, 2) for op in entry.ops for value in op.results))
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
    return (module, model)


def _emit(module, model):
    allocation = allocate_instruction_resources(module, **model)
    assert allocation.status == "feasible", allocation.reason
    return emit_resource_allocation(module, allocation, **model)


@pytest.fixture(scope="module")
def real_resource_toolchain(tmp_path_factory):
    manifest = os.environ.get("MERLIN_LEAN_TOOLCHAIN_MANIFEST")
    if not manifest:
        pytest.skip("explicit independently selected Lean toolchain unavailable")
    work = tmp_path_factory.mktemp("resource-geometry")
    selected = LeanToolchain.from_manifest(Path(manifest), work_root=work)
    source = repo_root() / "src/merlin/verification/lean/Merlin/ResourceGeometry.lean"
    product = work / "lib/Merlin/ResourceGeometry.olean"
    product.parent.mkdir(parents=True)
    before = file_digest(source)
    environment = dict(
        os.environ, LEAN_PATH=os.pathsep.join((str(root) for root in selected.library_roots)), TMPDIR=str(work)
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
        ((path, file_digest(path)) for path in (work / "lib").rglob("*") if path.is_file() and imported_product(path))
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
        LEAN_PATH=os.pathsep.join((str(root) for root in toolchain.library_roots)),
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


# Explicit fixture/reference reexports consumed by retained direct tests.
__all__ = [
    "ATTRIBUTE",
    "AddressConstraint",
    "GeometryObligation",
    "IRDLOperation",
    "IRIndex",
    "InstructionConstraints",
    "LeanToolchain",
    "MemoryWriteEffect",
    "Path",
    "Reservation",
    "ResourceBinding",
    "StorageBank",
    "_StateWrite",
    "_case",
    "_emit",
    "_run_lean",
    "allocate_instruction_resources",
    "builtin",
    "emit_resource_allocation",
    "file_digest",
    "func",
    "imported_product",
    "ir_digest",
    "irdl_op_definition",
    "json",
    "lean_resource_query",
    "operand_def",
    "os",
    "parse_ir",
    "project_resource_geometry",
    "pytest",
    "real_resource_toolchain",
    "replace",
    "repo_root",
    "subprocess",
    "traits_def",
]
