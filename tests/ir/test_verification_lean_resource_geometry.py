from __future__ import annotations

from verification_witnesses.lean_resource_geometry import (
    ATTRIBUTE,
    AddressConstraint,
    GeometryObligation,
    InstructionConstraints,
    IRDLOperation,
    IRIndex,
    LeanToolchain,
    MemoryWriteEffect,
    Path,
    Reservation,
    ResourceBinding,
    StorageBank,
    _case,
    _emit,
    _run_lean,
    _StateWrite,
    allocate_instruction_resources,
    builtin,
    emit_resource_allocation,
    file_digest,
    func,
    imported_product,
    ir_digest,
    irdl_op_definition,
    json,
    lean_resource_query,
    operand_def,
    os,
    parse_ir,
    project_resource_geometry,
    pytest,
    real_resource_toolchain,
    replace,
    repo_root,
    subprocess,
    traits_def,
)


def test_real_lean_rejects_wrong_geometry_after_a_valid_structural_projection(real_resource_toolchain, tmp_path):
    (module, model) = _case()
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
    assert all(("decide" in row["data"] for row in errors))
