"""Frozen public source selection cannot be supplied by candidate host code."""

import hashlib
import json
from dataclasses import asdict

import pytest
import yaml
from test_spatial_scratch_abi import fixture

from merlin.compile.owned_spatial_source import _local_document, _route, emit, normalize_unused_void_asm_results


def test_original_document_pin_rejects_mutation_links_and_external_paths(tmp_path):
    document = tmp_path / "source.json"
    document.write_text('{"source":"original"}')
    record = {"file": document.name, "sha256": hashlib.sha256(document.read_bytes()).hexdigest()}
    assert _local_document(tmp_path, record)[1] == document.read_bytes()
    document.write_text('{"source":"changed"}')
    with pytest.raises(ValueError, match="differs"):
        _local_document(tmp_path, record)
    with pytest.raises(ValueError, match="direct capsule-local"):
        _local_document(tmp_path, {**record, "file": str(document)})
    link = tmp_path / "linked.json"
    link.symlink_to(document)
    with pytest.raises(ValueError, match="indirect"):
        _local_document(tmp_path, {**record, "file": link.name})


def test_route_decodes_data_and_rechecks_source_plan_identity_and_stack_budget():
    _, plan, writer, workspace = fixture()
    source = "a" * 64
    record = dict(
        schema="operator.connected_spatial_original_route_inputs.v1",
        plan=asdict(plan),
        plan_sha256=plan.identity,
        source_request_sha256=source,
        required_scratch_bytes=workspace.bytes,
        scratch_alignment=workspace.alignment,
        writer_symbol=writer.symbol,
        result_argument=writer.result_argument,
        borrowed_symbol=writer.borrowed_symbol,
        allocation_alignment=writer.allocation_alignment,
        source_pair=["producer", "pool"],
        pointer_alignments=[1, 1, 4, 1],
        helper_symbol="private_source_helper",
    )
    record = json.loads(json.dumps(record))
    decoded = _route(
        record, source_sha256=source, effects_basis="original conditional source premise", stack_budget_bytes=303
    )
    assert decoded.plan == plan and decoded.stack_budget_bytes == 303
    assert decoded.plan.readouts[0].producer.scales == (0.5,)
    assert decoded.plan.readouts[0].pool.scales == (0.9900199770927429,)
    with pytest.raises(ValueError, match="exact frozen source"):
        _route(record, source_sha256="b" * 64, effects_basis="original premise", stack_budget_bytes=303)
    with pytest.raises(ValueError, match="identity"):
        _route(
            {**record, "plan_sha256": "b" * 64},
            source_sha256=source,
            effects_basis="original premise",
            stack_budget_bytes=303,
        )
    with pytest.raises(ValueError, match="byte budget"):
        _route(record, source_sha256=source, effects_basis="original premise", stack_budget_bytes=True)


@pytest.mark.parametrize("damage", [None, "old_schema", "kind", "source", "plan", "capacity", "alignment", "unknown"])
def test_prospective_packed_owner_uses_original_pins_and_exact_private_storage(damage):
    from merlin.compile.scheduling.program_limits import AddressWindow
    from merlin.compile.scheduling.spatial_producer import PackedSpatialProducerPlan

    _, source_plan, writer, _ = fixture()
    plan = PackedSpatialProducerPlan(source_plan, 7, AddressWindow("original_packed", 0, 512), 64)
    original = "a" * 64
    record = json.loads(
        json.dumps(
            dict(
                schema="operator.connected_spatial_original_route_inputs.v2",
                plan_kind="staged_packed",
                plan=asdict(plan),
                plan_sha256=plan.identity,
                source_request_sha256=original,
                required_scratch_bytes=512,
                scratch_alignment=64,
                writer_symbol=writer.symbol,
                result_argument=writer.result_argument,
                borrowed_symbol=writer.borrowed_symbol,
                allocation_alignment=writer.allocation_alignment,
                source_pair=["producer", "pool"],
                pointer_alignments=[1, 1, 4, 1],
                helper_symbol="private_source_helper",
            )
        )
    )
    if damage == "old_schema":
        record["schema"] = "operator.connected_spatial_original_route_inputs.v1"
    elif damage == "kind":
        record["plan_kind"] = "staged_gather"
    elif damage == "source":
        record["source_request_sha256"] = "b" * 64
    elif damage == "plan":
        record["plan_sha256"] = "b" * 64
    elif damage == "capacity":
        record["required_scratch_bytes"] = 511
    elif damage == "alignment":
        record["scratch_alignment"] = 32
    elif damage == "unknown":
        record["candidate_asserted_effect"] = "noescape"
    if damage is not None:
        with pytest.raises(ValueError):
            _route(
                record,
                source_sha256=original,
                effects_basis="original conditional source premises",
                stack_budget_bytes=575,
            )
        return
    route = _route(
        record, source_sha256=original, effects_basis="original conditional source premises", stack_budget_bytes=575
    )
    assert route.plan == plan and route.workspace.bytes == 512 and route.stack_budget_bytes == 575
    assert route.plan.source.readouts == source_plan.readouts


def test_in_memory_selection_cannot_enable_an_undeclared_frozen_source(tmp_path):
    frozen = {
        "name": "ordinary_public_source",
        "kind": "model_slice",
        "label": "public",
        "operation": {"op": "spatial_producer_pool"},
        "semantic": {"must_accelerate": True},
    }
    (tmp_path / "capsule.yaml").write_text(yaml.safe_dump(frozen))
    candidate = {**frozen, "__dir__": str(tmp_path), "owned_host_source": {"schema": "merlin.owned_spatial_source.v1"}}
    with pytest.raises(ValueError, match="explicit frozen public"):
        emit(
            candidate,
            "untrusted submitted whole host",
            {},
            tmp_path / "output",
            target="fixture",
            candidate_package=tmp_path,
            binding=None,
            compile_flags=(),
            completion_symbol="complete",
            entry_symbol="entry",
        )
    assert not (tmp_path / "output").exists()


def _private_source(tmp_path):
    selected = {"schema": "merlin.owned_spatial_source.v2", "stack_budget_bytes": 8192,
                "effects_basis": "original conditional source premise"}
    for role in ("source", "plan", "source_graph", "source_rewrite"):
        path = tmp_path / (role + ".json")
        path.write_text('{"original":"' + role + '"}')
        selected[role] = {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    declaration = {"name": "original_region", "kind": "model_slice", "label": "hidden",
                   "model": "synthetic_owner_model", "source_role": "model_derived",
                   "owned_host_source": selected, "operation": {"op": "spatial_producer_pool"},
                   "semantic": {"must_accelerate": True}}
    raw = yaml.safe_dump(declaration).encode()
    (tmp_path / "capsule.yaml").write_bytes(raw)
    return declaration, raw


def test_private_source_intake_requires_scoped_exact_owner_selection(tmp_path):
    from merlin.compile.owned_spatial_source import (
        SpatialSourceDiagnostic,
        _source_visibility_selected,
        scoped_spatial_source_diagnostic,
    )

    declaration, raw = _private_source(tmp_path)
    assert not _source_visibility_selected(declaration, raw)
    selected = SpatialSourceDiagnostic.from_frozen_capsule(
        tmp_path, declaration_sha256=hashlib.sha256(raw).hexdigest(),
    )
    with scoped_spatial_source_diagnostic(selected):
        assert _source_visibility_selected(declaration, raw)
        assert not _source_visibility_selected({**declaration, "model": "another_model"}, raw)
        assert not _source_visibility_selected(declaration, raw + b"\n")
        # Even a matching trusted selection cannot relabel the in-memory original.
        with pytest.raises(ValueError, match="differs from its frozen"):
            emit(
                {**declaration, "model": "another_model", "__dir__": str(tmp_path)}, "", {}, tmp_path / "out",
                target="fixture", candidate_package=tmp_path, binding=None, compile_flags=(),
                completion_symbol="complete", entry_symbol="entry",
            )
    assert not _source_visibility_selected(declaration, raw)


@pytest.mark.parametrize("mutation", ["wrong_pin", "public", "missing_model", "old_schema", "changed_document"])
def test_private_source_selection_cannot_grant_unknown_or_changed_source(tmp_path, mutation):
    from merlin.compile.owned_spatial_source import SpatialSourceDiagnostic

    declaration, raw = _private_source(tmp_path)
    if mutation == "public":
        declaration["label"] = "public"
    elif mutation == "missing_model":
        del declaration["model"]
    elif mutation == "old_schema":
        declaration["owned_host_source"]["schema"] = "merlin.owned_spatial_source.v1"
    elif mutation == "changed_document":
        (tmp_path / "source_graph.json").write_text("changed source")
    if mutation in ("public", "missing_model", "old_schema"):
        raw = yaml.safe_dump(declaration).encode()
        (tmp_path / "capsule.yaml").write_bytes(raw)
    with pytest.raises(ValueError):
        SpatialSourceDiagnostic.from_frozen_capsule(
            tmp_path, declaration_sha256="0" * 64 if mutation == "wrong_pin" else hashlib.sha256(raw).hexdigest(),
        )


def test_void_transport_normalization_preserves_effects_and_real_results():
    from xdsl.dialects import llvm
    from xdsl.dialects.builtin import ModuleOp, i32

    from merlin.llvmlower.host_builder import FnBuilder

    builder = FnBuilder([])
    discarded = builder.add(llvm.InlineAsmOp("fence", "~{memory}", [], has_side_effects=True))
    imported = type(discarded).create(result_types=[llvm.LLVMVoidType()], properties=dict(discarded.properties))
    builder.blk.insert_op_before(imported, discarded)
    builder.blk.erase_op(discarded)
    real = builder.add(llvm.InlineAsmOp("", "=r", [], res_types=[i32], has_side_effects=True))
    module = ModuleOp([llvm.FuncOp("entry", llvm.LLVMFunctionType([], llvm.LLVMVoidType()), body=builder.finish())])
    properties = dict(imported.properties)
    assert normalize_unused_void_asm_results(module) == 1
    normalized = next(op for op in module.walk() if isinstance(op, llvm.InlineAsmOp))
    assert not normalized.results and normalized.properties == properties
    assert real.results[0].type == i32
    assert normalize_unused_void_asm_results(module) == 0
