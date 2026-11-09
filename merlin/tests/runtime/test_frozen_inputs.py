"""Immutable source arguments are removed from the caller ABI, never guessed."""

import hashlib
import os
from dataclasses import replace

import numpy as np
import pytest

from merlin.compile.frozen_inputs import FrozenInput, FrozenInputs, _admit, _reference, specialize
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.deployable_outline import canonical_kernel, plan_deployable_chain
from merlin.runtime.outlined_chain import render_outlined_chain
from merlin.runtime.staged_dispatch import StagedRoles

SOURCE = """module {
func.func @forward(%w:tensor<3x2xi8>,%x:tensor<2xi32>)->tensor<2xi32> {
 %e=tensor.empty():tensor<3x2xi32>
 %wide=linalg.generic {indexing_maps=[affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>],
 iterator_types=["parallel","parallel"]} ins(%w:tensor<3x2xi8>)outs(%e:tensor<3x2xi32>){
 ^bb0(%a:i8,%o:i32):
 %v=arith.extsi %a:i8 to i32
 linalg.yield %v:i32
 }->tensor<3x2xi32>
 %e1=tensor.empty():tensor<2xi32>
 %z=arith.constant 0:i32
 %init=linalg.fill ins(%z:i32)outs(%e1:tensor<2xi32>)->tensor<2xi32>
 %sum="linalg.reduce"(%wide,%init) <{dimensions=array<i64:0>}> ({
 ^bb0(%a:i32,%b:i32):
 %v=arith.addi %a,%b:i32
 linalg.yield %v:i32
 }):(tensor<3x2xi32>,tensor<2xi32>)->tensor<2xi32>
 %e2=tensor.empty():tensor<2xi32>
 %out=linalg.generic {indexing_maps=[affine_map<(i)->(i)>,affine_map<(i)->(i)>,affine_map<(i)->(i)>],
 iterator_types=["parallel"]} ins(%sum,%x:tensor<2xi32>,tensor<2xi32>)outs(%e2:tensor<2xi32>){
 ^bb0(%a:i32,%b:i32,%o:i32):
 %v=arith.addi %a,%b:i32
 linalg.yield %v:i32
 }->tensor<2xi32>
 func.return %out:tensor<2xi32>
}}
"""


def plan(source=SOURCE):
    return plan_deployable_chain(
        parse_mlir_text(source), worker_slots=(0, 0, 0), worker_count=1, workspace_budget_bytes=4096, alignment_bytes=64
    )


def binding(tmp_path, p):
    weight = np.array([[-128, 127], [1, -2], [127, -128]], dtype=np.int8)
    raw = weight.tobytes()
    path = tmp_path / "weight.bin"
    path.write_bytes(raw)
    item = FrozenInput(0, "i8", (3, 2), path, hashlib.sha256(raw).hexdigest())
    return FrozenInputs(p.source_ir_sha256, (item,), 1 << 20, 4096)


def compiler_required():
    if not os.getenv("MERLIN_MLIR_INSTALL") or not os.getenv("MERLIN_COMPILER_PYTHON"):
        pytest.skip("explicit selected native compiler required")


def permutation_plan(shape=(3, 5), width=1):
    permutation = tuple(reversed(range(len(shape))))
    out = tuple(shape[i] for i in permutation)
    dtype = "i" + str(width * 8)
    src = "tensor<" + "x".join(map(str, (*shape, dtype))) + ">"
    dst = "tensor<" + "x".join(map(str, (*out, dtype))) + ">"
    dims = ",".join(f"d{i}" for i in range(len(shape)))
    amap = "affine_map<(" + dims + ")->(" + dims + ")>"
    source = f"""module {{func.func @forward(%w:{src},%x:{dst})->{dst} {{
    %e=tensor.empty():{dst}
    %t=linalg.transpose ins(%w:{src})outs(%e:{dst})permutation={list(permutation)}
    %e2=tensor.empty():{dst}
    %r=linalg.generic {{indexing_maps=[{amap},{amap},{amap}],
    iterator_types=[{",".join(['"parallel"'] * len(shape))}]}}
    ins(%t,%x:{dst},{dst})outs(%e2:{dst}){{
    ^bb0(%a:{dtype},%b:{dtype},%o:{dtype}):
    %v=arith.addi %a,%b:{dtype}
    linalg.yield %v:{dtype}
    }}->{dst}
    func.return %r:{dst}}}}}"""
    return plan_deployable_chain(
        parse_mlir_text(source),
        worker_slots=(0,),
        worker_count=1,
        workspace_budget_bytes=max(4096, 4 * np.prod(shape).item() * width + 256),
        alignment_bytes=64,
        data_preparation=True,
    )


@pytest.mark.parametrize("change", ["source", "shape", "dtype", "bytes", "duplicate", "budget", "index"])
def test_binding_refusals_before_evaluation(tmp_path, change):
    p = plan()
    request = binding(tmp_path, p)
    item = request.inputs[0]
    if change == "source":
        request = replace(request, source_ir_sha256="0" * 64)
    elif change == "shape":
        request = replace(request, inputs=(replace(item, shape=(2, 3)),))
    elif change == "dtype":
        request = replace(request, inputs=(replace(item, dtype="f32"),))
    elif change == "bytes":
        item.file.write_bytes(b"changed")
    elif change == "duplicate":
        request = replace(request, inputs=(item, item))
    elif change == "budget":
        request = replace(request, offline_budget_bytes=1)
    else:
        request = replace(request, inputs=(replace(item, arg_index=True),))
    with pytest.raises(ValueError):
        specialize(p, request, tmp_path / "offline")


@pytest.mark.parametrize("change", ["flags", "unknown", "quant", "float"])
def test_unknown_or_nonwrapping_contracts_do_not_gain_offline_permission(change):
    kernel = canonical_kernel(plan().outlined.module, plan().dispatches[0]["symbol"])
    from xdsl.dialects.builtin import StringAttr

    if change == "flags":
        kernel = canonical_kernel(plan().outlined.module, plan().dispatches[2]["symbol"])
        op = next(o for o in kernel.walk() if o.name == "arith.addi")
        op.properties["overflowFlags"] = StringAttr("nsw")
    elif change == "float":
        kernel = parse_mlir_text(
            SOURCE.replace("i32", "f32")
            .replace("arith.extsi", "arith.sitofp")
            .replace("arith.addi", "arith.addf")
            .replace("0:f32", "0.0:f32")
        )
    else:
        kernel.attributes["unknown.contract" if change == "unknown" else "prov.quant_inner"] = StringAttr("x")
    with pytest.raises(ValueError):
        _admit(kernel)


def test_wrapping_reference_full_range_and_uninitialized_refusal():
    p = plan()
    kernel = canonical_kernel(p.outlined.module, p.dispatches[2]["symbol"])
    fn = next(o for o in kernel.body.block.ops if o.name == "func.func")
    a = np.array([2**31 - 1, -(2**31)], dtype=np.int32)
    b = np.array([1, -1], dtype=np.int32)
    assert _reference(fn, [a, b])[0].tolist() == [-(2**31), 2**31 - 1]


def test_generic_reduction_integer_order_and_full_width_sum():
    source = SOURCE.replace(
        '%sum="linalg.reduce"(%wide,%init) <{dimensions=array<i64:0>}> ({',
        '%sum="linalg.generic"(%wide,%init) <{indexing_maps=[affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(j)>],'
        "iterator_types=[#linalg.iterator_type<reduction>,#linalg.iterator_type<parallel>],"
        "operandSegmentSizes=array<i32:1,1>}> ({",
    )
    p = plan(source)
    kernel = canonical_kernel(p.outlined.module, p.dispatches[1]["symbol"])
    fn = next(o for o in kernel.body.block.ops if o.name == "func.func")
    a = np.array([[2**31 - 1, -(2**31)], [1, -1], [2**31 - 1, 2**31 - 1]], dtype=np.int32)
    assert _reference(fn, [a])[0].tolist() == [-1, -2]


def test_actual_compiler_fold_removes_weight_abi_and_dead_cast_storage(tmp_path):
    compiler_required()
    p = plan()
    before = p.to_dict()
    request = binding(tmp_path, p)
    derived, blob = specialize(p, request, tmp_path / "offline")
    assert p.to_dict() == before
    assert len(derived.dispatches) == 1
    assert [r["folded"] for r in derived.frozen_inputs["dispatch_mapping"]] == [True, True, False]
    assert derived.frozen_inputs["external_input_source_indices"] == [1]
    assert np.frombuffer(blob, dtype="<i4").tolist() == [0, -3]
    assert derived.workspace_bytes < p.workspace_bytes
    emitted = render_outlined_chain(derived, ("runtime_add",), roles=StagedRoles((0, 1, 2), 2, 2, 2, 2, (0,), (0,)))
    assert "void *inputs[1]" in emitted["header"]
    assert "c->inputs[1]" not in emitted["source"]
    assert "extern const unsigned char merlin_chain_frozen_blob[8]" in emitted["source"]
    assert emitted["metadata"]["external_input_source_indices"] == [1]


def test_published_blob_budget_refuses(tmp_path):
    compiler_required()
    p = plan()
    request = replace(binding(tmp_path, p), published_budget_bytes=1)
    with pytest.raises(ValueError, match="published frozen blob"):
        specialize(p, request, tmp_path / "offline")


def test_public_build_links_incbin_and_binds_every_frozen_byte(tmp_path):
    compiler_required()
    import json

    from merlin.compile.outlined_program import build_outlined_program

    p = plan()
    request = binding(tmp_path, p)
    source = tmp_path / "source.mlir"
    source.write_text(SOURCE)
    built = build_outlined_program(
        source,
        tmp_path / "build",
        roles=StagedRoles((0, 1, 2), 2, 2, 2, 2, (0,), (0,)),
        worker_slots=(0, 0, 0),
        workspace_budget_bytes=4096,
        alignment_bytes=64,
        compute_cflags=("-march=rv64gcv",),
        frozen_inputs=request,
    )
    assert built["dispatches"] == 1
    assert len(built["link_objects"]) == 2
    assert all(path in built["input_sha256"] for path in built["link_objects"])
    assert str(request.inputs[0].file) in built["input_sha256"]
    record = json.loads((tmp_path / "build/harness_blobs.json").read_bytes())
    assert record["blobs"][0]["bytes"] == 8
    assert ".incbin" in (tmp_path / "build/harness_blob_merlin_chain_frozen_blob.S").read_text()
    assert built["live_storage_bytes"] == built["workspace_bytes"] + 8
    assert built["frozen_inputs"]["dispatch_mapping"][2]["runtime_dispatch_index"] == 0
    for row in built["frozen_inputs"]["dispatch_mapping"]:
        if row["folded"]:
            proof = row["proof"]
            retained = proof["retained_host_image"]
            assert built["input_sha256"][retained] == proof["host_image_sha256"]
            assert proof["input_sha256"][retained] == proof["host_image_sha256"]
            assert proof["host_image_policy"]["kind"] == "PrivateHostImagePolicy"
    assert not any(".nfs" in path for path in built["input_sha256"])


@pytest.mark.parametrize("width", [1, 2, 4, 8])
def test_offline_byte_permutation_preserves_full_bitpatterns_and_tails(width):
    from merlin.compile.frozen_inputs import _view

    raw = np.arange(3 * 5 * width, dtype=np.uint8)
    array = raw.view("<i" + str(width)).reshape(3, 5)
    view = permutation_plan(width=width).dispatches[0]["views_before"][0]
    assert _view(view, array).tobytes() == array.T.copy().tobytes()


def test_changed_bytes_during_read_are_refused_even_if_path_seal_looks_current(tmp_path, monkeypatch):
    from pathlib import Path

    p = plan()
    request = binding(tmp_path, p)
    real = Path.read_bytes

    def changed(path):
        raw = real(path)
        return bytes([raw[0] ^ 1]) + raw[1:] if path == request.inputs[0].file else raw

    monkeypatch.setattr(Path, "read_bytes", changed)
    with pytest.raises(ValueError, match="changed while reading"):
        specialize(p, request, tmp_path / "offline")


def test_cartesian_iteration_domain_is_budgeted_before_reference_allocation(tmp_path):
    from merlin.compile.frozen_inputs import _evaluate

    source = """module {func.func @forward(%a:tensor<3x7xi8>,%b:tensor<7x5xi8>)->tensor<3x5xi32>{
    %e=tensor.empty():tensor<3x5xi32>
    %z=arith.constant 0:i32
    %init=linalg.fill ins(%z:i32)outs(%e:tensor<3x5xi32>)->tensor<3x5xi32>
    %r=linalg.generic {indexing_maps=[affine_map<(i,j,k)->(i,k)>,affine_map<(i,j,k)->(k,j)>,
    affine_map<(i,j,k)->(i,j)>],iterator_types=["parallel","parallel","reduction"]}
    ins(%a,%b:tensor<3x7xi8>,tensor<7x5xi8>)outs(%init:tensor<3x5xi32>){
    ^bb0(%a0:i8,%b0:i8,%acc:i32):
    %a1=arith.extsi %a0:i8 to i32
    %b1=arith.extsi %b0:i8 to i32
    %prod=arith.muli %a1,%b1:i32
    %sum=arith.addi %acc,%prod:i32
    linalg.yield %sum:i32
    }->tensor<3x5xi32>
    func.return %r:tensor<3x5xi32>}}"""
    p = plan_deployable_chain(
        parse_mlir_text(source), worker_slots=(0,), worker_count=1, workspace_budget_bytes=4096, alignment_bytes=64
    )
    kernel = canonical_kernel(p.outlined.module, p.dispatches[0]["symbol"])
    with pytest.raises(ValueError, match="offline intermediate budget"):
        _evaluate(kernel, [np.zeros((3, 7), dtype=np.int8), np.zeros((7, 5), dtype=np.int8)], tmp_path, 5000)


@pytest.mark.parametrize("shape", [(1, 65537), (65537, 1), (65537,)])
def test_thin_and_rank1_budget_refusal_precedes_permutation_allocation(tmp_path, monkeypatch, shape):
    from merlin.compile import frozen_inputs as module

    p = permutation_plan(shape)
    raw = bytes(np.prod(shape).item())
    path = tmp_path / "weight.bin"
    path.write_bytes(raw)
    request = FrozenInputs(
        p.source_ir_sha256,
        (FrozenInput(0, "i8", shape, path, hashlib.sha256(raw).hexdigest()),),
        3 * len(raw) - 1,
        2 * len(raw),
    )

    def forbidden(*args):
        raise AssertionError("allocation reached despite insufficient budget")

    monkeypatch.setattr(module, "_view", forbidden)
    with pytest.raises(ValueError, match="offline view intermediate budget"):
        specialize(p, request, tmp_path / "offline")


def test_actual_thin_counterexample_within_declared_budget_after_transpose_fix(tmp_path):
    import tracemalloc

    n = 1 << 20
    p = permutation_plan((1, n))
    raw = np.arange(n, dtype=np.uint8).tobytes()
    path = tmp_path / "weight.bin"
    path.write_bytes(raw)
    request = FrozenInputs(
        p.source_ir_sha256, (FrozenInput(0, "i8", (1, n), path, hashlib.sha256(raw).hexdigest()),), 14_680_064, 2 * n
    )
    tracemalloc.start()
    try:
        derived, blob = specialize(p, request, tmp_path / "offline")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert blob == raw and derived.frozen_inputs["published_blob_bytes"] == n
    assert peak < request.offline_budget_bytes


@pytest.mark.parametrize("change", ["missing", "stride", "shape", "width", "duplicate"])
def test_offline_permutation_requires_complete_noncontradictory_proof(change):
    from merlin.compile.frozen_inputs import _view

    view = dict(permutation_plan().dispatches[0]["views_before"][0])
    if change == "missing":
        del view["permutation"]
    elif change == "stride":
        view["source_strides"] = (5, 1)
    elif change == "shape":
        view["source_shape"] = (5, 3)
    elif change == "width":
        view["element_bytes"] = 4
    else:
        view["permutation"] = (0, 0)
    with pytest.raises(ValueError, match="permutation"):
        _view(view, np.arange(15, dtype=np.int8).reshape(3, 5))


def test_retained_host_image_mutation_after_successful_call_refuses(tmp_path, monkeypatch):
    compiler_required()
    from merlin.compile.frozen_inputs import _evaluate
    from merlin.llvmlower.abi import HostModel

    p = plan()
    kernel = canonical_kernel(p.outlined.module, p.dispatches[0]["symbol"])
    original = HostModel.__call__
    work = tmp_path / "native"

    def mutate(self, args):
        original(self, args)
        path = work / "model_host.so"
        path.write_bytes(path.read_bytes() + b"post-call mutation")

    monkeypatch.setattr(HostModel, "__call__", mutate)
    with pytest.raises(ValueError, match="changed after native evaluation"):
        _evaluate(kernel, [np.arange(6, dtype=np.int8).reshape(3, 2)], work, 1 << 20)


def test_explicit_metadata_analysis_bound_does_not_increase_final_target_budget(tmp_path):
    compiler_required()
    from merlin.compile.outlined_program import build_outlined_program

    p = plan()
    request = binding(tmp_path, p)
    source = tmp_path / "source.mlir"
    source.write_text(SOURCE)
    kwargs = dict(
        roles=StagedRoles((0, 1, 2), 2, 2, 2, 2, (0,), (0,)),
        worker_slots=(0, 0, 0),
        workspace_budget_bytes=128,
        alignment_bytes=64,
        compute_cflags=("-march=rv64gcv",),
        frozen_inputs=request,
    )
    with pytest.raises(ValueError, match="unique-storage"):
        build_outlined_program(source, tmp_path / "old-limit", **kwargs)
    built = build_outlined_program(source, tmp_path / "analysis", analysis_workspace_budget_bytes=4096, **kwargs)
    assert built["analysis_plan"]["pre_specialization_workspace_bytes"] > 128
    assert built["live_storage_bytes"] <= 128
    assert built["analysis_plan"]["final_target_storage_budget_bytes"] == 128
    with pytest.raises(ValueError, match="whole-chain tensor/provider"):
        build_outlined_program(
            source,
            tmp_path / "final-refusal",
            analysis_workspace_budget_bytes=4096,
            **{**kwargs, "workspace_budget_bytes": 64},
        )


def test_rank_zero_offline_is_explicitly_unqualified():
    kernel = parse_mlir_text("module {func.func @forward(%a:tensor<i32>)->tensor<i32>{func.return %a:tensor<i32>}}")
    with pytest.raises(ValueError, match="rank-zero"):
        _admit(kernel)


def test_builder_independently_refuses_mismatched_loaded_image_manifest(tmp_path, monkeypatch):
    compiler_required()
    from merlin.compile import frozen_inputs as module
    from merlin.compile.outlined_program import build_outlined_program

    p = plan()
    request = binding(tmp_path, p)
    source = tmp_path / "source.mlir"
    source.write_text(SOURCE)
    original = module._evaluate

    def inconsistent(*args, **kwargs):
        values, proof = original(*args, **kwargs)
        proof["host_image_sha256"] = "0" * 64
        return values, proof

    monkeypatch.setattr(module, "_evaluate", inconsistent)
    with pytest.raises(ValueError, match="manifest differs from actually loaded image"):
        build_outlined_program(
            source,
            tmp_path / "bad-manifest",
            roles=StagedRoles((0, 1, 2), 2, 2, 2, 2, (0,), (0,)),
            worker_slots=(0, 0, 0),
            workspace_budget_bytes=4096,
            alignment_bytes=64,
            compute_cflags=("-march=rv64gcv",),
            frozen_inputs=request,
        )


@pytest.mark.parametrize("limit", [None, True, 0, -1, 1 << 63])
def test_analysis_bound_never_grants_unfrozen_or_invalid_budget_permission(tmp_path, limit):
    from merlin.compile.outlined_program import build_outlined_program

    p = plan()
    request = binding(tmp_path, p)
    source = tmp_path / "source.mlir"
    source.write_text(SOURCE)
    # None here exercises an explicit analysis bound on a dynamic-input build.
    frozen, bound = (None, 4096) if limit is None else (request, limit)
    with pytest.raises(ValueError, match="analysis workspace bound"):
        build_outlined_program(
            source,
            tmp_path / "bad-analysis",
            roles=StagedRoles((0, 1, 2), 2, 2, 2, 2, (0,), (0,)),
            worker_slots=(0, 0, 0),
            workspace_budget_bytes=4096,
            alignment_bytes=64,
            compute_cflags=("-march=rv64gcv",),
            frozen_inputs=frozen,
            analysis_workspace_budget_bytes=bound,
        )
    assert not (tmp_path / "bad-analysis").exists()


@pytest.mark.parametrize("width", [4, 8])
def test_byte_permutation_retains_ieee_payloads_without_float_arithmetic(width):
    from merlin.compile.frozen_inputs import _view

    word = np.array(
        [0, 1, (1 << (width * 8)) - 1, 1 << (width * 8 - 1), (1 << (width * 8 - 1)) - 1], dtype="<u" + str(width)
    )
    bits = np.tile(word, 3).reshape(3, 5)
    value = bits.view("<f" + str(width))
    view = permutation_plan(width=width).dispatches[0]["views_before"][0]
    assert _view(view, value).tobytes() == bits.T.copy().tobytes()
