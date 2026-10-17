"""Ordinary row normalization, complete source replay and independent execution."""

from __future__ import annotations

import ctypes
import hashlib
import json
import shutil
import subprocess
import time
from math import prod

import numpy as np
import pytest
from xdsl.dialects import arith, builtin, func, math, scf, tensor
from xdsl.dialects.linalg import ops as L
from xdsl.ir import Block, Region
from xdsl.ir.affine import AffineConstantExpr, AffineDimExpr, AffineMap
from xdsl.rewriter import Rewriter

from merlin.llvmlower.automatic_host import prepare_automatic_host
from merlin.transforms.row_normalization import (
    RowNormalizationGroup,
    RowNormalizationRewrite,
    emit_row_normalization,
    fuse_row_normalizations,
    match_row_normalization,
)
from merlin.transforms.row_normalization_check import verify_row_normalizations
from merlin.xdsl_dialects._common import text


def _maps(*maps):
    return builtin.ArrayAttr([builtin.AffineMapAttr(m) for m in maps])


def _generic(inputs, output, body, maps):
    rank = output.type.get_num_dims()
    return L.GenericOp(
        inputs,
        (output,),
        Region(body),
        _maps(*maps),
        builtin.ArrayAttr([L.IteratorTypeAttr(L.IteratorType.PARALLEL)] * rank),
        (output.type,),
    )


def _normalization(value, *, views=False):
    shape = value.type.get_shape()
    element = value.type.get_element_type()
    prefix = builtin.TensorType(element, shape[:-1])
    rank = len(shape)
    ops = []

    def emit(op):
        ops.append(op)
        return op.results[0]

    def init(number):
        scalar = emit(arith.ConstantOp(builtin.FloatAttr(number, element)))
        return emit(tensor.SplatOp(scalar, (), prefix))

    def reduce(input, init, scalar):
        body = Block(arg_types=[element, element])
        op = scalar(*body.args)
        body.add_ops((op, L.YieldOp(op.result)))
        return emit(L.ReduceOp(input, init, builtin.DenseArrayBase.from_list(builtin.i64, [rank - 1]), Region(body)))

    def broadcast(value):
        if not views:
            return value, AffineMap(rank, 0, tuple(AffineDimExpr(i) for i in range(rank - 1)))
        reassociation = builtin.ArrayAttr(
            [builtin.ArrayAttr([builtin.IntegerAttr(i, builtin.i64) for i in range(rank - 1)])]
        )
        collapsed = emit(
            tensor.CollapseShapeOp.create(
                operands=[value],
                result_types=[builtin.TensorType(element, (prod(shape[:-1]),))],
                properties={"reassociation": reassociation},
            )
        )
        expanded = emit(
            tensor.ExpandShapeOp.create(
                operands=[collapsed],
                result_types=[builtin.TensorType(element, (*shape[:-1], 1))],
                properties={
                    "reassociation": builtin.ArrayAttr(
                        [builtin.ArrayAttr([builtin.IntegerAttr(i, builtin.i64) for i in range(rank)])]
                    ),
                    "static_output_shape": builtin.DenseArrayBase.from_list(builtin.i64, (*shape[:-1], 1)),
                },
            )
        )
        return expanded, AffineMap(
            rank,
            0,
            (
                *[AffineDimExpr(i) for i in range(rank - 1)],
                AffineConstantExpr(0),
            ),
        )

    maximum = reduce(value, init(-float("inf")), arith.MaximumfOp)
    maximum, projection = broadcast(maximum)
    empty = emit(tensor.EmptyOp((), value.type))
    body = Block(arg_types=[element, element, element])
    sub = arith.SubfOp(body.args[0], body.args[1])
    body.add_ops((sub, L.YieldOp(sub.result)))
    shifted = emit(
        _generic((value, maximum), empty, body, (AffineMap.identity(rank), projection, AffineMap.identity(rank)))
    )
    empty = emit(tensor.EmptyOp((), value.type))
    body = Block(arg_types=[element, element])
    ex = math.ExpOp(body.args[0])
    body.add_ops((ex, L.YieldOp(ex.result)))
    exponentials = emit(_generic((shifted,), empty, body, (AffineMap.identity(rank),) * 2))
    summed = reduce(exponentials, init(0), arith.AddfOp)
    summed, projection = broadcast(summed)
    empty = emit(tensor.EmptyOp((), value.type))
    body = Block(arg_types=[element, element, element])
    div = arith.DivfOp(body.args[0], body.args[1])
    body.add_ops((div, L.YieldOp(div.result)))
    result = emit(
        _generic((exponentials, summed), empty, body, (AffineMap.identity(rank), projection, AffineMap.identity(rank)))
    )
    return ops, result


def module(shape=(8, 16, 19), dtype=builtin.f32, *, views=False):
    typ = builtin.TensorType(dtype, shape)
    body = Block(arg_types=[typ])
    ops, result = _normalization(body.args[0], views=views)
    body.add_ops((*ops, func.ReturnOp(result)))
    result = builtin.ModuleOp([func.FuncOp("forward", ([typ], [typ]), Region(body))])
    result.verify()
    return result


@pytest.mark.parametrize(
    "shape,dtype,views",
    [((53, 19), builtin.f32, False), ((2, 17, 17), builtin.f32, True), ((7, 3, 9), builtin.f64, False)],
)
def test_source_order_full_domains_and_clean_ssa_users(shape, dtype, views):
    original = module(shape, dtype, views=views)
    result, report = fuse_row_normalizations(original, numerical_policy="unchanged-source-operations")
    assert report["rewrites"] == 1
    loop = next(op for op in result.walk() if type(op) is scf.ForOp)
    assert loop.ub.owner.value.value.data == prod(shape[:-1])
    assert [op.name for op in loop.walk() if isinstance(op, L.ReduceOp)] == ["linalg.reduce", "linalg.reduce"]
    members = set(result.walk())
    assert all(use.operation in members for op in result.walk() for v in op.results for use in v.uses)
    assert report["checks"][-1]["name"] == "row_fiber_conformance"
    assert report["islands"][0]["intermediate_full_row_elements_after"] == shape[-1]


def _with_record(original):
    current = original.clone()
    group = next(g for op in current.walk() if (g := match_row_normalization(op)))
    mapper = {}
    snapshot = tuple(op.clone(value_mapper=mapper) for op in group.operations)
    emitted, result = emit_row_normalization(group)
    record = RowNormalizationRewrite(RowNormalizationGroup(group.source, snapshot, group.shape), emitted, result)
    Rewriter.replace_op(group.operations[-1], emitted, new_results=[result])
    for op in reversed(group.operations[:-1]):
        Rewriter.erase_op(op)
    return current, record


@pytest.mark.parametrize("fault", ["rows", "tail", "offset", "reset", "source", "sum_order", "extra_effect", "outer"])
def test_actual_replay_mutations_refuse(fault):
    original = module()
    modified, record = _with_record(original)
    loop = next(op for op in record.emitted if isinstance(op, scf.ForOp))
    if fault == "rows":
        loop.ub.owner.properties["value"] = builtin.IntegerAttr(2, builtin.IndexType())
    elif fault == "tail":
        sliced = next(op for op in loop.body.block.ops if isinstance(op, tensor.ExtractSliceOp))
        sliced.properties["static_sizes"] = builtin.DenseArrayBase.from_list(builtin.i64, [1, 1, 18])
    elif fault == "offset":
        sliced = next(op for op in loop.body.block.ops if isinstance(op, tensor.ExtractSliceOp))
        sliced.properties["static_offsets"] = builtin.DenseArrayBase.from_list(
            builtin.i64, [builtin.DYNAMIC_INDEX, builtin.DYNAMIC_INDEX, 1]
        )
    elif fault == "reset":
        summed = next(
            op
            for op in loop.walk()
            if isinstance(op, L.ReduceOp) and type(op.regions[0].block.first_op) is arith.AddfOp
        )
        summed.operands = (summed.operands[0], summed.operands[0])
    elif fault == "source":
        pointwise = next(op for op in loop.walk() if isinstance(op, L.GenericOp))
        pointwise.operands = (pointwise.operands[0], pointwise.operands[0], pointwise.operands[-1])
    elif fault == "sum_order":
        added = next(op for op in loop.walk() if type(op) is arith.AddfOp)
        added.operands = tuple(reversed(added.operands))
    elif fault == "extra_effect":
        loop.body.block.insert_op_before(arith.ConstantOp.from_int_and_width(7, 32), loop.body.block.last_op)
    else:
        modified.body.block.add_op(func.FuncOp("unexpected", ([], [])))
    with pytest.raises(ValueError, match="conformance"):
        verify_row_normalizations(original, modified, (record,))


def test_observer_effects_unknown_algorithm_and_code_budget_retain_source():
    for fault in ("observer", "effect", "algorithm", "budget"):
        original = module()
        function = original.body.block.first_op
        if fault == "observer":
            exps = next(
                op
                for op in original.walk()
                if isinstance(op, L.GenericOp) and type(op.regions[0].block.first_op) is math.ExpOp
            )
            ret = function.body.block.last_op
            ret.operands = (exps.results[0],)
        elif fault == "effect":
            reduction = next(op for op in function.body.block.ops if isinstance(op, L.ReduceOp))
            function.body.block.insert_op_after(func.CallOp("barrier", [], []), reduction)
            original.body.block.add_op(func.FuncOp("barrier", ([], []), Region()))
        elif fault == "algorithm":
            exp = next(op for op in original.walk() if type(op) is math.ExpOp)
            Rewriter.replace_op(exp, math.AbsFOp(exp.operands[0]))
        result, report = fuse_row_normalizations(
            original, numerical_policy="source", max_added_operations=0 if fault == "budget" else 4096
        )
        assert report["rewrites"] == 0 and result.is_structurally_equivalent(original)


def test_ordinary_source_selection_and_explicit_ablation(tmp_path, monkeypatch):
    from merlin.llvmlower.automatic_rows import prepare_automatic_rows

    source = tmp_path / "model.mlir"
    source.write_text(text(module(), generic=True))
    selected, report = prepare_automatic_rows(source.read_text(), serial=True)
    assert selected != source.read_text()
    assert report["rewrites"] == 1
    monkeypatch.setenv("MERLIN_AUTOMATIC_HOST_ROWS", "0")
    retained, report = prepare_automatic_rows(source.read_text(), serial=True)
    assert retained == source.read_text() and report["rewrites"] == 0
    monkeypatch.setenv("MERLIN_AUTOMATIC_HOST_ROWS", "1")
    retained, report = prepare_automatic_rows(source.read_text(), serial=False)
    assert retained == source.read_text() and report["rewrites"] == 0
    retained, report = prepare_automatic_rows(text(module((2, 113)), generic=True), serial=True)
    assert report["rewrites"] == 0 and "no closed normalization" in report["diagnostic"]
    assert report["source_candidates"] == {"row_state_not_smaller": 1}


def test_equal_flat_offsets_with_out_of_bounds_coordinates_are_not_row_fibers():
    original = module((128, 19))
    generic = next(op for op in original.walk() if isinstance(op, L.GenericOp))
    mapping = AffineMap(2, 0, (AffineDimExpr(0) + 1, AffineDimExpr(1) - 19))
    maps = list(generic.indexing_maps)
    maps[0] = builtin.AffineMapAttr(mapping)
    generic.properties["indexing_maps"] = builtin.ArrayAttr(maps)
    result, report = fuse_row_normalizations(original, numerical_policy="source")
    assert report["rewrites"] == 0 and result.is_structurally_equivalent(original)


_ALLOCATION_OBSERVER = r"""
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
void *__real_malloc(size_t);
void __real_free(void*);
static void *pointers[4096];
static size_t sizes[4096],live,peak,total,calls;
void *__wrap_malloc(size_t n) {
 void *p=__real_malloc(n);if(!p) return p;
 for(int i=0;i<4096;i++) if(!pointers[i]) {
  pointers[i]=p;sizes[i]=n;live+=n;total+=n;calls++;if(live>peak)peak=live;return p;
 }
 abort();
}
void __wrap_free(void *p) {
 for(int i=0;i<4096;i++) if(pointers[i]==p && p) {
  live-=sizes[i];pointers[i]=0;break;
 }
 __real_free(p);
}
uint64_t peak_requested(void){return peak;}
uint64_t total_requested(void){return total;}
uint64_t current_requested(void){return live;}
uint64_t allocation_calls(void){return calls;}
void reset_observer(void){if(live)abort();peak=total=calls=0;}
"""


def _compile_ordinary(tmp_path, original, enabled, monkeypatch, **lowering_options):
    from merlin.llvmlower import toolchain
    from merlin.llvmlower.abi import HostModel
    from merlin.llvmlower.codegen import mlir_runtime_c
    from merlin.llvmlower.lower import lower_model_file

    if not toolchain.available():
        pytest.skip("native MLIR toolchain unavailable")
    started = time.perf_counter()
    source = tmp_path / "source.mlir"
    source.write_text(text(original, generic=True))
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "schema": "merlin.quantization_policy.v1",
                "selection": "captured_scope",
                "prepared_mlir_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
        )
    )
    monkeypatch.setenv("MERLIN_AUTOMATIC_HOST_NUMERICS", "1")
    monkeypatch.setenv("MERLIN_AUTOMATIC_HOST_ROWS", "1" if enabled else "0")
    prepared = prepare_automatic_host(source, tmp_path / "prepare", policy_path=policy)
    lowered = lower_model_file(prepared, tmp_path / "lower", targets=(), textual=True, **lowering_options)
    report = lowered.stats["automatic_host_rows"]
    identity = hashlib.sha256(lowered.ll_path.read_bytes()).hexdigest()[:16]
    observer = tmp_path / "observe.c"
    observer.write_text(_ALLOCATION_OBSERVER)
    obj = tmp_path / f"program_{identity}.o"
    library = tmp_path / f"program_{identity}.so"
    subprocess.run(
        [str(toolchain.clang()), "-O2", "-ffp-contract=off", "-fPIC", "-c", str(lowered.ll_path), "-o", str(obj)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "cc",
            "-O2",
            "-shared",
            "-fPIC",
            str(obj),
            str(observer),
            str(mlir_runtime_c()),
            "-Wl,--wrap=malloc,--wrap=free",
            "-lm",
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
    )
    host = HostModel.load(str(library))
    observer_lib = ctypes.CDLL(str(library))
    for symbol in ("peak_requested", "total_requested", "current_requested", "allocation_calls"):
        getattr(observer_lib, symbol).restype = ctypes.c_uint64
    report["experiment"] = {
        "compile_seconds": time.perf_counter() - started,
        "compiler_peak_rss": "not_measured",
        "object_bytes": obj.stat().st_size,
        "object_sha256": hashlib.sha256(obj.read_bytes()).hexdigest(),
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "text_bytes": None,
        "accelerator_work": "none; native host-only program",
        "transfer_bytes": "not_measured",
    }
    if size := shutil.which("size"):
        sections = subprocess.run([size, "-A", str(obj)], check=True, capture_output=True, text=True).stdout
        report["experiment"]["text_bytes"] = sum(
            int(row.split()[1]) for row in sections.splitlines() if row.split() and row.split()[0].startswith(".text")
        )
    return host, observer_lib, report


def _reference_normalization(values):
    import math as pymath

    dtype = values.dtype.type
    libm = ctypes.CDLL("libm.so.6")
    exp = libm.expf if dtype is np.float32 else libm.exp
    scalar = ctypes.c_float if dtype is np.float32 else ctypes.c_double
    exp.argtypes = [scalar]
    exp.restype = scalar
    result = np.empty_like(values)
    for index in np.ndindex(values.shape[:-1]):
        row = values[index]
        maximum = dtype(-pymath.inf)
        for value in row:
            maximum = dtype(np.maximum(value, maximum))
        with np.errstate(invalid="ignore", over="ignore"):
            exponentials = [dtype(exp(float(dtype(value - maximum)))) for value in row]
            total = dtype(0)
            for value in exponentials:
                total = dtype(value + total)
            result[index] = [dtype(value / total) for value in exponentials]
    return result


@pytest.mark.parametrize(
    "shape,dtype,views",
    [((128, 19), builtin.f32, False), ((4, 128, 113), builtin.f32, True), ((8, 16, 23), builtin.f64, False)],
)
def test_native_complete_outputs_and_realized_allocation(tmp_path, monkeypatch, record_property, shape, dtype, views):
    original = module(shape, dtype, views=views)
    arms = []
    for name, enabled in (("baseline", False), ("optimized", True)):
        directory = tmp_path / name
        directory.mkdir()
        arms.append(_compile_ordinary(directory, original, enabled, monkeypatch))
    values = np.random.default_rng(11).normal(0, 4, shape).astype("float32" if dtype == builtin.f32 else "float64")
    values.reshape(-1, shape[-1])[1] = np.linspace(-120, 0, shape[-1])
    # Existing source semantics, including all-masked and nonfinite rows.
    special = values.copy()
    special.reshape(-1, shape[-1])[0] = -np.inf
    special.reshape(-1, shape[-1])[1, 1] = np.nan
    special.reshape(-1, shape[-1])[2, 2] = np.inf
    metrics = []
    for inputs in (values, special):
        results = []
        for host, observer, report in arms:
            output = np.empty_like(inputs)
            observer.reset_observer()
            original_inputs = inputs.copy()
            host([(inputs.ctypes.data, inputs.shape), (output.ctypes.data, output.shape)])
            np.testing.assert_array_equal(
                inputs.view("u4" if dtype == builtin.f32 else "u8"),
                original_inputs.view("u4" if dtype == builtin.f32 else "u8"),
            )
            assert observer.current_requested() == 0
            results.append(output.copy())
            if inputs is values:
                metrics.append(
                    dict(
                        peak=observer.peak_requested(),
                        total=observer.total_requested(),
                        calls=observer.allocation_calls(),
                    )
                )
        word = "u4" if dtype == builtin.f32 else "u8"
        np.testing.assert_array_equal(results[0].view(word), results[1].view(word))
        reference = _reference_normalization(inputs)
        np.testing.assert_array_equal(results[1], reference)
    assert arms[0][2]["library_sources_sha256"] == arms[1][2]["library_sources_sha256"]
    assert arms[1][2]["rewrites"] == 1
    (tmp_path / "allocation-pair.json").write_text(json.dumps(metrics, indent=2))
    record_property(
        "allocation_pair",
        json.dumps({"shape": shape, "dtype": str(dtype), "baseline": metrics[0], "optimized": metrics[1]}),
    )
    assert metrics[1]["peak"] < metrics[0]["peak"]


def _attention_module(
    rows=128,
    columns=19,
    head=7,
    width=5,
    *,
    masked=True,
    scale=False,
    broadcast_mask=False,
    generic_contractions=False,
    identity_readout=False,
):
    types = [
        builtin.TensorType(builtin.f32, shape)
        for shape in (
            (rows, head),
            (head, columns),
            (columns, width),
            (1 if broadcast_mask else rows, columns),
            (rows, width),
        )
    ]
    position_type = builtin.TensorType(builtin.i32, (1,))
    body = Block(arg_types=[*types, position_type])
    q, key, value, mask, state, position = body.args

    def emit(op):
        body.add_op(op)
        return op.results[0]

    def matmul(a, b, shape):
        typ = builtin.TensorType(builtin.f32, shape)
        zero = emit(arith.ConstantOp(builtin.FloatAttr(0.0, builtin.f32)))
        initial = emit(tensor.SplatOp(zero, (), typ))
        if not generic_contractions:
            return emit(L.MatmulOp((a, b), (initial,), (typ,)))
        block = Block(arg_types=[builtin.f32] * 3)
        product = arith.MulfOp(*block.args[:2])
        added = arith.AddfOp(block.args[2], product.result)
        block.add_ops((product, added, L.YieldOp(added.result)))
        maps = _maps(
            AffineMap(3, 0, (AffineDimExpr(0), AffineDimExpr(2))),
            AffineMap(3, 0, (AffineDimExpr(2), AffineDimExpr(1))),
            AffineMap(3, 0, (AffineDimExpr(0), AffineDimExpr(1))),
        )
        return emit(
            L.GenericOp(
                (a, b),
                (initial,),
                Region(block),
                maps,
                builtin.ArrayAttr(
                    [
                        L.IteratorTypeAttr(L.IteratorType.PARALLEL),
                        L.IteratorTypeAttr(L.IteratorType.PARALLEL),
                        L.IteratorTypeAttr(L.IteratorType.REDUCTION),
                    ]
                ),
                (typ,),
            )
        )

    def add(a, b):
        typ = a.type
        rank = typ.get_num_dims()
        empty = emit(tensor.EmptyOp((), typ))
        block = Block(arg_types=[typ.get_element_type()] * 3)
        added = (
            arith.AddfOp(*block.args[:2]) if typ.get_element_type() == builtin.f32 else arith.AddiOp(*block.args[:2])
        )
        block.add_ops((added, L.YieldOp(added.result)))
        bmap = AffineMap(
            rank,
            0,
            tuple(AffineConstantExpr(0) if n == 1 else AffineDimExpr(i) for i, n in enumerate(b.type.get_shape())),
        )
        return emit(_generic((a, b), empty, block, (AffineMap.identity(rank), bmap, AffineMap.identity(rank))))

    scores = matmul(q, key, (rows, columns))
    if scale:
        # Captured producers may place a scalar splat between their empty
        # destination and the pointwise operation. It is part of the island.
        empty = emit(tensor.EmptyOp((), scores.type))
        scalar = emit(arith.ConstantOp(builtin.FloatAttr(0.5, builtin.f32)))
        scalar = emit(tensor.SplatOp(scalar, (), builtin.TensorType(builtin.f32, ())))
        local = Block(arg_types=[builtin.f32] * 3)
        product = arith.MulfOp(*local.args[:2])
        local.add_ops((product, L.YieldOp(product.result)))
        scores = emit(
            _generic(
                (scores, scalar), empty, local, (AffineMap.identity(2), AffineMap(2, 0, ()), AffineMap.identity(2))
            )
        )
    admitted_scores = add(scores, mask) if masked else scores
    ops, normalized = _normalization(admitted_scores)
    body.add_ops(ops)
    if identity_readout:
        empty = emit(tensor.EmptyOp((), normalized.type))
        local = Block(arg_types=[builtin.f32] * 2)
        local.add_op(L.YieldOp(local.args[0]))
        normalized = emit(_generic((normalized,), empty, local, (AffineMap.identity(2),) * 2))
    output = matmul(normalized, value, (rows, width))
    carried = add(output, state)
    one = emit(arith.ConstantOp.from_int_and_width(1, builtin.i32))
    ones = emit(tensor.SplatOp(one, (), position_type))
    next_position = add(position, ones)
    body.add_op(func.ReturnOp(output, carried, next_position))
    module = builtin.ModuleOp(
        [func.FuncOp("forward", ([*types, position_type], [output.type, carried.type, position_type]), Region(body))]
    )
    module.verify()
    return module


@pytest.mark.parametrize(
    "masked,scale,broadcast,generic,identity",
    [
        (True, False, False, False, False),
        (False, False, False, False, False),
        (True, True, False, False, False),
        (True, True, True, False, False),
        (True, True, True, True, False),
        (True, True, True, True, True),
    ],
)
def test_ordinary_complete_attention_graph_and_carried_state(
    tmp_path, monkeypatch, record_property, masked, scale, broadcast, generic, identity
):
    original = _attention_module(
        masked=masked, scale=scale, broadcast_mask=broadcast, generic_contractions=generic, identity_readout=identity
    )
    arms = []
    for label, enabled in (("baseline", False), ("optimized", True)):
        directory = tmp_path / label
        directory.mkdir()
        arms.append(_compile_ordinary(directory, original, enabled, monkeypatch))
    assert arms[1][2]["rewrites"] == 1
    assert arms[0][2]["library_sources_sha256"] == arms[1][2]["library_sources_sha256"]
    rng = np.random.default_rng(17)
    q, key, value = [rng.normal(0, 0.5, shape).astype("f4") for shape in ((128, 7), (7, 19), (19, 5))]
    mask = np.where(np.arange(19)[None, :] <= np.arange(128)[:, None] % 19, np.float32(0), np.float32(-np.inf))
    if broadcast:
        mask = mask[7:8].copy()

    def ordered_product(a, b):
        result = np.zeros((a.shape[0], b.shape[1]), dtype="f4")
        for k in range(a.shape[1]):
            result = np.add(np.multiply(a[:, k : k + 1], b[k : k + 1, :], dtype="f4"), result, dtype="f4")
        return result

    scores = ordered_product(q, key)
    if scale:
        scores = np.multiply(scores, np.float32(0.5), dtype="f4")
    if masked:
        scores = np.add(scores, mask, dtype="f4")
    expected = ordered_product(_reference_normalization(scores), value)
    states = [np.zeros((128, 5), dtype="f4") for _ in arms]
    positions = [np.array([127], dtype="i4") for _ in arms]
    reference_state = states[0].copy()
    measurements = []
    for step in range(3):
        reference_state = np.add(expected, reference_state, dtype="f4")
        for arm, (host, observer, report) in enumerate(arms):
            output, carried, position = (
                np.empty((128, 5), dtype="f4"),
                np.empty((128, 5), dtype="f4"),
                np.empty((1,), dtype="i4"),
            )
            inputs = [q, key, value, mask, states[arm], positions[arm]]
            unchanged = [v.copy().view("u4") for v in inputs]
            arrays = [*inputs, output, carried, position]
            observer.reset_observer()
            started = time.perf_counter()
            host([(v.ctypes.data, v.shape) for v in arrays])
            elapsed = time.perf_counter() - started
            assert observer.current_requested() == 0
            for before, after in zip(unchanged, inputs, strict=True):
                np.testing.assert_array_equal(before, after.view("u4"))
            np.testing.assert_array_equal(output.view("u4"), expected.view("u4"))
            np.testing.assert_array_equal(carried.view("u4"), reference_state.view("u4"))
            np.testing.assert_array_equal(position, np.array([128 + step], dtype="i4"))
            states[arm], positions[arm] = carried.copy(), position.copy()
            measurements.append(
                dict(
                    arm=arm,
                    step=step,
                    host_wall_seconds=elapsed,
                    peak_requested_heap=observer.peak_requested(),
                    total_requested_heap=observer.total_requested(),
                    allocation_calls=observer.allocation_calls(),
                )
            )
    record_property(
        "attention_pair", json.dumps(dict(compilation=[r[2]["experiment"] for r in arms], execution=measurements))
    )


@pytest.mark.parametrize("fault", ["mask_source", "mask_tail", "prefix_math", "shared_prefix", "unknown_prefix"])
def test_pointwise_prefix_frontier_and_source_body_are_checked(fault):
    original = _attention_module(width=19, masked=True, scale=True)
    if fault in {"shared_prefix", "unknown_prefix"}:
        source = next(g.source for o in original.walk() if (g := match_row_normalization(o)))
        if fault == "shared_prefix":
            # An external observation prevents absorbing the private prefix.
            ret = source.owner.parent.last_op
            ret.operands = (source, *ret.operands[1:])
        else:
            source.owner.attributes["unknown.effect"] = builtin.UnitAttr()
        result, report = fuse_row_normalizations(original, numerical_policy="source")
        assert report["rewrites"] == 0 and result.is_structurally_equivalent(original)
        return
    modified, record = _with_record(original)
    loop = next(o for o in record.emitted if isinstance(o, scf.ForOp))
    extracts = [o for o in loop.body.block.ops if isinstance(o, tensor.ExtractSliceOp)]
    assert len(extracts) == 2
    if fault == "mask_source":
        extracts[-1].operands = (extracts[0].operands[0], *extracts[-1].operands[1:])
    elif fault == "mask_tail":
        extracts[-1].properties["static_sizes"] = builtin.DenseArrayBase.from_list(builtin.i64, [1, 18])
    else:
        product = next(o for o in loop.walk() if isinstance(o, arith.MulfOp))
        Rewriter.replace_op(product, arith.AddfOp(*product.operands))
    with pytest.raises(ValueError, match="conformance"):
        verify_row_normalizations(original, modified, (record,))


def _broadcast_module(shape, mask_shape, dtype):
    typ, mask_type = (builtin.TensorType(dtype, s) for s in (shape, mask_shape))
    rank = len(shape)
    body = Block(arg_types=[typ, mask_type])
    empty = tensor.EmptyOp((), typ)
    local = Block(arg_types=[dtype] * 3)
    add = arith.AddfOp(*local.args[:2])
    local.add_ops((add, L.YieldOp(add.result)))
    mapping = AffineMap(
        rank,
        0,
        tuple(
            AffineConstantExpr(0) if n == 1 else AffineDimExpr(i + rank - len(mask_shape))
            for i, n in enumerate(mask_shape)
        ),
    )
    prefix = _generic(body.args, empty.tensor, local, (AffineMap.identity(rank), mapping, AffineMap.identity(rank)))
    ops, result = _normalization(prefix.results[0])
    body.add_ops((empty, prefix, *ops, func.ReturnOp(result)))
    original = builtin.ModuleOp([func.FuncOp("forward", ([typ, mask_type], [typ]), Region(body))])
    original.verify()
    return original


@pytest.mark.parametrize(
    "shape,mask_shape,dtype",
    [
        ((2, 3, 7, 19), (1, 3, 1, 19), builtin.f32),
        ((2, 3, 7, 11), (2, 1, 7, 11), builtin.f64),
        ((53, 19), (1,), builtin.f32),
    ],
)
def test_native_projected_broadcasts_preserve_rows_and_complete_columns(
    tmp_path, monkeypatch, shape, mask_shape, dtype
):
    original = _broadcast_module(shape, mask_shape, dtype)
    rng = np.random.default_rng(47)
    native_type = "f4" if dtype == builtin.f32 else "f8"
    scores = rng.normal(0, 3, shape).astype(native_type)
    mask = rng.normal(0, 1, mask_shape).astype(native_type)
    if mask_shape[-1] == shape[-1]:
        mask[..., 3::5] = -np.inf
    expected = _reference_normalization(np.add(scores, mask, dtype=native_type))
    results = []
    for name, enabled in (("baseline", False), ("optimized", True)):
        directory = tmp_path / name
        directory.mkdir()
        host, observer, report = _compile_ordinary(directory, original, enabled, monkeypatch)
        output = np.empty_like(scores)
        unchanged = [scores.copy(), mask.copy()]
        observer.reset_observer()
        arrays = [scores, mask, output]
        host([(v.ctypes.data, v.shape) for v in arrays])
        assert observer.current_requested() == 0
        word = "u4" if dtype == builtin.f32 else "u8"
        for old, actual in zip(unchanged, arrays, strict=False):
            np.testing.assert_array_equal(old.view(word), actual.view(word))
        np.testing.assert_array_equal(output, expected)
        assert report["rewrites"] == int(enabled)
        results.append(output.copy().view(word))
    np.testing.assert_array_equal(*results)


@pytest.mark.parametrize("fault", ["row_projection", "broadcast_offset", "source_columns"])
def test_actual_broadcast_projection_mutations_refuse(fault):
    original = _broadcast_module((2, 3, 7, 19), (1, 3, 1, 19), builtin.f32)
    modified, record = _with_record(original)
    loop = record.emitted[-1]
    extracts = [o for o in loop.body.block.ops if isinstance(o, tensor.ExtractSliceOp)]
    mask = extracts[-1]
    if fault == "row_projection":
        mask.operands = (mask.operands[0], extracts[0].operands[1])
    elif fault == "broadcast_offset":
        mask.properties["static_offsets"] = builtin.DenseArrayBase.from_list(
            builtin.i64, [1, builtin.DYNAMIC_INDEX, 0, 0]
        )
    else:
        mask.properties["static_sizes"] = builtin.DenseArrayBase.from_list(builtin.i64, [1, 1, 1, 18])
    with pytest.raises(ValueError, match="conformance"):
        verify_row_normalizations(original, modified, (record,))


def test_selected_native_layout_and_64_bit_index_use_the_same_row_path(tmp_path, monkeypatch):
    from merlin.llvmlower import toolchain

    if not toolchain.available():
        pytest.skip("native MLIR toolchain unavailable")
    query = subprocess.run(
        [str(toolchain.clang()), "-S", "-emit-llvm", "-x", "c", "-", "-o", "-"],
        input="void layout_probe(void) {}",
        text=True,
        check=True,
        capture_output=True,
    ).stdout
    line = next(row for row in query.splitlines() if row.startswith("target datalayout = "))
    layout = json.loads(line.partition(" = ")[2])
    original = module((128, 19), builtin.f64)
    values = np.random.default_rng(59).normal(0, 2, (128, 19)).astype("f8")
    expected = _reference_normalization(values)
    outputs = []
    for name, enabled in (("baseline", False), ("optimized", True)):
        directory = tmp_path / name
        directory.mkdir()
        host, observer, report = _compile_ordinary(
            directory, original, enabled, monkeypatch, data_layout=layout, index_bits=64
        )
        output = np.empty_like(values)
        observer.reset_observer()
        host([(v.ctypes.data, v.shape) for v in (values, output)])
        assert observer.current_requested() == 0 and report["rewrites"] == int(enabled)
        assert report["hardware_identity"] == "llvm-datalayout:" + hashlib.sha256(layout.encode()).hexdigest()
        if enabled:
            assert report["realization"]["context"]["hardware"] == report["hardware_identity"]
            assert "index_bits=64" in report["realization"]["context"]["resources"]
        np.testing.assert_array_equal(output.view("u8"), expected.view("u8"))
        outputs.append(output.copy().view("u8"))
    np.testing.assert_array_equal(*outputs)


def test_unsupported_index_width_retains_the_ordinary_source():
    from merlin.llvmlower.automatic_rows import prepare_automatic_rows

    source = text(module(), generic=True)
    result, report = prepare_automatic_rows(source, serial=True, index_bits=32)
    assert result == source and report["rewrites"] == 0 and report["index_bits"] == 32
    with pytest.raises(ValueError, match="64-bit index"):
        fuse_row_normalizations(module(), numerical_policy="source", index_bits=32)


def test_identity_readout_is_checked_instead_of_assumed():
    original = _attention_module(
        masked=True, scale=True, broadcast_mask=True, generic_contractions=True, identity_readout=True
    )
    modified, record = _with_record(original)
    # Last replayed generic is the source identity readout. Reading its
    # undefined destination instead of its input changes the actual program.
    copied = tuple(record.emitted[-1].body.block.ops)[-3]
    assert isinstance(copied, L.GenericOp)
    yielded = copied.regions[0].block.last_op
    yielded.operands = (copied.regions[0].block.args[-1],)
    with pytest.raises(ValueError, match="conformance"):
        verify_row_normalizations(original, modified, (record,))
