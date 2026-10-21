"""Explicit packed-LHS alternatives; independent complete values, not costs."""

import numpy as np
import pytest
from test_dense_row_batching import plan, source
from xdsl.dialects import arith, builtin, func, tensor
from xdsl.dialects.linalg import ops as linalg
from xdsl.ir import Block, Region
from xdsl.ir.affine import AffineMap
from xdsl.utils.exceptions import VerifyException

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.prepared_model_transform import apply_prepared_model_transform
from merlin.runtime.linalg_numpy import evaluate
from merlin.transforms.certificates import ir_text
from merlin.transforms.dense_row_batching_check import geometry, verify_dense_row_batching


def computed_rows(*, rank3=False, **kwargs):
    module = source(rank3=rank3, **kwargs)
    for dot in tuple(module.walk()):
        if type(dot) is not linalg.GenericOp:
            continue
        try:
            geometry(dot, index_bits=64)
        except ValueError:
            continue
        value = dot.inputs[0]
        empty = tensor.EmptyOp((), value.type)
        body = Block(arg_types=[value.type.element_type] * 2)
        body.add_op(linalg.YieldOp(body.args[0]))
        copy = linalg.GenericOp(
            inputs=(value,),
            outputs=(empty.tensor,),
            body=Region(body),
            indexing_maps=builtin.ArrayAttr([builtin.AffineMapAttr(AffineMap.identity(2))] * 2),
            iterator_types=builtin.ArrayAttr([linalg.IteratorTypeAttr(linalg.IteratorType.PARALLEL)] * 2),
            result_types=(value.type,),
        )
        dot.parent.insert_ops_before((empty, copy), dot.outputs[0].owner)
        dot.operands = (copy.results[0], *dot.operands[1:])
    module.verify()
    return module


def packed(module, cap=4096):
    return plan(module, allow_lhs_materialization=True, max_materialized_bytes=cap)


@pytest.mark.parametrize(
    "parts,k,n,dtype,accum",
    [
        ((1, 1, 1), 5, 7, "i8", "i32"),
        ((2, 1), 3, 5, "i16", "i32"),
        ((1, 2, 1), 2, 3, "i32", "i32"),
    ],
)
def test_explicit_materialization_consumes_actual_rows_and_preserves_complete_outputs(parts, k, n, dtype, accum):
    module = computed_rows(parts=parts, k=k, n=n, dtype=dtype, accum=accum, update_state=True)
    original = ir_text(module)
    assert plan(module) is None
    selected = packed(module)
    assert selected is not None and ir_text(module) == original
    selected.revalidate()
    assert len([op for op in selected.candidate.walk() if type(op) is tensor.ConcatOp]) == 1
    record = selected.record()["batches"][0]
    assert record["lhs_materialization_bytes"] == sum(parts) * k * int(dtype[1:]) // 8
    lhs = (np.arange(sum(parts) * k) % 13 - 6).astype(dtype.replace("i", "int")).reshape(sum(parts), k)
    rhs = (np.arange(k * n) % 11 - 5).astype(lhs.dtype).reshape(k, n)
    state = np.full_like(lhs, 3)
    for _ in range(3):
        old = evaluate(module, [lhs, rhs, state.copy()])
        actual = evaluate(selected.candidate, [lhs, rhs, state.copy()])
        assert len(old) == len(actual) == 2 * len(parts) + 3
        for a, b in zip(actual, old, strict=True):
            assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
        state = actual[-1].copy()


def dots(module):
    result = []
    for op in module.walk():
        if type(op) is linalg.GenericOp:
            try:
                geometry(op, index_bits=64)
                result.append(op)
            except ValueError:
                pass
    return result


def combine(*modules):
    functions = [m.body.block.first_op for m in modules]
    inputs = [v.type for f in functions for v in f.body.block.args]
    outputs = [v.type for f in functions for v in f.body.block.last_op.arguments]
    block = Block(arg_types=inputs)
    returned, cursor = [], 0
    for f in functions:
        mapping = dict(zip(f.body.block.args, block.args[cursor : cursor + len(f.body.block.args)], strict=True))
        for op in f.body.block.ops:
            if type(op) is func.ReturnOp:
                returned.extend(mapping[v] for v in op.arguments)
            else:
                block.add_op(op.clone(value_mapper=mapping))
        cursor += len(f.body.block.args)
    block.add_op(func.ReturnOp(*returned))
    module = builtin.ModuleOp([func.FuncOp("forward", (inputs, outputs), Region(block))])
    module.verify()
    return module


def test_two_materialized_groups_preserve_an_existing_zero_copy_batch_and_complete_results():
    first = plan(source(k=3, n=4, rank3=False)).candidate
    module = combine(first, computed_rows(k=5, n=7), computed_rows(k=9, n=7))
    selected = packed(module, cap=42)
    assert selected is not None
    assert [r["lhs_materialization_bytes"] for r in selected.record()["batches"]] == [15, 27]
    assert len(dots(module)) == 7 and len(dots(selected.candidate)) == 3
    assert packed(selected.candidate, cap=42) is None
    arguments = []
    for value in module.body.block.first_op.body.block.args:
        shape = value.type.get_shape()
        arguments.append((np.arange(np.prod(shape)) % 9 - 4).astype(np.int8).reshape(shape))
    expected = evaluate(module, arguments)
    actual = evaluate(selected.candidate, arguments)
    assert len(actual) == len(expected) == 27
    assert all(
        a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
        for a, b in zip(actual, expected, strict=True)
    )


def test_repeated_identical_lhs_values_are_materialized_as_real_duplicate_rows():
    module = computed_rows(k=3, n=4)
    members = dots(module)
    members[2].operands = (members[0].inputs[0], *members[2].operands[1:])
    selected = packed(module, cap=9)
    assert selected is not None
    concat = next(op for op in selected.candidate.walk() if type(op) is tensor.ConcatOp)
    assert concat.operands[0] is concat.operands[2]
    assert selected.record()["batches"][0]["lhs_materialization_bytes"] == 9


@pytest.mark.parametrize(
    "kwargs",
    [
        {"allow_lhs_materialization": True},
        {"max_materialized_bytes": 30},
        {"allow_lhs_materialization": 1, "max_materialized_bytes": 30},
        {"allow_lhs_materialization": True, "max_materialized_bytes": -1},
        {"allow_lhs_materialization": True, "max_materialized_bytes": True},
    ],
)
def test_materialization_requires_both_original_optin_and_exact_byte_budget(kwargs):
    with pytest.raises(ValueError, match="materialization"):
        plan(computed_rows(), **kwargs)


@pytest.mark.parametrize(
    "damage", ["early_use", "unknown_effect", "numeric_hint", "flags", "nonzero_init", "budget", "index_range"]
)
def test_unproved_materialization_preserves_original(damage):
    module = computed_rows(uniform_epilogue=("addi", 4) if damage == "early_use" else None)
    members = dots(module)
    kwargs = {}
    if damage == "unknown_effect":
        members[1].parent.insert_op_before(func.CallOp("observe", [], []), members[1])
        module.body.block.add_op(func.FuncOp.external("observe", [], []))
    elif damage == "numeric_hint":
        members[1].attributes["prov.numeric_policy"] = builtin.StringAttr("unproved")
    elif damage == "flags":
        mul = next(op for op in members[1].walk() if type(op) is arith.MuliOp)
        mul.properties["overflowFlags"] = arith.IntegerOverflowAttr((arith.IntegerOverflowFlag.NSW,))
    elif damage == "nonzero_init":
        zero = next(op for op in module.walk() if type(op) is arith.ConstantOp)
        zero.properties["value"] = builtin.IntegerAttr(1, builtin.i32)
    elif damage == "budget":
        kwargs["max_materialized_bytes"] = 20
    elif damage == "index_range":
        kwargs["index_bits"] = 5
    original = ir_text(module)
    assert (
        plan(
            module,
            allow_lhs_materialization=True,
            max_materialized_bytes=kwargs.pop("max_materialized_bytes", 4096),
            **kwargs,
        )
        is None
    )
    assert ir_text(module) == original


@pytest.mark.parametrize(
    "damage",
    [
        "concat_order",
        "concat_duplicate",
        "concat_dim",
        "concat_hint",
        "wrong_rhs",
        "wrong_body",
        "flags",
        "slice_order",
        "early_concat",
        "late_concat",
        "changed_return",
        "budget",
        "default_budget",
    ],
)
def test_independent_checker_refuses_actual_materialized_ir_mutations(damage):
    module = computed_rows(k=3, n=3, update_state=True)
    selected = packed(module, cap=9)
    actual = selected.candidate.clone()
    combined = dots(actual)[0]
    concat = combined.inputs[0].owner
    block = combined.parent
    kwargs = {"max_materialized_bytes": 9}
    if damage == "concat_order":
        concat.operands = tuple(reversed(concat.operands))
    elif damage == "concat_duplicate":
        concat.operands = (concat.operands[0],) * 3
    elif damage == "concat_dim":
        concat.properties["dim"] = builtin.IntegerAttr(1, builtin.i64)
    elif damage == "concat_hint":
        concat.attributes["prov.unknown_effect"] = builtin.StringAttr("unproved")
    elif damage == "wrong_rhs":
        combined.operands = (combined.inputs[0], block.args[0], combined.outputs[0])
    elif damage == "wrong_body":
        mul = next(op for op in combined.walk() if type(op) is arith.MuliOp)
        mul.operands = (mul.operands[0],) * 2
    elif damage == "flags":
        mul = next(op for op in combined.walk() if type(op) is arith.MuliOp)
        mul.properties["overflowFlags"] = arith.IntegerOverflowAttr((arith.IntegerOverflowFlag.NSW,))
    elif damage == "slice_order":
        views = sorted(
            (u.operation for u in combined.results[0].uses), key=lambda op: op.static_offsets.get_values()[0]
        )
        views[0].properties["static_offsets"] = builtin.DenseArrayBase.from_list(builtin.i64, (1, 0))
        views[1].properties["static_offsets"] = builtin.DenseArrayBase.from_list(builtin.i64, (0, 0))
    elif damage == "early_concat":
        block.detach_op(concat)
        block.insert_op_before(concat, block.first_op)
    elif damage == "late_concat":
        cluster = [concat, combined.outputs[0].owner, combined]
        cluster += [op for op in block.ops if op in {u.operation for u in combined.results[0].uses}]
        for op in cluster:
            block.detach_op(op)
        block.insert_ops_before(cluster, block.last_op)
        actual.verify()
    elif damage == "changed_return":
        block.last_op.operands = (*block.last_op.operands[:-1], block.args[0])
    elif damage == "budget":
        kwargs["max_materialized_bytes"] = 8
    else:
        kwargs = {}
    with pytest.raises((ValueError, VerifyException)):
        verify_dense_row_batching(module, actual, index_bits=64, **kwargs)


@pytest.mark.parametrize("damage", ["slice_overlap", "concat_dim", "concat_hint"])
def test_malformed_existing_batches_are_refused_even_without_a_new_group(damage):
    actual = packed(computed_rows(k=3, n=3), cap=9).candidate
    combined = dots(actual)[0]
    if damage == "slice_overlap":
        for use in combined.results[0].uses:
            use.operation.properties["static_offsets"] = builtin.DenseArrayBase.from_list(builtin.i64, (0, 0))
    elif damage == "concat_dim":
        combined.inputs[0].owner.properties["dim"] = builtin.IntegerAttr(1, builtin.i64)
    elif damage == "concat_hint":
        combined.inputs[0].owner.attributes["prov.unknown_effect"] = builtin.StringAttr("unproved")
    with pytest.raises((ValueError, VerifyException)):
        packed(actual, cap=9)


def authored_shell(mode, floating):
    """Valid original programs, without a prior batching plan or provenance."""
    dtype, accum = ("f32", "f32") if floating else ("i8", "i32")
    args = f"%a:tensor<1x2x{dtype}>,%b:tensor<1x2x{dtype}>,%r:tensor<2x3x{dtype}>"
    if mode == "concat":
        lines = [
            f"%lhs = tensor.concat dim(0) %a, %b : (tensor<1x2x{dtype}>,tensor<1x2x{dtype}>) -> tensor<2x2x{dtype}>"
        ]
    else:
        args = f"%rows:tensor<2x2x{dtype}>,%r:tensor<2x3x{dtype}>"
        lines = [
            f"%flat=tensor.collapse_shape %rows [[0,1]] : tensor<2x2x{dtype}> into tensor<4x{dtype}>",
            f"%lhs=tensor.expand_shape %flat [[0,1]] output_shape [2,2] : tensor<4x{dtype}> into tensor<2x2x{dtype}>",
        ]
    lines.insert(0, f"%initial=arith.constant {'0.0' if floating else '1'} : {accum}")
    lines += [
        f"%init=tensor.splat %initial : tensor<2x3x{accum}>",
        "%dot=linalg.generic {indexing_maps=[affine_map<(m,n,k)->(m,k)>,affine_map<(m,n,k)->(k,n)>,"
        'affine_map<(m,n,k)->(m,n)>],iterator_types=["parallel","parallel","reduction"]} '
        f"ins(%lhs,%r:tensor<2x2x{dtype}>,tensor<2x3x{dtype}>) outs(%init:tensor<2x3x{accum}>) {{",
        f"^bb0(%x:{dtype},%y:{dtype},%z:{accum}):",
    ]
    if floating:
        lines += ["%p=arith.mulf %x,%y : f32", "%v=arith.addf %p,%z : f32", "linalg.yield %v : f32"]
    else:
        lines += [
            "%xx=arith.extsi %x : i8 to i32",
            "%yy=arith.extsi %y : i8 to i32",
            "%p=arith.muli %xx,%yy : i32",
            "%v=arith.addi %p,%z : i32",
            "linalg.yield %v : i32",
        ]
    lines += [f"}} -> tensor<2x3x{accum}>"]
    for i in range(2):
        lines += [
            f'%out{i}="tensor.extract_slice"(%dot) <{{static_offsets=array<i64:{i},0>,'
            "static_sizes=array<i64:1,3>,static_strides=array<i64:1,1>,operandSegmentSizes=array<i32:1,0,0,0>}> "
            f": (tensor<2x3x{accum}>) -> tensor<1x3x{accum}>"
        ]
    lines += [f"return %out0,%out1 : tensor<1x3x{accum}>,tensor<1x3x{accum}>"]
    module = parse_mlir_text(
        f"module {{ func.func @forward({args}) -> (tensor<1x3x{accum}>,tensor<1x3x{accum}>) {{\n"
        + "\n".join(lines)
        + "\n} }"
    )
    module.verify()
    return module


@pytest.mark.parametrize("mode", ["concat", "view"])
@pytest.mark.parametrize("floating", [False, True])
@pytest.mark.parametrize("materialize", [False, True])
def test_authored_unsupported_shell_retains_exact_original_in_selected_ordinary_path(
    tmp_path, mode, floating, materialize
):
    module = authored_shell(mode, floating)
    dtype = np.float32 if floating else np.int8
    a = np.asarray([[1, 2]], dtype=dtype)
    b = np.asarray([[3, 4]], dtype=dtype)
    r = np.arange(6, dtype=dtype).reshape(2, 3)
    args = [a, b, r] if mode == "concat" else [np.concatenate((a, b)), r]
    expected = [
        np.asarray(
            [[sum(int(v) * int(r[k, n]) for k, v in enumerate(row)) + (0 if floating else 1) for n in range(3)]],
            dtype=np.float32 if floating else np.int32,
        )
        for row in (a[0], b[0])
    ]
    actual = evaluate(module, args)
    assert all(x.dtype == y.dtype and x.tobytes() == y.tobytes() for x, y in zip(actual, expected, strict=True))
    original = ir_text(module)
    path = tmp_path / "original.mlir"
    path.write_text(original)

    def choose(snapshot, work):
        current = parse_mlir_text(snapshot.read_text())
        selected = packed(current) if materialize else plan(current)
        assert selected is None
        return snapshot

    selected = apply_prepared_model_transform(path, tmp_path / "selected", choose)
    assert selected.read_text() == path.read_text() == original
    # Original-only fallback does not turn unsupported candidates into proof.
    with pytest.raises(ValueError, match="unproved scalar"):
        verify_dense_row_batching(module, module.clone(), index_bits=64, max_materialized_bytes=4096)


@pytest.mark.parametrize("mode", ["concat", "view"])
@pytest.mark.parametrize("floating", [False, True])
@pytest.mark.parametrize("unsupported_first", [False, True])
def test_unsupported_shell_declines_whole_plan_even_with_an_unrelated_legal_group(mode, floating, unsupported_first):
    unsupported = authored_shell(mode, floating)
    legal = computed_rows(k=3, n=4)
    assert packed(legal) is not None
    module = combine(unsupported, legal) if unsupported_first else combine(legal, unsupported)
    original = ir_text(module)
    assert packed(module) is None and plan(module) is None
    assert ir_text(module) == original


def test_issued_original_source_scalar_mutation_refuses_instead_of_becoming_fallback():
    module = computed_rows(k=3, n=3)
    selected = packed(module, cap=9)
    multiply = next(op for op in dots(module)[0].walk() if type(op) is arith.MuliOp)
    multiply.properties["overflowFlags"] = arith.IntegerOverflowAttr((arith.IntegerOverflowFlag.NSW,))
    with pytest.raises(ValueError, match="changed"):
        selected.revalidate()
