"""Exact tensor CSE must preserve source roots, scalar boundaries and metadata."""

import pytest
from xdsl.dialects import arith, builtin, func, tensor
from xdsl.dialects.linalg import ops
from xdsl.dialects.linalg.attrs import IteratorTypeAttr
from xdsl.ir import Block, Region
from xdsl.ir.affine import AffineMap

from merlin.llvmlower.tensor_provenance_cse import deduplicate_tensor_preparation, rewrite_text


def fixture(integer=builtin.i8):
    typ = builtin.TensorType(integer, [2, 3])
    out = builtin.TensorType(builtin.f32, [2, 3])
    block = Block(arg_types=[typ, typ])
    f = func.FuncOp("ordinary", ([typ, typ], [out, out]), Region(block))
    return builtin.ModuleOp([f]), block


def cast(block, value, label, *, use_seed=False, unsigned=False):
    out = builtin.TensorType(builtin.f32, [2, 3])
    empty = tensor.EmptyOp((), out)
    block.add_op(empty)
    region = Block(arg_types=[value.type.element_type, builtin.f32])
    scalar = (arith.UIToFPOp if unsigned else arith.SIToFPOp)(region.args[0], builtin.f32)
    scalar.attributes["prov.source_node_ids"] = builtin.ArrayAttr([builtin.StringAttr(label + "_scalar")])
    region.add_ops([scalar, ops.YieldOp(region.args[1] if use_seed else scalar)])
    op = ops.GenericOp([value], [empty.tensor], Region(region),
                       [builtin.AffineMapAttr(AffineMap.identity(2))] * 2,
                       [IteratorTypeAttr.parallel()] * 2, [out])
    op.attributes["prov.region_id"] = builtin.StringAttr(label)
    op.attributes["prov.source_node_ids"] = builtin.ArrayAttr([builtin.StringAttr(label)])
    block.add_op(op)
    return op


def finish(block, a, b):
    block.add_op(func.ReturnOp(a.results[0], b.results[0]))


def consumer(block, value, increment):
    empty = tensor.EmptyOp((), value.type)
    block.add_op(empty)
    body = Block(arg_types=[builtin.f32, builtin.f32])
    constant = arith.ConstantOp(builtin.FloatAttr(increment, builtin.f32))
    add = arith.AddfOp(body.args[0], constant)
    body.add_ops([constant, add, ops.YieldOp(add)])
    op = ops.GenericOp([value], [empty.tensor], Region(body),
                       [builtin.AffineMapAttr(AffineMap.identity(2))] * 2,
                       [IteratorTypeAttr.parallel()] * 2, [value.type])
    block.add_op(op)
    return op


def test_repeated_exact_cast_ignores_only_provenance_and_unread_empty_seed():
    module, block = fixture()
    a, b = cast(block, block.args[0], "first"), cast(block, block.args[0], "second")
    ca, cb = consumer(block, a.results[0], 1.), consumer(block, b.results[0], -1.)
    finish(block, ca, cb)
    assert deduplicate_tensor_preparation(module) == {"linalg.generic": 1}
    assert ca.operands[0] is cb.operands[0] is a.results[0]
    assert [x.data for x in a.attributes["prov.source_node_ids"]] == ["first", "second"]
    assert len(a.attributes["prov.cse_sources"]) == 2
    scalar = a.body.block.first_op
    assert [x.data for x in scalar.attributes["prov.source_node_ids"]] == ["first_scalar", "second_scalar"]
    assert deduplicate_tensor_preparation(module) == {}


def test_independent_tensor_return_preparations_keep_upstream_storage_fallback():
    module, block = fixture()
    a, b = cast(block, block.args[0], "first"), cast(block, block.args[0], "second")
    finish(block, a, b)
    assert deduplicate_tensor_preparation(module) == {}


@pytest.mark.parametrize("case", ["root", "unsigned", "seed", "semantic_attr", "strictfp", "inexact_integer"])
def test_unknown_or_distinct_semantics_stay_separate(case):
    module, block = fixture(builtin.i32 if case == "inexact_integer" else builtin.i8)
    a = cast(block, block.args[0], "first", unsigned=case == "unsigned")
    b = cast(block, block.args[1] if case == "root" else block.args[0], "second",
             use_seed=case == "seed", unsigned=case == "unsigned")
    if case == "semantic_attr":
        b.attributes["unknown.numeric.mode"] = builtin.StringAttr("different")
    if case == "strictfp":
        module.body.block.first_op.attributes["strictfp"] = builtin.UnitAttr()
    ca, cb = consumer(block, a.results[0], 1.), consumer(block, b.results[0], -1.)
    finish(block, ca, cb)
    assert deduplicate_tensor_preparation(module) == {}
    assert (ca.operands[0], cb.operands[0]) == (a.results[0], b.results[0])


def test_exact_slices_merge_but_different_offsets_and_roots_do_not():
    typ = builtin.TensorType(builtin.i8, [4, 3])
    block = Block(arg_types=[typ, typ])
    out = builtin.TensorType(builtin.i8, [2, 3])
    module = builtin.ModuleOp([func.FuncOp("views", ([typ, typ], []), Region(block)),
                              func.FuncOp.external("read", [out] * 4, [])])
    views = []
    for source, offset, name in [(0, 0, "a"), (0, 0, "b"), (0, 1, "c"), (1, 0, "d")]:
        op = tensor.ExtractSliceOp.from_static_parameters(block.args[source], [offset, 0], [2, 3])
        op.attributes["prov.region_id"] = builtin.StringAttr(name)
        block.add_op(op)
        views.append(op)
    observe = func.CallOp("read", [v.results[0] for v in views], [])
    block.add_ops([observe, func.ReturnOp()])
    assert deduplicate_tensor_preparation(module) == {"tensor.extract_slice": 1}
    assert tuple(observe.operands) == tuple(views[i].results[0] for i in (0, 0, 2, 3))


def test_unselected_text_is_byte_identical_and_opaque_calls_are_retained():
    source = ('builtin.module { func.func private @opaque() func.func @f() { '
              'func.call @opaque() : () -> () func.call @opaque() : () -> () func.return } }')
    output, report = rewrite_text(source)
    assert output == source and report == {}


@pytest.mark.parametrize("width", [8, 16])
@pytest.mark.parametrize("direct_return", [False, True])
def test_actual_host_cast_boundaries_keep_all_bits_and_independent_outputs(tmp_path, width, direct_return):
    import io

    import numpy as np
    from xdsl.printer import Printer

    from merlin.llvmlower.abi import HostModel
    from merlin.llvmlower.lower import lower_model

    module, block = fixture(builtin.IntegerType(width))
    module.body.block.first_op.sym_name = builtin.StringAttr("forward")
    a, b = cast(block, block.args[0], "first"), cast(block, block.args[0], "second")
    if direct_return:
        finish(block, a, b)
    else:
        finish(block, consumer(block, a.results[0], 1.), consumer(block, b.results[0], -1.))
    stream = io.StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(module)
    result = lower_model(stream.getvalue(), tmp_path / f"cast_{width}_{direct_return}",
                         features=frozenset({"share_exact_tensor_preparations"}))
    assert result.stats["tensor_preparation_cse"]["rewrites"] == ({} if direct_return else {"linalg.generic": 1})
    model = HostModel.load(str(result.host_so))
    values = np.array([-(1 << (width - 1)), -1, 0, 1, (1 << (width - 1)) - 2,
                       (1 << (width - 1)) - 1], dtype=f"int{width}").reshape(2, 3)
    original = values.copy()
    for _ in range(2):
        buffers = [np.full(22, 73, np.float32) for _ in range(2)]
        outputs = [b[8:14].reshape(2, 3) for b in buffers]
        model([(v.ctypes.data, v.shape) for v in (values, original, *outputs)])
        increments = (0., 0.) if direct_return else (1., -1.)
        for output, storage, increment in zip(outputs, buffers, increments, strict=True):
            expected = np.add(values.astype(np.float32), np.float32(increment))
            np.testing.assert_array_equal(output.view(np.uint32), expected.view(np.uint32))
            np.testing.assert_array_equal(storage[:8], np.full(8, 73, np.float32))
            np.testing.assert_array_equal(storage[14:], np.full(8, 73, np.float32))
        np.testing.assert_array_equal(values, original)
