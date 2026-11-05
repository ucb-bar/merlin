"""Maximum-number requires typed exclusion of NaNs and negative-zero ties."""

import ctypes
import io
import math
import subprocess

import numpy as np
import pytest
from xdsl.dialects import arith, builtin, func, tensor
from xdsl.dialects.linalg import ops
from xdsl.dialects.linalg.attrs import IteratorTypeAttr
from xdsl.ir import Block, Region
from xdsl.ir.affine import AffineMap
from xdsl.printer import Printer

from merlin.llvmlower.proved_float_maximum import rewrite_proved_maxima, rewrite_text


def fixture(*, dtype=builtin.i32, scale=8., fill=-np.finfo(np.float32).max, seed=-math.inf,
            scalar=arith.DivfOp):
    shape = (3, 7)
    itype, ftype = builtin.TensorType(dtype, shape), builtin.TensorType(builtin.f32, shape)
    masktype = builtin.TensorType(builtin.i1, shape)
    output = builtin.TensorType(builtin.f32, (3,))
    block = Block(arg_types=[itype, masktype])
    function = func.FuncOp("forward", ([itype, masktype], [output]), Region(block))
    function.attributes["llvm.emit_c_interface"] = builtin.UnitAttr()
    module, nodes = builtin.ModuleOp([function]), {"function": function}

    def splat(value, target):
        constant = arith.ConstantOp(builtin.FloatAttr(float(value), builtin.f32))
        operation = tensor.SplatOp(constant, (), builtin.TensorType(builtin.f32, target))
        block.add_ops([constant, operation])
        return operation.result

    def generic(inputs, body_builder):
        empty = tensor.EmptyOp((), ftype)
        body = Block(arg_types=[value.type.element_type for value in inputs] + [builtin.f32])
        operation = body_builder(body.args)
        body.add_ops([operation, ops.YieldOp(operation)])
        generic = ops.GenericOp(inputs, [empty.tensor], Region(body),
                                [builtin.AffineMapAttr(AffineMap.identity(2))] * (len(inputs) + 1),
                                [IteratorTypeAttr.parallel()] * 2, [ftype])
        block.add_ops([empty, generic])
        return generic.results[0], generic

    if dtype == builtin.f32:
        cast = block.args[0]
    else:
        cast, nodes["cast"] = generic([block.args[0]], lambda a: arith.SIToFPOp(a[0], builtin.f32))
    scaled, nodes["scaled"] = generic([cast, splat(scale, shape)], lambda a: scalar(a[0], a[1]))
    masked, nodes["masked"] = generic([block.args[1], scaled, splat(fill, shape)],
                                       lambda a: arith.SelectOp(*a[:3]))
    initial = splat(seed, (3,))
    body = Block(arg_types=[builtin.f32, builtin.f32])
    maximum = arith.MaximumfOp(*body.args)
    maximum.attributes["prov.source_node_ids"] = builtin.ArrayAttr([builtin.StringAttr("ordinary")])
    body.add_ops([maximum, ops.YieldOp(maximum)])
    reduce = ops.ReduceOp(masked, initial, builtin.DenseArrayBase.from_list(builtin.i64, [1]), Region(body))
    block.add_ops([reduce, func.ReturnOp(reduce.results[0])])
    nodes.update(maximum=maximum, reduction=reduce)
    return module, nodes


def text(module):
    stream = io.StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(module)
    return stream.getvalue()


@pytest.mark.parametrize("dtype,scale,scalar", [(builtin.i8, 8., arith.DivfOp),
                                              (builtin.i32, 8., arith.DivfOp),
                                              (builtin.i64, np.finfo(np.float32).max, arith.DivfOp),
                                              (builtin.i64, np.nextafter(np.float32(0), np.float32(1)), arith.MulfOp)])
def test_typed_integer_scale_select_reduction_retains_metadata(dtype, scale, scalar):
    module, nodes = fixture(dtype=dtype, scale=scale, scalar=scalar)
    report = rewrite_proved_maxima(module)
    assert report == {"proved_nan_and_negative_zero_free_maximum_reductions": 1}
    replacement = nodes["reduction"].regions[0].block.first_op
    assert replacement.name == "arith.maxnumf"
    assert replacement.attributes["prov.source_node_ids"] == nodes["maximum"].attributes["prov.source_node_ids"]
    assert rewrite_proved_maxima(module) == {}


@pytest.mark.parametrize("case", ["unknown_float", "negative_scale", "zero_scale", "nan_fill", "negative_zero_fill",
                                   "negative_zero_seed", "strict", "unknown_body", "unknown_attribute", "relaxed"])
def test_unproved_domain_or_semantics_preserves_original_bytes(case):
    kwargs = ({"dtype": builtin.f32} if case == "unknown_float" else
              {"scale": -8.} if case == "negative_scale" else
              {"scale": 0.} if case == "zero_scale" else
              {"fill": math.nan} if case == "nan_fill" else
              {"fill": -0.} if case == "negative_zero_fill" else
              {"seed": -0.} if case == "negative_zero_seed" else {})
    module, nodes = fixture(**kwargs)
    if case == "strict":
        nodes["function"].attributes["strictfp"] = builtin.UnitAttr()
    if case == "unknown_body":
        # Subtraction need not preserve the absence of negative zero.
        operation = nodes["scaled"].body.block.first_op
        replacement = arith.SubfOp(*operation.operands)
        block = operation.parent_block()
        block.insert_op_before(replacement, operation)
        operation.results[0].replace_all_uses_with(replacement.result)
        block.erase_op(operation)
    if case == "unknown_attribute":
        nodes["maximum"].attributes["numeric.unknown"] = builtin.UnitAttr()
    if case == "relaxed":
        nodes["maximum"].properties["fastmath"] = arith.FastMathFlagsAttr("fast")
    original = text(module)
    assert rewrite_text(original) == (original, {})


def test_opaque_floating_producer_is_not_admitted_from_its_result_type():
    module, nodes = fixture()
    function = nodes["function"]
    value = nodes["scaled"].inputs[0]
    declaration = func.FuncOp.external("opaque", [function.body.block.args[0].type], [value.type])
    module.body.block.add_op(declaration)
    call = func.CallOp("opaque", [function.body.block.args[0]], [value.type])
    function.body.block.insert_op_before(call, nodes["scaled"])
    nodes["scaled"].operands = [call.results[0], *nodes["scaled"].operands[1:]]
    original = text(module)
    assert rewrite_text(original) == (original, {})


def test_second_shrinking_scale_cannot_inherit_integer_nonzero_magnitude():
    module, nodes = fixture(scale=np.finfo(np.float32).max)
    first = nodes["scaled"]
    second = first.clone()
    second.operands = [first.results[0], *second.operands[1:]]
    first.parent_block().insert_op_after(second, first)
    mask = nodes["masked"]
    mask.operands = [mask.operands[0], second.results[0], *mask.operands[2:]]
    original = text(module)
    assert rewrite_text(original) == (original, {})


def test_selected_ordinary_lowering_records_live_source_and_rewrite(tmp_path):
    import hashlib
    import json

    from merlin.llvmlower.lower import lower_model
    from merlin.llvmlower.proved_float_maximum import FEATURE

    module, _ = fixture()
    original = text(module)
    result = lower_model(original, tmp_path / "ordinary", features=frozenset({FEATURE}))
    report = result.stats["proved_float_maximum"]
    assert report["source_sha256"] == hashlib.sha256(original.encode()).hexdigest()
    assert report["rewrites"] == {"proved_nan_and_negative_zero_free_maximum_reductions": 1}
    assert json.loads((tmp_path / "ordinary/proved-float-maximum.json").read_text()) == report
    assert "llvm.maxnum.f32" in result.ll_path.read_text()


@pytest.mark.parametrize("scale,scalar", [(8., arith.DivfOp),
                                         (np.finfo(np.float32).max, arith.MulfOp),
                                         (np.finfo(np.float32).max, arith.DivfOp)])
def test_complete_native_words_and_flags_in_every_rounding_mode(tmp_path, scale, scalar):
    from merlin.llvmlower.abi import HostModel
    from merlin.llvmlower.codegen import build_host_shared
    from merlin.llvmlower.pipeline import lower_to_llvm_ir
    from merlin.llvmlower.toolchain import clang

    module, _ = fixture(dtype=builtin.i64, scale=scale, scalar=scalar)
    original = text(module)
    candidate, report = rewrite_text(original)
    assert report
    models = []
    for name, source in [("control", original), ("candidate", candidate)]:
        root = tmp_path / name
        root.mkdir()
        llvm = lower_to_llvm_ir(source, workdir=root)
        ll = root / "model.ll"
        ll.write_text(llvm)
        models.append(HostModel.load(str(build_host_shared(ll, root / "model.so"))))
    helper = tmp_path / "environment.c"
    helper.write_text("""#include <fenv.h>
      int mode(int n){const int modes[]={FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO};return fesetround(modes[n]);}
      int flags(void){return fetestexcept(FE_ALL_EXCEPT);}
      int clear(void){return feclearexcept(FE_ALL_EXCEPT);}
    """)
    so = tmp_path / "environment.so"
    subprocess.run([str(clang()), "-shared", "-fPIC", str(helper), "-lm", "-o", str(so)],
                   check=True, capture_output=True)
    environment = ctypes.CDLL(str(so))
    values = np.array([-(1 << 63), (1 << 63) - 1, 0, 1, -1, (1 << 24) + 1, -(1 << 24) - 1], dtype="i8")
    try:
        for mode in range(4):
            assert environment.mode(mode) == 0
            for shift in range(7):
                data = np.tile(np.roll(values, shift), (3, 1))
                for kind in (0, 1, 2):
                    mask = np.zeros((3, 7), dtype="u1") if kind == 0 else np.ones((3, 7), dtype="u1")
                    if kind == 2:
                        mask[:, ::2] = 0
                    outputs, flags = [], []
                    for model in models:
                        store = np.full(19, 0xA5A5A5A5, dtype="u4")
                        out = store[8:11].view("f4")
                        for _ in range(2):
                            environment.clear()
                            model([(v.ctypes.data, v.shape) for v in (data, mask, out)])
                            flags.append(environment.flags())
                        outputs.append(out.view("u4").copy())
                        assert np.all(store[:8] == 0xA5A5A5A5) and np.all(store[11:] == 0xA5A5A5A5)
                    np.testing.assert_array_equal(outputs[1], outputs[0])
                    assert len(set(flags)) == 1
    finally:
        assert environment.mode(0) == 0
