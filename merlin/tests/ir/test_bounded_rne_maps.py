"""Source graph acceptance/refusals independent of provenance names."""

import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.bounded_rne_maps import prove_scalar_bounded_rne
from merlin.llvmlower.quant_round import fuse_round_clamp_convert


def fixture(bits=8):
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    module = parse_mlir_text(f"""module {{
      func.func @any_name(%x:f32, %scale:f32) -> i{bits} {{
        %v = arith.mulf %x, %scale : f32
        %r = math.roundeven %v : f32
        %lo = arith.constant {float(lo):.6e} : f32
        %hi = arith.constant {float(hi):.6e} : f32
        %a = arith.maximumf %r, %lo : f32
        %b = arith.minimumf %a, %hi : f32
        %q = arith.fptosi %b : f32 to i{bits}
        return %q : i{bits}
      }} }}""")
    assert fuse_round_clamp_convert(module) == 1
    module.verify()
    return module, next(op for op in module.walk() if op.name == "func.func").body.block


@pytest.mark.parametrize("bits", [8, 16])
def test_complete_original_scalar_graph(bits):
    _, block = fixture(bits)
    before = tuple(block.ops)
    proof = prove_scalar_bounded_rne(block)
    assert proof is not None and proof.integer_bits == bits
    assert proof.bounds == (-(1 << (bits - 1)), (1 << (bits - 1)) - 1)
    assert proof.raw_input.owner.name == "arith.mulf"
    assert proof.result is block.last_op.operands[0]
    assert tuple(block.ops) == before


@pytest.mark.parametrize("change", ["fast", "overflow", "half", "parity", "wrong_sign", "unknown_op"])
def test_semantic_near_misses_refuse(change):
    from xdsl.dialects import arith, func
    from xdsl.dialects.builtin import FloatAttr, IntegerAttr, f32, i8

    _, block = fixture()
    if change == "fast":
        op = next(o for o in block.ops if o.name == "arith.mulf")
        op.properties["fastmath"] = arith.FastMathFlagsAttr("fast")
    elif change == "overflow":
        op = next(o for o in block.ops if o.name == "arith.addi")
        op.properties["overflowFlags"] = arith.IntegerOverflowAttr([arith.IntegerOverflowFlag.NSW])
    elif change == "half":
        op = next(
            o
            for o in block.ops
            if o.name == "arith.constant" and isinstance(o.value, FloatAttr) and o.value.value.data == 0.5
        )
        op.properties["value"] = FloatAttr(0.499, f32)
    elif change == "parity":
        op = next(
            o
            for o in block.ops
            if o.name == "arith.constant"
            and isinstance(o.value, IntegerAttr)
            and o.result.type == i8
            and o.value.value.data == 1
        )
        op.properties["value"] = IntegerAttr(2, i8)
    elif change == "wrong_sign":
        op = next(o for o in block.ops if o.name == "arith.cmpf" and o.predicate.value.data == 4)
        op.properties["predicate"] = IntegerAttr(2, 64)
    else:
        call = func.CallOp("opaque_effect", [], [f32])
        block.insert_op_before(call, block.last_op)
    assert prove_scalar_bounded_rne(block) is None


@pytest.mark.parametrize("n,pad,lanes", [(7, 1, 4), (9, 0, 4), (10, 0, 4), (6, 0, 8), (15, 1, 8), (17, 0, 8)])
def test_stripmine_transpose_tails_live_destination_native(tmp_path, n, pad, lanes):
    import subprocess

    import numpy as np

    from merlin.llvmlower.abi import HostModel
    from merlin.llvmlower.bounded_rne_maps import schedule_bounded_rne_maps
    from merlin.llvmlower.codegen import mlir_runtime_c
    from merlin.llvmlower.pipeline import lower_to_llvm_ir
    from merlin.llvmlower.toolchain import clang
    from merlin.xdsl_dialects._common import text

    m = 3
    pn, pm = n + 2 * pad, m + 2 * pad
    view = (
        f'%view="tensor.extract_slice"(%old) <{{static_offsets=array<i64:{pad},{pad}>,'
        f"static_sizes=array<i64:{n},{m}>,static_strides=array<i64:1,1>,"
        f"operandSegmentSizes=array<i32:1,0,0,0>}}> : (tensor<{pn}x{pm}xi8>) -> tensor<{n}x{m}xi8>"
        if pad
        else ""
    )
    insert = (
        f'%answer="tensor.insert_slice"(%r,%old) <{{static_offsets=array<i64:{pad},{pad}>,'
        f"static_sizes=array<i64:{n},{m}>,static_strides=array<i64:1,1>,"
        f"operandSegmentSizes=array<i32:1,1,0,0,0>}}> : "
        f"(tensor<{n}x{m}xi8>,tensor<{pn}x{pm}xi8>) -> tensor<{pn}x{pm}xi8>"
        if pad
        else ""
    )
    module = parse_mlir_text(f"""module {{
      func.func @forward(%a:tensor<{m}x{n}xf32>,%old:tensor<{pn}x{pm}xi8>)
        -> (tensor<{pn}x{pm}xi8>,tensor<{pn}x{pm}xi8>) attributes {{llvm.emit_c_interface}} {{
        %e=tensor.empty():tensor<{n}x{m}xf32>
        %t=linalg.transpose ins(%a:tensor<{m}x{n}xf32>) outs(%e:tensor<{n}x{m}xf32>) permutation=[1,0]
        {view}
        %r=linalg.generic {{indexing_maps=[affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>],
        iterator_types=["parallel","parallel"]}}
        ins(%t:tensor<{n}x{m}xf32>) outs({"%view" if pad else "%old"}:tensor<{n}x{m}xi8>) {{
        ^bb0(%x:f32,%o:i8):
          %s=arith.constant 2.500000e+00:f32
          %v=arith.mulf %x,%s:f32
          %rd=math.roundeven %v:f32
          %lo=arith.constant -1.280000e+02:f32
          %hi=arith.constant 1.270000e+02:f32
          %lower=arith.maximumf %rd,%lo:f32
          %b=arith.minimumf %lower,%hi:f32
          %q=arith.fptosi %b:f32 to i8
          linalg.yield %q:i8
        }}->tensor<{n}x{m}xi8>
        {insert}
        return %old,{"%answer" if pad else "%r"}:tensor<{pn}x{pm}xi8>,tensor<{pn}x{pm}xi8>
      }} }}""")
    assert fuse_round_clamp_convert(module) == 1
    reports = schedule_bounded_rne_maps(module, lanes=lanes)
    assert len(reports) == 1 and reports[0]["packet_axis"] == 0 and reports[0]["tail"] == n % lanes
    module.verify()
    llvm = lower_to_llvm_ir(text(module, generic=True), workdir=tmp_path / "lower", vectorize=False)
    assert "call void @free(" in llvm
    src = tmp_path / "model.ll"
    src.write_text(llvm)
    obj = tmp_path / "model.o"
    lib = tmp_path / f"quant_{n}_{pad}.so"
    subprocess.run([str(clang()), "-O2", "-fPIC", "-c", str(src), "-o", str(obj)], check=True, capture_output=True)
    subprocess.run(
        ["cc", "-fPIC", "-shared", str(obj), str(mlir_runtime_c()), "-lm", "-o", str(lib)],
        check=True,
        capture_output=True,
    )
    a = np.linspace(-60, 60, m * n, dtype=np.float32).reshape(m, n)
    old = np.full((pn, pm), 71, dtype=np.int8)
    expected = old.copy()
    expected[pad : pad + n, pad : pad + m] = np.rint(np.clip(np.multiply(a.T, np.float32(2.5)), -128, 127)).astype(
        np.int8
    )
    first = np.zeros_like(old)
    second = np.zeros_like(old)
    invoke = HostModel.load(str(lib))
    original_a = a.copy()
    for _ in range(3):
        invoke([(x.ctypes.data, x.shape) for x in (a, old, first, second)])
        np.testing.assert_array_equal(old, 71)
        np.testing.assert_array_equal(a, original_a)
        np.testing.assert_array_equal(first, old)
        np.testing.assert_array_equal(second, expected)


@pytest.mark.parametrize("effectful", [False, True])
def test_dead_tensor_branch_does_not_gain_opaque_calls(effectful):
    from xdsl.dialects import func, tensor
    from xdsl.dialects.builtin import AffineMapAttr, ModuleOp, TensorType, f32, i8
    from xdsl.dialects.linalg.ops import GenericOp, IteratorType, IteratorTypeAttr
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineMap

    from merlin.llvmlower.bounded_rne_maps import schedule_bounded_rne_maps

    _, scalar = fixture()
    types = [TensorType(f32, [7]), TensorType(f32, [7]), TensorType(i8, [7])]
    block = Block(arg_types=types)
    body = Block(arg_types=[f32, f32, i8])
    mapping = dict(zip(scalar.args, body.args[:2]))
    for old in list(scalar.ops)[:-1]:
        body.add_op(old.clone(mapping))
    from xdsl.dialects.linalg.ops import YieldOp

    body.add_op(YieldOp(mapping[scalar.last_op.operands[0]]))
    g = GenericOp(
        inputs=block.args[:2],
        outputs=[block.args[2]],
        body=Region(body),
        indexing_maps=[AffineMapAttr(AffineMap.identity(1))] * 3,
        iterator_types=[IteratorTypeAttr(IteratorType.PARALLEL)],
        result_types=[types[2]],
    )
    block.add_op(g)
    cast = tensor.CastOp(g.results[0], TensorType(i8, [-1]))
    block.add_op(cast)
    if effectful:
        block.add_op(func.CallOp("opaque_sink", [cast.results[0]], []))
    block.add_op(func.ReturnOp())
    module = ModuleOp(
        [
            func.FuncOp("forward", (types, []), Region(block)),
            func.FuncOp.external("opaque_sink", [TensorType(i8, [-1])], []),
        ]
    )
    before = len(list(module.walk()))
    routes = schedule_bounded_rne_maps(module)
    assert len(routes) == int(effectful)
    if not effectful:
        assert len(list(module.walk())) == before
