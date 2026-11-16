"""Real source operands and closed numeric contracts at the host callback seam."""

import pytest
from xdsl.dialects import arith, builtin, func
from xdsl.dialects.linalg.ops import MatmulOp

from merlin.frontends.linalg_batch_matmul import BatchMatmulOp
from merlin.frontends.linalg_tensor_graph import read_tensor_graph
from merlin.frontends.linalg_text import parse_mlir_text
from merlin.llvmlower.floating_matmul_outline import callback_bridge, outline_floating_matmuls
from merlin.xdsl_dialects._common import text


def source(batch=None, dtype="f32", shape=(3, 5, 7)):
    m, k, n = shape
    prefix = "" if batch is None else f"{batch}x"
    a, b, c = (f"tensor<{prefix}{x}x{y}x{dtype}>" for x, y in ((m, k), (k, n), (m, n)))
    op = "matmul" if batch is None else "batch_matmul"
    return f"""module {{
      func.func @compute(%a: {a}, %b: {b}) -> ({c}, {a}) {{
        %zero = arith.constant 0.0 : {dtype}
        %empty = tensor.empty() : {c}
        %init = linalg.fill ins(%zero: {dtype}) outs(%empty: {c}) -> {c}
        %result = linalg.{op} ins(%a, %b: {a}, {b}) outs(%init: {c}) -> {c}
        func.return %result, %a : {c}, {a}
      }}
    }}"""


@pytest.mark.parametrize("batch", [None, 1, 3])
def test_outlining_records_actual_geometry_and_keeps_other_results(batch):
    original = source(batch)
    host, calls = outline_floating_matmuls(original, entry="compute", numerical_policy="explicit-test-policy")
    assert len(calls) == 1
    prefix = [] if batch is None else [batch]
    assert calls[0]["input_shapes"] == [prefix + [3, 5], prefix + [5, 7]]
    assert calls[0]["output_shape"] == prefix + [3, 7]
    graph = read_tensor_graph(calls[0]["source"], entry="contraction", target_identity="test-device")
    assert graph.nodes[-1].op == ("matmul" if batch is None else "batch_matmul")
    module = parse_mlir_text(host)
    module.verify()
    fn = module.body.block.first_op
    assert len(fn.function_type.outputs) == 2
    assert fn.body.block.last_op.arguments[1] is fn.body.block.args[0]
    assert not any(type(op) in (MatmulOp, BatchMatmulOp) for op in fn.walk())
    assert sum(type(op) is func.CallOp for op in fn.walk()) == 1
    assert source(batch) == original
    assert "dispatch(0,a,b,c)" in callback_bridge(calls)


def test_duplicate_source_operands_keep_both_abi_positions():
    original = source(shape=(5, 5, 5)).replace("ins(%a, %b:", "ins(%a, %a:")
    _, calls = outline_floating_matmuls(original, entry="compute", numerical_policy="explicit")
    graph = read_tensor_graph(calls[0]["source"], entry="contraction", target_identity="test-device")
    assert graph.nodes[-1].inputs == ("input0", "input1")


@pytest.mark.parametrize("batch", [None, 2])
@pytest.mark.parametrize("mutation", ["negative_zero", "nonzero", "wrong_scalar", "fastmath", "map"])
def test_mutated_numeric_contract_is_rejected(batch, mutation):
    original = source(batch)
    if mutation in ("negative_zero", "nonzero"):
        original = original.replace("0.0 : f32", ("-0.0" if mutation == "negative_zero" else "1.0") + " : f32")
    else:
        module = parse_mlir_text(original)
        mm = next(op for op in module.walk() if type(op) in (MatmulOp, BatchMatmulOp))
        if mutation == "map":
            mm.properties["indexing_maps"] = builtin.ArrayAttr(list(reversed(tuple(mm.indexing_maps))))
        else:
            mul = next(op for op in mm.walk() if type(op) is arith.MulfOp)
            if mutation == "wrong_scalar":
                mul.operands = (mm.regions[0].block.args[0], mm.regions[0].block.args[2])
            else:
                mul.properties["fastmath"] = arith.FastMathFlagsAttr("fast")
        original = text(module, generic=True)
    with pytest.raises(Exception):
        outline_floating_matmuls(original, entry="compute", numerical_policy="explicit")


@pytest.mark.parametrize("dtype", ["f16", "bf16", "f64"])
def test_callback_precision_does_not_silently_change_abi(dtype):
    with pytest.raises(ValueError, match="f32"):
        outline_floating_matmuls(source(dtype=dtype), entry="compute", numerical_policy="explicit")


def test_policy_and_symbol_collision_refuse():
    with pytest.raises(ValueError, match="explicit"):
        outline_floating_matmuls(source(), entry="compute", numerical_policy="")
    with pytest.raises(ValueError, match="collides"):
        outline_floating_matmuls(
            source().replace("@compute", "@merlin_matmul_0"), entry="merlin_matmul_0", numerical_policy="explicit"
        )


@pytest.mark.parametrize("old,new", [("2x5x7", "3x5x7"), ("2x5x7", "2x6x7"), ("2x3x7", "2x3x8")])
def test_batch_geometry_mismatch_refuses(old, new):
    with pytest.raises(Exception):
        outline_floating_matmuls(source(2).replace(old, new), entry="compute", numerical_policy="explicit")


def test_nested_matmul_requires_control_flow_dispatch():
    original = source()
    line = next(line for line in original.splitlines() if "%result = linalg.matmul" in line)
    nested = """        %yes = arith.constant true
        %result = scf.if %yes -> (tensor<3x7xf32>) {
          %inner = linalg.matmul ins(%a, %b: tensor<3x5xf32>, tensor<5x7xf32>) outs(%init: tensor<3x7xf32>) -> tensor<3x7xf32>
          scf.yield %inner : tensor<3x7xf32>
        } else {
          scf.yield %init : tensor<3x7xf32>
        }"""
    with pytest.raises(ValueError, match="nested"):
        outline_floating_matmuls(original.replace(line, nested), entry="compute", numerical_policy="explicit")


@pytest.mark.parametrize("dtype", ["f16", "bf16", "f32", "f64"])
def test_batch_source_reader_preserves_dtype_without_selecting_a_target_policy(dtype):
    original = (
        source(3, dtype)
        .replace(
            f" -> (tensor<3x3x7x{dtype}>, tensor<3x3x5x{dtype}>)",
            f" -> tensor<3x3x7x{dtype}>",
        )
        .replace(
            f"func.return %result, %a : tensor<3x3x7x{dtype}>, tensor<3x3x5x{dtype}>",
            f"func.return %result : tensor<3x3x7x{dtype}>",
        )
    )
    graph = read_tensor_graph(original, entry="compute", target_identity="test-device")
    assert graph.nodes[-1].type.dtype == dtype
    assert graph.nodes[-1].type.shape == (3, 3, 7)
    assert dict(graph.nodes[-1].attrs)["reduction_order"] == "source-unspecified"
