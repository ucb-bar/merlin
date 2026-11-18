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
          %inner = linalg.matmul ins(%a, %b: tensor<3x5xf32>, tensor<5x7xf32>)
              outs(%init: tensor<3x7xf32>) -> tensor<3x7xf32>
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


def generic_source(batch=None, *, splat=True, acc_first=True):
    original = source(batch)
    prefix = "" if batch is None else f"{batch}x"
    a, b, c = (f"tensor<{prefix}{m}x{n}xf32>" for m, n in ((3, 5), (5, 7), (3, 7)))
    dims = "b,m,n,k" if batch is not None else "m,n,k"
    lead = "b," if batch is not None else ""
    maps = ", ".join(f"affine_map<({dims})->({lead}{v})>" for v in ("m,k", "k,n", "m,n"))
    iterators = ", ".join(['"parallel"'] * (3 if batch is not None else 2) + ['"reduction"'])
    add = "%old, %p" if acc_first else "%p, %old"
    generic = f"""        %result = linalg.generic {{indexing_maps=[{maps}], iterator_types=[{iterators}]}}
      ins(%a,%b : {a},{b}) outs(%init : {c}) {{
        ^bb0(%x:f32,%y:f32,%old:f32):
          %p=arith.mulf %x,%y : f32
          %sum=arith.addf {add} : f32
          linalg.yield %sum : f32
      }} -> {c}"""
    line = next(line for line in original.splitlines() if "%result = linalg." in line)
    original = original.replace(line, generic)
    if splat:
        old = next(line for line in original.splitlines() if "%init = linalg.fill" in line)
        original = original.replace(old, f"        %init = tensor.splat %zero : {c}")
    return original


@pytest.mark.parametrize("acc_first", [False, True])
def test_batched_mixed_precision_preserves_individual_operands_and_callback_abi(acc_first):
    # Widen just the left input. The right input is already in the compute format.
    source = generic_source(3, splat=False, acc_first=acc_first).replace("3x3x5xf32", "3x3x5xf16")
    source = source.replace("%x:f32", "%x:f16").replace(
        "%p=arith.mulf %x,%y", "%wide=arith.extf %x : f16 to f32\n          %p=arith.mulf %wide,%y"
    )
    graph = read_tensor_graph(source, entry="compute", target_identity="another-accelerator")
    result = graph.nodes[-1]
    assert result.op == "widening_batch_matmul"
    assert tuple(n.type.dtype for n in graph.nodes) == ("f16", "f32", "f32")
    assert dict(result.attrs) == dict(
        accumulator_first=acc_first, compute_dtype="f32", reduction_order="source-unspecified"
    )
    assert graph.outputs == (result.id, "input0")
    with pytest.raises(ValueError, match="requires f32"):
        outline_floating_matmuls(source, entry="compute", numerical_policy="explicit")


@pytest.mark.parametrize("batch", [None, 3])
@pytest.mark.parametrize("splat", [False, True])
@pytest.mark.parametrize("acc_first", [False, True])
def test_generic_contraction_checks_maps_body_and_seed(batch, splat, acc_first):
    _, calls = outline_floating_matmuls(
        generic_source(batch, splat=splat, acc_first=acc_first), entry="compute", numerical_policy="explicit"
    )
    assert len(calls) == 1
    assert calls[0]["output_shape"] == ([3] if batch else []) + [3, 7]


@pytest.mark.parametrize(
    "old,new",
    [
        ("0.0 : f32", "-0.0 : f32"),
        ("0.0 : f32", "1.0 : f32"),
        ("arith.mulf %x,%y", "arith.mulf %x,%old"),
        ("arith.addf %old, %p : f32", "arith.addf %old, %p fastmath<fast> : f32"),
    ],
)
def test_generic_contraction_does_not_accept_wrong_arithmetic(old, new):
    with pytest.raises(ValueError):
        outline_floating_matmuls(generic_source(3).replace(old, new), entry="compute", numerical_policy="explicit")


def test_wrong_generic_maps_remain_host_operations():
    _, calls = outline_floating_matmuls(
        generic_source().replace("->(k,n)", "->(n,k)").replace("tensor<5x7xf32>", "tensor<7x5xf32>"),
        entry="compute",
        numerical_policy="explicit",
    )
    assert not calls


def test_host_signature_preserves_integer_tokens_boolean_masks_and_scalar_buffers():
    from merlin.llvmlower.floating_matmul_outline import static_tensor_signature

    mlir = """module {func.func @entry(%a:tensor<2x3xf32>,%tokens:tensor<1x7xi64>,
            %mask:tensor<1x7xi1>,%count:tensor<i64>) -> tensor<2x3xf32> {return %a : tensor<2x3xf32>}}"""
    sig = static_tensor_signature(mlir, entry="entry")
    assert sig["inputs_dtypes"] == ["f32", "i64", "i1", "i64"]
    assert sig["inputs"][-1] == []
    with pytest.raises(ValueError):
        static_tensor_signature(mlir.replace("tensor<2x3xf32>", "tensor<?x3xf32>"), entry="entry")
