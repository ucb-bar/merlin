"""Bit-exact typed scalar semantics, independent of the target implementation."""

import numpy as np
import pytest
from mlir_fixtures import pointwise

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.targetgen.core_aten_w8a8_reference import evaluate


def source(arguments, result, body):
    inputs = [f"tensor<1x{t}>" for t in arguments]
    return pointwise(inputs, f"tensor<1x{result}>", body, args=[f"a{i}" for i in range(len(inputs))])


def run(module, arrays):
    pytest.importorskip("torch")
    values, _ = evaluate(module, arrays, [], quant_min=-128, quant_max=127, eps=2**-12)
    return values[0].reshape(-1)


@pytest.mark.parametrize(
    "token,np_type,word",
    [
        ("f16", np.float16, 0xFE15),
        ("bf16", np.float32, 0xFFA1),
        ("f32", np.float32, 0xFFA12345),
        ("f64", np.float64, 0xFFF0123456789ABC),
    ],
)
def test_bitcast_preserves_signaling_nan_storage(token, np_type, word):
    width = 16 if token in {"f16", "bf16"} else 32 if token == "f32" else 64
    signed = word - (1 << width)
    integer = f"i{width}"
    module = source(
        [integer], integer, f"%fp=arith.bitcast %a0 : {integer} to {token}\n%v=arith.bitcast %fp : {token} to {integer}"
    )
    value = np.array([signed], dtype={16: np.int16, 32: np.int32, 64: np.int64}[width])
    assert run(module, [value]).item() == signed


@pytest.mark.parametrize("width", [16, 32, 64])
def test_width_correct_logical_shifts_and_bits(width):
    integer = f"i{width}"
    module = source(
        [integer],
        integer,
        f"""%one=arith.constant 1 : {integer}
      %hi=arith.shrui %a0,%one : {integer}
      %lo=arith.shli %a0,%one : {integer}
      %bits=arith.ori %hi,%lo : {integer}
      %v=arith.andi %bits,%hi : {integer}""",
    )
    value = np.array([-1], dtype={16: np.int16, 32: np.int32, 64: np.int64}[width])
    assert run(module, [value]).item() == (1 << (width - 1)) - 1


INTEGER_PREDICATES = ["slt", "sle", "sgt", "sge", "ult", "ule", "ugt", "uge"]


@pytest.mark.parametrize("predicate,expected", [("ult", False), ("ule", False), ("ugt", True), ("uge", True)])
def test_unsigned_i64_predicates(predicate, expected):
    module = source(["i64", "i64"], "i1", f"%v=arith.cmpi {predicate},%a0,%a1 : i64")
    assert run(module, [np.array([-1], np.int64), np.array([0], np.int64)]).item() is expected


@pytest.mark.parametrize(
    "predicate,expected",
    [(p, False) for p in ("false", "oeq", "ord")] + [(p, True) for p in ("ueq ugt uge ult ule une uno true".split())],
)
def test_unordered_float_predicates(predicate, expected):
    module = source(["f32", "f32"], "i1", f"%v=arith.cmpf {predicate},%a0,%a1 : f32")
    assert run(module, [np.array([np.nan], np.float32), np.array([1], np.float32)]).item() is expected


@pytest.mark.parametrize("token,p", [("f32", 24), ("f64", 53)])
def test_fma_has_one_rounding_and_preserves_negative_zero(token, p):
    dtype = np.float32 if token == "f32" else np.float64
    module = source([token] * 3, token, f"%v=math.fma %a0,%a1,%a2 : {token}")
    a, b, c = 1 + 2.0 ** (1 - p), 1 - 2.0 ** (1 - p), -1.0
    expected = -(2.0 ** (2 - 2 * p))
    assert run(module, [np.array([v], dtype) for v in [a, b, c]]).item() == expected
    actual = run(module, [np.array([v], dtype) for v in [-0.0, 1.0, -0.0]])
    torch = pytest.importorskip("torch")
    assert actual.view(torch.int32 if token == "f32" else torch.int64).item() < 0


@pytest.mark.parametrize("token,p", [("f32", 24), ("f64", 53)])
def test_unsigned_i64_conversion_avoids_double_rounding(token, p):
    dtype = np.float32 if token == "f32" else np.float64
    module = source(["i64"], token, f"%v=arith.uitofp %a0 : i64 to {token}")
    midpoint = (1 << 63) + (1 << (63 - p))
    for value in [midpoint - 1, midpoint, midpoint + 1, (1 << 64) - 1]:
        signed = value - (1 << 64)
        actual = run(module, [np.array([signed], np.int64)]).item()
        expected = (1 << 63) if value <= midpoint else (1 << 63) + (1 << (64 - p))
        if value == (1 << 64) - 1:
            expected = 1 << 64
        assert actual == float(expected)
        assert actual == dtype(expected)


def test_nan_select_keeps_the_selected_payload():
    module = source(
        ["f32", "f32"],
        "f32",
        """%sum=arith.addf %a0,%a1 : f32
      %nan=arith.cmpf uno,%sum,%sum : f32
      %v=arith.select %nan,%a0,%sum : f32""",
    )
    words = [np.array([word], np.uint32).view(np.float32) for word in [0x7FC00001, 0xFFC00002]]
    assert run(module, words).view(__import__("torch").int32).item() == 0x7FC00001


def test_nan_select_follows_masks_across_tensor_regions():
    module = parse_mlir_text("""module {
      func.func @forward(%x:tensor<1xf32>,%y:tensor<1xf32>) -> tensor<1xf32> {
        %e=tensor.empty() : tensor<1xf32>
        %sum=linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>,
          affine_map<(d0)->(d0)>],iterator_types=["parallel"]}
          ins(%x,%y:tensor<1xf32>,tensor<1xf32>) outs(%e:tensor<1xf32>) {
          ^bb0(%a:f32,%b:f32,%unused:f32):
            %s=arith.addf %a,%b : f32
            linalg.yield %s : f32
        } -> tensor<1xf32>
        %r=linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>,
          affine_map<(d0)->(d0)>],iterator_types=["parallel"]}
          ins(%sum,%x:tensor<1xf32>,tensor<1xf32>) outs(%e:tensor<1xf32>) {
          ^bb1(%s:f32,%a:f32,%unused:f32):
            %nan=arith.cmpf uno,%s,%s : f32
            %v=arith.select %nan,%a,%s : f32
            linalg.yield %v : f32
        } -> tensor<1xf32>
        return %r : tensor<1xf32>
      }}""")
    words = [np.array([word], np.uint32).view(np.float32) for word in [0x7FC00001, 0xFFC00002]]
    result = run(module, words)
    assert result.view(__import__("torch").int32).item() == 0x7FC00001


def test_scalar_coefficients_outside_tensor_regions_use_the_same_semantics():
    module = parse_mlir_text("""module {
      func.func @forward() -> tensor<1xf16> {
        %integer=arith.constant -9223371487098961919 : i64
        %fp=arith.uitofp %integer : i64 to f32
        %one=arith.constant 1.0 : f32
        %minus=arith.constant -1.0 : f32
        %combined=math.fma %one,%one,%minus : f32
        %half=arith.truncf %combined : f32 to f16
        %out=tensor.splat %half : tensor<1xf16>
        return %out : tensor<1xf16>
      }}""")
    # Keep the unsigned conversion live as a second output, so neither scalar
    # conversion is silently skipped by source interpretation.
    from xdsl.dialects import func
    from xdsl.dialects.builtin import FunctionType, TensorType, f32
    from xdsl.dialects.tensor import SplatOp
    from xdsl.rewriter import Rewriter

    function = next(op for op in module.walk() if op.name == "func.func")
    converted = next(op for op in module.walk() if op.name == "arith.uitofp").results[0]
    splat = SplatOp(converted, [], TensorType(f32, [1]))
    original = function.body.block.last_op.operands[0]
    Rewriter.replace_op(function.body.block.last_op, [splat, func.ReturnOp(original, splat.result)])
    function.properties["function_type"] = FunctionType.from_lists([], [original.type, splat.result.type])
    pytest.importorskip("torch")
    values, _ = evaluate(module, [], [], quant_min=-128, quant_max=127, eps=2**-12)
    assert values[0].item() == 0.0
    assert values[1].item() == float((1 << 63) + (1 << 40))


@pytest.mark.parametrize("value", [2, 3, -2, -1])
def test_truncation_to_i1_keeps_only_the_low_bit(value):
    module = source(["i8"], "i1", "%v=arith.trunci %a0 : i8 to i1")
    assert run(module, [np.array([value], np.int8)]).item() is bool(value & 1)


@pytest.mark.parametrize("operation,result", [("arith.extsi", "i8"), ("arith.sitofp", "f32")])
def test_signed_i1_conversion_sign_extends(operation, result):
    module = source(["i1"], result, f"%v={operation} %a0 : i1 to {result}")
    assert run(module, [np.array([True], np.bool_)]).item() == -1


@pytest.mark.parametrize("operation", ["addi", "subi"])
def test_i1_integer_arithmetic_wraps(operation):
    module = source(["i1", "i1"], "i1", f"%v=arith.{operation} %a0,%a1 : i1")
    assert run(module, [np.array([True], np.bool_)] * 2).item() is False


@pytest.mark.parametrize("predicate,expected", [(p, p in {"slt", "sle", "ugt", "uge"}) for p in INTEGER_PREDICATES])
def test_i1_ordering_uses_the_predicate_signedness(predicate, expected):
    module = source(["i1", "i1"], "i1", f"%v=arith.cmpi {predicate},%a0,%a1 : i1")
    assert run(module, [np.array([True], np.bool_), np.array([False], np.bool_)]).item() is expected


@pytest.mark.parametrize("token,p", [("f16", 11), ("bf16", 8)])
def test_direct_reduced_cast_rounds_once_and_explicit_f32_bridge_twice(token, p):
    direct = source(["f64"], token, f"%v=arith.truncf %a0 : f64 to {token}")
    bridge = source(["f64"], token, f"%f=arith.truncf %a0 : f64 to f32\n%v=arith.truncf %f : f32 to {token}")
    value = np.array([1.0 + 2.0**-p + 2.0**-30], np.float64)
    assert run(direct, [value]).item() == 1.0 + 2.0 ** (1 - p)
    assert run(bridge, [value]).item() == 1.0


@pytest.mark.parametrize("token,p,exponent", [("f16", 11, 15), ("bf16", 8, 63)])
def test_unsigned_integer_to_reduced_storage_has_one_rounding(token, p, exponent):
    module = source(["i64"], token, f"%v=arith.uitofp %a0 : i64 to {token}")
    midpoint = (1 << exponent) + (1 << (exponent - p))
    for value in [midpoint - 1, midpoint, midpoint + 1]:
        word = value - (1 << 64) if value >= (1 << 63) else value
        expected = (1 << exponent) if value <= midpoint else (1 << exponent) + (1 << (exponent + 1 - p))
        assert run(module, [np.array([word], np.int64)]).item() == float(expected)


@pytest.mark.parametrize("token", ["f16", "bf16", "f32", "f64"])
def test_ieee_conversion_keeps_zero_sign(token):
    torch = pytest.importorskip("torch")
    module = source(
        ["f64"], token, f"%v=arith.truncf %a0 : f64 to {token}" if token != "f64" else "%v=arith.negf %a0 : f64"
    )
    value = np.array([-0.0 if token != "f64" else 0.0], np.float64)
    assert torch.signbit(run(module, [value])).item() is True


def test_nested_static_control_selects_the_untaken_branch_payload():
    module = source(
        ["f32", "f32"],
        "f32",
        """%lo = arith.constant 0 : index
        %hi = arith.constant 1 : index
        %loop = scf.for %i = %lo to %hi step %hi iter_args(%acc = %a0) -> (f32) {
          %sum = arith.addf %acc, %a1 : f32
          scf.yield %sum : f32
        }
        %condition = arith.constant 0 : i1
        %v = scf.if %condition -> (f32) {
          scf.yield %loop : f32
        } else {
          scf.yield %a0 : f32
        }""",
    )
    left = np.array([0x7FC00001], dtype=np.uint32).view(np.float32)
    right = np.array([0x7FC00002], dtype=np.uint32).view(np.float32)
    assert run(module, [left, right]).numpy().tobytes() == left.tobytes()


def test_unsigned_byte_conversion_has_no_sign_extension():
    module = source(["i8"], "f32", "%v=arith.uitofp %a0 : i8 to f32")
    assert run(module, [np.array([-1], np.int8)]).item() == 255.0
