"""Typed scalar SSA, exact binary64 checks and explicit source limitations."""

from dataclasses import replace

import numpy as np
import pytest

from merlin.semantic_compiler.linalg_bridge import translate_linalg_text
from merlin.semantic_compiler.reference import TensorValue, evaluate_graph


def program(inputs, result, body, size=8):
    tensors = [f"tensor<{size}x{dtype}>" for dtype in inputs]
    output = f"tensor<{size}x{result}>"
    args = ", ".join(f"%arg{i}: {t}" for i, t in enumerate(tensors))
    values = ", ".join(f"%arg{i}" for i in range(len(inputs)))
    types = ", ".join(tensors)
    maps = ", ".join(["affine_map<(d) -> (d)>"] * (len(inputs) + 1))
    scalar_args = ", ".join([*(f"%x{i}: {t}" for i, t in enumerate(inputs)), f"%unused: {result}"])
    return f"""module {{
      func.func @forward({args}) -> {output} {{
        %empty = tensor.empty() : {output}
        %result = linalg.generic {{indexing_maps = [{maps}], iterator_types = ["parallel"]}}
          ins({values} : {types}) outs(%empty : {output}) {{
        ^bb0({scalar_args}):
          {body}
          linalg.yield %value : {result}
        }} -> {output}
        return %result : {output}
      }}
    }}"""


def translate(source):
    return translate_linalg_text(source, entry="forward", target_identity="synthetic")


@pytest.mark.parametrize("operation,constant", (("xori", "true"), ("andi", "false"), ("andi", "true")))
def test_boolean_literal_logic_preserves_exact_mask_bits(operation, constant):
    body = f"""%constant = arith.constant {constant}
      %value = arith.{operation} %x0, %constant : i1"""
    translated = translate(program(("i1",), "i1", body))
    bits = (0, 1, 1, 0, 1, 0, 0, 1)
    literal = int(constant == "true")
    expected = tuple((x ^ literal) if operation == "xori" else (x & literal) for x in bits)
    assert execute(translated, bits) == expected
    assert translated.request.nodes[-1].op == "mask"


def test_boolean_multiplication_is_exactly_conjunction():
    translated = translate(program(("i1", "i1"), "i1", "%value = arith.muli %x0, %x1 : i1"))
    assert execute(translated, (0, 0, 1, 1) * 2, (0, 1, 0, 1) * 2) == (0, 0, 0, 1) * 2


@pytest.mark.parametrize("operation", ("add", "sub", "mul", "div"))
def test_source_bf16_arithmetic_matches_independent_cpu_bf16(operation):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    rng = np.random.default_rng(129)
    left = rng.integers(0, 1 << 16, size=512, dtype=np.uint16)
    right = rng.integers(0, 1 << 16, size=512, dtype=np.uint16)
    left[:12] = [0, 0x8000, 0x3F80, 1, 0x80, 0x7F80, 0, 0x7F7F, 0x3F81, 0xBF80, 0x7F80, 0x8001]
    right[:12] = [0x8000, 0x8000, 0xBF80, 2, 0x3F00, 0x3F80, 0, 0x4000, 0x3F80, 0, 0xFF80, 0x3F00]
    function = {"add": torch.add, "sub": torch.sub, "mul": torch.mul, "div": torch.div}[operation]
    output = function(torch.from_numpy(left).view(torch.bfloat16), torch.from_numpy(right).view(torch.bfloat16))
    expected = output.view(torch.uint16).numpy()
    valid = ((expected & 0x7F80) != 0x7F80) | ((expected & 0x7F) == 0)
    translated = translate(
        program(("bf16", "bf16"), "bf16", f"%value = arith.{operation}f %x0, %x1 : bf16", int(valid.sum()))
    )
    assert execute(translated, left[valid], right[valid]) == tuple(map(int, expected[valid]))
    assert translated.request.nodes[-1].op == f"bf16_{operation}"
    assert dict(translated.request.nodes[-1].attrs) == {"numeric_semantics": f"ieee754-bf16-{operation}-rne-v1"}
    if not valid.all():
        one = translate(program(("bf16", "bf16"), "bf16", f"%value = arith.{operation}f %x0, %x1 : bf16", 1))
        at = int(np.flatnonzero(~valid)[0])
        with pytest.raises(ValueError, match="NaN"):
            execute(one, left[at : at + 1], right[at : at + 1])


def test_large_projected_select_stays_symbolic_and_broadcasts_a_scalar_tensor():
    source = """module {
      func.func @forward(%condition: tensor<2x1xi1>, %scalar: tensor<f32>,
                         %input: tensor<2x3xf32>) -> tensor<2x3xf32> {
        %empty = tensor.empty() : tensor<2x3xf32>
        %result = linalg.generic {indexing_maps = [affine_map<(a,b)->(a,0)>,
          affine_map<(a,b)->()>, affine_map<(a,b)->(a,b)>, affine_map<(a,b)->(a,b)>],
          iterator_types = ["parallel", "parallel"]}
          ins(%condition, %scalar, %input : tensor<2x1xi1>, tensor<f32>, tensor<2x3xf32>)
          outs(%empty : tensor<2x3xf32>) {
        ^bb0(%c: i1, %s: f32, %x: f32, %unused: f32):
          %selected = arith.select %c, %s, %x : f32
          linalg.yield %selected : f32
        } -> tensor<2x3xf32>
        return %result : tensor<2x3xf32>
      }
    }"""
    translated = translate(source)
    bits = (0, 0x80000000, 1, 0x7FC00033, 0x7F800000, 0xFF800000)
    assert execute(translated, (1, 0), (0x7FC01234,), bits) == (0x7FC01234,) * 3 + bits[3:]
    large = translate(source.replace("2x3", "2x2000000"))
    assert len(large.request.nodes) == 6
    assert not large.constants  # No expanded mask or scalar tensor is materialized.
    assert large.request.nodes[-1].type.shape == (2, 2000000)


def test_large_literal_expansion_retains_a_resource_limit():
    from merlin.semantic_compiler.target_binding import NativeCompilationError

    source = program(("i1",), "i1", "%one = arith.constant true\n%value = arith.xori %x0, %one : i1", 2000000)
    with pytest.raises(NativeCompilationError, match="materialized element budget"):
        translate(source)


def execute(translated, *arrays):
    request = translated.request
    return evaluate_graph(
        request,
        {
            f"arg{i}": TensorValue(request.node(f"arg{i}").type, tuple(map(int, array)))
            for i, array in enumerate(arrays)
        },
        constants=translated.constants,
    )[0].elements


def test_ordered_clamp_preserves_nan_payloads_and_equal_signed_zero():
    body = """%limit = arith.constant -0.0 : f32
      %above = arith.cmpf ogt, %x0, %limit : f32
      %value = arith.select %above, %limit, %x0 : f32"""
    translated = translate(program(("f32",), "f32", body))
    bits = (0, 0x80000000, 0x3F800000, 0xBF800000, 0x7F800000, 0xFF800000, 0x7FC01234, 1)
    assert execute(translated, bits) == (
        0,
        0x80000000,
        0x80000000,
        0xBF800000,
        0x80000000,
        0xFF800000,
        0x7FC01234,
        0x80000000,
    )
    assert [n.op for n in translated.request.nodes[1:]] == ["constant", "mask", "select"]
    assert type(translated.request).from_record(translated.request.record()).digest() == translated.request.digest()


@pytest.mark.parametrize("operation", ("add", "sub", "mul", "div"))
def test_binary64_matches_native_rounding_on_seeded_bits_and_boundaries(operation):
    rng = np.random.default_rng(837)
    left = rng.integers(0, 1 << 64, size=256, dtype=np.uint64)
    right = rng.integers(0, 1 << 64, size=256, dtype=np.uint64)
    # Add cancellation, both zeros, infinities and the normal/subnormal boundary.
    left[:8] = [0, 1 << 63, 0x3FF0000000000000, 1, 0x0010000000000000, 0x7FF0000000000000, 0, 0x7FEFFFFFFFFFFFFF]
    right[:8] = [1 << 63, 1 << 63, 0xBFF0000000000000, 2, 0x3FE0000000000000, 0x3FF0000000000000, 0, 0x4000000000000000]
    with np.errstate(all="ignore"):
        expected = {"add": np.add, "sub": np.subtract, "mul": np.multiply, "div": np.divide}[operation](
            left.view(np.float64), right.view(np.float64)
        )
    valid = ~np.isnan(expected)
    source = program(("f64", "f64"), "f64", f"%value = arith.{operation}f %x0, %x1 : f64", int(valid.sum()))
    translated = translate(source)
    assert execute(translated, left[valid], right[valid]) == tuple(map(int, expected[valid].view(np.uint64)))
    if not valid.all():
        one = translate(program(("f64", "f64"), "f64", f"%value = arith.{operation}f %x0, %x1 : f64", 1))
        at = int(np.flatnonzero(~valid)[0])
        with pytest.raises(ValueError, match="NaN"):
            execute(one, left[at : at + 1], right[at : at + 1])


def test_float_casts_preserve_width_rounding_and_signed_zero():
    values = np.array([0, 1, -1, (1 << 53) + 1, (1 << 53) + 3, -(1 << 63), (1 << 63) - 1, 7], dtype=np.int64)
    cast = translate(program(("i64",), "f64", "%value = arith.sitofp %x0 : i64 to f64"))
    assert execute(cast, values) == tuple(map(int, values.astype(np.float64).view(np.uint64)))
    doubles = np.array(
        [0.0, -0.0, 1.0 + 2.0**-24, 1.0 + 3 * 2.0**-24, 2.0**-149, 2.0**-150, float("inf"), -1.0e300], dtype=np.float64
    )
    narrow = translate(program(("f64",), "f32", "%value = arith.truncf %x0 : f64 to f32"))
    with np.errstate(over="ignore"):
        expected = doubles.astype(np.float32).view(np.uint32)
    assert execute(narrow, doubles.view(np.uint64)) == tuple(map(int, expected))
    nodes = tuple(
        replace(n, attrs=(("numeric_semantics", "wrong-rounding"),)) if n.op == "f64_to_f32" else n
        for n in narrow.request.nodes
    )
    with pytest.raises(ValueError, match="policy"):
        execute(replace(narrow, request=replace(narrow.request, nodes=nodes)), doubles.view(np.uint64))


def test_mixed_bf16_f32_region_retains_widening_before_multiplication():
    translated = translate(
        program(
            ("bf16", "f32"),
            "f32",
            """%wide = arith.extf %x0 : bf16 to f32
      %value = arith.mulf %wide, %x1 : f32""",
        )
    )
    source = np.array([0x3F80, 0xBF80, 0, 0x8000, 0x3F81, 0x3F7F, 1, 0x7F7F], dtype=np.uint16)
    factors = np.array([1, -2, 3, -4, 1.5, -2.5, 1, 0.5], dtype=np.float32)
    expected = (source.astype(np.uint32) << 16).view(np.float32) * factors
    assert execute(translated, source, factors.view(np.uint32)) == tuple(map(int, expected.view(np.uint32)))
    assert [node.op for node in translated.request.nodes[-2:]] == ["bf16_to_f32", "f32_mul"]


def test_signed_i64_pointwise_cast_keeps_its_own_f32_rounding():
    source = np.array([0, -1, (1 << 24) + 1, (1 << 24) + 3, -(1 << 63), (1 << 63) - 1, 1, -7], dtype=np.int64)
    translated = translate(program(("i64",), "f32", "%value = arith.sitofp %x0 : i64 to f32"))
    assert execute(translated, source) == tuple(map(int, source.astype(np.float32).view(np.uint32)))


@pytest.mark.parametrize("operation,inputs", (("sin", ("f32",)), ("cos", ("f32",)), ("powf", ("f32", "f32"))))
def test_positional_math_preserves_source_type_without_granting_an_implementation(operation, inputs):
    operands = "%x0, %x1" if len(inputs) == 2 else "%x0"
    translated = translate(program(inputs, "f32", f"%value = math.{operation} {operands} : f32"))
    graph = translated.request
    node = graph.nodes[-1]
    assert node.op == f"source_f32_{operation}_unqualified"
    assert dict(node.attrs) == {"numeric_semantics": f"source-f32-{operation}-unqualified-v1"}
    assert node.inputs == tuple(f"arg{i}" for i in range(len(inputs)))
    assert type(graph).from_record(graph.record()).digest() == graph.digest()
    with pytest.raises(ValueError):
        execute(translated, *((0,) * 8 for _ in inputs))


def test_bf16_cast_and_float_to_integer_keep_their_own_rounding():
    values = (0, 0x80000000, 0x3F808000, 0x3F818000, 0x00008000, 0x00008001, 0xBF818000, 0x7F800000)
    cast = translate(program(("f32",), "bf16", "%value = arith.truncf %x0 : f32 to bf16"))
    assert execute(cast, values) == (0, 0x8000, 0x3F80, 0x3F82, 0, 1, 0xBF82, 0x7F80)
    integers = translate(program(("f64",), "i64", "%value = arith.fptosi %x0 : f64 to i64", 3))
    assert execute(integers, np.array([-2.9, 0.0, 3.1], dtype=np.float64).view(np.uint64)) == (-2, 0, 3)
    with pytest.raises(ValueError, match="defined domain"):
        execute(integers, np.array([0.0, float("inf"), 0.0], dtype=np.float64).view(np.uint64))


def test_binary64_constants_and_unqualified_intrinsics_are_explicit():
    source = """module {
      func.func @forward() -> tensor<2xf64> {
        %negative_zero = arith.constant -0.0 : f64
        %values = tensor.splat %negative_zero : tensor<2xf64>
        return %values : tensor<2xf64>
      }
    }"""
    translated = translate(source)
    assert translated.request.constants[0].encoding == "f64-le-raw"
    assert execute(translated) == (1 << 63, 1 << 63)
    pow_ = translate(program(("f64", "f64"), "f64", "%value = math.powf %x0, %x1 : f64", 1))
    assert pow_.request.nodes[-1].op == "source_f64_powf_unqualified"
    with pytest.raises(ValueError, match="source"):
        execute(pow_, (0x4000000000000000,), (0x4008000000000000,))


def test_projected_operands_and_integer_wrap_preserve_source_geometry():
    source = """module {
      func.func @forward(%rows: tensor<2x1xf64>, %columns: tensor<1x3xf64>) -> tensor<2x3xf64> {
        %empty = tensor.empty() : tensor<2x3xf64>
        %sum = linalg.generic {indexing_maps = [affine_map<(i,j)->(i,0)>,
          affine_map<(i,j)->(0,j)>, affine_map<(i,j)->(i,j)>], iterator_types = ["parallel", "parallel"]}
          ins(%rows, %columns : tensor<2x1xf64>, tensor<1x3xf64>) outs(%empty : tensor<2x3xf64>) {
        ^bb0(%a: f64, %b: f64, %unused: f64):
          %next = arith.addf %a, %b : f64
          linalg.yield %next : f64
        } -> tensor<2x3xf64>
        return %sum : tensor<2x3xf64>
      }
    }"""
    translated = translate(source)
    a, b = np.array([2.0, -4.0]), np.array([1.0, 2.0, 8.0])
    assert execute(translated, a.view(np.uint64), b.view(np.uint64)) == tuple(
        map(int, (a[:, None] + b[None, :]).ravel().view(np.uint64))
    )
    assert sum(n.op == "gather" for n in translated.request.nodes) == 2
    subtract = translate(program(("i64", "i64"), "i64", "%value = arith.subi %x0, %x1 : i64", 2))
    assert execute(subtract, (-(1 << 63), 7), (1, 8)) == ((1 << 63) - 1, -1)


@pytest.mark.parametrize("literal,expected", [("true", (1, 1, 1)), ("false", (0, 0, 0))])
def test_boolean_constants_use_canonical_logical_bytes(literal, expected):
    source = f"""module {{func.func @forward() -> tensor<3xi1> {{
      %value = arith.constant {literal}
      %tensor = tensor.splat %value : tensor<3xi1>
      return %tensor : tensor<3xi1>
    }}}}"""
    result = translate(source)
    assert execute(result) == expected
    assert result.request.constants[0].data_hex == bytes(expected).hex()
    assert type(result.request).from_record(result.request.record()).digest() == result.request.digest()
    dense = translate("""module {func.func @forward() -> tensor<3xi1> {
      %values = arith.constant dense<[true, false, true]> : tensor<3xi1>
      return %values : tensor<3xi1>
    }}""")
    assert execute(dense) == (1, 0, 1)


def test_boolean_projection_preserves_each_runtime_mask_bit():
    source = """module {func.func @forward(%mask: tensor<1x1x3xi1>) -> tensor<1x2x3xi1> {
      %empty = tensor.empty() : tensor<1x2x3xi1>
      %expanded = linalg.generic {indexing_maps = [affine_map<(a,b,c)->(a,0,c)>,
        affine_map<(a,b,c)->(a,b,c)>], iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%mask : tensor<1x1x3xi1>) outs(%empty : tensor<1x2x3xi1>) {
      ^bb0(%value: i1, %unused: i1):
        linalg.yield %value : i1
      } -> tensor<1x2x3xi1>
      return %expanded : tensor<1x2x3xi1>
    }}"""
    translated = translate(source)
    assert execute(translated, (1, 0, 1)) == (1, 0, 1, 1, 0, 1)


def test_single_operand_concat_is_a_checked_identity():
    source = """module {func.func @forward(%x: tensor<2x3xf32>) -> tensor<2x3xf32> {
      %result = tensor.concat dim(1) %x : (tensor<2x3xf32>) -> tensor<2x3xf32>
      return %result : tensor<2x3xf32>
    }}"""
    result = translate(source)
    bits = (0, 0x80000000, 0x7FC01234, 1, 0x7F800000, 0xBF800000)
    assert execute(result, bits) == bits
    assert result.request.nodes[-1].op == "identity"
    with pytest.raises(ValueError):
        translate(source.replace("dim(1)", "dim(2)"))


def test_rank_zero_integer_reduction_preserves_the_scalar_accumulator_map():
    source = """module {func.func @forward(%x: tensor<4xi64>) -> tensor<i64> {
      %zero = arith.constant 0 : i64
      %init = tensor.splat %zero : tensor<i64>
      %sum = linalg.reduce ins(%x : tensor<4xi64>) outs(%init : tensor<i64>)
        dimensions = [0] (%value: i64, %acc: i64) {
          %next = arith.addi %value, %acc : i64
          linalg.yield %next : i64
        }
      return %sum : tensor<i64>
    }}"""
    translated = translate(source)
    assert execute(translated, ((1 << 63) - 1, 1, -3, 4)) == (-(1 << 63) + 1,)
    assert translated.request.nodes[-1].index_maps[-1].coefficients == ()
    assert type(translated.request).from_record(translated.request.record()).digest() == translated.request.digest()


@pytest.mark.parametrize("right,predicate", [(False, "ult"), (True, "ule")])
def test_literal_bucketize_count_handles_duplicate_limits_nan_and_infinities(right, predicate):
    source = f"""module {{func.func @forward(%input: tensor<8xf32>) -> tensor<8xi64> {{
      %limits = arith.constant dense<[0.0, 0.25, 0.25, 1.0]> : tensor<4xf32>
      %zero = arith.constant 0 : i64
      %one = arith.constant 1 : i64
      %init = tensor.splat %zero : tensor<8xi64>
      %result = linalg.generic {{indexing_maps = [affine_map<(i,k)->(i)>,
        affine_map<(i,k)->(k)>, affine_map<(i,k)->(i)>], iterator_types = ["parallel", "reduction"]}}
        ins(%input, %limits : tensor<8xf32>, tensor<4xf32>) outs(%init : tensor<8xi64>) {{
      ^bb0(%value: f32, %boundary: f32, %acc: i64):
        %less = arith.cmpf {predicate}, %boundary, %value : f32
        %increment = arith.select %less, %one, %zero : i64
        %next = arith.addi %acc, %increment : i64
        linalg.yield %next : i64
      }} -> tensor<8xi64>
      return %result : tensor<8xi64>
    }}}}"""
    values = np.array([-np.inf, -0.0, 0.0, 0.25, 0.5, 1.0, np.inf, np.nan], dtype=np.float32)
    result = translate(source)
    expected = np.searchsorted(
        np.array([0.0, 0.25, 0.25, 1.0], dtype=np.float32), values, side="right" if right else "left"
    )
    assert execute(result, values.view(np.uint32)) == tuple(map(int, expected))
    assert dict(result.request.nodes[-1].attrs)["right"] is right
    for old, new in (
        ("0.0, 0.25, 0.25, 1.0", "0.0, 1.0, 0.25, 0.25"),
        ("arith.constant 0 : i64", "arith.constant 1 : i64"),
        (f"cmpf {predicate}", "cmpf olt"),
    ):
        with pytest.raises(ValueError):
            translate(source.replace(old, new))


@pytest.mark.parametrize(
    "body",
    (
        "%value = arith.addf %x0, %unused : f64",
        "%value = arith.addf %x0, %x0 fastmath<reassoc> : f64",
        "%value = arith.maximumf %x0, %x0 : f64",
    ),
)
def test_unknown_or_uninitialized_scalar_computation_is_refused(body):
    with pytest.raises(ValueError):
        translate(program(("f64",), "f64", body))
