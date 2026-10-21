"""Exact mask conversion and bounded compile-time index generation."""

import pytest

from merlin.semantic_compiler.linalg_bridge import translate_linalg_text
from merlin.semantic_compiler.reference import TensorValue, evaluate_graph
from merlin.semantic_compiler.target_binding import NativeCompilationError


def signed_reciprocal_source(size=12):
    return f"""module {{
      func.func @forward(%input: tensor<{size}xi64>) -> tensor<{size}xf32> {{
        %empty = tensor.empty() : tensor<{size}xf32>
        %result = linalg.generic {{indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
          iterator_types = ["parallel"]}} ins(%input : tensor<{size}xi64>) outs(%empty : tensor<{size}xf32>) {{
        ^bb0(%value: i64, %unused: f32):
          %cast = arith.sitofp %value : i64 to f32
          %one = arith.constant 1.0 : f32
          %inverse = arith.divf %one, %cast : f32
          linalg.yield %inverse : f32
        }} -> tensor<{size}xf32>
        return %result : tensor<{size}xf32>
      }}
    }}"""


def test_signed_reciprocal_preserves_both_roundings_and_zero_division():
    from dataclasses import replace

    import numpy as np

    integers = (
        0,
        1,
        -1,
        3,
        (1 << 24) + 1,
        (1 << 24) + 3,
        -(1 << 24) - 3,
        (1 << 62) + (1 << 38) - 1,
        (1 << 62) + (1 << 38) + 1,
        -(1 << 62) - (1 << 38) - 1,
        -(1 << 63),
        (1 << 63) - 1,
    )
    translated = translate_linalg_text(signed_reciprocal_source(), entry="forward", target_identity="synthetic")
    request = translated.request
    cast = next(node for node in request.nodes if node.op == "signed_i64_to_f32")
    inverse, rounded = evaluate_graph(
        replace(request, outputs=(*request.outputs, cast.id), output_storages=("external", "external")),
        {"arg0": TensorValue(request.node("arg0").type, integers)},
        constants=translated.constants,
    )
    native_cast = np.array(integers, dtype=np.int64).astype(np.float32)
    with np.errstate(divide="ignore"):
        native_inverse = np.float32(1) / native_cast
    assert rounded.elements == tuple(map(int, native_cast.view(np.uint32)))
    assert inverse.elements == tuple(map(int, native_inverse.view(np.uint32)))
    # This integer distinguishes direct rounding from an intermediate f64 cast.
    assert rounded.elements[8] != np.array(float(integers[8]), dtype=np.float32).view(np.uint32).item()
    restored = type(request).from_record(request.record())
    assert restored.digest() == request.digest()


@pytest.mark.parametrize(
    "old,new",
    [
        ("constant 1.0", "constant 2.0"),
        ("arith.sitofp", "arith.uitofp"),
        ("%one, %cast", "%cast, %one"),
        ("arith.divf %one, %cast", "arith.divf %one, %cast fastmath<fast>"),
    ],
)
def test_signed_reciprocal_rejects_different_scalar_semantics(old, new):
    with pytest.raises(ValueError):
        translate_linalg_text(
            signed_reciprocal_source().replace(old, new), entry="forward", target_identity="synthetic"
        )


def iota_source(size=5, start=7, step=-2):
    return f"""module {{
      func.func @forward() -> tensor<{size}xi64> {{
        %empty = tensor.empty() : tensor<{size}xi64>
        %result = linalg.generic {{indexing_maps = [affine_map<(i) -> (i)>],
          iterator_types = ["parallel"]}} outs(%empty : tensor<{size}xi64>) {{
        ^bb0(%unused: i64):
          %index = linalg.index 0 : index
          %wide = arith.index_cast %index : index to i64
          %step = arith.constant {step} : i64
          %scaled = arith.muli %wide, %step : i64
          %start = arith.constant {start} : i64
          %value = arith.addi %start, %scaled : i64
          linalg.yield %value : i64
        }} -> tensor<{size}xi64>
        return %result : tensor<{size}xi64>
      }}
    }}"""


def test_static_sequence_materializes_declared_constants_with_signed_steps():
    translated = translate_linalg_text(iota_source(), entry="forward", target_identity="synthetic")
    request = translated.request
    assert len(request.constants) == 1
    assert tuple(value.elements for value in evaluate_graph(request, {}, constants=translated.constants)) == (
        (7, 5, 3, 1, -1),
    )
    assert len(bytes.fromhex(request.constants[0].data_hex)) == 40


def test_sequence_materialization_budget_is_not_hardware_unsupported():
    with pytest.raises(NativeCompilationError) as error:
        translate_linalg_text(iota_source(size=(1 << 20) + 1), entry="forward", target_identity="synthetic")
    assert error.value.status == "resource_limit"


@pytest.mark.parametrize(
    "type_,literal,expected",
    [
        ("tensor<3xf32>", "dense<[3.125000e-02, -0.0, 1.0]>", (0x3D000000, 0x80000000, 0x3F800000)),
        ("tensor<f32>", "dense<-0.0>", (0x80000000,)),
        ("tensor<2xi64>", "dense<[-9223372036854775808, 9223372036854775807]>", (-(1 << 63), (1 << 63) - 1)),
    ],
)
def test_dense_tensor_literals_preserve_declared_bytes_and_signed_zero(type_, literal, expected):
    text = f"""module {{
      func.func @forward() -> {type_} {{
        %values = arith.constant {literal} : {type_}
        return %values : {type_}
      }}
    }}"""
    translated = translate_linalg_text(text, entry="forward", target_identity="synthetic")
    assert evaluate_graph(translated.request, {}, constants=translated.constants)[0].elements == expected


def test_integer_reduction_preserves_signed_wraparound_and_explicit_seed():
    text = """module {
      func.func @forward(%input: tensor<2x3xi64>) -> tensor<2xi64> {
        %zero = arith.constant 0 : i64
        %init = tensor.splat %zero : tensor<2xi64>
        %sum = linalg.reduce ins(%input : tensor<2x3xi64>) outs(%init : tensor<2xi64>)
          dimensions = [1] (%value: i64, %acc: i64) {
            %next = arith.addi %value, %acc : i64
            linalg.yield %next : i64
          }
        return %sum : tensor<2xi64>
      }
    }"""
    translated = translate_linalg_text(text, entry="forward", target_identity="synthetic")
    request = translated.request
    inputs = {"arg0": TensorValue(request.node("arg0").type, ((1 << 63) - 1, 1, 0, -5, -6, -7))}
    assert evaluate_graph(request, inputs, constants=translated.constants)[0].elements == (-(1 << 63), -18)
    with pytest.raises(ValueError, match="zero"):
        translate_linalg_text(text.replace("constant 0", "constant 1"), entry="forward", target_identity="synthetic")


def test_f32_truth_conversion_preserves_nan_subnormal_and_signed_zero_semantics():
    source = """module {
      func.func @forward(%input: tensor<8xf32>) -> tensor<8xi1> {
        %empty = tensor.empty() : tensor<8xi1>
        %result = linalg.generic {indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
          iterator_types = ["parallel"]} ins(%input : tensor<8xf32>) outs(%empty : tensor<8xi1>) {
        ^bb0(%value: f32, %unused: i1):
          %zero = arith.constant 0.0 : f32
          %truth = arith.cmpf une, %value, %zero : f32
          linalg.yield %truth : i1
        } -> tensor<8xi1>
        return %result : tensor<8xi1>
      }
    }"""
    translated = translate_linalg_text(source, entry="forward", target_identity="synthetic")
    request = translated.request
    bits = (0, 0x80000000, 0x3F800000, 0xBF800000, 0x7F800000, 0x7FC00000, 1, 0x80000001)
    result = evaluate_graph(request, {"arg0": TensorValue(request.node("arg0").type, bits)})
    assert result[0].elements == (0, 0, 1, 1, 1, 1, 1, 1)
    with pytest.raises(ValueError):
        translate_linalg_text(source.replace("cmpf une", "cmpf one"), entry="forward", target_identity="synthetic")
