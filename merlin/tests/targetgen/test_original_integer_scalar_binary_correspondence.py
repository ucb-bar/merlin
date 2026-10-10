"""Synthetic exact integer IR and registry controls confer no native credit."""

import copy
from fractions import Fraction

import pytest
from test_original_integer_scalar_binary_sources import declared_form
from test_original_scalar_binary_correspondence import SOURCES, checked, fixture
from test_original_scalar_binary_sources import declarations
from xdsl.dialects import arith
from xdsl.dialects.builtin import FloatAttr, IntegerAttr, f32, i64

from merlin.targetgen import original_scalar_binary_correspondence as C
from merlin.targetgen import original_scalar_binary_sources as S
from merlin.targetgen.frontend_trace import _digest


def integer_fixture(target="aten.div.Tensor", scalar=2**60 + 2**36 + 1):
    _, _, module, trace, registry = fixture(target, 1.0)
    original = declared_form(target, scalar)
    source = S.scalar_binary_source(original, extent=2, max_tensor_elements=100)
    block = next(iter(module.body.block.ops)).body.block
    old = next(iter(block.ops))
    constant = arith.ConstantOp.from_int_and_width(scalar, 64)
    converted = arith.SIToFPOp(constant.result, f32)
    for operation in (constant, converted):
        operation.attributes.update(copy.deepcopy(old.attributes))
    block.insert_ops_before([constant, converted], old)
    old.result.replace_all_uses_with(converted.result)
    block.erase_op(old)
    for graph in trace["graphs"].values():
        graph["nodes"][1]["args"][1] = scalar
        graph.pop("sha256")
        graph["sha256"] = _digest(graph)
    registry["schema"] = C.INTEGER_REGISTRY_SCHEMA
    registry["events"][0]["literal"] = {"kind": "int", "value": scalar}
    registry["events"][0]["emitted_operations"].insert(1, "arith.sitofp")
    descriptor = {
        "kind": "tensor",
        "dtype": "torch.float32",
        "shape": [2, 3],
        "layout": "torch.strided",
        "device": "cpu",
    }
    registry["tensor_argument"] = {
        "original_binding": copy.deepcopy(original["tensor_binding"]),
        "native_binding": copy.deepcopy(original["tensor_binding"]["native"]),
        "common_dtype": "torch.float32",
        "input": descriptor,
        "outputs": [copy.deepcopy(descriptor)],
        "boxed_outputs": [copy.deepcopy(descriptor)],
    }
    return original, source, module, trace, registry


def nearest_exact_integer(scalar):
    """Independent distance search near integer's exponent, using exact rationals."""
    magnitude = abs(scalar)
    if not magnitude:
        return "00000000"
    exponent = magnitude.bit_length() - 1
    spacing = Fraction(2) ** (exponent - 23)
    lower = int(Fraction(magnitude) // spacing)
    candidates = [
        (abs(Fraction(magnitude) - spacing * mantissa), mantissa % 2, mantissa)
        for mantissa in range(max(1, lower - 1), lower + 3)
    ]
    mantissa = min(candidates)[2]
    if mantissa == 1 << 24:
        mantissa >>= 1
        exponent += 1
    bits = (int(scalar < 0) << 31) | ((exponent + 127) << 23) | (mantissa - (1 << 23))
    return bits.to_bytes(4, "big").hex()


@pytest.mark.parametrize(
    "scalar",
    [
        0,
        1,
        -1,
        2**53 + 2**29 - 1,
        2**53 + 2**29,
        2**53 + 2**29 + 1,
        -(2**53 + 2**29 + 1),
        2**60 + 2**36 - 1,
        2**60 + 2**36,
        2**60 + 2**36 + 1,
        -(2**60 + 2**36 + 1),
        -(2**63),
        2**63 - 1,
    ],
)
@pytest.mark.parametrize("target", sorted(S.TARGETS))
def test_direct_exact_signed64_constant_cast_body_and_complete_promoted_roster(target, scalar):
    record = checked(integer_fixture(target, scalar))
    assert record["schema"] == C.INTEGER_SCHEMA
    assert record["coefficient_f32_bits"] == nearest_exact_integer(scalar)
    assert record["ordered_abi"] == {"inputs": ["tensor<2x3xf32>"], "outputs": ["tensor<2x3xf32>"]}
    assert "no numerical" in record["scope"]


@pytest.mark.parametrize(
    "change",
    [
        "integer",
        "float_constant",
        "cast_input",
        "cast_type",
        "cast_attr",
        "splat_input",
        "div_order",
        "missing_promotion",
        "promoted_dtype",
        "wrapped",
        "native_literal",
        "output_drop",
        "boxed_shape",
        "original_binding",
        "trace_float",
        "trace_integer",
        "registry_float",
        "source_inventory",
    ],
)
def test_same_type_coefficients_order_cast_wrapper_and_complete_promotion_cannot_be_substituted(change):
    values = integer_fixture()
    block = next(iter(values[2].body.block.ops)).body.block
    constant, converted, splat, _, generic, _ = tuple(block.ops)
    promotion = values[4]["tensor_argument"]
    if change == "integer":
        constant.properties["value"] = IntegerAttr(2**60 + 2**36 - 1, i64)
    elif change == "float_constant":
        constant.properties["value"] = FloatAttr(float(2**60 + 2**36 + 1), f32)
        constant.result._type = f32
    elif change == "cast_input":
        converted.operands = [generic.body.block.args[0]]
    elif change == "cast_type":
        converted.result._type = i64
    elif change == "cast_attr":
        converted.attributes["saturate"] = IntegerAttr(1, i64)
    elif change == "splat_input":
        splat.operands = [constant.result]
    elif change == "div_order":
        operation = next(iter(generic.body.block.ops))
        operation.operands = list(reversed(operation.operands))
    elif change == "missing_promotion":
        values[4]["tensor_argument"] = None
    elif change == "promoted_dtype":
        promotion["common_dtype"] = "torch.float64"
    elif change == "wrapped":
        promotion["native_binding"]["wrapped_number"] = False
    elif change == "native_literal":
        promotion["native_binding"]["literal"]["value"] = "1"
    elif change == "output_drop":
        promotion["outputs"].clear()
    elif change == "boxed_shape":
        promotion["boxed_outputs"][0]["shape"] = [6]
    elif change == "original_binding":
        promotion["original_binding"]["request"]["node"] = "other-original"
    elif change in {"trace_float", "trace_integer"}:
        graph = values[3]["graphs"]["prepared"]
        graph["nodes"][1]["args"][1] = float(2**60 + 2**36 + 1) if change == "trace_float" else 1
        graph.pop("sha256")
        graph["sha256"] = _digest(graph)
    elif change == "registry_float":
        values[4]["events"][0]["literal"] = {"kind": "float", "value_hex": float(2**60 + 2**36 + 1).hex()}
    else:
        values[4]["function"]["sha256"] = "c" * 64
        assert SOURCES["/selected/decompositions.py"] != "c" * 64
    with pytest.raises(ValueError):
        checked(values)


def test_v2_float_keeps_exact_legacy_body_without_integer_box_or_cast():
    old, _, module, trace, registry = fixture(scalar=-0.0)
    form = S.scalar_binary_forms(*declarations(old["target"], scalar=-0.0), version=2)[0]
    source = S.scalar_binary_source(form, extent=2, max_tensor_elements=100)
    registry["schema"] = C.INTEGER_REGISTRY_SCHEMA
    registry["tensor_argument"] = None
    assert form["parameters"] == old["parameters"]
    assert checked((form, source, module, trace, registry))["coefficient_f32_bits"] == "80000000"
    registry["tensor_argument"] = {}
    with pytest.raises(ValueError):
        checked((form, source, module, trace, registry))
