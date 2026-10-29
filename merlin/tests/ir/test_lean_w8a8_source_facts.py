"""Qualify source calibration facts from executed scalar constants and rank-zero splats.

These facts identify source parameters; they do not prove the numerical domain of arbitrary
activation data or install the source quantization candidate in frozen semantics.
"""

from __future__ import annotations

import json

import pytest
from test_lean_w8a8_body import exported
from test_lean_w8a8_source_contract import inspect

from merlin.frontends.linalg_mlir import parse_mlir_text

pytestmark = pytest.mark.usefixtures("lean_oracle")


def source(kind="quantize", width=64, *, scale="0.1", zero=0, unknown=None):
    input_type = "tensor<2x3xf32>" if kind == "quantize" else "tensor<2x3xi8>"
    result_type = "tensor<2x3xi8>" if kind == "quantize" else "tensor<2x3xf32>"
    parameters = ""
    if unknown == "argument":
        parameters = ", %s: tensor<f32>"
        scale_ops = ""
    elif unknown == "producer":
        scale_ops = '%s = "frontend.parameter"() : () -> tensor<f32>'
    else:
        scale_ops = f"%sv = arith.constant {scale} : f32\n%s = tensor.splat %sv : tensor<f32>"
    metadata = "quant_min = -128 : i64, quant_max = 127 : i64"
    if kind == "quantize":
        metadata += ', output_dtype = "int8"'
    return parse_mlir_text(f"""func.func @forward(%x: {input_type}{parameters}) -> {result_type} {{
      {scale_ops}
      %zv = arith.constant {zero} : i{width}
      %z = tensor.splat %zv : tensor<i{width}>
      %r = "quant_ext.{kind}_per_tensor"(%x, %s, %z) <{{{metadata}}}>
        : ({input_type}, tensor<f32>, tensor<i{width}>) -> {result_type}
      return %r : {result_type}
    }}""")


def checked(document, work):
    report = inspect(document, work)
    assert not report["refusals"] and len(report["sites"]) == 1
    return report["sites"][0]["calibration_parameters_checked"]


@pytest.mark.parametrize("kind", ["quantize", "dequantize"])
@pytest.mark.parametrize("width", [32, 64])
def test_actual_constant_splat_parameters_are_checked(tmp_path, lean_tools, kind, width):
    document = exported(source(kind, width), tmp_path, lean_tools, generic=True, allow_unregistered=True)
    assert checked(document, tmp_path)


@pytest.mark.parametrize("zero", [-3, 17])
def test_initialized_zero_point_is_not_a_symmetric_reuse_assertion(tmp_path, lean_tools, zero):
    document = exported(source(zero=zero), tmp_path, lean_tools, generic=True, allow_unregistered=True)
    assert checked(document, tmp_path)


@pytest.mark.parametrize("scale", ["0.0", "-0.0", "-0.1", "0x7F800000", "0xFF800000", "0x7FC00001"])
def test_known_invalid_calibration_scale_refuses(tmp_path, lean_tools, scale):
    document = exported(source(scale=scale), tmp_path, lean_tools, generic=True, allow_unregistered=True)
    assert not checked(document, tmp_path)


@pytest.mark.parametrize("unknown", ["argument", "producer"])
def test_declared_scalar_tensor_type_does_not_supply_a_value_fact(tmp_path, lean_tools, unknown):
    document = exported(source(unknown=unknown), tmp_path, lean_tools, generic=True, allow_unregistered=True)
    assert not checked(document, tmp_path)


@pytest.mark.parametrize("overwrite", ["scalar", "tensor"])
def test_ssa_overwrite_invalidates_parameter_facts(tmp_path, lean_tools, overwrite):
    # Duplicate SSA definitions are not compiler-valid source witnesses. Exercise them directly
    # in structured JSON to check that the fact cache never reuses a stale value.
    document = exported(source(), tmp_path, lean_tools, generic=True, allow_unregistered=True)
    data = json.loads(document.read_text())
    operations = data["module"]["ops"][0]["regions"][0]["blocks"][0]["ops"]
    original = operations[0 if overwrite == "scalar" else 1]["results"][0]
    operations.insert(
        1 if overwrite == "scalar" else 2,
        {
            "name": "frontend.unknown",
            "results": [original],
            "operands": [],
            "attrs": {},
            "props": {},
            "regions": [],
            "successors": [],
        },
    )
    document.write_text(json.dumps(data))
    assert not checked(document, tmp_path)


def test_changed_constant_is_recomputed(tmp_path, lean_tools):
    document = exported(source(), tmp_path, lean_tools, generic=True, allow_unregistered=True)
    data = json.loads(document.read_text())
    constant = data["module"]["ops"][0]["regions"][0]["blocks"][0]["ops"][0]
    table = constant["props"] if "value" in constant["props"] else constant["attrs"]
    table["value"]["bits"] = "00000000"
    document.write_text(json.dumps(data))
    assert not checked(document, tmp_path)
