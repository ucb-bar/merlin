"""Calibrated source operations inside structured regions use actual captured SSA facts.

These are support/refinement claims; initialized quantizer execution remains a separate check.
"""

from __future__ import annotations

import json

import pytest
from test_lean_w8a8_body import exported
from test_lean_w8a8_defined_policy import qualify
from test_lean_w8a8_source_program import execute
from test_lean_w8a8_source_quantize import word
from test_lean_w8a8_source_support import policy, support

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.verify.lean import differential as diff

pytestmark = pytest.mark.usefixtures("lean_oracle")


def source(width=64, *, depth=1, local=False, runtime_dequant=False, dynamic_scale=False, reference=False):
    element, output = ("i8", "f32") if runtime_dequant else ("f32", "i8")
    parameters = f", %s: tensor<f32>, %z: tensor<i{width}>" if runtime_dequant else ""
    calibration = (
        ""
        if runtime_dequant
        else f"""%sv = arith.constant 0.1 : f32
      %s = tensor.splat %sv : tensor<f32>
      %zv = arith.constant 17 : i{width}
      %z = tensor.splat %zv : tensor<i{width}>"""
    )
    operation = "dequantize" if runtime_dequant else "quantize"
    attrs = "" if runtime_dequant else '<{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}>'
    scale = "%s"
    pre = ""
    if local or dynamic_scale:
        pre = (
            "%local_s = tensor.splat %value : tensor<f32>"
            if dynamic_scale
            else """%local_sv = arith.constant 0.1 : f32
          %local_s = tensor.splat %local_sv : tensor<f32>"""
        )
        scale = "%local_s"
    body = f"""{pre}
      %scalar = tensor.splat %value : tensor<{element}>
      %converted = "quant_ext.{operation}_per_tensor"(%scalar, {scale}, %z) {attrs}
        : (tensor<{element}>, tensor<f32>, tensor<i{width}>) -> tensor<{output}>
      %result = tensor.extract %converted[] : tensor<{output}>
      linalg.yield %result : {output}"""
    if reference:
        arithmetic = (
            f"""%qf = arith.sitofp %value : i8 to f32
          %zf = arith.sitofp %zero : i{width} to f32
          %unscaled = arith.subf %qf, %zf : f32
          %result = arith.mulf %unscaled, %scale : f32"""
            if runtime_dequant
            else f"""%one = arith.constant 1.0 : f64
          %scale64 = arith.extf %scale : f32 to f64
          %inverse64 = arith.divf %one, %scale64 : f64
          %inverse = arith.truncf %inverse64 : f64 to f32
          %product = arith.mulf %value, %inverse : f32
          %rounded = math.roundeven %product : f32
          %zf = arith.sitofp %zero : i{width} to f32
          %shifted = arith.addf %rounded, %zf : f32
          %lower = arith.constant -128.0 : f32
          %upper = arith.constant 127.0 : f32
          %lo = arith.maximumf %shifted, %lower : f32
          %hi = arith.minimumf %lo, %upper : f32
          %result = arith.fptosi %hi : f32 to i8"""
        )
        body = f"""{pre}
          %scale = tensor.extract {scale}[] : tensor<f32>
          %zero = tensor.extract %z[] : tensor<i{width}>
          {arithmetic}
          linalg.yield %result : {output}"""
    for level in range(depth - 1):
        body = body.replace("%value", f"%nested_value{level}")
        body = f"""%inner_input{level} = tensor.splat %value : tensor<1x{element}>
          %inner_empty{level} = tensor.empty() : tensor<1x{output}>
          %inner{level} = linalg.generic {{indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
            iterator_types = ["parallel"]}}
            ins(%inner_input{level} : tensor<1x{element}>) outs(%inner_empty{level} : tensor<1x{output}>) {{
              ^bb0(%nested_value{level}: {element}, %unused{level}: {output}):
                {body}
          }} -> tensor<1x{output}>
          %index{level} = arith.constant 0 : index
          %inner_result{level} = tensor.extract %inner{level}[%index{level}] : tensor<1x{output}>
          linalg.yield %inner_result{level} : {output}"""
    return parse_mlir_text(f"""func.func @forward(%x: tensor<3x{element}>{parameters}) -> tensor<3x{output}> {{
      {calibration}
      %empty = tensor.empty() : tensor<3x{output}>
      %r = linalg.generic {{indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
        iterator_types = ["parallel"]}} ins(%x : tensor<3x{element}>) outs(%empty : tensor<3x{output}>) {{
          ^bb0(%value: {element}, %unused: {output}):
            {body}
      }} -> tensor<3x{output}>
      return %r : tensor<3x{output}>
    }}""")


@pytest.mark.parametrize("width", [32, 64])
@pytest.mark.parametrize("depth,local", [(1, False), (1, True), (2, False), (3, True)])
def test_quantizers_in_nested_regions_have_universal_typed_support(tmp_path, lean_tools, width, depth, local):
    document = exported(
        source(width, depth=depth, local=local), tmp_path, lean_tools, generic=True, allow_unregistered=True
    )
    assert support(document, tmp_path)["supported"]
    # The stronger initialized check must remain separate from recursive typed support.
    assert not qualify(document, document, tmp_path)["initialized_policy_refinement"]
    actual = execute(document, [[[str(word(v)) for v in [-1.0, 0.0, 1.0]]]], tmp_path)
    assert actual[0]["values"][0]["bits"] == ["7", "17", "27"]


@pytest.mark.parametrize("width", [32, 64])
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_runtime_dequantization_parameters_are_captured_with_actual_types(tmp_path, lean_tools, width, depth):
    program = source(width, depth=depth, runtime_dequant=True)
    document = exported(program, tmp_path, lean_tools, generic=True, allow_unregistered=True)
    assert support(document, tmp_path)["supported"]
    actual = execute(document, [[["1", "255", "127"], [str(word(1))], ["0"]]], tmp_path)
    assert actual[0]["values"][0]["bits"] == [str(word(v)) for v in [1.0, -1.0, 127.0]]


def test_whole_policy_qualification_uses_the_new_region_support(tmp_path, lean_tools):
    before = source(64, depth=2, local=True)
    assert policy(before, before.clone(), tmp_path, lean_tools, scope="captured")["policy_refinement"]


@pytest.mark.parametrize("width", [32, 64])
@pytest.mark.parametrize("depth", [1, 3])
@pytest.mark.parametrize("dequant", [False, True])
def test_nested_source_execution_matches_independent_native_reference(
    tmp_path, lean_tools, fidelity_tools, width, depth, dequant
):
    directory = tmp_path / "source"
    directory.mkdir()
    document = exported(
        source(width, depth=depth, runtime_dequant=dequant),
        directory,
        lean_tools,
        generic=True,
        allow_unregistered=True,
    )
    arguments = (
        [[["1", "255", "127"], [str(word(1))], ["0"]]] if dequant else [[[str(word(v)) for v in [-1.0, 0.0, 1.0]]]]
    )
    actual = execute(document, arguments, directory)
    reference = tmp_path / "reference"
    reference.mkdir()
    standard = exported(source(width, depth=depth, runtime_dequant=dequant, reference=True), reference, lean_tools)
    inputs, outputs = diff.signature(json.loads(standard.read_text()), "forward")
    native = diff.execute_mlir(
        reference / "actual.mlir",
        arguments,
        reference,
        fidelity_tools,
        inputs=inputs,
        outputs=outputs,
        function="forward",
    )
    assert [sample["values"][0]["bits"] for sample in actual] == native["results"]


def test_region_argument_shadowing_invalidates_captured_calibration_facts(tmp_path, lean_tools):
    # A runtime value cannot act as a known-positive scale. Rebinding its SSA id to the
    # captured constant's id must not revive that constant fact in the region environment.
    document = exported(source(dynamic_scale=True), tmp_path, lean_tools, generic=True, allow_unregistered=True)
    assert not support(document, tmp_path)["supported"]
    data = json.loads(document.read_text())
    operations = data["module"]["ops"][0]["regions"][0]["blocks"][0]["ops"]
    scalar = operations[0]["results"][0]["id"]
    generic = next(op for op in operations if op["name"] == "linalg.generic")
    body = generic["regions"][0]["blocks"][0]
    previous = body["args"][0]["id"]
    body["args"][0]["id"] = scalar
    for op in body["ops"]:
        op["operands"] = [scalar if value == previous else value for value in op["operands"]]
    document.write_text(json.dumps(data))
    assert not support(document, tmp_path)["supported"]


@pytest.mark.parametrize("mutation", ["unknown_metadata", "extra_region", "overwrite_parameter", "unbound_capture"])
def test_nested_source_failures_cannot_hide_inside_an_accepted_parent(tmp_path, lean_tools, mutation):
    document = exported(source(local=True), tmp_path, lean_tools, generic=True, allow_unregistered=True)
    data = json.loads(document.read_text())
    operations = data["module"]["ops"][0]["regions"][0]["blocks"][0]["ops"]
    generic = next(op for op in operations if op["name"] == "linalg.generic")
    body = generic["regions"][0]["blocks"][0]
    quantizer = next(op for op in body["ops"] if op["name"].startswith("quant_ext."))
    if mutation == "unknown_metadata":
        quantizer["attrs"]["unreviewed_numeric_mode"] = {"kind": "string", "value": "changed"}
    elif mutation == "extra_region":
        quantizer["regions"] = [{"blocks": []}]
    elif mutation == "overwrite_parameter":
        parameter = body["ops"][1]["results"][0]
        body["ops"].insert(
            2,
            {
                "name": "tensor.empty",
                "results": [parameter],
                "operands": [],
                "attrs": {},
                "props": {},
                "regions": [],
                "successors": [],
            },
        )
    else:
        quantizer["operands"][1] = 999999
    document.write_text(json.dumps(data))
    assert not support(document, tmp_path)["supported"]
