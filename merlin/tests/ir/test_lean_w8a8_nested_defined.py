"""Initialized source DQ inside structured bodies needs its own body and overwrite proofs.

Runtime scale/zero-point parameters are initialized inputs, without positivity assumptions.
Poison, unchecked body outputs and out-of-bounds tensor reads cannot acquire that guarantee.
"""

from __future__ import annotations

import json

import pytest
from test_lean_w8a8_body import exported
from test_lean_w8a8_defined_policy import qualify
from test_lean_w8a8_nested_support import source
from test_lean_w8a8_source_program import execute
from test_lean_w8a8_source_quantize import word
from test_lean_w8a8_source_support import support

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.verify.lean import differential as diff

pytestmark = pytest.mark.usefixtures("lean_oracle")


@pytest.mark.parametrize("width", [32, 64])
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_nested_source_dequantization_is_universally_initialized(tmp_path, lean_tools, width, depth):
    document = exported(
        source(width, depth=depth, runtime_dequant=True), tmp_path, lean_tools, generic=True, allow_unregistered=True
    )
    assert qualify(document, document, tmp_path)["initialized_policy_refinement"]
    actual = execute(document, [[["1", "255", "127"], [str(word(1))], ["0"]]], tmp_path)
    assert actual[0]["values"][0]["bits"] == [str(word(v)) for v in [1.0, -1.0, 127.0]]


@pytest.mark.parametrize("width", [32, 64])
def test_initialized_nested_dq_matches_native_mlir(tmp_path, lean_tools, fidelity_tools, width):
    directory = tmp_path / "source"
    directory.mkdir()
    document = exported(
        source(width, depth=3, runtime_dequant=True), directory, lean_tools, generic=True, allow_unregistered=True
    )
    assert qualify(document, document, directory)["initialized_policy_refinement"]
    arguments = [[["1", "255", "127"], [str(word(1))], ["0"]]]
    actual = execute(document, arguments, directory)
    reference = tmp_path / "reference"
    reference.mkdir()
    standard = exported(source(width, depth=3, runtime_dequant=True, reference=True), reference, lean_tools)
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


def inner_block(document):
    body = document["module"]["ops"][0]["regions"][0]["blocks"][0]
    while True:
        generic = next((op for op in body["ops"] if op["name"] == "linalg.generic"), None)
        if generic is None:
            return body
        body = generic["regions"][0]["blocks"][0]


@pytest.mark.parametrize("mutation", ["data", "scale", "zero", "destination", "metadata", "region"])
def test_nested_initialization_obligations_are_checked(tmp_path, lean_tools, mutation):
    document = exported(
        source(64, depth=3, runtime_dequant=True), tmp_path, lean_tools, generic=True, allow_unregistered=True
    )
    data = json.loads(document.read_text())
    body = inner_block(data)
    dq = next(op for op in body["ops"] if op["name"].startswith("quant_ext."))
    if mutation in {"data", "scale", "zero"}:
        argument = {"data": 0, "scale": 1, "zero": 2}[mutation]
        ty = (
            {"kind": "tensor", "shape": [], "element": {"kind": "int", "width": 8}, "encoding": None}
            if mutation == "data"
            else data["module"]["ops"][0]["regions"][0]["blocks"][0]["args"][argument]["type"]
        )
        body["ops"].insert(
            0,
            {
                "name": "tensor.empty",
                "results": [{"id": 999999, "type": ty}],
                "operands": [],
                "attrs": {},
                "props": {},
                "regions": [],
                "successors": [],
            },
        )
        dq["operands"][argument] = 999999
    elif mutation == "destination":
        body["ops"][-1]["operands"] = [body["args"][-1]["id"]]
    elif mutation == "metadata":
        dq["attrs"]["unknown_numeric_mode"] = {"kind": "string", "value": "changed"}
    else:
        dq["regions"] = [{"blocks": []}]
    document.write_text(json.dumps(data))
    if mutation in {"data", "scale", "zero", "destination"}:
        assert support(document, tmp_path)["supported"]
    assert not qualify(document, document, tmp_path)["initialized_policy_refinement"]


@pytest.mark.parametrize("index", [0, 1])
@pytest.mark.parametrize("width", [32, 64])
def test_source_dq_body_retains_proved_index_bounds(tmp_path, lean_tools, index, width):
    program = parse_mlir_text(f"""func.func @forward(%q: tensor<3xi8>, %s: tensor<f32>,
        %z: tensor<i{width}>) -> tensor<3xf32> {{
      %empty = tensor.empty() : tensor<3xf32>
      %r = linalg.generic {{indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
        iterator_types = ["parallel"]}} ins(%q : tensor<3xi8>) outs(%empty : tensor<3xf32>) {{
          ^bb0(%value: i8, %unused: f32):
            %scalar = tensor.splat %value : tensor<1xi8>
            %dq = "quant_ext.dequantize_per_tensor"(%scalar, %s, %z)
              : (tensor<1xi8>, tensor<f32>, tensor<i{width}>) -> tensor<1xf32>
            %index = arith.constant {index} : index
            %result = tensor.extract %dq[%index] : tensor<1xf32>
            linalg.yield %result : f32
      }} -> tensor<3xf32>
      return %r : tensor<3xf32>
    }}""")
    document = exported(program, tmp_path, lean_tools, generic=True, allow_unregistered=True)
    assert qualify(document, document, tmp_path)["initialized_policy_refinement"] == (index == 0)


@pytest.mark.parametrize("width", [32, 64])
@pytest.mark.parametrize("index", [0, 2, -1, 3])
def test_bounds_survive_between_top_level_operations_in_mixed_functions(tmp_path, lean_tools, width, index):
    program = parse_mlir_text(f"""func.func @forward(%q: tensor<3xi8>, %s: tensor<f32>,
        %z: tensor<i{width}>) -> f32 {{
      %dq = "quant_ext.dequantize_per_tensor"(%q, %s, %z)
        : (tensor<3xi8>, tensor<f32>, tensor<i{width}>) -> tensor<3xf32>
      %index = arith.constant {index} : index
      %result = tensor.extract %dq[%index] : tensor<3xf32>
      return %result : f32
    }}""")
    document = exported(program, tmp_path, lean_tools, generic=True, allow_unregistered=True)
    assert qualify(document, document, tmp_path)["initialized_policy_refinement"] == (0 <= index < 3)
