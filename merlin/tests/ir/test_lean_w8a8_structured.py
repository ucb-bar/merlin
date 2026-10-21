"""Carry conditional W8A8 numerical facts through actual structured execution.

The same finite-input and explicit model-law premises remain in force. Incorrect maps,
poison destinations, metadata and SSA writes cannot supply a missing numerical fact.
"""

from __future__ import annotations

import json

import pytest
from test_lean_w8a8_body import exported
from test_lean_w8a8_calls import qualify as called
from test_lean_w8a8_finite_defined import qualify
from test_lean_w8a8_finite_policy import qualify as policy
from test_lean_w8a8_finite_policy import recipe_values, reused_reference
from test_lean_w8a8_finite_policy import source as policy_source
from test_lean_w8a8_source_program import execute
from test_lean_w8a8_source_quantize import samples_for, source_program, word

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.passes_quant_int import lower_contraction_int8
from merlin.verify.lean import differential as diff

pytestmark = pytest.mark.usefixtures("lean_oracle")


def transported(shape=(2, 8), *, width=64, mode="identity", reduction=False, scale="0.1"):
    dims = "".join(f"{size}x" for size in shape)
    source_type = f"tensor<{dims}f32>"
    output_shape = () if reduction else shape
    out_dims = "".join(f"{size}x" for size in output_shape)
    data_type, byte_type = f"tensor<{out_dims}f32>", f"tensor<{out_dims}i8>"
    coordinates = ", ".join(f"d{i}" for i in range(len(shape)))
    identity = f"affine_map<({coordinates}) -> ({coordinates})>"
    destination = f"affine_map<({coordinates}) -> ({'' if reduction else coordinates})>"
    iters = ", ".join(f'"{"reduction" if reduction else "parallel"}"' for _ in shape)
    body = {
        "identity": "linalg.yield %value : f32",
        "arithmetic": "%v = arith.addf %value, %value : f32\n linalg.yield %v : f32",
        "old": "linalg.yield %old : f32",
        "call": "%v = func.call @identity(%value) : (f32) -> f32\n linalg.yield %v : f32",
    }[mode]
    initial = "%empty"
    initialization = ""
    if reduction:
        initial = "%filled"
        initialization = f"""%zero = arith.constant 0.0 : f32
          %filled = linalg.fill ins(%zero : f32) outs(%empty : {data_type}) -> {data_type}"""
    helper = "func.func @identity(%x: f32) -> f32 { return %x : f32 }" if mode == "call" else ""
    return parse_mlir_text(f"""module {{
      {helper}
      func.func @forward(%x: {source_type}) -> {byte_type} {{
        %empty = tensor.empty() : {data_type}
        {initialization}
        %data = linalg.generic {{indexing_maps = [{identity}, {destination}], iterator_types = [{iters}]}}
          ins(%x : {source_type}) outs({initial} : {data_type}) {{
          ^bb0(%value: f32, %old: f32):
            {body}
        }} -> {data_type}
        %sv = arith.constant {scale} : f32
        %s = tensor.splat %sv : tensor<f32>
        %zv = arith.constant 17 : i{width}
        %z = tensor.splat %zv : tensor<i{width}>
        %q = "quant_ext.quantize_per_tensor"(%data, %s, %z)
          <{{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}}>
          : ({data_type}, tensor<f32>, tensor<i{width}>) -> {byte_type}
        return %q : {byte_type}
      }}
    }}""")


def region_quantizer(width=64, *, scale="0.1", via_call=False, nested=False):
    operation = (
        "%q = func.call @kernel(%scalar) : (tensor<f32>) -> tensor<i8>"
        if via_call
        else f"""%q = "quant_ext.quantize_per_tensor"(%scalar, %s, %z)
          <{{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}}>
          : (tensor<f32>, tensor<f32>, tensor<i{width}>) -> tensor<i8>"""
    )
    helper = ""
    if via_call:
        from test_lean_w8a8_source_program import source

        module = source(width, (), scale=scale, wrapper=True)
        helper = next(iter(module.body.block.ops)).clone()
    body = f"""%scalar = tensor.splat %value : tensor<f32>
      {operation}
      %byte = tensor.extract %q[] : tensor<i8>
      linalg.yield %byte : i8"""
    if nested:
        inner_body = body.replace("%value", "%nested_value")
        body = f"""%inner_data = tensor.splat %value : tensor<2xf32>
          %inner_empty = tensor.empty() : tensor<i8>
          %inner_zero = arith.constant 0 : i8
          %inner_init = linalg.fill ins(%inner_zero : i8) outs(%inner_empty : tensor<i8>) -> tensor<i8>
          %inner = linalg.generic {{indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],
            iterator_types = ["reduction"]}}
            ins(%inner_data : tensor<2xf32>) outs(%inner_init : tensor<i8>) {{
            ^bb0(%nested_value: f32, %nested_unused: i8):
              {inner_body}
          }} -> tensor<i8>
          %byte = tensor.extract %inner[] : tensor<i8>
          linalg.yield %byte : i8"""
    module = parse_mlir_text(f"""module {{
      func.func @forward(%x: tensor<2x8xf32>) -> tensor<2x8xi8> {{
        %sv = arith.constant {scale} : f32
        %s = tensor.splat %sv : tensor<f32>
        %zv = arith.constant 17 : i{width}
        %z = tensor.splat %zv : tensor<i{width}>
        %empty = tensor.empty() : tensor<2x8xi8>
        %r = linalg.generic {{
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel"]}}
          ins(%x : tensor<2x8xf32>) outs(%empty : tensor<2x8xi8>) {{
          ^bb0(%value: f32, %unused: i8):
            {body}
        }} -> tensor<2x8xi8>
        return %r : tensor<2x8xi8>
      }}
    }}""")
    if helper:
        module.body.block.add_op(helper)
    return module


@pytest.mark.parametrize("width", [32, 64])
@pytest.mark.parametrize("shape", [(), (2, 8), (0, 3)])
def test_overwrite_preserves_finite_input_for_source_quantization(tmp_path, lean_tools, width, shape):
    document = exported(transported(shape, width=width), tmp_path, lean_tools, generic=True, allow_unregistered=True)
    assert qualify(document, tmp_path)["conditional_source_initialized"]


@pytest.mark.parametrize("width", [32, 64])
@pytest.mark.parametrize("mode", ["identity", "old"])
@pytest.mark.parametrize("shape", [(2, 8), (0, 3)])
def test_initialized_reduction_preserves_a_checked_finite_invariant(tmp_path, lean_tools, width, mode, shape):
    document = exported(
        transported(shape, width=width, reduction=True, mode=mode),
        tmp_path,
        lean_tools,
        generic=True,
        allow_unregistered=True,
    )
    assert qualify(document, tmp_path)["conditional_source_initialized"]


@pytest.mark.parametrize("reduction,mode", [(False, "old"), (False, "arithmetic"), (True, "arithmetic")])
def test_loop_execution_cannot_invent_initialization_or_finiteness(tmp_path, lean_tools, reduction, mode):
    document = exported(
        transported(reduction=reduction, mode=mode), tmp_path, lean_tools, generic=True, allow_unregistered=True
    )
    assert not qualify(document, tmp_path)["conditional_source_initialized"]


@pytest.mark.parametrize("width", [32, 64])
@pytest.mark.parametrize("nested", [False, True])
def test_source_quantizer_is_checked_inside_the_actual_region(tmp_path, lean_tools, width, nested):
    document = exported(
        region_quantizer(width, nested=nested), tmp_path, lean_tools, generic=True, allow_unregistered=True
    )
    assert qualify(document, tmp_path)["conditional_source_initialized"]


@pytest.mark.parametrize("program", ["transport", "quantizer", "nested_quantizer"])
def test_checked_internal_call_contracts_are_available_in_regions(tmp_path, lean_tools, program):
    module = (
        transported(mode="call")
        if program == "transport"
        else region_quantizer(via_call=True, nested=program == "nested_quantizer")
    )
    document = exported(module, tmp_path, lean_tools, generic=True, allow_unregistered=True)
    assert not qualify(document, tmp_path)["conditional_source_initialized"]
    assert called(document, tmp_path)["conditional_source_initialized"]


@pytest.mark.parametrize("width", [32, 64])
@pytest.mark.parametrize("placement", ["after", "inside", "nested", "nested_call"])
def test_checked_structured_execution_agrees_with_native_mlir(tmp_path, lean_tools, fidelity_tools, width, placement):
    module = (
        transported(width=width)
        if placement == "after"
        else region_quantizer(width, nested=placement in ("nested", "nested_call"), via_call=placement == "nested_call")
    )
    document = exported(module, tmp_path, lean_tools, generic=True, allow_unregistered=True)
    checker = called if placement == "nested_call" else qualify
    assert checker(document, tmp_path)["conditional_source_initialized"]
    _, arguments = samples_for(width, (2, 8))
    actual = execute(document, [[sample[0]] for sample in arguments], tmp_path)
    native_work = tmp_path / "native"
    native_work.mkdir()
    native_document = exported(source_program(width, (2, 8)), native_work, lean_tools)
    inputs, outputs = diff.signature(json.loads(native_document.read_text()), "forward")
    native = diff.execute_mlir(
        native_work / "actual.mlir",
        [[sample[0], [str(word(0.1))], ["17"]] for sample in arguments],
        native_work,
        fidelity_tools,
        inputs=inputs,
        outputs=outputs,
        function="forward",
    )
    assert all(sample["status"] == "ok" for sample in actual)
    assert [sample["values"][0]["bits"] for sample in actual] == native["results"]


@pytest.mark.parametrize("scale", ["0.0", "-0.1", "0x00000001", "0x7F800000", "0x7FC00001"])
def test_bad_calibration_is_refused_inside_a_region(tmp_path, lean_tools, scale):
    document = exported(region_quantizer(scale=scale), tmp_path, lean_tools, generic=True, allow_unregistered=True)
    assert not qualify(document, tmp_path)["conditional_source_initialized"]


@pytest.mark.parametrize("scale,accepted", [("0.1", True), ("-0.1", False), ("0x7FC00001", False)])
@pytest.mark.parametrize("width", [32, 64])
def test_fill_preserves_the_actual_scale_class(tmp_path, lean_tools, scale, accepted, width):
    from test_lean_w8a8_source_program import source
    from xdsl.dialects import tensor
    from xdsl.dialects.linalg.ops import FillOp

    module = source(width, (2, 8), scale=scale)
    block = next(op for op in module.walk() if op.name == "func.func").body.block
    splat = next(op for op in block.ops if op.name == "tensor.splat")
    empty = tensor.EmptyOp([], splat.result.type)
    fill = FillOp(inputs=[splat.input], outputs=[empty.tensor], res=[splat.result.type])
    block.insert_op_before(empty, splat)
    block.insert_op_before(fill, splat)
    splat.result.replace_all_uses_with(fill.res[0])
    block.erase_op(splat)
    document = exported(module, tmp_path, lean_tools, generic=True, allow_unregistered=True)
    assert qualify(document, tmp_path)["conditional_source_initialized"] == accepted


@pytest.mark.parametrize("change", ["output_map", "body_type", "metadata", "shadow", "captured_nan"])
def test_actual_region_geometry_captures_and_ssa_are_rechecked(tmp_path, lean_tools, change):
    document = exported(transported(), tmp_path, lean_tools, generic=True, allow_unregistered=True)
    data = json.loads(document.read_text())
    ops = data["module"]["ops"][0]["regions"][0]["blocks"][0]["ops"]
    loop = next(op for op in ops if op["name"] == "linalg.generic")
    body = loop["regions"][0]["blocks"][0]
    if change == "output_map":
        table = loop["props"] if "indexing_maps" in loop["props"] else loop["attrs"]
        # Wrong dimensions are not an overwrite proof, even if the body itself is harmless.
        table["indexing_maps"]["elements"][1]["results"] = []
    elif change == "body_type":
        body["args"][0]["type"] = {"kind": "int", "width": 32}
    elif change == "metadata":
        loop["attrs"]["unknown_numerical_mode"] = {"kind": "unit"}
    elif change == "shadow":
        # Adversarial protocol input: a later SSA write must replace the finite argument fact.
        argument = body["args"][0]
        body["ops"].insert(
            0,
            {
                "name": "arith.addf",
                "results": [argument],
                "operands": [argument["id"]] * 2,
                "attrs": {},
                "props": {},
                "regions": [],
                "successors": [],
            },
        )
    else:
        original = next(
            op for op in ops if op["name"] == "arith.constant" and op["results"][0]["type"]["kind"] == "float"
        )
        constant = json.loads(json.dumps(original))
        constant["results"][0]["id"] = 900001
        table = constant["props"] if "value" in constant["props"] else constant["attrs"]
        table["value"]["bits"] = "7fc00001"
        ops.insert(0, constant)
        body["ops"][-1]["operands"] = [constant["results"][0]["id"]]
    document.write_text(json.dumps(data))
    assert not qualify(document, tmp_path)["conditional_source_initialized"]


def test_real_reuse_pass_qualifies_after_a_proved_finite_copy(tmp_path, lean_tools):
    import numpy as np
    from xdsl.dialects import tensor
    from xdsl.dialects.builtin import UnregisteredOp
    from xdsl.dialects.linalg.ops import GenericOp

    before = policy_source()
    block = next(op for op in before.walk() if op.name == "func.func").body.block
    original = block.args[0]
    copy_module = transported((2, 3))
    copy_block = next(op for op in copy_module.walk() if op.name == "func.func").body.block
    mapping = {copy_block.args[0]: original}
    copy_ops = []
    for operation in copy_block.ops:
        if isinstance(operation, (tensor.EmptyOp, GenericOp)):
            cloned = operation.clone(mapping)
            copy_ops.append(cloned)
            mapping.update(zip(operation.results, cloned.results, strict=True))
    assert len(copy_ops) == 2
    first = block.first_op
    for operation in copy_ops:
        block.insert_op_before(operation, first)
    quantizer = next(
        op for op in block.ops if isinstance(op, UnregisteredOp) and op.op_name.data == "quant_ext.quantize_per_tensor"
    )
    quantizer.operands = [copy_ops[-1].results[0], *quantizer.operands[1:]]
    before.verify()
    after = before.clone()
    assert lower_contraction_int8(after) == 1
    report, documents = policy(before, after, tmp_path, lean_tools)
    assert report["conditional_initialized_policy_refinement"]
    a, b = recipe_values()
    samples = [[[str(int(v)) for v in value.reshape(-1).view(np.uint32)] for value in (a, b)]]
    result = execute(documents[1], samples, tmp_path)
    expected = [str(int(v)) for v in reused_reference(a, b).reshape(-1).view(np.uint32)]
    assert result[0]["status"] == "ok" and result[0]["values"][0]["bits"] == expected
