"""Successful, nonpoison dequantization source execution and defined refinement chains."""

from __future__ import annotations

import json

import pytest
from test_lean_dequant_c1 import export_pair, invoke
from test_lean_dequant_scalar import dequant_source

from merlin.verify.lean.check import check_definedness, check_support


@pytest.mark.parametrize("index_bits", [32, 64])
@pytest.mark.parametrize(
    "kind,shape,axis",
    [
        ("per_tensor", (), 0),
        ("per_tensor", (2,), 0),
        ("per_tensor", (2, 3), 0),
        ("per_tensor", (0, 3), 0),
        ("per_channel", (2,), 0),
        ("per_channel", (2, 3), 1),
        ("per_channel", (0, 3), 0),
        ("per_channel", (2, 3, 4), 2),
    ],
)
def test_dequantization_source_is_defined(tmp_path, lean_tools, kind, shape, axis, index_bits):
    before, _ = export_pair(tmp_path, lean_tools, kind, shape, axis)
    result = invoke(lean_tools, "check-definedness", before, "forward", index_bits)
    assert result["verdict"] == "verified", result


@pytest.mark.parametrize("operand", ["q", "s", "z"])
def test_dequantization_requires_initialized_arithmetic_inputs(tmp_path, lean_tools, operand):
    source = str(dequant_source("per_tensor"))
    # Keep the program well typed while replacing one actual arithmetic input with poison storage.
    from merlin.frontends.linalg_mlir import parse_mlir_text

    module = parse_mlir_text(source)
    from xdsl.dialects import tensor

    block = next(op for op in module.walk() if op.name == "func.func").body.block
    index = ("q", "s", "z").index(operand)
    empty = tensor.EmptyOp((), block.args[index].type)
    block.insert_op_before(empty, block.first_op)
    for op in list(block.ops):
        if getattr(op, "op_name", None) is not None:
            operands = list(op.operands)
            operands[index] = empty.results[0]
            op.operands = operands
    module.verify()
    program = tmp_path / "poison.mlir"
    program.write_text(str(module))
    support = check_support(program, "forward", tmp_path / "support", m2m_python=lean_tools.m2m_python)
    assert support["verdict"] == "verified", support
    result = check_definedness(program, "forward", tmp_path / "defined", m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "abstracted", result


@pytest.mark.parametrize("mutation", ["axis", "negative_axis", "bounds", "attribute", "kind", "parameter_shape"])
def test_dequantization_definedness_refuses_unchecked_contracts(tmp_path, lean_tools, mutation):
    before, _ = export_pair(tmp_path, lean_tools, "per_channel", (2, 3), 1)
    document = json.loads(before.read_text())
    block = document["module"]["ops"][0]["regions"][0]["blocks"][0]
    source = block["ops"][0]
    if mutation in ("axis", "negative_axis"):
        source["props"]["axis"]["value"] = "2" if mutation == "axis" else "-1"
    elif mutation == "bounds":
        source["props"]["quant_min"] = {"kind": "int", "value": "0", "type": {"kind": "int", "width": 64}}
    elif mutation == "attribute":
        source["attrs"]["numeric_contract"] = {"kind": "string", "value": "unreviewed"}
    elif mutation == "kind":
        source["name"] = "quant_ext.dequantize_per_group"
    else:
        block["args"][1]["type"]["shape"] = [2]
    before.write_text(json.dumps(document))
    assert invoke(lean_tools, "check-definedness", before, "forward")["verdict"] == "unsupported"
