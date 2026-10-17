"""Certify actual MLIR copy generalization, including source definedness and chain replay."""

import subprocess

import pytest
from test_lean_generalize import _generalize

from merlin.verify.lean.check import check_definedness, check_support, check_typed_stage, compose_chain


def _source(shape=(2, 3), element="i32", target_element=None, *, dirty=True, escape=False, dirty_input=False):
    target_element = target_element or element
    dims = "".join(f"{d}x" for d in shape)
    source_type, target_type = f"tensor<{dims}{element}>", f"tensor<{dims}{target_element}>"
    destination = "" if dirty else f", %init: {target_type}"
    allocation = f"%init = tensor.empty() : {target_type}" if dirty else ""
    poison = f"%poison = tensor.empty() : {source_type}" if dirty_input else ""
    result_types = f"({target_type}, {source_type})" if escape else target_type
    returns = f"%r, %a : {target_type}, {source_type}" if escape else f"%r : {target_type}"
    return f"""func.func @forward(%a: {source_type}, %alternate: {source_type}{destination}) -> {result_types} {{
      {allocation}
      {poison}
      %r = linalg.copy ins({"%poison" if dirty_input else "%a"} : {source_type})
        outs(%init : {target_type}) -> {target_type}
      return {returns}
    }}"""


@pytest.mark.parametrize("index_bits", [32, 64])
@pytest.mark.parametrize(
    "shape,element,target_element,dirty",
    [
        ((), "i32", None, True),
        ((0, 3), "i8", None, True),
        ((2, 3), "i13", None, True),
        ((2, 3, 2), "i8", None, False),
        ((2, 3), "i8", "i32", True),
        ((2, 3), "f32", None, True),
        ((2,), "f64", None, False),
        ((2,), "bf16", None, True),
    ],
)
def test_actual_copy_generalization(tmp_path, lean_tools, shape, element, target_element, dirty, index_bits):
    stage, before, _ = _generalize(
        tmp_path, lean_tools, _source(shape, element, target_element, dirty=dirty), boundary="copy-generalization"
    )
    rewritten = check_typed_stage(stage, index_bits=index_bits, m2m_python=lean_tools.m2m_python)
    assert rewritten["verdict"] == "verified", rewritten.get("reason")
    assert rewritten["index_bits"] == index_bits
    defined = check_definedness(
        before, "forward", tmp_path / "defined", index_bits=index_bits, m2m_python=lean_tools.m2m_python
    )
    assert defined["verdict"] == "verified", defined.get("reason")


@pytest.mark.parametrize("index_bits", [32, 64])
def test_copy_defined_chain_and_stale_output_rejection(tmp_path, lean_tools, index_bits):
    stage, before, after = _generalize(tmp_path, lean_tools, _source(escape=True), boundary="copy-generalization")
    for checker, name in [(check_support, "support"), (check_definedness, "defined")]:
        result = checker(before, "forward", tmp_path / name, index_bits=index_bits, m2m_python=lean_tools.m2m_python)
        assert result["verdict"] == "verified", result.get("reason")
    assert check_typed_stage(stage, index_bits=index_bits, m2m_python=lean_tools.m2m_python)["verdict"] == "verified"

    def replay():
        return compose_chain(
            [stage],
            tmp_path / "support/receipt.json",
            "forward",
            tmp_path / "chain.json",
            definedness=tmp_path / "defined/receipt.json",
            m2m_python=lean_tools.m2m_python,
        )

    result = replay()
    assert result["verdict"] == "verified", result.get("reason")
    assert result["theorem"] == "MerlinVerify.Check.chain_typed_defined_refines"
    after.write_text(after.read_text() + "\n// changed live output after proof publication\n")
    assert replay()["verdict"] != "verified"


MUTATE = """
import sys
from pathlib import Path
from torch_mlir import ir
with ir.Context():
    module = ir.Module.parse(Path(sys.argv[1]).read_text())
    function = module.body.operations[0]
    block = function.regions[0].blocks[0]
    generic = next(op for op in block.operations if op.operation.name == 'linalg.generic')
    body = generic.regions[0].blocks[0]
    terminator = list(block.operations)[-1]
    fault = sys.argv[2]
    if fault == 'input':
        generic.operation.operands[0] = block.arguments[1]
    elif fault == 'body':
        list(body.operations)[-1].operation.operands[0] = body.arguments[1]
    elif fault == 'returned_storage':
        terminator.operation.operands[0] = generic.operation.operands[-1]
    elif fault == 'escaping_input':
        terminator.operation.operands[1] = block.arguments[1]
    elif fault == 'attribute':
        generic.operation.attributes['unreviewed.meaning'] = ir.UnitAttr.get()
    else:
        maps = ['affine_map<(d0,d1)->(d0,d1)>'] * 2
        maps[0 if fault == 'input_map' else 1] = 'affine_map<(d0,d1)->(d1,d0)>'
        generic.operation.attributes['indexing_maps'] = ir.Attribute.parse('[' + ', '.join(maps) + ']')
    module.operation.verify()
    Path(sys.argv[1]).write_text(str(module))
"""


@pytest.mark.parametrize(
    "fault", ["input", "body", "input_map", "output_map", "returned_storage", "escaping_input", "attribute"]
)
def test_copy_rejects_verified_mlir_mutations(tmp_path, lean_tools, fault):
    stage, _, after = _generalize(tmp_path, lean_tools, _source((2, 2), escape=True), boundary="copy-generalization")
    original = after.read_bytes()
    result = subprocess.run(
        [str(lean_tools.m2m_python), "-c", MUTATE, str(after), fault], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert after.read_bytes() != original
    receipt = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert receipt["verdict"] == "abstracted", receipt.get("reason")


@pytest.mark.parametrize("dirty", [True, False])
def test_copy_of_poison_is_refinement_but_not_defined_execution(tmp_path, lean_tools, dirty):
    stage, before, _ = _generalize(tmp_path, lean_tools, _source(dirty=dirty, dirty_input=True), boundary="poison-copy")
    support = check_support(before, "forward", tmp_path / "support", m2m_python=lean_tools.m2m_python)
    assert support["verdict"] == "verified", support.get("reason")
    rewritten = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert rewritten["verdict"] == "verified", rewritten.get("reason")
    defined = check_definedness(before, "forward", tmp_path / "defined", m2m_python=lean_tools.m2m_python)
    assert defined["verdict"] == "abstracted", defined.get("reason")


def test_copy_and_fill_use_actual_intermediate_types(tmp_path, lean_tools):
    source = """func.func @forward(%value: i32) -> tensor<6xi32> {
      %e = tensor.empty() : tensor<2x3xi32>
      %filled = linalg.fill ins(%value : i32) outs(%e : tensor<2x3xi32>) -> tensor<2x3xi32>
      %flat = tensor.collapse_shape %filled [[0, 1]] : tensor<2x3xi32> into tensor<6xi32>
      %out = tensor.empty() : tensor<6xi32>
      %r = linalg.copy ins(%flat : tensor<6xi32>) outs(%out : tensor<6xi32>) -> tensor<6xi32>
      return %r : tensor<6xi32>
    }"""
    stage, before, _ = _generalize(tmp_path, lean_tools, source, boundary="fill-reshape-copy")
    for checker, name in [(check_support, "support"), (check_definedness, "defined")]:
        result = checker(before, "forward", tmp_path / name, m2m_python=lean_tools.m2m_python)
        assert result["verdict"] == "verified", result.get("reason")
    receipt = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert receipt["verdict"] == "verified", receipt.get("reason")
