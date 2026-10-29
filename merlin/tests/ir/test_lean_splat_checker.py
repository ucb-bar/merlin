"""Certify the actual splat pass with independent source/target typed execution checks."""

import json
import subprocess

import pytest
from test_lean_splat_inputs import _captured_source, _compact, _repeated_source, _source, _tensor_capture_source

from merlin.verify.lean.check import check_definedness, check_support, check_typed_stage, compose_chain


def _stage(tmp_path, source):
    before, after, count = _compact(source)
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "before.mlir").write_text(before)
    (stage / "after.mlir").write_text(after)
    (stage / "meta.json").write_text(json.dumps({"boundary": "splat-input-compaction"}))
    return stage, count


@pytest.mark.parametrize("index_bits", [32, 64])
@pytest.mark.parametrize(
    "shape,element,escape,multiple,affine",
    [
        ((2, 3), "i32", False, False, False),
        ((2, 3), "i32", True, False, False),
        ((2, 3), "i32", True, True, False),
        ((4, 3), "i32", False, False, True),
        ((0, 3), "i32", False, False, False),
        ((), "i32", False, False, False),
        ((2, 3, 2), "i32", False, False, False),
        ((2, 3), "i8", True, True, False),
        ((2, 3), "i13", True, False, False),
        ((2, 3), "f32", True, False, False),
        ((2,), "f64", False, False, False),
        ((2,), "bf16", True, False, False),
    ],
)
def test_actual_splat_checker(tmp_path, lean_tools, shape, element, escape, multiple, affine, index_bits):
    stage, count = _stage(tmp_path, _source(shape, element, escape=escape, multiple=multiple, affine=affine))
    assert count == (1 + int(multiple) if shape else 0)
    result = check_typed_stage(stage, index_bits=index_bits, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "verified", result.get("reason")


@pytest.mark.parametrize("index_bits", [32, 64])
def test_splat_checker_preserves_captures_and_defined_chain(tmp_path, lean_tools, index_bits):
    stage, count = _stage(tmp_path, _captured_source())
    assert count == 2
    before = stage / "before.mlir"
    for checker, name in [(check_support, "support"), (check_definedness, "defined")]:
        result = checker(before, "subject", tmp_path / name, index_bits=index_bits, m2m_python=lean_tools.m2m_python)
        assert result["verdict"] == "verified", result.get("reason")
    result = check_typed_stage(stage, index_bits=index_bits, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "verified", result.get("reason")
    replay = compose_chain(
        [stage],
        tmp_path / "support/receipt.json",
        "subject",
        tmp_path / "chain.json",
        definedness=tmp_path / "defined/receipt.json",
        m2m_python=lean_tools.m2m_python,
    )
    assert replay["verdict"] == "verified", replay.get("reason")
    after = stage / "after.mlir"
    after.write_text(after.read_text() + "\n// mutated after receipt publication\n")
    stale = compose_chain(
        [stage],
        tmp_path / "support/receipt.json",
        "subject",
        tmp_path / "stale.json",
        definedness=tmp_path / "defined/receipt.json",
        m2m_python=lean_tools.m2m_python,
    )
    assert stale["verdict"] != "verified"


MUTATE = """
import sys
from pathlib import Path
from torch_mlir import ir
with ir.Context():
    module = ir.Module.parse(Path(sys.argv[1]).read_text())
    function = module.body.operations[0]
    block = function.regions[0].blocks[0]
    generic = next(op for op in block.operations if op.operation.name == 'linalg.generic')
    splats = [op for op in block.operations if op.operation.name == 'tensor.splat']
    compact = next(op for op in splats if ir.RankedTensorType(op.results[0].type).rank == 0)
    full = next(op for op in splats if ir.RankedTensorType(op.results[0].type).rank != 0)
    body = generic.regions[0].blocks[0]
    terminator = list(block.operations)[-1]
    fault = sys.argv[2]
    if fault == 'payload':
        compact.operation.operands[0] = block.arguments[3]
    elif fault == 'escaping_tensor':
        terminator.operation.operands[1] = block.arguments[1]
    elif fault == 'full_payload':
        full.operation.operands[0] = block.arguments[3]
    elif fault == 'capture':
        first = next(op for op in body.operations if op.operation.name == 'arith.addi')
        first.operation.operands[1] = block.arguments[0]
    elif fault == 'body':
        list(body.operations)[-1].operation.operands[0] = body.arguments[-1]
    elif fault == 'output':
        generic.operation.operands[-1] = block.arguments[1]
    elif fault == 'result':
        terminator.operation.operands[0] = block.arguments[2]
    elif fault == 'attribute':
        generic.operation.attributes['unreviewed.meaning'] = ir.UnitAttr.get()
    else:
        maps = ['affine_map<(d0,d1)->(d0,d1)>', 'affine_map<(d0,d1)->()>',
                'affine_map<(d0,d1)->()>', 'affine_map<(d0,d1)->(d0,d1)>']
        maps[0 if fault == 'input_map' else 3] = 'affine_map<(d0,d1)->(d1,d0)>'
        generic.operation.attributes['indexing_maps'] = ir.Attribute.parse('[' + ', '.join(maps) + ']')
    module.operation.verify()
    Path(sys.argv[1]).write_text(str(module))
"""


@pytest.mark.parametrize(
    "fault",
    [
        "payload",
        "escaping_tensor",
        "full_payload",
        "capture",
        "body",
        "output",
        "result",
        "attribute",
        "input_map",
        "output_map",
    ],
)
def test_splat_checker_rejects_verified_mlir_mutants(tmp_path, lean_tools, fault):
    source = _captured_source().replace("2x3", "2x2")
    stage, count = _stage(tmp_path, source)
    assert count == 2
    after = stage / "after.mlir"
    original = after.read_bytes()
    changed = subprocess.run(
        [str(lean_tools.m2m_python), "-c", MUTATE, str(after), fault], capture_output=True, text=True, check=False
    )
    assert changed.returncode == 0, changed.stderr
    assert original != after.read_bytes()
    result = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "abstracted", result.get("reason")


@pytest.mark.parametrize("index_bits", [32, 64])
def test_splat_checker_retains_tensor_used_only_by_capture(tmp_path, lean_tools, index_bits):
    stage, count = _stage(tmp_path, _tensor_capture_source())
    assert count == 1
    result = check_typed_stage(stage, index_bits=index_bits, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "verified", result.get("reason")


@pytest.mark.parametrize("index_bits", [32, 64])
def test_splat_checker_multiple_generic_uses(tmp_path, lean_tools, index_bits):
    stage, count = _stage(tmp_path, _repeated_source())
    assert count == 2
    result = check_typed_stage(stage, index_bits=index_bits, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "verified", result.get("reason")
