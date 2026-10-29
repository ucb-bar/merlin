"""Observed dense rank-zero extraction, checked materialization and conservative scope."""

import subprocess

import pytest
from test_lean_reshape_compose import _canonicalize, _verify_mlir

from merlin.verify.lean.check import check_definedness, check_support, check_typed_stage, compose_chain

CASES = [
    ("i1", "true"),
    ("i13", "-1"),
    ("i32", "-2147483648"),
    ("index", "-1"),
    ("f16", "0x8000"),
    ("bf16", "0x8000"),
    ("f32", "0x7FC01234"),
    ("f64", "0x7FF8000000001234"),
]


def _stage(tmp_path, compiler, source):
    stage, before, after = _canonicalize(tmp_path, compiler, source)
    (stage / "meta.json").write_text('{"boundary": "dense-scalar-extraction"}')
    return stage, before, after


def _source(element="f32", literal="0x7FC01234", *, escape=False):
    tensor = f"tensor<{element}>"
    return f"""func.func @forward(%x: {element}) -> {f"({element}, {tensor})" if escape else element} {{
      %carrier = arith.constant dense<{literal}> : {tensor}
      %value = tensor.extract %carrier[] : {tensor}
      return {f"%value, %carrier : {element}, {tensor}" if escape else f"%value : {element}"}
    }}"""


def _captured_source(*, local=False):
    carrier = """%carrier = arith.constant dense<0x7FC01234> : tensor<f32>
      %value = tensor.extract %carrier[] : tensor<f32>"""
    return f"""func.func @forward(%other: f32, %condition: i1,
        %payload: tensor<2xf32>, %init: tensor<2xf32>) -> tensor<2xf32> {{
      {"" if local else carrier}
      %result = linalg.generic {{indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
          iterator_types = ["parallel"]}} ins(%payload : tensor<2xf32>) outs(%init : tensor<2xf32>) {{
        ^bb0(%p: f32, %old: f32):
          {carrier if local else ""}
          %selected = arith.select %condition, %value, %p : f32
          linalg.yield %selected : f32
      }} -> tensor<2xf32>
      return %result : tensor<2xf32>
    }}"""


@pytest.mark.parametrize("element,literal", CASES)
@pytest.mark.parametrize("escape", [False, True])
def test_actual_dense_rankzero_extraction(tmp_path, lean_tools, element, literal, escape):
    stage, _, after = _stage(tmp_path, lean_tools.m2m_python, _source(element, literal, escape=escape))
    assert "tensor.extract" not in after.read_text()
    assert after.read_text().count("arith.constant") == 1 + int(escape)
    checked = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert checked["verdict"] == "verified", checked.get("reason")


@pytest.mark.parametrize("element,literal", [CASES[1], CASES[3], CASES[6]])
@pytest.mark.parametrize("index_bits", [32, 64])
@pytest.mark.parametrize("escape", [False, True])
def test_dense_extraction_strict_chain(tmp_path, lean_tools, element, literal, index_bits, escape):
    stage, before, after = _stage(tmp_path, lean_tools.m2m_python, _source(element, literal, escape=escape))
    checked = check_typed_stage(stage, index_bits=index_bits, m2m_python=lean_tools.m2m_python)
    assert checked["verdict"] == "verified", checked.get("reason")
    for checker, name in [(check_support, "support"), (check_definedness, "defined")]:
        checked = checker(before, "forward", tmp_path / name, index_bits=index_bits, m2m_python=lean_tools.m2m_python)
        assert checked["verdict"] == "verified", checked.get("reason")

    def replay():
        return compose_chain(
            [stage],
            tmp_path / "support/receipt.json",
            "forward",
            tmp_path / "chain.json",
            definedness=tmp_path / "defined/receipt.json",
            m2m_python=lean_tools.m2m_python,
        )

    checked = replay()
    assert checked["verdict"] == "verified", checked.get("reason")
    after.write_text(after.read_text() + "\n// stale native output\n")
    assert replay()["verdict"] == "abstracted"


def test_dense_scalar_captured_by_structured_body(tmp_path, lean_tools):
    stage, before, after = _stage(tmp_path, lean_tools.m2m_python, _captured_source())
    assert "linalg.generic" in after.read_text()
    assert "tensor.extract" not in after.read_text()
    checked = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert checked["verdict"] == "verified", checked.get("reason")
    for checker, name in [(check_support, "support"), (check_definedness, "defined")]:
        checked = checker(before, "forward", tmp_path / name, m2m_python=lean_tools.m2m_python)
        assert checked["verdict"] == "verified", checked.get("reason")


MUTATE = """
import sys
from pathlib import Path
from torch_mlir import ir
with ir.Context(), ir.Location.unknown():
    m=ir.Module.parse(Path(sys.argv[1]).read_text())
    block=m.body.operations[0].regions[0].blocks[0]
    constants=[op for op in block.operations if op.operation.name=='arith.constant']
    scalar=next(op for op in constants if not isinstance(op.results[0].type, ir.RankedTensorType))
    if sys.argv[2]=='attribute':
        scalar.operation.attributes['unknown.meaning']=ir.UnitAttr.get()
    elif sys.argv[2]=='escape':
        tensor=next(op for op in constants if isinstance(op.results[0].type, ir.RankedTensorType))
        tensor.operation.attributes['value']=ir.DenseElementsAttr.parse('dense<1.0> : tensor<f32>')
    elif sys.argv[2]=='capture':
        generic=next(op for op in block.operations if op.operation.name=='linalg.generic')
        select=next(op for op in generic.regions[0].blocks[0].operations if op.operation.name=='arith.select')
        select.operation.operands[1]=block.arguments[0]
    else:
        scalar.operation.attributes['value']=ir.FloatAttr.get(ir.F32Type.get(), 1.0)
    m.operation.verify()
    Path(sys.argv[1]).write_text(str(m))
"""


@pytest.mark.parametrize("fault", ["bits", "attribute", "escape", "capture"])
def test_dense_extraction_valid_mlir_mutations(tmp_path, lean_tools, fault):
    source = _captured_source() if fault == "capture" else _source(escape=True)
    stage, _, after = _stage(tmp_path, lean_tools.m2m_python, source)
    assert check_typed_stage(stage, m2m_python=lean_tools.m2m_python)["verdict"] == "verified"
    changed = subprocess.run(
        [str(lean_tools.m2m_python), "-c", MUTATE, str(after), fault], capture_output=True, text=True, check=False
    )
    assert changed.returncode == 0, changed.stderr
    assert check_typed_stage(stage, m2m_python=lean_tools.m2m_python)["verdict"] == "abstracted"


def test_local_dense_constant_hoisting_is_outside_current_family(tmp_path, lean_tools):
    stage, before, after = _stage(tmp_path, lean_tools.m2m_python, _captured_source(local=True))
    assert "tensor.extract" not in after.read_text()
    # The actual pass also hoists the materialized scalar across the generic region boundary.
    # This family proves local extraction, not that additional movement/capture relation.
    assert check_typed_stage(stage, m2m_python=lean_tools.m2m_python)["verdict"] == "abstracted"
    for checker, name in [(check_support, "support"), (check_definedness, "defined")]:
        checked = checker(before, "forward", tmp_path / name, m2m_python=lean_tools.m2m_python)
        assert checked["verdict"] == "verified", checked.get("reason")


def test_local_dense_extraction_with_scope_preserved(tmp_path, lean_tools):
    source = _captured_source(local=True)
    stage, _, after = _stage(tmp_path, lean_tools.m2m_python, source)
    # Propose only local materialization; the actual canonicalizer additionally hoists it.
    target = source.replace(
        "%carrier = arith.constant dense<0x7FC01234> : tensor<f32>\n"
        "      %value = tensor.extract %carrier[] : tensor<f32>",
        "%value = arith.constant 0x7FC01234 : f32",
    )
    assert target != source
    after.write_text(target)
    _verify_mlir(after, lean_tools.m2m_python)
    checked = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert checked["verdict"] == "verified", checked.get("reason")


@pytest.mark.parametrize("shape,literal", [("0x", ""), ("2x", "7")])
def test_positive_rank_dense_extraction_requires_bounds(tmp_path, lean_tools, shape, literal):
    source = f"""func.func @forward(%index: index) -> i32 {{
      %carrier = arith.constant dense<{literal}> : tensor<{shape}i32>
      %value = tensor.extract %carrier[%index] : tensor<{shape}i32>
      return %value : i32
    }}"""
    stage, before, after = _stage(tmp_path, lean_tools.m2m_python, source)
    after.write_text("func.func @forward(%index: index) -> i32 { %v = arith.constant 7 : i32 return %v : i32 }")
    _verify_mlir(after, lean_tools.m2m_python)
    assert check_typed_stage(stage, m2m_python=lean_tools.m2m_python)["verdict"] == "abstracted"
    checked = check_definedness(before, "forward", tmp_path / "defined", m2m_python=lean_tools.m2m_python)
    assert checked["verdict"] == "abstracted"
