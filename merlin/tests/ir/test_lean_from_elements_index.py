"""Constructed scalar selection learns coordinates from actual source arithmetic, without hints."""

import subprocess

import pytest
from test_lean_defined_conversions import certify
from test_lean_from_elements_select import MUTATE, _captured_source, _source, _stage
from test_lean_reshape_compose import _verify_mlir

from merlin.verify.lean.check import check_definedness, check_typed_stage

OPERATIONS = [("addi", 0, 1), ("subi", 2, 1), ("muli", 1, 1), ("andi", 3, 1), ("ori", 0, 1), ("xori", 3, 2)]


def computed_source(name="addi", a=0, b=1, *, element="f32", escape=False, capture=False, length=0):
    source = _captured_source(element) if capture else _source(element, escape=escape)
    statements = [f"%left = arith.constant {a} : index", f"%right = arith.constant {b} : index"]
    previous = "%left"
    for n in range(length):
        statements += [f"%p{n} = arith.addi {previous}, %left : index"]
        previous = f"%p{n}"
    statements += [f"%position = arith.{name} {previous}, %right : index"]
    return source.replace("%position = arith.constant 1 : index", "\n".join(statements))


@pytest.mark.parametrize("name,a,b", OPERATIONS)
@pytest.mark.parametrize("index_bits", [32, 64])
@pytest.mark.parametrize("escape", [False, True])
def test_actual_computed_index_selection(tmp_path, lean_tools, name, a, b, index_bits, escape):
    stage, before, after = _stage(tmp_path, lean_tools.m2m_python, computed_source(name, a, b, escape=escape))
    assert "tensor.extract" not in after.read_text()
    assert "arith.constant" not in after.read_text()  # No opposite-side index materialization hint exists.
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)


@pytest.mark.parametrize("index_bits", [32, 64])
def test_computed_selection_scalar_remains_captured(tmp_path, lean_tools, index_bits):
    stage, before, after = _stage(tmp_path, lean_tools.m2m_python, computed_source(capture=True))
    assert "linalg.generic" in after.read_text() and "tensor.extract" not in after.read_text()
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)


@pytest.mark.parametrize("index_bits", [32, 64])
def test_source_chain_needs_no_intermediate_materialization(tmp_path, lean_tools, index_bits):
    stage, before, after = _stage(tmp_path, lean_tools.m2m_python, computed_source(length=32, escape=True))
    assert "arith.constant" not in after.read_text() and "tensor.extract" not in after.read_text()
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)


@pytest.mark.parametrize("index_bits", [32, 64])
def test_computed_coordinate_uses_actual_index_width(tmp_path, lean_tools, index_bits):
    stage, before, after = _stage(tmp_path, lean_tools.m2m_python, computed_source(a=1 << 32))
    assert "tensor.extract" not in after.read_text()
    result = check_typed_stage(stage, m2m_python=lean_tools.m2m_python, index_bits=index_bits)
    assert result["verdict"] == ("verified" if index_bits == 32 else "abstracted"), result.get("reason")
    result = check_definedness(
        before, "forward", tmp_path / "defined", m2m_python=lean_tools.m2m_python, index_bits=index_bits
    )
    assert result["verdict"] == ("verified" if index_bits == 32 else "abstracted"), result.get("reason")


@pytest.mark.parametrize("fault", ["coordinate", "payload", "return", "attribute"])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_computed_index_valid_mlir_mutations_refuse(tmp_path, lean_tools, fault, index_bits):
    stage, before, after = _stage(tmp_path, lean_tools.m2m_python, computed_source())
    mutated = before if fault in {"coordinate", "payload", "attribute"} else after
    if fault == "coordinate":
        before.write_text(before.read_text().replace("constant 0 : index", "constant 1 : index"))
        _verify_mlir(before, lean_tools.m2m_python)
    else:
        changed = subprocess.run(
            [str(lean_tools.m2m_python), "-c", MUTATE, str(mutated), fault],
            capture_output=True,
            text=True,
            check=False,
        )
        assert changed.returncode == 0, changed.stderr
    result = check_typed_stage(stage, m2m_python=lean_tools.m2m_python, index_bits=index_bits)
    assert result["verdict"] == "abstracted", result.get("reason")


def test_computed_coordinate_requires_actual_values(tmp_path, lean_tools):
    stage, before, after = _stage(tmp_path, lean_tools.m2m_python, computed_source())
    after.write_text(after.read_text().replace(") ->", ", %unknown: index) ->", 1))
    text = before.read_text().replace("%x2: f32)", "%x2: f32, %unknown: index)")
    text = text.replace("%left = arith.constant 0 : index", "%left = arith.addi %unknown, %unknown : index")
    before.write_text(text)
    _verify_mlir(before, lean_tools.m2m_python)
    _verify_mlir(after, lean_tools.m2m_python)
    assert check_typed_stage(stage, m2m_python=lean_tools.m2m_python)["verdict"] == "abstracted"
