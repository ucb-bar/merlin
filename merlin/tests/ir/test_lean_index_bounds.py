"""Actual modular index arithmetic contributes producer-proved extraction bounds."""

import pytest
from test_lean_defined_bounds import _captured_source, _chain, _program
from test_lean_reshape_compose import _canonicalize, _verify_mlir
from test_lean_splat_nested import _stage as _splat_stage

from merlin.verify.lean.check import check_definedness, check_support, check_typed_stage

SAFE = [("addi", 1, 1, 3), ("subi", 1, 0, 2), ("muli", 1, 2, 3), ("muli", 0, 0, 1)]


def _source(kind="addi", lhs=1, rhs=1, size=3, *, dynamic=False, flags="", extract=True):
    extra = ", %unknown: index" if dynamic else ""
    result_type = "i32" if extract else "index"
    return f"""func.func @subject(%data: tensor<{size}xi32>{extra}) -> {result_type} {{
      %lhs = arith.constant {lhs} : index
      %rhs = arith.constant {rhs} : index
      %index = arith.{kind} {"%unknown" if dynamic else "%lhs"}, %rhs {flags}: index
      {f"%value = tensor.extract %data[%index] : tensor<{size}xi32>" if extract else ""}
      return {"%value : i32" if extract else "%index : index"}
    }}"""


def _captured_arithmetic():
    return _captured_source().replace(
        "%coordinate1 = arith.constant 2 : index",
        "%one = arith.constant 1 : index\n%coordinate1 = arith.addi %one, %one : index",
    )


@pytest.mark.parametrize("index_bits", [32, 64])
@pytest.mark.parametrize("kind,lhs,rhs,size", SAFE)
def test_actual_index_arithmetic_bounds(tmp_path, lean_tools, index_bits, kind, lhs, rhs, size):
    stage, before, after = _canonicalize(tmp_path, lean_tools.m2m_python, _source(kind, lhs, rhs, size))
    assert f"arith.{kind}" in before.read_text()
    assert f"arith.{kind}" not in after.read_text()
    for side, path in [("before", before), ("after", after)]:
        result = check_definedness(
            path, "subject", tmp_path / side, index_bits=index_bits, m2m_python=lean_tools.m2m_python
        )
        assert result["verdict"] == "verified", result.get("reason")


@pytest.mark.parametrize("index_bits", [32, 64])
@pytest.mark.parametrize("kind,lhs,rhs,size", [("subi", 0, 1, 2), ("addi", 1, 1, 2), ("muli", 0, 0, 0)])
def test_unsafe_arithmetic_indices_refuse(tmp_path, lean_tools, index_bits, kind, lhs, rhs, size):
    path = _program(tmp_path, _source(kind, lhs, rhs, size), lean_tools.m2m_python)
    support = check_support(
        path, "subject", tmp_path / "support", index_bits=index_bits, m2m_python=lean_tools.m2m_python
    )
    assert support["verdict"] == "verified", support.get("reason")
    result = check_definedness(
        path, "subject", tmp_path / "defined", index_bits=index_bits, m2m_python=lean_tools.m2m_python
    )
    assert result["verdict"] == "abstracted", result.get("reason")


@pytest.mark.parametrize("index_bits", [32, 64])
@pytest.mark.parametrize("kind,lhs,rhs", [("muli", 2147483647, 2147483647), ("addi", 4294967296, 1)])
def test_arithmetic_bounds_use_actual_width(tmp_path, lean_tools, index_bits, kind, lhs, rhs):
    path = _program(tmp_path, _source(kind, lhs, rhs, 2), lean_tools.m2m_python)
    result = check_definedness(
        path, "subject", tmp_path / "defined", index_bits=index_bits, m2m_python=lean_tools.m2m_python
    )
    assert result["verdict"] == ("verified" if index_bits == 32 else "abstracted"), result.get("reason")


@pytest.mark.parametrize("index_bits", [32, 64])
@pytest.mark.parametrize("extract", [False, True])
def test_unknown_index_arithmetic_retains_initialization_only(tmp_path, lean_tools, index_bits, extract):
    path = _program(tmp_path, _source(dynamic=True, extract=extract), lean_tools.m2m_python)
    result = check_definedness(
        path, "subject", tmp_path / "defined", index_bits=index_bits, m2m_python=lean_tools.m2m_python
    )
    assert result["verdict"] == ("abstracted" if extract else "verified"), result.get("reason")


@pytest.mark.parametrize("index_bits", [32, 64])
def test_arithmetic_capture_composes_with_actual_splat_pass(tmp_path, lean_tools, index_bits):
    stage, count = _splat_stage(tmp_path, _captured_arithmetic())
    assert count == 1
    _chain(tmp_path, stage, "subject", lean_tools, index_bits)


@pytest.mark.parametrize("flags", ["overflow<nsw>", "overflow<nuw>"])
def test_flagged_arithmetic_does_not_create_bounds(tmp_path, lean_tools, flags):
    path = _program(tmp_path, _source(flags=flags), lean_tools.m2m_python)
    result = check_definedness(path, "subject", tmp_path / "defined", m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "abstracted", result.get("reason")


def test_changed_arithmetic_capture_is_not_a_verified_rewrite(tmp_path, lean_tools):
    stage, _ = _splat_stage(tmp_path, _captured_arithmetic())
    _chain(tmp_path, stage, "subject", lean_tools, 64)
    after = stage / "after.mlir"
    text = after.read_text()
    # Actual MLIR prints constants by their value; c1 also supplies the other in-bounds coordinate.
    assert "arith.constant 1 : index" in text
    after.write_text(text.replace("arith.constant 1 : index", "arith.constant 0 : index", 1))
    _verify_mlir(after, lean_tools.m2m_python)
    result = check_definedness(after, "subject", tmp_path / "mutant-defined", m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "verified", result.get("reason")
    result = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] != "verified"
