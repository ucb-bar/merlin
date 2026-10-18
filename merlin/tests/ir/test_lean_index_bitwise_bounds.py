"""Actual index bitwise operations preserve initialization and only proved singleton bounds."""

import pytest
from test_lean_defined_bounds import _captured_source, _chain, _program
from test_lean_reshape_compose import _canonicalize, _verify_mlir
from test_lean_splat_nested import _stage as _splat_stage

from merlin.verify.lean.check import check_definedness, check_support, check_typed_stage

CASES = [("andi", 3, 1, 2), ("ori", 0, 1, 2), ("xori", 3, 1, 3), ("andi", 0, 0, 1)]


def _source(kind="andi", lhs=3, rhs=1, size=2, *, dynamic=False, extract=True, attributes=""):
    extra = ", %unknown: index" if dynamic else ""
    result = "i32" if extract else "index"
    return f"""func.func @subject(%data: tensor<{size}xi32>{extra}) -> {result} {{
      %lhs = arith.constant {lhs} : index
      %rhs = arith.constant {rhs} : index
      %index = "arith.{kind}"({"%unknown" if dynamic else "%lhs"}, %rhs) {attributes} : (index, index) -> index
      {f"%value = tensor.extract %data[%index] : tensor<{size}xi32>" if extract else ""}
      return {"%value : i32" if extract else "%index : index"}
    }}"""


def _captured_bitwise():
    return _captured_source().replace(
        "%coordinate1 = arith.constant 2 : index",
        "%three = arith.constant 3 : index\n%one = arith.constant 1 : index\n"
        "%coordinate1 = arith.xori %three, %one : index",
    )


@pytest.mark.parametrize("index_bits", [32, 64])
@pytest.mark.parametrize("kind,lhs,rhs,size", CASES)
def test_actual_bitwise_constants_establish_extraction_bounds(tmp_path, lean_tools, index_bits, kind, lhs, rhs, size):
    _, before, after = _canonicalize(tmp_path, lean_tools.m2m_python, _source(kind, lhs, rhs, size))
    assert f"arith.{kind}" in before.read_text()
    assert f"arith.{kind}" not in after.read_text()
    # This unit proves safety; the separately qualified IndexFold family owns rewrite refinement.
    for label, path in [("before", before), ("after", after)]:
        result = check_definedness(
            path, "subject", tmp_path / label, index_bits=index_bits, m2m_python=lean_tools.m2m_python
        )
        assert result["verdict"] == "verified", result.get("reason")


@pytest.mark.parametrize("index_bits", [32, 64])
@pytest.mark.parametrize("kind,lhs,rhs,size", [("andi", 4294967296, 4294967296, 1), ("ori", 4294967296, 1, 2)])
def test_bitwise_singletons_use_actual_wrapped_width(tmp_path, lean_tools, index_bits, kind, lhs, rhs, size):
    path = _program(tmp_path, _source(kind, lhs, rhs, size), lean_tools.m2m_python)
    result = check_definedness(
        path, "subject", tmp_path / "defined", index_bits=index_bits, m2m_python=lean_tools.m2m_python
    )
    assert result["verdict"] == ("verified" if index_bits == 32 else "abstracted"), result.get("reason")


@pytest.mark.parametrize("index_bits", [32, 64])
@pytest.mark.parametrize("kind", ["andi", "ori", "xori"])
@pytest.mark.parametrize("extract", [False, True])
def test_unknown_bitwise_values_are_initialized_without_invented_bounds(
    tmp_path, lean_tools, index_bits, kind, extract
):
    path = _program(tmp_path, _source(kind, rhs=0, dynamic=True, extract=extract), lean_tools.m2m_python)
    support = check_support(
        path, "subject", tmp_path / "support", index_bits=index_bits, m2m_python=lean_tools.m2m_python
    )
    assert support["verdict"] == "verified", support.get("reason")
    result = check_definedness(
        path, "subject", tmp_path / "defined", index_bits=index_bits, m2m_python=lean_tools.m2m_python
    )
    assert result["verdict"] == ("abstracted" if extract else "verified"), result.get("reason")


@pytest.mark.parametrize("attributes", ["{disjoint}", "{fastmath = #arith.fastmath<nnan>}", "{unreviewed.meaning}"])
def test_valid_mlir_bitwise_promises_or_unknown_attributes_refuse(tmp_path, lean_tools, attributes):
    path = _program(tmp_path, _source("ori", 0, 1, 2, attributes=attributes), lean_tools.m2m_python)
    result = check_definedness(path, "subject", tmp_path / "defined", m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "abstracted", result.get("reason")


@pytest.mark.parametrize("index_bits", [32, 64])
def test_bitwise_capture_has_actual_splat_defined_chain(tmp_path, lean_tools, index_bits):
    stage, count = _splat_stage(tmp_path, _captured_bitwise())
    assert count == 1
    _chain(tmp_path, stage, "subject", lean_tools, index_bits)


def test_wrong_inbounds_bitwise_operator_is_not_verified(tmp_path, lean_tools):
    stage, _ = _splat_stage(tmp_path, _captured_bitwise())
    _chain(tmp_path, stage, "subject", lean_tools, 64)
    after = stage / "after.mlir"
    assert "arith.xori" in after.read_text()
    after.write_text(after.read_text().replace("arith.xori", "arith.andi", 1))
    _verify_mlir(after, lean_tools.m2m_python)
    result = check_definedness(after, "subject", tmp_path / "mutant-defined", m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "verified", result.get("reason")
    assert check_typed_stage(stage, m2m_python=lean_tools.m2m_python)["verdict"] != "verified"
