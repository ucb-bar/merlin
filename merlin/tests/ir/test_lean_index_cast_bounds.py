"""Unsigned widening bounds cover every initialized bit pattern at its actual fixed width."""

import pytest
from test_lean_defined_bounds import _captured_source, _chain, _program
from test_lean_reshape_compose import _verify_mlir
from test_lean_splat_nested import _stage as _splat_stage

from merlin.verify.lean.check import check_definedness, check_support, check_typed_stage


def _source(width=8, size=256, *, signed=False, flags="", extract=True):
    name = "index_cast" if signed else "index_castui"
    result = "i32" if extract else "index"
    return f"""func.func @subject(%data: tensor<{size}xi32>, %input: i{width}) -> {result} {{
      %index = arith.{name} %input {flags}: i{width} to index
      {f"%value = tensor.extract %data[%index] : tensor<{size}xi32>" if extract else ""}
      return {"%value : i32" if extract else "%index : index"}
    }}"""


def _captured_cast():
    # The production xDSL context carries index_castui as an unregistered op; generic syntax
    # represents the same authoritative MLIR operation without relying on a custom xDSL parser.
    source = _captured_source(shape=(256,), coordinates=(0,))
    source = source.replace("%alternate: i32)", "%alternate: i32, %index_source: i8)")
    return source.replace(
        "%coordinate0 = arith.constant 0 : index",
        '%coordinate0 = "arith.index_castui"(%index_source) : (i8) -> index',
    )


@pytest.mark.parametrize("index_bits", [32, 64])
@pytest.mark.parametrize("width", [1, 3, 8])
def test_unsigned_widening_proves_whole_input_range(tmp_path, lean_tools, index_bits, width):
    path = _program(tmp_path, _source(width, 1 << width), lean_tools.m2m_python)
    result = check_definedness(
        path, "subject", tmp_path / "defined", index_bits=index_bits, m2m_python=lean_tools.m2m_python
    )
    assert result["verdict"] == "verified", result.get("reason")


@pytest.mark.parametrize("index_bits", [32, 64])
@pytest.mark.parametrize("width,size", [(1, 0), (1, 1), (8, 255), (32, 2), (64, 2)])
def test_insufficient_unsigned_range_refuses(tmp_path, lean_tools, index_bits, width, size):
    path = _program(tmp_path, _source(width, size), lean_tools.m2m_python)
    support = check_support(
        path, "subject", tmp_path / "support", index_bits=index_bits, m2m_python=lean_tools.m2m_python
    )
    assert support["verdict"] == "verified", support.get("reason")
    result = check_definedness(
        path, "subject", tmp_path / "defined", index_bits=index_bits, m2m_python=lean_tools.m2m_python
    )
    assert result["verdict"] == "abstracted", result.get("reason")


@pytest.mark.parametrize("index_bits", [32, 64])
@pytest.mark.parametrize("signed,flags", [(True, ""), (False, "nneg")])
def test_signed_or_promised_input_does_not_invent_bounds(tmp_path, lean_tools, index_bits, signed, flags):
    path = _program(tmp_path, _source(signed=signed, flags=flags), lean_tools.m2m_python)
    result = check_definedness(
        path, "subject", tmp_path / "defined", index_bits=index_bits, m2m_python=lean_tools.m2m_python
    )
    assert result["verdict"] == "abstracted", result.get("reason")


@pytest.mark.parametrize("index_bits", [32, 64])
@pytest.mark.parametrize("signed", [False, True])
def test_casts_without_extraction_are_initialized(tmp_path, lean_tools, index_bits, signed):
    path = _program(tmp_path, _source(64, signed=signed, extract=False), lean_tools.m2m_python)
    result = check_definedness(
        path, "subject", tmp_path / "defined", index_bits=index_bits, m2m_python=lean_tools.m2m_python
    )
    assert result["verdict"] == "verified", result.get("reason")


@pytest.mark.parametrize("index_bits", [32, 64])
def test_actual_splat_pass_retains_unsigned_capture_range(tmp_path, lean_tools, index_bits):
    stage, count = _splat_stage(tmp_path, _captured_cast())
    assert count == 1
    _chain(tmp_path, stage, "subject", lean_tools, index_bits)


def test_unsigned_to_signed_mutation_refuses_proof(tmp_path, lean_tools):
    stage, _ = _splat_stage(tmp_path, _captured_cast())
    _chain(tmp_path, stage, "subject", lean_tools, 64)
    after = stage / "after.mlir"
    after.write_text(after.read_text().replace("arith.index_castui", "arith.index_cast", 1))
    _verify_mlir(after, lean_tools.m2m_python)
    result = check_definedness(after, "subject", tmp_path / "mutant-defined", m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "abstracted", result.get("reason")
    result = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] != "verified"
