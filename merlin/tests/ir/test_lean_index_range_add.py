"""Actual unsigned-cast ranges survive addition only when the whole signed result cannot wrap."""

import pytest
from test_lean_defined_bounds import _chain, _program
from test_lean_index_cast_bounds import _captured_cast
from test_lean_reshape_compose import _verify_mlir
from test_lean_splat_nested import _stage as _splat_stage

from merlin.verify.lean.check import check_definedness, check_support, check_typed_stage


def _source(width=8, size=257, offset=1, *, extract=True, unknown=False):
    input_type = "index" if unknown else f"i{width}"
    result = "i32" if extract else "index"
    cast = "" if unknown else f'%cast = "arith.index_castui"(%input) : (i{width}) -> index'
    return f"""func.func @subject(%data: tensor<{size}xi32>, %input: {input_type}) -> {result} {{
      {cast}
      %offset = arith.constant {offset} : index
      %index = arith.addi {"%input" if unknown else "%cast"}, %offset : index
      {f"%value = tensor.extract %data[%index] : tensor<{size}xi32>" if extract else ""}
      return {"%value : i32" if extract else "%index : index"}
    }}"""


def _captured_offset():
    source = _captured_cast().replace("tensor<256xi32>", "tensor<257xi32>")
    return source.replace(
        '%coordinate0 = "arith.index_castui"(%index_source) : (i8) -> index',
        '%cast = "arith.index_castui"(%index_source) : (i8) -> index\n'
        "%offset = arith.constant 1 : index\n%coordinate0 = arith.addi %cast, %offset : index",
    )


@pytest.mark.parametrize("index_bits", [32, 64])
@pytest.mark.parametrize("width,offset", [(1, 0), (1, 1), (3, 1), (8, 1), (8, 7)])
def test_unsigned_range_plus_offset_is_defined(tmp_path, lean_tools, index_bits, width, offset):
    path = _program(tmp_path, _source(width, (1 << width) + offset, offset), lean_tools.m2m_python)
    result = check_definedness(
        path, "subject", tmp_path / "defined", index_bits=index_bits, m2m_python=lean_tools.m2m_python
    )
    assert result["verdict"] == "verified", result.get("reason")


@pytest.mark.parametrize("index_bits", [32, 64])
@pytest.mark.parametrize("options", [{"size": 256}, {"offset": -1}, {"size": 0}, {"unknown": True}])
def test_unsafe_or_unknown_range_does_not_establish_bounds(tmp_path, lean_tools, index_bits, options):
    path = _program(tmp_path, _source(**options), lean_tools.m2m_python)
    support = check_support(
        path, "subject", tmp_path / "support", index_bits=index_bits, m2m_python=lean_tools.m2m_python
    )
    assert support["verdict"] == "verified", support.get("reason")
    result = check_definedness(
        path, "subject", tmp_path / "defined", index_bits=index_bits, m2m_python=lean_tools.m2m_python
    )
    assert result["verdict"] == "abstracted", result.get("reason")


@pytest.mark.parametrize("index_bits", [32, 64])
@pytest.mark.parametrize("offset", [0, 1])
def test_signed_limit_is_checked_at_actual_index_width(tmp_path, lean_tools, index_bits, offset):
    path = _program(tmp_path, _source(31, (1 << 31) + offset, offset), lean_tools.m2m_python)
    result = check_definedness(
        path, "subject", tmp_path / "defined", index_bits=index_bits, m2m_python=lean_tools.m2m_python
    )
    assert result["verdict"] == ("abstracted" if index_bits == 32 and offset == 1 else "verified"), result.get("reason")


def test_lost_overflow_bounds_retain_initialized_arithmetic(tmp_path, lean_tools):
    path = _program(tmp_path, _source(31, 2, 1, extract=False), lean_tools.m2m_python)
    result = check_definedness(path, "subject", tmp_path / "defined", index_bits=32, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "verified", result.get("reason")


@pytest.mark.parametrize("index_bits", [32, 64])
def test_actual_splat_pass_preserves_captured_interval_addition(tmp_path, lean_tools, index_bits):
    stage, count = _splat_stage(tmp_path, _captured_offset())
    assert count == 1
    _chain(tmp_path, stage, "subject", lean_tools, index_bits)


def test_wrong_inbounds_offset_is_not_verified(tmp_path, lean_tools):
    stage, _ = _splat_stage(tmp_path, _captured_offset())
    _chain(tmp_path, stage, "subject", lean_tools, 64)
    after = stage / "after.mlir"
    after.write_text(after.read_text().replace("arith.constant 1 : index", "arith.constant 0 : index", 1))
    _verify_mlir(after, lean_tools.m2m_python)
    result = check_definedness(after, "subject", tmp_path / "mutant-defined", m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "verified", result.get("reason")
    result = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] != "verified"
