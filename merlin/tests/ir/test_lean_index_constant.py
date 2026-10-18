"""Literal spelling is normalized from actual checked-width values on both real pass sides."""

import pytest
from test_lean_defined_conversions import certify
from test_lean_index_fold import NAMES, VALUES, mutate, source, stage_for

from merlin.verify.lean.check import check_typed_stage


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("a,b", VALUES)
@pytest.mark.parametrize("escape", [False, True])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_full_actual_index_fold_matrix_has_defined_chains(tmp_path, lean_tools, name, a, b, escape, index_bits):
    stage, before, after = stage_for(tmp_path, lean_tools, source(name, a, b, escape=escape))
    assert f"arith.{name}" not in after.read_text()
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)


@pytest.mark.parametrize("name", ["addi", "ori", "xori"])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_wrapped_literal_aliases_preserve_escaping_values_and_captures(tmp_path, lean_tools, name, index_bits):
    stage, before, after = stage_for(tmp_path, lean_tools, source(name, -(1 << 63), 3, escape=True, capture=True))
    assert "linalg.generic" in after.read_text()
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)


@pytest.mark.parametrize("name", ["addi", "ori", "xori"])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_actual_wrapped_literal_bit_corruption_still_refuses(tmp_path, lean_tools, name, index_bits):
    stage, before, after = stage_for(tmp_path, lean_tools, source(name, -(1 << 63), 3, escape=True))
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)
    mutate(after, lean_tools, "value")
    result = check_typed_stage(stage, m2m_python=lean_tools.m2m_python, index_bits=index_bits)
    assert result["verdict"] == "abstracted", result.get("reason")
