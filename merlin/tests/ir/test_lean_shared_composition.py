"""Independent normalizers compose across actual shared-pass rewrite families."""

import pytest
from test_lean_scalar_carrier import _mixed_source, _stage

from merlin.verify.lean import check


@pytest.mark.parametrize("index_bits", [32, 64])
def test_carrier_exposes_nested_cast_composition(tmp_path, lean_tools, index_bits):
    stage, before, after = _stage(tmp_path, lean_tools.m2m_python, _mixed_source())
    target = after.read_text()
    assert "tensor.from_elements" not in target
    assert "tensor.extract" not in target
    assert "arith.extsi" not in target
    assert "linalg.generic" in target
    boundary = check.check_typed_stage(stage, m2m_python=lean_tools.m2m_python, index_bits=index_bits)
    assert boundary["verdict"] == "verified", boundary.get("reason")
    for checker, label in [(check.check_support, "support"), (check.check_definedness, "defined")]:
        receipt = checker(before, "forward", tmp_path / label, m2m_python=lean_tools.m2m_python, index_bits=index_bits)
        assert receipt["verdict"] == "verified", receipt.get("reason")
    result = check.compose_chain(
        [stage],
        tmp_path / "support/receipt.json",
        "forward",
        tmp_path / "chain.json",
        definedness=tmp_path / "defined/receipt.json",
        m2m_python=lean_tools.m2m_python,
    )
    assert result["verdict"] == "verified", result.get("reason")
    assert result["theorem"] == check.TYPED_EXTENDED_DEFINED_CHAIN_THEOREM
