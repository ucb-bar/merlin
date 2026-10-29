"""Different local rule families compose on one actual upstream canonicalization boundary."""

import pytest
from test_lean_integer_fold import canonicalize

from merlin.verify.lean.check import check_definedness, check_stage, check_support, compose_chain


def combined_source(escape_constant=False):
    extra_type, extra_value = (", i13", ", %zero") if escape_constant else ("", "")
    return f"""func.func @forward(%x: i13, %a: tensor<6x4xi32>, %keep: i32)
        -> (i13, tensor<24xi32>, tensor<6x4xi32>, i32{extra_type}) {{
      %zero = arith.constant 0 : i13
      %identity = arith.addi %x, %zero : i13
      %mid = tensor.expand_shape %a [[0,1],[2]] output_shape [2,3,4]
        : tensor<6x4xi32> into tensor<2x3x4xi32>
      %reshape = tensor.collapse_shape %mid [[0,1,2]] : tensor<2x3x4xi32> into tensor<24xi32>
      return %identity, %reshape, %a, %keep{extra_value}
        : i13, tensor<24xi32>, tensor<6x4xi32>, i32{extra_type}
    }}"""


@pytest.mark.parametrize("index_bits", [32, 64])
@pytest.mark.parametrize("escape_constant", [False, True])
def test_shared_rule_families_compose(tmp_path, lean_tools, index_bits, escape_constant):
    stage, before, after = canonicalize(tmp_path, lean_tools, combined_source(escape_constant))
    assert "arith.addi" not in after.read_text()
    assert "tensor.expand_shape" not in after.read_text()
    assert after.read_text().count("tensor.collapse_shape") == 1
    receipt = check_stage(stage, m2m_python=lean_tools.m2m_python)
    assert receipt["verdict"] == "verified", receipt.get("reason")
    for checker, name in [(check_support, "support"), (check_definedness, "defined")]:
        receipt = checker(before, "forward", tmp_path / name, index_bits=index_bits, m2m_python=lean_tools.m2m_python)
        assert receipt["verdict"] == "verified", receipt.get("reason")

    def replay():
        return compose_chain(
            [stage],
            tmp_path / "support/receipt.json",
            "forward",
            tmp_path / "chain.json",
            definedness=tmp_path / "defined/receipt.json",
            m2m_python=lean_tools.m2m_python,
        )

    receipt = replay()
    assert receipt["verdict"] == "verified", receipt.get("reason")
    before.write_text(before.read_text() + "\n// live source bytes changed\n")
    assert replay()["verdict"] == "abstracted"
