"""Actual canonicalization composes partial casts, insertion aliases and comparison order."""

import pytest
from test_lean_defined_conversions import certify
from test_lean_integer_compare_order import mutate
from test_lean_integer_fold import canonicalize

from merlin.verify.lean.check import check_definedness, check_typed_stage


def source(kind="extui", escape="none"):
    values, types = ["%mixed", "%keep"], ["i13", "i32"]
    if escape in {"tensor", "both"}:
        values.append("%inserted")
        types.append("tensor<i13>")
    if escape in {"wide", "both"}:
        values.append("%wide")
        types.append("i32")
    return f"""func.func @forward(%x: i8, %destination: tensor<i13>, %keep: i32)
        -> ({", ".join(types)}) {{
      %cap = arith.constant 5 : i13
      %wide = arith.{kind} %x : i8 to i32
      %partial = arith.trunci %wide : i32 to i13
      %inserted = tensor.insert %partial into %destination[] : tensor<i13>
      %read = tensor.extract %inserted[] : tensor<i13>
      %compared = arith.cmpi slt, %cap, %read : i13
      %bit = arith.extui %compared : i1 to i13
      %mixed = arith.addi %bit, %read : i13
      return {", ".join(values)} : {", ".join(types)}
    }}"""


@pytest.mark.parametrize("kind", ["extui", "extsi"])
@pytest.mark.parametrize("escape", ["none", "tensor", "wide", "both"])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_actual_three_family_rewrite_has_strict_chain(tmp_path, lean_tools, kind, escape, index_bits):
    stage, before, after = canonicalize(tmp_path, lean_tools, source(kind, escape))
    target = after.read_text()
    assert "tensor.extract" not in target and "arith.trunci" not in target
    assert target.count("tensor.insert") == int(escape in {"tensor", "both"})
    assert "arith.cmpi sgt" in target
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)


@pytest.mark.parametrize("fault", ["predicate", "input", "bits", "keep"])
def test_composed_target_mutation_is_defined_but_refused(tmp_path, lean_tools, fault):
    stage, before, after = canonicalize(tmp_path, lean_tools, source(escape="none" if fault == "keep" else "both"))
    certify(stage, before, lean_tools, tmp_path)
    mutate(after, lean_tools, fault)
    defined = check_definedness(after, "forward", tmp_path / "mutant-defined", m2m_python=lean_tools.m2m_python)
    assert defined["verdict"] == "verified", defined.get("reason")
    refused = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert refused["verdict"] == "abstracted", refused.get("reason")
