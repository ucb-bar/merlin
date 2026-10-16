"""Closed-expression swaps preserve escaping values; unproved motion is refused."""

import json

import pytest

from merlin.verify.lean.check import check_definedness, check_stage, check_support, compose_chain


def _stage(tmp_path, before, after):
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "before.mlir").write_text(before)
    (stage / "after.mlir").write_text(after)
    (stage / "meta.json").write_text(json.dumps({"boundary": "closed-expression-motion"}))
    return stage


def _constants(scalar="i32", a="7", b="8"):
    before = f"""func.func @forward(%keep: i32) -> ({scalar}, {scalar}, i32) {{
      %a = arith.constant {a} : {scalar}
      %b = arith.constant {b} : {scalar}
      %sum = arith.addi %keep, %keep : i32
      return %a, %b, %sum : {scalar}, {scalar}, i32
    }}"""
    after = f"""func.func @forward(%saved: i32) -> ({scalar}, {scalar}, i32) {{
      %u = arith.constant {b} : {scalar}
      %v = arith.constant {a} : {scalar}
      %sum = arith.addi %saved, %saved : i32
      return %v, %u, %sum : {scalar}, {scalar}, i32
    }}"""
    return before, after


@pytest.mark.parametrize("scalar,a,b", [
    ("i1", "0", "1"), ("i8", "-128", "127"), ("i13", "-17", "19"),
    ("i32", "7", "8"), ("i64", "-9223372036854775808", "9223372036854775807"),
    ("index", "7", "8"), ("f32", "-0.0", "1.5"), ("f64", "-0.0", "1.5"),
])
def test_swapped_constants_and_defined_chain(tmp_path, lean_tools, scalar, a, b):
    stage = _stage(tmp_path, *_constants(scalar, a, b))
    receipt = check_stage(stage, m2m_python=lean_tools.m2m_python)
    assert receipt["verdict"] == "verified", receipt.get("reason")
    assert receipt["theorem"] == "MerlinVerify.Check.moduleCheck_sound"
    before = stage / "before.mlir"
    assert check_support(before, "forward", tmp_path / "support",
                         m2m_python=lean_tools.m2m_python)["verdict"] == "verified"
    receipt = check_definedness(before, "forward", tmp_path / "defined",
                               m2m_python=lean_tools.m2m_python)
    assert receipt["verdict"] == "verified", receipt.get("reason")
    chain = compose_chain([stage], tmp_path / "support/receipt.json", "forward", tmp_path / "chain.json",
                          m2m_python=lean_tools.m2m_python, definedness=tmp_path / "defined/receipt.json")
    assert chain["verdict"] == "verified", chain.get("reason")


@pytest.mark.parametrize("fault", ["constant", "escaping_uses", "argument_use", "unknown_attribute"])
def test_swapped_constants_reject_changed_computation(tmp_path, lean_tools, fault):
    before, after = _constants()
    if fault == "constant":
        after = after.replace("constant 7", "constant 9")
    elif fault == "escaping_uses":
        after = after.replace("return %v, %u", "return %u, %v")
    elif fault == "argument_use":
        after = after.replace("%saved, %saved", "%saved, %v")
    else:
        after = after.replace("constant 7 :", "constant 7 {unrecognized = true} :")
    stage = _stage(tmp_path, before, after)
    receipt = check_stage(stage, m2m_python=lean_tools.m2m_python)
    assert receipt["verdict"] == "abstracted", receipt.get("reason")


def test_independent_operand_taking_motion_remains_unsupported(tmp_path, lean_tools):
    before = """func.func @forward(%x: i32) -> (i32, i32) {
      %a = arith.constant 7 : i32
      %b = arith.addi %x, %x : i32
      return %a, %b : i32, i32
    }"""
    after = """func.func @forward(%x: i32) -> (i32, i32) {
      %b = arith.addi %x, %x : i32
      %a = arith.constant 7 : i32
      return %a, %b : i32, i32
    }"""
    stage = _stage(tmp_path, before, after)
    receipt = check_stage(stage, m2m_python=lean_tools.m2m_python)
    assert receipt["verdict"] == "abstracted", receipt.get("reason")


def test_motion_across_external_call_remains_unsupported(tmp_path, lean_tools):
    declaration = "func.func private @external() -> i32\n"
    before = declaration + """func.func @forward() -> (i32, i32) {
      %a = arith.constant 7 : i32
      %b = func.call @external() : () -> i32
      return %a, %b : i32, i32
    }"""
    after = declaration + """func.func @forward() -> (i32, i32) {
      %b = func.call @external() : () -> i32
      %a = arith.constant 7 : i32
      return %a, %b : i32, i32
    }"""
    stage = _stage(tmp_path, before, after)
    receipt = check_stage(stage, m2m_python=lean_tools.m2m_python)
    assert receipt["verdict"] == "abstracted", receipt.get("reason")


def test_multiple_disjoint_swaps_preserve_every_escaping_result(tmp_path, lean_tools):
    before = """func.func @forward() -> (i32, i32, i32, i32) {
      %a = arith.constant 7 : i32
      %b = arith.constant 8 : i32
      %c = arith.constant 9 : i32
      %d = arith.constant 10 : i32
      return %a, %b, %c, %d : i32, i32, i32, i32
    }"""
    after = """func.func @forward() -> (i32, i32, i32, i32) {
      %b = arith.constant 8 : i32
      %a = arith.constant 7 : i32
      %d = arith.constant 10 : i32
      %c = arith.constant 9 : i32
      return %a, %b, %c, %d : i32, i32, i32, i32
    }"""
    stage = _stage(tmp_path, before, after)
    receipt = check_stage(stage, m2m_python=lean_tools.m2m_python)
    assert receipt["verdict"] == "verified", receipt.get("reason")


def test_nonadjacent_hoisting_needs_additional_synthesis(tmp_path, lean_tools):
    before = """func.func @forward() -> (i32, i32, i32) {
      %a = arith.constant 7 : i32
      %b = arith.constant 8 : i32
      %c = arith.constant 9 : i32
      return %a, %b, %c : i32, i32, i32
    }"""
    after = """func.func @forward() -> (i32, i32, i32) {
      %c = arith.constant 9 : i32
      %a = arith.constant 7 : i32
      %b = arith.constant 8 : i32
      return %a, %b, %c : i32, i32, i32
    }"""
    stage = _stage(tmp_path, before, after)
    receipt = check_stage(stage, m2m_python=lean_tools.m2m_python)
    assert receipt["verdict"] == "abstracted", receipt.get("reason")
