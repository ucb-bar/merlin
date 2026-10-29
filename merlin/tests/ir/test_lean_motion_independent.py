"""Actual canonicalize constant hoisting uses proved adjacent SSA independence."""

import json

import pytest
from test_lean_defined_bounds import _chain
from test_lean_reshape_compose import _canonicalize, _verify_mlir

from merlin.verify.lean.check import check_definedness, check_support, check_typed_stage

CASTS = [("extsi", 8, 32), ("extui", 8, 32), ("trunci", 32, 8), ("extsi", 13, 64)]


def _source(kind="extsi", source_width=8, target_width=32, *, attributes=""):
    src, dst = f"i{source_width}", f"i{target_width}"
    return f"""func.func @subject(%x: {src}, %other: {src}) -> ({dst}, i32, {src}) {{
      %cast = "arith.{kind}"(%x) {attributes} : (i{source_width}) -> i{target_width}
      %c = arith.constant 22 : i32
      return %cast, %c, %other : i{target_width}, i32, i{source_width}
    }}"""


def _stage(tmp_path, before, after):
    path = tmp_path / "stage"
    path.mkdir()
    (path / "before.mlir").write_text(before)
    (path / "after.mlir").write_text(after)
    (path / "meta.json").write_text(json.dumps({"boundary": "independent-expression-motion"}))
    return path


def _independent(*, dependency=False, reverse=False, operation="arith.muli"):
    first = "%a = arith.addi %x, %y : i32"
    second = f"%b = {operation} {'%a' if dependency else '%x'}, %y : i32"
    body = f"{second}\n{first}" if reverse else f"{first}\n{second}"
    return f"""func.func @subject(%x: i32, %y: i32) -> (i32, i32, i32) {{
      {body}
      return %a, %b, %x : i32, i32, i32
    }}"""


@pytest.mark.parametrize("index_bits", [32, 64])
@pytest.mark.parametrize("kind,source_width,target_width", CASTS)
def test_actual_constant_hoist_has_defined_strict_chain(
    tmp_path, lean_tools, index_bits, kind, source_width, target_width
):
    stage, before, after = _canonicalize(tmp_path, lean_tools.m2m_python, _source(kind, source_width, target_width))
    assert before.read_text().index(f'"arith.{kind}"') < before.read_text().index("arith.constant")
    assert after.read_text().index("arith.constant") < after.read_text().index(f"arith.{kind}")
    _chain(tmp_path, stage, "subject", lean_tools, index_bits)


@pytest.mark.parametrize("operation", ["arith.muli", "arith.subi", "arith.andi", "arith.ori", "arith.xori"])
def test_both_operand_taking_ops_retain_all_escaping_values(tmp_path, lean_tools, operation):
    stage = _stage(tmp_path, _independent(operation=operation), _independent(reverse=True, operation=operation))
    for path in [stage / "before.mlir", stage / "after.mlir"]:
        _verify_mlir(path, lean_tools.m2m_python)
    _chain(tmp_path, stage, "subject", lean_tools, 64)


@pytest.mark.parametrize("fault", ["constant", "signedness", "operand", "escaping-use", "attribute"])
def test_valid_changed_actual_pass_output_refuses(tmp_path, lean_tools, fault):
    stage, _, after = _canonicalize(tmp_path, lean_tools.m2m_python, _source())
    text = after.read_text()
    if fault == "constant":
        text = text.replace("constant 22", "constant 23")
    elif fault == "signedness":
        text = text.replace("arith.extsi", "arith.extui")
    elif fault == "operand":
        text = text.replace("arith.extsi %arg0", "arith.extsi %arg1")
    elif fault == "escaping-use":
        text = text.replace("return %0, %c22_i32", "return %c22_i32, %0")
    else:
        text = text.replace("arith.extsi %arg0 :", "arith.extsi %arg0 {unreviewed.meaning} :")
    assert text != after.read_text()
    after.write_text(text)
    _verify_mlir(after, lean_tools.m2m_python)
    result = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "abstracted", result.get("reason")
    assert "function body did not verify" in result["reason"]


def test_valid_dependency_change_is_not_independent_motion(tmp_path, lean_tools):
    stage = _stage(tmp_path, _independent(dependency=True), _independent(reverse=True))
    for path in [stage / "before.mlir", stage / "after.mlir"]:
        _verify_mlir(path, lean_tools.m2m_python)
    assert check_typed_stage(stage, m2m_python=lean_tools.m2m_python)["verdict"] == "abstracted"


def test_conditional_division_motion_does_not_prove_source_definedness(tmp_path, lean_tools):
    before = """func.func @subject(%x: i32, %y: i32) -> (i32, i32) {
      %quotient = arith.divsi %x, %y : i32
      %c = arith.constant 22 : i32
      return %quotient, %c : i32, i32
    }"""
    after = before.replace(
        "%quotient = arith.divsi %x, %y : i32\n      %c = arith.constant 22 : i32",
        "%c = arith.constant 22 : i32\n      %quotient = arith.divsi %x, %y : i32",
    )
    stage = _stage(tmp_path, before, after)
    for path in [stage / "before.mlir", stage / "after.mlir"]:
        _verify_mlir(path, lean_tools.m2m_python)
    assert check_typed_stage(stage, m2m_python=lean_tools.m2m_python)["verdict"] == "verified"
    source = stage / "before.mlir"
    assert (
        check_support(source, "subject", tmp_path / "support", m2m_python=lean_tools.m2m_python)["verdict"]
        == "verified"
    )
    assert (
        check_definedness(source, "subject", tmp_path / "defined", m2m_python=lean_tools.m2m_python)["verdict"]
        == "abstracted"
    )


def test_unmodelled_source_attributes_never_gain_strict_source_support(tmp_path, lean_tools):
    before = _source(attributes="{unreviewed.meaning}")
    marker = '%cast = "arith.extsi"(%x) {unreviewed.meaning} : (i8) -> i32'
    after = before.replace(
        marker + "\n      %c = arith.constant 22 : i32", "%c = arith.constant 22 : i32\n      " + marker
    )
    stage = _stage(tmp_path, before, after)
    for path in [stage / "before.mlir", stage / "after.mlir"]:
        _verify_mlir(path, lean_tools.m2m_python)
    assert check_typed_stage(stage, m2m_python=lean_tools.m2m_python)["verdict"] == "verified"
    assert (
        check_support(stage / "before.mlir", "subject", tmp_path / "support", m2m_python=lean_tools.m2m_python)[
            "verdict"
        ]
        == "abstracted"
    )
