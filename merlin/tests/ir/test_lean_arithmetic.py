"""Proved exact integer canonicalization, qualified against actual upstream MLIR output."""

import json
import subprocess

import pytest

from merlin.verify.lean.check import check_definedness, check_stage, check_support, compose_chain

RUN_CANONICALIZE = """
import sys
from pathlib import Path
from torch_mlir import ir, passmanager
with ir.Context():
    module = ir.Module.parse(Path(sys.argv[1]).read_text())
    def arithmetic():
        return next(o for o in module.body.operations[0].regions[0].blocks[0].operations
                    if o.name == 'arith.' + sys.argv[3])
    operands = list(arithmetic().operands)
    passmanager.PassManager.parse('builtin.module(canonicalize)').run(module.operation)
    assert list(arithmetic().operands) == operands[::-1], 'the real pass did not commute the operands'
    Path(sys.argv[2]).write_text(str(module))
"""


def _stage(tmp_path, source):
    stage = tmp_path / "stage"
    stage.mkdir()
    before, after = stage / "before.mlir", stage / "after.mlir"
    before.write_text(source)
    (stage / "meta.json").write_text(json.dumps({"boundary": "integer-canonicalization"}))
    return stage, before, after


@pytest.mark.parametrize("name", ["addi", "muli", "andi", "ori", "xori"])
@pytest.mark.parametrize("width", [8, 32])
def test_actual_integer_canonicalization_and_chain(tmp_path, lean_tools, name, width):
    stage, before, after = _stage(
        tmp_path,
        f"""func.func @forward(%a: i{width}) -> i{width} {{
      %c = arith.constant 7 : i{width}
      %r = arith.{name} %c, %a : i{width}
      return %r : i{width}
    }}""",
    )
    result = subprocess.run(
        [str(lean_tools.m2m_python), "-c", RUN_CANONICALIZE, str(before), str(after), name],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    receipt = check_stage(stage, m2m_python=lean_tools.m2m_python)
    assert receipt["verdict"] == "verified", receipt.get("reason")
    assert (
        check_support(before, "forward", tmp_path / "support", m2m_python=lean_tools.m2m_python)["verdict"]
        == "verified"
    )
    assert (
        check_definedness(before, "forward", tmp_path / "defined", m2m_python=lean_tools.m2m_python)["verdict"]
        == "verified"
    )
    chain = compose_chain(
        [stage],
        tmp_path / "support/receipt.json",
        "forward",
        tmp_path / "chain.json",
        m2m_python=lean_tools.m2m_python,
        definedness=tmp_path / "defined/receipt.json",
    )
    assert chain["verdict"] == "verified", chain.get("reason")


@pytest.mark.parametrize(
    "name,ty,flags",
    [
        ("subi", "i32", ""),
        ("divsi", "i32", ""),
        ("addf", "f32", ""),
        ("mulf", "f32", ""),
        ("addi", "i32", " overflow<nsw>"),
    ],
)
def test_unproved_commutations_are_refused(tmp_path, lean_tools, name, ty, flags):
    source = f"""func.func @forward(%a: {ty}, %b: {ty}) -> {ty} {{
      %r = arith.{name} %a, %b{flags} : {ty}
      return %r : {ty}
    }}"""
    stage, _, after = _stage(tmp_path, source)
    after.write_text(source.replace("%a, %b", "%b, %a"))
    receipt = check_stage(stage, m2m_python=lean_tools.m2m_python)
    assert receipt["verdict"] == "abstracted", receipt.get("reason")


def test_commutation_cannot_change_flags(tmp_path, lean_tools):
    source = """func.func @forward(%a: i32, %b: i32) -> i32 {
      %r = arith.addi %a, %b : i32
      return %r : i32
    }"""
    stage, _, after = _stage(tmp_path, source)
    after.write_text(source.replace("addi %a, %b :", "addi %b, %a overflow<nsw> :"))
    receipt = check_stage(stage, m2m_python=lean_tools.m2m_python)
    assert receipt["verdict"] == "abstracted", receipt.get("reason")
