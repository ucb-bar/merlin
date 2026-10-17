"""Closed integer canonicalization replays actual SSA definitions and preserves exact bits."""

import json
import subprocess

import pytest

from merlin.verify.lean.check import (
    check_definedness,
    check_stage,
    check_support,
    check_typed_stage,
    compose_chain,
)

CANONICALIZE = """
import sys
from pathlib import Path
from torch_mlir import ir, passmanager
with ir.Context():
    module = ir.Module.parse(Path(sys.argv[1]).read_text())
    module.operation.verify()
    passmanager.PassManager.parse('builtin.module(canonicalize)').run(module.operation)
    module.operation.verify()
    Path(sys.argv[2]).write_text(str(module))
"""

BINARY = [
    "addi",
    "subi",
    "muli",
    "andi",
    "ori",
    "xori",
    "maxsi",
    "minsi",
    "maxui",
    "minui",
    "divui",
    "divsi",
    "remui",
    "remsi",
    "ceildivui",
    "ceildivsi",
    "floordivsi",
    "shli",
    "shrui",
    "shrsi",
]


def canonicalize(tmp_path, tools, source):
    stage = tmp_path / "stage"
    stage.mkdir(parents=True)
    before, after = stage / "before.mlir", stage / "after.mlir"
    before.write_text(source)
    (stage / "meta.json").write_text(json.dumps({"boundary": "closed-integer-canonicalization"}))
    result = subprocess.run(
        [str(tools.m2m_python), "-c", CANONICALIZE, str(before), str(after)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert before.read_text() != after.read_text()
    return stage, before, after


def binary_source(name, width=32, *, a=7, b=3, flags=""):
    ty = f"i{width}"
    return f"""func.func @forward(%keep: i32) -> ({ty}, i32) {{
      %a = arith.constant {a} : {ty}
      %b = arith.constant {b} : {ty}
      %r = arith.{name} %a, %b{flags} : {ty}
      return %r, %keep : {ty}, i32
    }}"""


@pytest.mark.parametrize("name", BINARY)
@pytest.mark.parametrize("width", [8, 13, 32, 64])
def test_actual_integer_constant_folds(tmp_path, lean_tools, name, width):
    stage, before, after = canonicalize(tmp_path, lean_tools, binary_source(name, width))
    assert f"arith.{name} " not in after.read_text()
    receipt = check_stage(stage, m2m_python=lean_tools.m2m_python)
    assert receipt["verdict"] == "verified", receipt.get("reason")
    supported = check_support(before, "forward", tmp_path / "support", m2m_python=lean_tools.m2m_python)
    assert supported["verdict"] == "verified", supported.get("reason")


@pytest.mark.parametrize("predicate", ["eq", "ne", "slt", "sle", "sgt", "sge", "ult", "ule", "ugt", "uge"])
def test_actual_comparison_folds(tmp_path, lean_tools, predicate):
    source = f"""func.func @forward() -> i1 {{
      %a = arith.constant -7 : i13
      %b = arith.constant 3 : i13
      %r = arith.cmpi {predicate}, %a, %b : i13
      return %r : i1
    }}"""
    stage, before, after = canonicalize(tmp_path, lean_tools, source)
    assert "arith.cmpi" not in after.read_text()
    assert (
        check_support(before, "forward", tmp_path / "support", m2m_python=lean_tools.m2m_python)["verdict"]
        == "verified"
    )
    receipt = check_stage(stage, m2m_python=lean_tools.m2m_python)
    assert receipt["verdict"] == "verified", receipt.get("reason")


@pytest.mark.parametrize(
    "name,source_width,target_width,value",
    [
        ("extsi", 1, 8, -1),
        ("extui", 1, 8, 1),
        ("extsi", 8, 32, -128),
        ("extui", 8, 32, 255),
        ("extsi", 13, 64, -4096),
        ("extui", 13, 64, 8191),
        ("trunci", 32, 8, 65535),
        ("trunci", 64, 13, -4097),
    ],
)
def test_actual_cast_folds(tmp_path, lean_tools, name, source_width, target_width, value):
    source = f"""func.func @forward() -> i{target_width} {{
      %a = arith.constant {value} : i{source_width}
      %r = arith.{name} %a : i{source_width} to i{target_width}
      return %r : i{target_width}
    }}"""
    stage, _, after = canonicalize(tmp_path, lean_tools, source)
    assert f"arith.{name}" not in after.read_text()
    receipt = check_stage(stage, m2m_python=lean_tools.m2m_python)
    assert receipt["verdict"] == "verified", receipt.get("reason")


@pytest.mark.parametrize("condition", ["true", "false"])
def test_actual_select_folds(tmp_path, lean_tools, condition):
    source = f"""func.func @forward() -> i13 {{
      %c = arith.constant {condition}
      %a = arith.constant -17 : i13
      %b = arith.constant 19 : i13
      %r = arith.select %c, %a, %b : i13
      return %r : i13
    }}"""
    stage, _, after = canonicalize(tmp_path, lean_tools, source)
    assert "arith.select" not in after.read_text()
    receipt = check_stage(stage, m2m_python=lean_tools.m2m_python)
    assert receipt["verdict"] == "verified", receipt.get("reason")


@pytest.mark.parametrize("name", ["addi", "subi", "muli", "shli"])
def test_flags_are_evaluated_before_folding(tmp_path, lean_tools, name):
    stage, _, _ = canonicalize(tmp_path, lean_tools, binary_source(name, 8, flags=" overflow<nsw, nuw>"))
    receipt = check_stage(stage, m2m_python=lean_tools.m2m_python)
    assert receipt["verdict"] == "verified", receipt.get("reason")


@pytest.mark.parametrize("index_bits", [32, 64])
def test_mixed_integer_fragment_and_defined_chain(tmp_path, lean_tools, index_bits):
    source = """func.func @forward(%keep: i32) -> (i13, i32) {
      %a = arith.constant -17 : i8
      %b = arith.constant 9 : i8
      %sum = arith.addi %a, %b : i8
      %wide = arith.extsi %sum : i8 to i32
      %r = arith.trunci %wide : i32 to i13
      return %r, %keep : i13, i32
    }"""
    stage, before, after = canonicalize(tmp_path, lean_tools, source)
    assert after.read_text().count("arith.constant") == 1
    result = check_typed_stage(stage, index_bits=index_bits, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "verified", result.get("reason")
    for checker, name in [(check_support, "support"), (check_definedness, "defined")]:
        result = checker(before, "forward", tmp_path / name, index_bits=index_bits, m2m_python=lean_tools.m2m_python)
        assert result["verdict"] == "verified", result.get("reason")
    result = compose_chain(
        [stage],
        tmp_path / "support/receipt.json",
        "forward",
        tmp_path / "chain.json",
        m2m_python=lean_tools.m2m_python,
        definedness=tmp_path / "defined/receipt.json",
    )
    assert result["verdict"] == "verified", result.get("reason")
    after.write_text(after.read_text() + "\n// live output changed after proof\n")
    replay = compose_chain(
        [stage],
        tmp_path / "support/receipt.json",
        "forward",
        tmp_path / "chain-mutated.json",
        m2m_python=lean_tools.m2m_python,
        definedness=tmp_path / "defined/receipt.json",
    )
    assert replay["verdict"] != "verified"


MUTATE = """
import sys
from pathlib import Path
from torch_mlir import ir
with ir.Context():
    module = ir.Module.parse(Path(sys.argv[1]).read_text())
    block = module.body.operations[0].regions[0].blocks[0]
    constant = next(o for o in block.operations if o.name == 'arith.constant')
    fault = sys.argv[2]
    if fault == 'bits':
        constant.operation.attributes['value'] = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 11)
    elif fault == 'attribute':
        constant.operation.attributes['unknown.numeric_meaning'] = ir.UnitAttr.get()
    elif fault == 'escaping_argument':
        list(block.operations)[-1].operation.operands[1] = constant.results[0]
    module.operation.verify()
    Path(sys.argv[1]).write_text(str(module))
"""


@pytest.mark.parametrize("fault", ["bits", "attribute", "escaping_argument"])
def test_fold_rejects_verified_target_mutations(tmp_path, lean_tools, fault):
    stage, _, after = canonicalize(tmp_path, lean_tools, binary_source("addi"))
    assert check_stage(stage, m2m_python=lean_tools.m2m_python)["verdict"] == "verified"
    original = after.read_bytes()
    result = subprocess.run(
        [str(lean_tools.m2m_python), "-c", MUTATE, str(after), fault],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert after.read_bytes() != original
    receipt = check_stage(stage, m2m_python=lean_tools.m2m_python)
    assert receipt["verdict"] == "abstracted", receipt.get("reason")


@pytest.mark.parametrize(
    "source",
    [
        binary_source("addi", 8, a=127, b=1, flags=" overflow<nsw>"),
        binary_source("muli", 8, a=128, b=2, flags=" overflow<nuw>"),
        binary_source("shli", 8, a=1, b=8),
        binary_source("divui", 8, a=7, b=0),
        binary_source("divsi", 8, a=-128, b=-1),
    ],
)
def test_poison_and_ub_are_not_replaced_by_initialized_constants(tmp_path, lean_tools, source):
    stage = tmp_path / "stage"
    stage.mkdir()
    before, after = stage / "before.mlir", stage / "after.mlir"
    before.write_text(source)
    after.write_text("""func.func @forward(%keep: i32) -> (i8, i32) {
      %r = arith.constant 0 : i8
      return %r, %keep : i8, i32
    }""")
    (stage / "meta.json").write_text(json.dumps({"boundary": "unsafe-constant-fold"}))
    supported = check_support(before, "forward", tmp_path / "support", m2m_python=lean_tools.m2m_python)
    assert supported["verdict"] == "verified", supported.get("reason")
    receipt = check_stage(stage, m2m_python=lean_tools.m2m_python)
    assert receipt["verdict"] == "abstracted", receipt.get("reason")


def test_fold_refuses_erased_intermediate_that_still_escapes(tmp_path, lean_tools):
    source = """func.func @forward() -> (i32, i32) {
      %a = arith.constant 7 : i32
      %b = arith.constant 3 : i32
      %r = arith.addi %a, %b : i32
      return %r, %a : i32, i32
    }"""
    stage, _, after = canonicalize(tmp_path, lean_tools, source)
    # Actual canonicalization preserves the escaped value. Dropping it and substituting the
    # folded final result is valid MLIR but changes the program's second output.
    after.write_text("""func.func @forward() -> (i32, i32) {
      %r = arith.constant 10 : i32
      return %r, %r : i32, i32
    }""")
    _verify_mlir(after, lean_tools)
    receipt = check_stage(stage, m2m_python=lean_tools.m2m_python)
    assert receipt["verdict"] == "abstracted", receipt.get("reason")


VERIFY = """
import sys
from pathlib import Path
from torch_mlir import ir
with ir.Context():
    module = ir.Module.parse(Path(sys.argv[1]).read_text())
    module.operation.verify()
"""


def _verify_mlir(path, lean_tools):
    verified = subprocess.run(
        [str(lean_tools.m2m_python), "-c", VERIFY, str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert verified.returncode == 0, verified.stderr


@pytest.mark.parametrize("fault", ["overflow_flag", "intermediate_width"])
def test_source_flags_and_widths_are_recomputed(tmp_path, lean_tools, fault):
    if fault == "overflow_flag":
        source = binary_source("addi", 8, a=127, b=1)
        mutated = source.replace("addi %a, %b :", "addi %a, %b overflow<nsw> :")
    else:
        source = """func.func @forward() -> i64 {
          %a = arith.constant 65536 : i64
          %b = arith.trunci %a : i64 to i32
          %r = arith.extui %b : i32 to i64
          return %r : i64
        }"""
        mutated = source.replace("to i32", "to i16").replace(": i32 to", ": i16 to")
    stage, before, _ = canonicalize(tmp_path, lean_tools, source)
    assert check_stage(stage, m2m_python=lean_tools.m2m_python)["verdict"] == "verified"
    before.write_text(mutated)
    _verify_mlir(before, lean_tools)
    receipt = check_stage(stage, m2m_python=lean_tools.m2m_python)
    assert receipt["verdict"] == "abstracted", receipt.get("reason")


def test_runtime_input_is_not_treated_as_a_closed_constant(tmp_path, lean_tools):
    stage = tmp_path / "stage"
    stage.mkdir()
    before, after = stage / "before.mlir", stage / "after.mlir"
    before.write_text("""func.func @forward(%a: i32) -> i32 {
      %b = arith.constant 3 : i32
      %r = arith.addi %a, %b : i32
      return %r : i32
    }""")
    after.write_text("""func.func @forward(%a: i32) -> i32 {
      %r = arith.constant 10 : i32
      return %r : i32
    }""")
    for path in [before, after]:
        _verify_mlir(path, lean_tools)
    (stage / "meta.json").write_text(json.dumps({"boundary": "nonclosed-fold"}))
    receipt = check_stage(stage, m2m_python=lean_tools.m2m_python)
    assert receipt["verdict"] == "abstracted", receipt.get("reason")


def test_float_constant_folding_still_requires_float_laws(tmp_path, lean_tools):
    source = """func.func @forward() -> f32 {
      %a = arith.constant 1.0 : f32
      %b = arith.constant 2.0 : f32
      %r = arith.addf %a, %b : f32
      return %r : f32
    }"""
    stage, before, after = canonicalize(tmp_path, lean_tools, source)
    assert "arith.addf" not in after.read_text()
    supported = check_support(before, "forward", tmp_path / "support", m2m_python=lean_tools.m2m_python)
    assert supported["verdict"] == "verified", supported.get("reason")
    receipt = check_stage(stage, m2m_python=lean_tools.m2m_python)
    assert receipt["verdict"] == "abstracted", receipt.get("reason")


@pytest.mark.parametrize("index_bits", [32, 64])
def test_index_constant_folding_is_outside_fixed_width_rules(tmp_path, lean_tools, index_bits):
    source = """func.func @forward() -> index {
      %a = arith.constant 7 : index
      %b = arith.constant 3 : index
      %r = arith.addi %a, %b : index
      return %r : index
    }"""
    stage, before, _ = canonicalize(tmp_path, lean_tools, source)
    supported = check_support(
        before, "forward", tmp_path / "support", index_bits=index_bits, m2m_python=lean_tools.m2m_python
    )
    assert supported["verdict"] == "verified", supported.get("reason")
    receipt = check_typed_stage(stage, index_bits=index_bits, m2m_python=lean_tools.m2m_python)
    assert receipt["verdict"] == "abstracted", receipt.get("reason")
