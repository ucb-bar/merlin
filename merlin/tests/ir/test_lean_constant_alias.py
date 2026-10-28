"""Exact local scalar constants compose with actual folding without greedy SSA alignment gaps."""

import subprocess

import pytest
from test_lean_defined_conversions import certify
from test_lean_index_fold import source as fold_source
from test_lean_reshape_compose import _canonicalize, _verify_mlir

from merlin.verify.lean.check import check_definedness, check_support, check_typed_stage

CASES = [
    ("i1", "true"),
    ("i8", "-1"),
    ("i32", "-2147483648"),
    ("index", "-1"),
    ("f16", "0x8000"),
    ("bf16", "0x8000"),
    ("f32", "0x7FC01234"),
    ("f64", "0x7FF8000000001234"),
]
FOLDS = [("addi", 17, 0), ("subi", 17, 0), ("muli", -1, 1), ("andi", 17, 17), ("ori", 17, 0), ("xori", 17, 0)]


def duplicate_source(element="i32", literal="7"):
    constant = literal if element == "i1" else f"{literal} : {element}"
    return f"""func.func @forward(%unknown: {element}, %keep: i32) -> ({element}, {element}, i32) {{
      %a = arith.constant {constant}
      %b = arith.constant {constant}
      return %b, %a, %keep : {element}, {element}, i32
    }}"""


def captured_source():
    return """func.func @forward(%payload: tensor<3xi32>, %init: tensor<3xi32>)
        -> (tensor<3xi32>, i32, i32) {
      %a = arith.constant 7 : i32
      %b = arith.constant 7 : i32
      %out = linalg.generic {indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
          iterator_types = ["parallel"]} ins(%payload : tensor<3xi32>) outs(%init : tensor<3xi32>) {
        ^bb0(%x: i32, %old: i32):
          %sum = arith.addi %x, %b : i32
          linalg.yield %sum : i32
      } -> tensor<3xi32>
      return %out, %a, %b : tensor<3xi32>, i32, i32
    }"""


def local_source(*, alias=False):
    # A scope-preserving control qualifies the recursive adapter independently of MLIR's
    # separate constant-hoisting transform. Fresh inner destinations avoid known native S4.
    outer = """%out = linalg.generic {indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
        iterator_types = ["parallel"]} ins(%payload : tensor<2xi32>) outs(%init : tensor<2xi32>) {
      ^bb0(%x: i32, %old: i32):
        %one = arith.constant 1 : i32
        %fresh = tensor.splat %one : tensor<i32>
        %inner = linalg.generic {indexing_maps = [affine_map<()->()>, affine_map<()->()>],
            iterator_types = []} ins(%fresh : tensor<i32>) outs(%fresh : tensor<i32>) {
          ^bb0(%value: i32, %unused: i32):
            %a = arith.constant 7 : i32
            DUPLICATE
            %sum = arith.addi %x, SELECTED : i32
            %mixed = arith.xori %sum, %value : i32
            linalg.yield %mixed : i32
        } -> tensor<i32>
        %scalar = tensor.extract %inner[] : tensor<i32>
        linalg.yield %scalar : i32
    } -> tensor<2xi32>"""
    outer = outer.replace("DUPLICATE", "" if alias else "%b = arith.constant 7 : i32")
    outer = outer.replace("SELECTED", "%a" if alias else "%b")
    return f"""func.func @forward(%payload: tensor<2xi32>, %init: tensor<2xi32>) -> tensor<2xi32> {{
      {outer}
      return %out : tensor<2xi32>
    }}"""


def stage_for(tmp_path, compiler, source):
    stage, before, after = _canonicalize(tmp_path, compiler, source)
    (stage / "meta.json").write_text('{"boundary":"exact-scalar-constant-alias"}')
    return stage, before, after


@pytest.mark.parametrize("element,literal", CASES)
@pytest.mark.parametrize("index_bits", [32, 64])
def test_actual_duplicate_scalar_constants(tmp_path, lean_tools, element, literal, index_bits):
    stage, before, after = stage_for(tmp_path, lean_tools.m2m_python, duplicate_source(element, literal))
    assert after.read_text().count("arith.constant") == 1
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)


@pytest.mark.parametrize("name,a,b", FOLDS)
@pytest.mark.parametrize("index_bits", [32, 64])
def test_actual_fold_to_escaped_existing_constant(tmp_path, lean_tools, name, a, b, index_bits):
    stage, before, after = stage_for(tmp_path, lean_tools.m2m_python, fold_source(name, a, b, escape=True))
    assert f"arith.{name}" not in after.read_text()
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)


def test_equal_runtime_bits_with_different_index_spellings(tmp_path, lean_tools):
    stage, before, after = stage_for(tmp_path, lean_tools.m2m_python, fold_source("andi", -(1 << 63), 3, escape=True))
    # Actual MLIR retains both the original escaped literal and the zero folded result.
    assert "-9223372036854775808" in after.read_text() and "constant 0 : index" in after.read_text()
    certify(stage, before, lean_tools, tmp_path, index_bits=32)


@pytest.mark.parametrize("index_bits", [32, 64])
def test_cyclic_128_fold_chain_with_duplicate_values(tmp_path, lean_tools, index_bits):
    stage, before, after = stage_for(tmp_path, lean_tools.m2m_python, fold_source(length=128, cycle=True))
    assert after.read_text().count("arith.constant") == 1
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)


@pytest.mark.parametrize("index_bits", [32, 64])
def test_alias_updates_escaped_values_and_region_captures(tmp_path, lean_tools, index_bits):
    stage, before, after = stage_for(tmp_path, lean_tools.m2m_python, captured_source())
    assert "linalg.generic" in after.read_text() and after.read_text().count("arith.constant") == 1
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)


@pytest.mark.parametrize("index_bits", [32, 64])
def test_fold_alias_updates_real_nested_capture(tmp_path, lean_tools, index_bits):
    stage, before, after = stage_for(
        tmp_path, lean_tools.m2m_python, fold_source("muli", -1, 1, escape=True, capture=True)
    )
    assert "linalg.generic" in after.read_text() and "arith.muli" not in after.read_text()
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)


def test_scope_preserving_nested_local_alias(tmp_path, lean_tools):
    stage = tmp_path / "stage"
    stage.mkdir()
    before, after = stage / "before.mlir", stage / "after.mlir"
    before.write_text(local_source())
    after.write_text(local_source(alias=True))
    (stage / "meta.json").write_text('{"boundary":"nested-local-constant-alias-control"}')
    for program in [before, after]:
        _verify_mlir(program, lean_tools.m2m_python)
    certify(stage, before, lean_tools, tmp_path)


MUTATE = """
from pathlib import Path
from torch_mlir import ir
import sys
with ir.Context():
    path, fault = Path(sys.argv[1]), sys.argv[2]
    m=ir.Module.parse(path.read_text()); block=m.body.operations[0].regions[0].blocks[0]
    if fault == 'value':
        op=next(o for o in block.operations if o.name=='arith.constant')
        attr=ir.IntegerAttr(op.attributes['value'])
        op.operation.attributes['value']=ir.IntegerAttr.get(attr.type,attr.value+1)
    elif fault == 'unknown':
        returned=list(block.operations)[-1]
        returned.operation.operands[0]=block.arguments[0]
    elif fault == 'capture':
        generic=next(o for o in block.operations if o.name=='linalg.generic')
        body=generic.regions[0].blocks[0]
        add=next(o for o in body.operations if o.name=='arith.addi')
        add.operation.operands[1]=body.arguments[1]
    else:
        op=next(o for o in block.operations if o.name=='arith.constant')
        op.operation.attributes['unreviewed.meaning']=ir.UnitAttr.get()
    m.operation.verify(); path.write_text(str(m))
"""


@pytest.mark.parametrize("fault", ["value", "unknown", "attribute", "capture"])
def test_valid_mlir_alias_mutations_refuse(tmp_path, lean_tools, fault):
    source = captured_source() if fault == "capture" else duplicate_source()
    stage, _, after = stage_for(tmp_path, lean_tools.m2m_python, source)
    changed = subprocess.run(
        [str(lean_tools.m2m_python), "-c", MUTATE, str(after), fault], capture_output=True, text=True, check=False
    )
    assert changed.returncode == 0, changed.stderr
    checked = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert checked["verdict"] == "abstracted", checked.get("reason")


def test_captured_function_argument_is_not_a_constant_fact(tmp_path, lean_tools):
    source = duplicate_source().replace("%b = arith.constant 7 : i32", "%b = arith.addi %unknown, %unknown : i32")
    stage, before, after = stage_for(tmp_path, lean_tools.m2m_python, source)
    # A wrong target selecting the unrelated constant must not acquire a fact from its type.
    after.write_text(duplicate_source().replace("return %b, %a, %keep", "return %a, %a, %keep"))
    _verify_mlir(after, lean_tools.m2m_python)
    assert check_typed_stage(stage, m2m_python=lean_tools.m2m_python)["verdict"] == "abstracted"
    for checker, name in [(check_support, "support"), (check_definedness, "defined")]:
        checked = checker(before, "forward", tmp_path / name, m2m_python=lean_tools.m2m_python)
        assert checked["verdict"] == "verified", checked.get("reason")
