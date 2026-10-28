"""Certify actual closed index folding at the checked width, retaining excluded comparison cases."""

import json
import subprocess

import pytest
from test_lean_defined_conversions import certify, verify
from test_lean_integer_fold import canonicalize

from merlin.verify.lean.check import check_definedness, check_support, check_typed_stage

NAMES = ["addi", "subi", "muli", "andi", "ori", "xori"]
VALUES = [(17, 5), (-1, 1), (2147483647, 1), (-(1 << 63), 3)]


def source(name="addi", a=17, b=5, *, escape=False, capture=False, length=0, cycle=False, flags=""):
    statements = [f"%a = arith.constant {a} : index", f"%b = arith.constant {b} : index"]
    previous = "%a"
    if length:
        for n in range(length):
            statements += [
                f"%c{n} = arith.constant {137 * n + 5} : index",
                f"%v{n} = arith.{NAMES[n % len(NAMES)] if cycle else 'addi'} {previous}, %c{n} : index",
            ]
            previous = f"%v{n}"
    else:
        statements += [f"%r = arith.{name} %a, %b {flags} : index"]
        previous = "%r"
    arguments = "%input: index, %keep: i32"
    if capture:
        arguments += ", %data: tensor<3xindex>, %init: tensor<3xindex>"
        statements += [
            f"""%out = linalg.generic {{indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
            iterator_types = ["parallel"]}} ins(%data : tensor<3xindex>) outs(%init : tensor<3xindex>) {{
          ^bb0(%x: index, %old: index):
            %sum = arith.addi %x, {previous} : index
            linalg.yield %sum : index
        }} -> tensor<3xindex>"""
        ]
        previous = "%out"
    types = ["tensor<3xindex>" if capture else "index"] + (["index"] if escape else []) + ["i32"]
    returned = [previous] + (["%a"] if escape else []) + ["%keep"]
    return f"""func.func @forward({arguments}) -> ({", ".join(types)}) {{
      {chr(10).join(statements)}
      return {", ".join(returned)} : {", ".join(types)}
    }}"""


def stage_for(tmp_path, tools, text):
    stage, before, after = canonicalize(tmp_path, tools, text)
    (stage / "meta.json").write_text(json.dumps({"boundary": "closed-index-fold"}))
    return stage, before, after


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("a,b", VALUES)
@pytest.mark.parametrize("escape", [False, True])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_actual_closed_index_arithmetic(tmp_path, lean_tools, name, a, b, escape, index_bits):
    stage, before, after = stage_for(tmp_path, lean_tools, source(name, a, b, escape=escape))
    assert f"arith.{name}" not in after.read_text()
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("index_bits", [32, 64])
def test_folded_index_result_keeps_real_nested_captures(tmp_path, lean_tools, name, index_bits):
    stage, before, after = stage_for(tmp_path, lean_tools, source(name, capture=True, escape=True))
    assert "linalg.generic" in after.read_text()
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)


@pytest.mark.parametrize("length", [64, 128])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_index_chains_learn_unmaterialized_intermediate_values(tmp_path, lean_tools, length, index_bits):
    stage, before, after = stage_for(tmp_path, lean_tools, source(length=length))
    assert after.read_text().count("arith.constant") == 1
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)


MUTATE = """
from pathlib import Path
from torch_mlir import ir
import sys
with ir.Context():
    path, fault = Path(sys.argv[1]), sys.argv[2]
    m=ir.Module.parse(path.read_text()); block=m.body.operations[0].regions[0].blocks[0]
    if fault == 'value':
        constant=next(o for o in block.operations if o.name=='arith.constant')
        attr=ir.IntegerAttr(constant.attributes['value'])
        constant.operation.attributes['value']=ir.IntegerAttr.get(attr.type,attr.value+1)
    elif fault == 'input':
        returned=list(block.operations)[-1]
        returned.operation.operands[0]=block.arguments[0]
    elif fault == 'keep':
        returned=list(block.operations)[-1]
        with ir.InsertionPoint(returned):
            zero=ir.Operation.create('arith.constant',results=[ir.IntegerType.get_signless(32)],
                attributes={'value':ir.IntegerAttr.get(ir.IntegerType.get_signless(32),0)},loc=ir.Location.unknown())
        returned.operation.operands[-1]=zero.results[0]
    elif fault == 'source_input':
        operation=next(o for o in block.operations if o.name=='arith.addi')
        operation.operation.operands[0]=block.arguments[0]
    else:
        operation=next(o for o in block.operations if o.name in ['arith.constant','arith.addi'])
        operation.operation.attributes['unreviewed.meaning']=ir.UnitAttr.get()
    m.operation.verify(); path.write_text(str(m))
"""


def mutate(path, tools, fault):
    result = subprocess.run([str(tools.m2m_python), "-c", MUTATE, str(path), fault], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("fault", ["value", "input", "keep", "attribute"])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_actual_index_target_mutations_are_refused(tmp_path, lean_tools, fault, index_bits):
    stage, before, after = stage_for(tmp_path, lean_tools, source())
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)
    mutate(after, lean_tools, fault)
    result = check_typed_stage(stage, m2m_python=lean_tools.m2m_python, index_bits=index_bits)
    assert result["verdict"] == "abstracted", result.get("reason")


@pytest.mark.parametrize("fault", ["source_input", "attribute"])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_index_source_replay_requires_actual_current_values(tmp_path, lean_tools, fault, index_bits):
    stage, before, _ = stage_for(tmp_path, lean_tools, source())
    mutate(before, lean_tools, fault)
    result = check_typed_stage(stage, m2m_python=lean_tools.m2m_python, index_bits=index_bits)
    assert result["verdict"] == "abstracted", result.get("reason")


@pytest.mark.parametrize("index_bits", [32, 64])
def test_actual_index_overflow_promises_are_replayed(tmp_path, lean_tools, index_bits):
    stage, before, _ = stage_for(tmp_path, lean_tools, source(a=2147483647, b=1, flags="overflow<nsw>"))
    result = check_typed_stage(stage, m2m_python=lean_tools.m2m_python, index_bits=index_bits)
    assert result["verdict"] == ("abstracted" if index_bits == 32 else "verified"), result.get("reason")
    result = check_definedness(
        before, "forward", tmp_path / "defined", m2m_python=lean_tools.m2m_python, index_bits=index_bits
    )
    assert result["verdict"] == "abstracted", result.get("reason")


def wrapped_compare(*, explicit_layout=False):
    header = (
        "module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32>>}" if explicit_layout else "module"
    )
    return (
        header
        + """ {
      func.func @forward(%keep: i32) -> (index, i1, i32) {
        %a = arith.constant 2147483647 : index
        %b = arith.constant 1 : index
        %zero = arith.constant 0 : index
        %sum = arith.addi %a, %b : index
        %negative = arith.cmpi slt, %sum, %zero : index
        return %sum, %negative, %keep : index, i1, i32
      }
    }"""
    )


@pytest.mark.parametrize("explicit_layout,index_bits", [(False, 32), (False, 64), (True, 32)])
def test_s6_wrapped_signed_compare_checks_actual_width(tmp_path, lean_tools, explicit_layout, index_bits):
    stage, before, after = stage_for(tmp_path, lean_tools, wrapped_compare(explicit_layout=explicit_layout))
    verify(before, lean_tools)
    assert "arith.constant false" in after.read_text()
    for checker, directory in [(check_support, "support"), (check_definedness, "defined")]:
        result = checker(
            before, "forward", tmp_path / directory, m2m_python=lean_tools.m2m_python, index_bits=index_bits
        )
        assert result["verdict"] == "verified", result.get("reason")
    result = check_typed_stage(stage, m2m_python=lean_tools.m2m_python, index_bits=index_bits)
    assert result["verdict"] == ("verified" if index_bits == 64 else "abstracted"), result.get("reason")
    if index_bits == 64:
        certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)


@pytest.mark.parametrize("index_bits", [32, 64])
def test_cyclic_chain_certifies_repeated_final_constant(tmp_path, lean_tools, index_bits):
    stage, before, _ = stage_for(tmp_path, lean_tools, source(length=128, cycle=True))
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)
