"""Actual index comparison folds use checked scalar types and current runtime constants."""

import json
import subprocess

import pytest
from test_lean_defined_conversions import certify
from test_lean_integer_fold import canonicalize

from merlin.verify.lean.check import check_typed_stage

PREDICATES = ["eq", "ne", "slt", "sle", "sgt", "sge", "ult", "ule", "ugt", "uge"]


def source(predicate="slt", a=17, b=5, *, escape=False, length=0):
    body = [f"%a = arith.constant {a} : index", f"%b = arith.constant {b} : index"]
    previous = "%a"
    for n in range(length):
        body += [f"%c{n} = arith.constant {137 * n + 5} : index", f"%v{n} = arith.addi {previous}, %c{n} : index"]
        previous = f"%v{n}"
    body += [f"%r = arith.cmpi {predicate}, {previous}, %b : index"]
    types = ["i1"] + (["index"] if escape else []) + ["i32"]
    results = ["%r"] + (["%a"] if escape else []) + ["%keep"]
    return f"""func.func @forward(%input: i1, %unknown: index, %keep: i32) -> ({", ".join(types)}) {{
      {chr(10).join(body)}
      return {", ".join(results)} : {", ".join(types)}
    }}"""


def stage_for(tmp_path, tools, text):
    stage, before, after = canonicalize(tmp_path, tools, text)
    (stage / "meta.json").write_text(json.dumps({"boundary": "closed-index-compare"}))
    return stage, before, after


@pytest.mark.parametrize("predicate", PREDICATES)
@pytest.mark.parametrize("a,b", [(17, 5), (-7, 5)])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_actual_index_comparison_folds(tmp_path, lean_tools, predicate, a, b, index_bits):
    stage, before, after = stage_for(tmp_path, lean_tools, source(predicate, a, b))
    assert "arith.cmpi" not in after.read_text()
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)


@pytest.mark.parametrize("predicate", ["slt", "ugt"])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_index_compare_preserves_escaping_operand(tmp_path, lean_tools, predicate, index_bits):
    stage, before, _ = stage_for(tmp_path, lean_tools, source(predicate, -7, 5, escape=True))
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)


@pytest.mark.parametrize("index_bits", [32, 64])
def test_index_comparison_uses_unmaterialized_arithmetic_facts(tmp_path, lean_tools, index_bits):
    stage, before, after = stage_for(tmp_path, lean_tools, source("sgt", length=64))
    assert after.read_text().count("arith.constant") == 1
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)


MUTATE = """
from pathlib import Path
from torch_mlir import ir
import sys
with ir.Context():
    path, fault = Path(sys.argv[1]), sys.argv[2]
    m=ir.Module.parse(path.read_text()); block=m.body.operations[0].regions[0].blocks[0]
    returned=list(block.operations)[-1]
    if fault == 'value':
        constant=next(o for o in block.operations if o.name=='arith.constant')
        attr=ir.IntegerAttr(constant.attributes['value'])
        constant.operation.attributes['value']=ir.IntegerAttr.get(attr.type,int(attr.value)^1)
    elif fault == 'input':
        returned.operation.operands[0]=block.arguments[0]
    elif fault == 'keep':
        with ir.InsertionPoint(returned):
            zero=ir.Operation.create('arith.constant',results=[ir.IntegerType.get_signless(32)],
                attributes={'value':ir.IntegerAttr.get(ir.IntegerType.get_signless(32),0)},loc=ir.Location.unknown())
        returned.operation.operands[-1]=zero.results[0]
    elif fault == 'predicate':
        comparison=next(o for o in block.operations if o.name=='arith.cmpi')
        comparison.operation.attributes['predicate']=ir.IntegerAttr.get(ir.IntegerType.get_signless(64),4)
    elif fault == 'source_input':
        comparison=next(o for o in block.operations if o.name=='arith.cmpi')
        comparison.operation.operands[0]=block.arguments[1]
    else:
        comparison=next(o for o in block.operations if o.name=='arith.cmpi')
        comparison.operation.attributes['unreviewed.meaning']=ir.UnitAttr.get()
    m.operation.verify(); path.write_text(str(m))
"""


def mutate(path, tools, fault):
    result = subprocess.run([str(tools.m2m_python), "-c", MUTATE, str(path), fault], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("fault", ["value", "input", "keep"])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_index_compare_refuses_actual_target_mutations(tmp_path, lean_tools, fault, index_bits):
    stage, before, after = stage_for(tmp_path, lean_tools, source())
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)
    mutate(after, lean_tools, fault)
    result = check_typed_stage(stage, m2m_python=lean_tools.m2m_python, index_bits=index_bits)
    assert result["verdict"] == "abstracted", result.get("reason")


@pytest.mark.parametrize("fault", ["predicate", "source_input", "attribute"])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_index_compare_replays_source_predicate_and_current_values(tmp_path, lean_tools, fault, index_bits):
    stage, before, _ = stage_for(tmp_path, lean_tools, source())
    mutate(before, lean_tools, fault)
    result = check_typed_stage(stage, m2m_python=lean_tools.m2m_python, index_bits=index_bits)
    assert result["verdict"] == "abstracted", result.get("reason")
