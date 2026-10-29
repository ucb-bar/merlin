"""Exact local integer constants justify poison-preserving neutral body aliases."""

import json
import subprocess

import pytest
from test_lean_integer_fold import canonicalize
from test_lean_integer_idempotent import certify
from test_lean_integer_idempotent import nested_source as idempotent_nested

from merlin.verify.lean.check import check_definedness, check_support, check_typed_stage

NAMES = ["addi", "subi", "muli", "andi", "ori", "xori", "minui", "maxui", "minsi", "maxsi"]


def neutral(name, width):
    if name == "muli":
        return 1
    if name in {"andi", "minui"}:
        return -1
    if name == "minsi":
        return (1 << (width - 1)) - 1
    if name == "maxsi":
        return -(1 << (width - 1))
    return 0


def source(name="addi", width=13, *, reverse=False, reduction=False, escape=False, dirty=False, captured=False):
    ty = f"i{width}"
    tensor, output = f"tensor<3x{ty}>", f"tensor<{ty}>" if reduction else f"tensor<3x{ty}>"
    operands = "%neutral, %x" if reverse else "%x, %neutral"
    constant = f"%neutral = arith.constant {neutral(name, width)} : {ty}"
    mixed = f"%mixed = arith.xori %r, %old : {ty}"
    # Reversed operand IDs must be sorted after alias substitution, including repeated uses.
    continuation = f"%result = arith.addi %mixed, %r : {ty}" if escape else ""
    return f"""func.func @forward(%data: {tensor}, %initial: {output}, %keep: i32) -> ({output}, i32) {{
      {constant if captured else ""}
      {f"%empty = tensor.empty() : {output}" if dirty else ""}
      %out = linalg.generic {{indexing_maps = [affine_map<(d0)->(d0)>,
          {"affine_map<(d0)->()>" if reduction else "affine_map<(d0)->(d0)>"}],
          iterator_types = ["{"reduction" if reduction else "parallel"}"]}}
          ins(%data : {tensor}) outs({"%empty" if dirty else "%initial"} : {output}) {{
        ^bb0(%x: {ty}, %old: {ty}):
          {"" if captured else constant}
          %r = arith.{name} {operands} : {ty}
          {mixed}
          {continuation}
          linalg.yield {"%result" if escape else "%mixed"} : {ty}
      }} -> {output}
      return %out, %keep : {output}, i32
    }}"""


def chain_source(length=128):
    body, previous = [], "%x"
    for i in range(length):
        name = NAMES[i % len(NAMES)]
        body.extend(
            [
                f"%neutral{i} = arith.constant {neutral(name, 13)} : i13",
                f"%r{i} = arith.{name} {previous}, %neutral{i} : i13",
            ]
        )
        previous = f"%r{i}"
    return (
        source()
        .replace(
            "%neutral = arith.constant 0 : i13\n          %r = arith.addi %x, %neutral : i13",
            "\n          ".join(body),
        )
        .replace("arith.xori %r, %old", f"arith.xori {previous}, %old")
    )


def nested_source(name):
    # Reuse the already-qualified capture fixture, replacing idempotence with a local literal.
    text = idempotent_nested("andi")
    return text.replace(
        "%same = arith.andi %x, %x : i13",
        f"%neutral = arith.constant {neutral(name, 13)} : i13\n          %same = arith.{name} %x, %neutral : i13",
    )


def stage_for(work, tools, text):
    stage, before, after = canonicalize(work, tools, text)
    (stage / "meta.json").write_text(json.dumps({"boundary": "integer-neutral-local-body"}))
    return stage, before, after


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("width", [1, 13, 64])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_actual_body_neutral_alias_has_defined_chain(tmp_path, lean_tools, name, width, index_bits):
    stage, before, after = stage_for(tmp_path, lean_tools, source(name, width))
    assert "linalg.generic" in after.read_text()
    assert after.read_text().count(f"arith.{name}") < before.read_text().count(f"arith.{name}")
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)


@pytest.mark.parametrize("name", [n for n in NAMES if n != "subi"])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_reverse_neutral_and_escaping_uses_sort_after_alias(tmp_path, lean_tools, name, index_bits):
    stage, before, _ = stage_for(tmp_path, lean_tools, source(name, reverse=True, escape=True))
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)


@pytest.mark.parametrize("name", NAMES)
def test_neutral_alias_inside_reduction(tmp_path, lean_tools, name):
    stage, before, _ = stage_for(tmp_path, lean_tools, source(name, reduction=True))
    certify(stage, before, lean_tools, tmp_path)


def test_128_mixed_neutral_aliases_share_one_checked_walk(tmp_path, lean_tools):
    stage, before, after = stage_for(tmp_path, lean_tools, chain_source())
    assert "linalg.generic" in after.read_text()
    assert before.read_text().count("arith.") >= 129
    assert after.read_text().count("arith.") == 1
    certify(stage, before, lean_tools, tmp_path)


@pytest.mark.parametrize("name", ["addi", "andi", "minsi", "maxui"])
def test_local_alias_updates_nested_region_captures(tmp_path, lean_tools, name):
    stage, before, _ = stage_for(tmp_path, lean_tools, nested_source(name))
    certify(stage, before, lean_tools, tmp_path)


@pytest.mark.parametrize("name", ["addi", "minui", "maxsi"])
def test_poison_accumulator_stays_conditional(tmp_path, lean_tools, name):
    stage, before, _ = stage_for(tmp_path, lean_tools, source(name, dirty=True))
    certify(stage, before, lean_tools, tmp_path, defined=False)


@pytest.mark.parametrize("name", ["addi", "minsi"])
def test_captured_numeric_constant_remains_explicit_coverage_gap(tmp_path, lean_tools, name):
    stage, before, _ = stage_for(tmp_path, lean_tools, source(name, captured=True))
    for checker, directory in [(check_support, "support"), (check_definedness, "defined")]:
        result = checker(before, "forward", tmp_path / directory, m2m_python=lean_tools.m2m_python)
        assert result["verdict"] == "verified", result.get("reason")
    result = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "abstracted", result.get("reason")


@pytest.mark.parametrize("name", ["addi", "minui", "maxsi"])
def test_actual_poison_input_is_preserved_by_neutral_alias(tmp_path, lean_tools, name):
    text = (
        source(name)
        .replace("%out = linalg.generic", "%poison = tensor.empty() : tensor<3xi13>\n      %out = linalg.generic")
        .replace("ins(%data : tensor<3xi13>)", "ins(%poison : tensor<3xi13>)")
    )
    stage, before, _ = stage_for(tmp_path, lean_tools, text)
    certify(stage, before, lean_tools, tmp_path, defined=False)


@pytest.mark.parametrize("name", ["addi", "subi", "muli"])
def test_overflow_promises_remain_outside_exact_family(tmp_path, lean_tools, name):
    text = source(name).replace(f"arith.{name} %x, %neutral :", f"arith.{name} %x, %neutral overflow<nsw> :")
    stage, before, _ = stage_for(tmp_path, lean_tools, text)
    result = check_support(before, "forward", tmp_path / "support", m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "verified", result.get("reason")
    result = check_definedness(before, "forward", tmp_path / "defined", m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "abstracted", result.get("reason")
    result = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "abstracted", result.get("reason")


MUTATE = """
from pathlib import Path
from torch_mlir import ir
import sys
with ir.Context():
    path,fault=Path(sys.argv[1]),sys.argv[2]
    module=ir.Module.parse(path.read_text()); function=module.body.operations[0].regions[0].blocks[0]
    generic=next(o for o in function.operations if o.name=='linalg.generic'); body=generic.regions[0].blocks[0]
    if fault=='input':
        op=next(o for o in body.operations if o.name=='arith.xori')
        op.operation.operands[0]=body.arguments[1]
    elif fault=='attribute':
        op=next(o for o in body.operations if o.name.startswith('arith.') and o.name!='arith.constant')
        op.operation.attributes['unreviewed.meaning']=ir.UnitAttr.get()
    elif fault=='keep':
        ret=list(function.operations)[-1]
        with ir.InsertionPoint(ret):
            c=ir.Operation.create('arith.constant',results=[ir.IntegerType.get_signless(32)],
                attributes={'value':ir.IntegerAttr.get(ir.IntegerType.get_signless(32),0)},loc=ir.Location.unknown())
        ret.operation.operands[-1]=c.results[0]
    elif fault=='constant':
        c=next(o for o in body.operations if o.name=='arith.constant')
        a=ir.IntegerAttr(c.attributes['value']);c.operation.attributes['value']=ir.IntegerAttr.get(a.type,a.value+1)
    elif fault=='direction':
        op=next(o for o in body.operations if o.name=='arith.subi')
        a,b=list(op.operands);op.operation.operands[0]=b;op.operation.operands[1]=a
    else: raise ValueError(fault)
    module.operation.verify();path.write_text(str(module))
"""


def mutate(path, tools, fault):
    original = path.read_bytes()
    result = subprocess.run([str(tools.m2m_python), "-c", MUTATE, str(path), fault], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert path.read_bytes() != original


@pytest.mark.parametrize("fault", ["input", "attribute", "keep", "constant", "direction"])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_mutations_recheck_exact_alias_obligations(tmp_path, lean_tools, fault, index_bits):
    stage, before, after = stage_for(tmp_path, lean_tools, source("subi"))
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)
    mutate(before if fault in {"constant", "direction"} else after, lean_tools, fault)
    result = check_typed_stage(stage, m2m_python=lean_tools.m2m_python, index_bits=index_bits)
    assert result["verdict"] == "abstracted", result.get("reason")
