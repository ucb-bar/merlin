"""Unsigned remainder derives initialized safe coordinates from actual positive divisor facts."""

import json
import subprocess

import pytest
from test_lean_defined_conversions import certify, verify
from test_lean_integer_fold import canonicalize

from merlin.verify.lean.check import check_definedness, check_support, check_typed_stage


def source(mode="constant", divisor=3, *, size=8, extract=True, dirty=False, attributes=""):
    ty = f"tensor<{size}xi13>"
    bound_type = "i2" if mode == "widened" else "index"
    if mode == "constant":
        producer = f"%divisor = arith.constant {divisor} : index"
    elif mode == "widened":
        producer = """%wide = arith.index_castui %bound : i2 to index
          %one = arith.constant 1 : index
          %divisor = arith.addi %wide, %one : index"""
    elif mode == "unknown":
        producer = ""
    else:
        raise ValueError(mode)
    denominator = "%bound" if mode == "unknown" else "%divisor"
    result_type = "i13" if extract else "index"
    empty = f"%empty = tensor.empty() : {ty}" if dirty else ""
    read = f"%value = tensor.extract {'%empty' if dirty else '%data'}[%remainder] : {ty}" if extract else ""
    return f"""func.func @forward(%data: {ty}, %idx: index, %bound: {bound_type}, %keep: i32)
        -> ({result_type}, i32) {{
      %dead = arith.addi %keep, %keep : i32
      {producer}
      %remainder = "arith.remui"(%idx, {denominator}) {attributes} : (index, index) -> index
      {empty}
      {read}
      return %{"value" if extract else "remainder"}, %keep : {result_type}, i32
    }}"""


def stage_for(tmp_path, tools, text):
    stage, before, after = canonicalize(tmp_path, tools, text)
    (stage / "meta.json").write_text(json.dumps({"boundary": "index-remainder-definedness"}))
    return stage, before, after


@pytest.mark.parametrize("mode,divisor,size", [("constant", 3, 3), ("constant", 7, 7), ("widened", 3, 4)])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_actual_positive_remainder_read_has_a_defined_chain(tmp_path, lean_tools, mode, divisor, size, index_bits):
    stage, before, after = stage_for(tmp_path, lean_tools, source(mode, divisor, size=size))
    assert "arith.remui" in after.read_text()
    assert "arith.addi" in before.read_text() and "%keep, %keep" not in after.read_text()
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)


@pytest.mark.parametrize("options", [{"divisor": 0}, {"mode": "unknown"}, {"size": 2}, {"dirty": True}, {"size": 0}])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_zero_unknown_or_unsafe_remainder_read_refuses(tmp_path, lean_tools, options, index_bits):
    before = tmp_path / "source.mlir"
    before.write_text(source(**options))
    verify(before, lean_tools)
    result = check_support(
        before, "forward", tmp_path / "support", m2m_python=lean_tools.m2m_python, index_bits=index_bits
    )
    assert result["verdict"] == "verified", result.get("reason")
    result = check_definedness(
        before, "forward", tmp_path / "defined", m2m_python=lean_tools.m2m_python, index_bits=index_bits
    )
    assert result["verdict"] == "abstracted", result.get("reason")


@pytest.mark.parametrize("attribute", ["nonNeg", "isExact", "disjoint", "unreviewed.meaning"])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_remainder_promises_do_not_gain_definedness(tmp_path, lean_tools, attribute, index_bits):
    before = tmp_path / "source.mlir"
    before.write_text(source(attributes="{" + attribute + "}"))
    verify(before, lean_tools)
    result = check_definedness(
        before, "forward", tmp_path / "defined", m2m_python=lean_tools.m2m_python, index_bits=index_bits
    )
    assert result["verdict"] == "abstracted", result.get("reason")


MUTATE = """
from pathlib import Path
from torch_mlir import ir
import sys
with ir.Context():
    path,fault=Path(sys.argv[1]),sys.argv[2]
    text=path.read_text()
    if fault=='signedness':
        text=text.replace('arith.remui','arith.remsi',1)
    m=ir.Module.parse(text); block=m.body.operations[0].regions[0].blocks[0]
    rem=next(o for o in block.operations if o.name in ['arith.remui','arith.remsi'])
    if fault in ['constant_read','remove_remainder']:
        read=next(o for o in block.operations if o.name=='tensor.extract')
        if fault=='constant_read':
            with ir.InsertionPoint(read):
                c=ir.Operation.create('arith.constant',results=[ir.IndexType.get()],
                  attributes={'value':ir.IntegerAttr.get(ir.IndexType.get(),0)},loc=ir.Location.unknown())
            read.operation.operands[1]=c.results[0]
        else:
            read.operation.operands[1]=block.arguments[1]
    elif fault=='divisor':
        c=next(o for o in block.operations if o.name=='arith.constant')
        a=ir.IntegerAttr(c.attributes['value'])
        c.operation.attributes['value']=ir.IntegerAttr.get(a.type,a.value+1)
    elif fault=='attribute':
        rem.operation.attributes['unreviewed.meaning']=ir.UnitAttr.get()
    m.operation.verify(); path.write_text(str(m))
"""


def mutate(path, tools, fault):
    result = subprocess.run([str(tools.m2m_python), "-c", MUTATE, str(path), fault], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("fault", ["constant_read", "remove_remainder", "divisor", "attribute"])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_actual_remainder_target_mutations_are_rejected(tmp_path, lean_tools, fault, index_bits):
    stage, before, after = stage_for(tmp_path, lean_tools, source())
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)
    mutate(after, lean_tools, fault)
    result = check_typed_stage(stage, m2m_python=lean_tools.m2m_python, index_bits=index_bits)
    assert result["verdict"] == "abstracted", result.get("reason")


def captured_source():
    return (
        source(size=3)
        .replace(
            "%value = tensor.extract %data[%remainder] : tensor<3xi13>",
            """%value = linalg.generic {
          indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>], iterator_types = ["parallel"]}
          ins(%xs : tensor<3xi13>) outs(%initial : tensor<3xi13>) {
          ^bb0(%x: i13, %old: i13):
            %loaded = tensor.extract %data[%remainder] : tensor<3xi13>
            %sum = arith.addi %loaded, %x : i13
            linalg.yield %sum : i13
        } -> tensor<3xi13>""",
        )
        .replace(", %keep: i32)", ", %xs: tensor<3xi13>, %initial: tensor<3xi13>, %keep: i32)")
        .replace("-> (i13, i32)", "-> (tensor<3xi13>, i32)")
        .replace("return %value, %keep : i13, i32", "return %value, %keep : tensor<3xi13>, i32")
    )


@pytest.mark.parametrize("index_bits", [32, 64])
def test_actual_remainder_bound_survives_generic_capture(tmp_path, lean_tools, index_bits):
    stage, before, _ = stage_for(tmp_path, lean_tools, captured_source())
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)


def body_source():
    return """func.func @forward(%data: tensor<3xi13>, %indices: tensor<3xindex>,
        %initial: tensor<3xi13>, %keep: i32) -> (tensor<3xi13>, i32) {
      %dead = arith.addi %keep, %keep : i32
      %divisor = arith.constant 3 : index
      %value = linalg.generic {
        indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>], iterator_types = ["parallel"]}
        ins(%indices : tensor<3xindex>) outs(%initial : tensor<3xi13>) {
        ^bb0(%idx: index, %old: i13):
          %remainder = arith.remui %idx, %divisor : index
          %loaded = tensor.extract %data[%remainder] : tensor<3xi13>
          linalg.yield %loaded : i13
      } -> tensor<3xi13>
      return %value, %keep : tensor<3xi13>, i32
    }"""


@pytest.mark.parametrize("index_bits", [32, 64])
def test_positive_remainder_rule_runs_inside_generic_bodies(tmp_path, lean_tools, index_bits):
    stage, before, _ = stage_for(tmp_path, lean_tools, body_source())
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)


@pytest.mark.parametrize("index_bits", [32, 64])
def test_scalar_remainder_signedness_mutation_refuses(tmp_path, lean_tools, index_bits):
    stage, before, after = stage_for(tmp_path, lean_tools, source(extract=False))
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)
    mutate(after, lean_tools, "signedness")
    result = check_typed_stage(stage, m2m_python=lean_tools.m2m_python, index_bits=index_bits)
    assert result["verdict"] == "abstracted", result.get("reason")


@pytest.mark.parametrize("operand", ["numerator", "divisor"])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_poison_remainder_operands_do_not_gain_initialized_facts(tmp_path, lean_tools, operand, index_bits):
    text = source(extract=False).replace(
        "      %remainder =",
        """      %empty = tensor.empty() : tensor<index>
      %poison = tensor.extract %empty[] : tensor<index>
      %remainder =""",
    )
    text = text.replace(
        '"arith.remui"(%idx, %divisor)',
        '"arith.remui"(%poison, %divisor)' if operand == "numerator" else '"arith.remui"(%idx, %poison)',
    )
    before = tmp_path / "source.mlir"
    before.write_text(text)
    verify(before, lean_tools)
    result = check_support(
        before, "forward", tmp_path / "support", m2m_python=lean_tools.m2m_python, index_bits=index_bits
    )
    assert result["verdict"] == "verified", result.get("reason")
    result = check_definedness(
        before, "forward", tmp_path / "defined", m2m_python=lean_tools.m2m_python, index_bits=index_bits
    )
    assert result["verdict"] == "abstracted", result.get("reason")


@pytest.mark.parametrize("index_bits", [32, 64])
def test_divisor_is_decoded_at_actual_index_width(tmp_path, lean_tools, index_bits):
    before = tmp_path / "source.mlir"
    before.write_text(source(divisor=1 << 32, extract=False))
    verify(before, lean_tools)
    result = check_definedness(
        before, "forward", tmp_path / "defined", m2m_python=lean_tools.m2m_python, index_bits=index_bits
    )
    assert result["verdict"] == ("abstracted" if index_bits == 32 else "verified"), result.get("reason")
