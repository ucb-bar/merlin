"""Actual integer comparison reversal uses the inverse predicate and exact source obligations."""

import json
import subprocess

import pytest
from test_lean_defined_conversions import certify, verify
from test_lean_integer_fold import canonicalize
from test_lean_integer_idempotent import certify as certify_conditional

from merlin.verify.lean.check import check_definedness, check_support, check_typed_stage

PREDICATES = ["eq", "ne", "slt", "sle", "sgt", "sge", "ult", "ule", "ugt", "uge"]
CODES = {name: i for i, name in enumerate(PREDICATES)}
INVERSE = dict(zip(PREDICATES, ["eq", "ne", "sgt", "sge", "slt", "sle", "ugt", "uge", "ult", "ule"], strict=True))


def source(predicate="slt", width=13):
    ty = "index" if width == "index" else f"i{width}"
    return f"""func.func @forward(%x: {ty}, %keep: i32) -> (i1, {ty}, i32) {{
      %cap = arith.constant {1 if width == 1 else 5} : {ty}
      %result = arith.cmpi {predicate}, %cap, %x : {ty}
      return %result, %x, %keep : i1, {ty}, i32
    }}"""


def body_source(predicate="slt", *, reduction=False, captured=False, dirty=False, nested=False):
    constant = "%cap = arith.constant 5 : i13"
    output = "tensor<i13>" if reduction else "tensor<3xi13>"
    compare = f"%compared = arith.cmpi {predicate}, %cap, %x : i13"
    body = f"""{compare}
          %wide = arith.extui %compared : i1 to i13
          %mixed = arith.xori %wide, %old : i13
          linalg.yield %mixed : i13"""
    if nested:
        assert not reduction
        body = f"""%fresh = tensor.splat %old : tensor<i13>
          %inner = linalg.generic {{indexing_maps = [affine_map<()->()>], iterator_types = []}}
              outs(%fresh : tensor<i13>) {{
            ^bb0(%previous: i13):
              {compare}
              %wide = arith.extui %compared : i1 to i13
              %mixed = arith.xori %wide, %previous : i13
              linalg.yield %mixed : i13
          }} -> tensor<i13>
          %value = tensor.extract %inner[] : tensor<i13>
          linalg.yield %value : i13"""
    return f"""func.func @forward(%data: tensor<3xi13>, %initial: {output}, %keep: i32)
        -> ({output}, i32) {{
      {constant if captured else ""}
      {"%empty = tensor.empty() : tensor<3xi13>" if dirty else ""}
      %result = linalg.generic {{indexing_maps = [affine_map<(d0)->(d0)>,
          {"affine_map<(d0)->()>" if reduction else "affine_map<(d0)->(d0)>"}],
          iterator_types = ["{"reduction" if reduction else "parallel"}"]}}
          ins({"%empty" if dirty else "%data"} : tensor<3xi13>) outs(%initial : {output}) {{
        ^bb0(%x: i13, %old: i13):
          {"" if captured else constant}
          {body}
      }} -> {output}
      return %result, %keep : {output}, i32
    }}"""


def stage_for(work, tools, text):
    stage, before, after = canonicalize(work, tools, text)
    (stage / "meta.json").write_text(json.dumps({"boundary": "integer-comparison-order"}))
    return stage, before, after


def manual_i1_stage(work, tools, predicate):
    stage = work / "stage"
    stage.mkdir()
    before, after = stage / "before.mlir", stage / "after.mlir"
    text = source(predicate, 1)
    before.write_text(text)
    after.write_text(text.replace(f"cmpi {predicate}, %cap, %x", f"cmpi {INVERSE[predicate]}, %x, %cap"))
    verify(before, tools)
    verify(after, tools)
    (stage / "meta.json").write_text(json.dumps({"boundary": "mlir-verified-i1-predicate-reversal-control"}))
    return stage, before, after


@pytest.mark.parametrize("predicate", PREDICATES)
@pytest.mark.parametrize("width", [8, 13, 32, 64])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_actual_scalar_comparison_order_has_defined_chain(tmp_path, lean_tools, predicate, width, index_bits):
    stage, before, after = stage_for(tmp_path, lean_tools, source(predicate, width))
    assert f"arith.cmpi {INVERSE[predicate]}" in after.read_text()
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)


@pytest.mark.parametrize("predicate", PREDICATES)
@pytest.mark.parametrize("index_bits", [32, 64])
def test_actual_index_comparison_order_uses_actual_width(tmp_path, lean_tools, predicate, index_bits):
    stage, before, _ = stage_for(tmp_path, lean_tools, source(predicate, "index"))
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)


@pytest.mark.parametrize("predicate", PREDICATES)
@pytest.mark.parametrize("reduction,captured", [(False, False), (True, False), (False, True)])
def test_actual_body_comparison_order_preserves_accumulator_and_capture(
    tmp_path, lean_tools, predicate, reduction, captured
):
    stage, before, after = stage_for(
        tmp_path, lean_tools, body_source(predicate, reduction=reduction, captured=captured)
    )
    assert "linalg.generic" in after.read_text() and f"arith.cmpi {INVERSE[predicate]}" in after.read_text()
    certify(stage, before, lean_tools, tmp_path)


@pytest.mark.parametrize("predicate", ["eq", "slt", "uge"])
def test_actual_depth_two_comparisons_use_captured_scalars(tmp_path, lean_tools, predicate):
    stage, before, after = stage_for(tmp_path, lean_tools, body_source(predicate, captured=True, nested=True))
    assert after.read_text().count("linalg.generic") == 2
    certify(stage, before, lean_tools, tmp_path)


@pytest.mark.parametrize("predicate", PREDICATES)
@pytest.mark.parametrize("index_bits", [32, 64])
def test_manual_i1_predicate_reverse_is_exact(tmp_path, lean_tools, predicate, index_bits):
    stage, before, _ = manual_i1_stage(tmp_path, lean_tools, predicate)
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)


@pytest.mark.parametrize("predicate", ["eq", "slt"])
def test_poison_comparison_input_stays_conditional(tmp_path, lean_tools, predicate):
    stage, before, _ = stage_for(tmp_path, lean_tools, body_source(predicate, dirty=True))
    certify_conditional(stage, before, lean_tools, tmp_path, defined=False)


def test_reflexive_fold_is_explicitly_outside_operand_ordering(tmp_path, lean_tools):
    text = source().replace("cmpi slt, %cap, %x", "cmpi slt, %x, %x")
    stage, before, after = stage_for(tmp_path, lean_tools, text)
    assert "arith.cmpi" not in after.read_text()
    for checker, directory in [(check_support, "support"), (check_definedness, "defined")]:
        result = checker(before, "forward", tmp_path / directory, m2m_python=lean_tools.m2m_python)
        assert result["verdict"] == "verified", result.get("reason")
    result = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "abstracted", result.get("reason")


MUTATE = """
from pathlib import Path
from torch_mlir import ir
import sys
with ir.Context():
    path,fault=Path(sys.argv[1]),sys.argv[2]
    module=ir.Module.parse(path.read_text());block=module.body.operations[0].regions[0].blocks[0]
    cmp=next(o for o in block.operations if o.name=='arith.cmpi')
    if fault=='predicate':
        attr=ir.IntegerAttr(cmp.attributes['predicate'])
        cmp.operation.attributes['predicate']=ir.IntegerAttr.get(attr.type,2)
    elif fault=='input':cmp.operation.operands[0]=cmp.operands[1]
    elif fault=='bits':
        c=next(o for o in block.operations if o.name=='arith.constant')
        attr=ir.IntegerAttr(c.attributes['value']);c.operation.attributes['value']=ir.IntegerAttr.get(attr.type,6)
    elif fault=='keep':
        ret=list(block.operations)[-1]
        with ir.InsertionPoint(ret):
            c=ir.Operation.create('arith.constant',results=[ir.IntegerType.get_signless(32)],
                attributes={'value':ir.IntegerAttr.get(ir.IntegerType.get_signless(32),0)},loc=ir.Location.unknown())
        ret.operation.operands[-1]=c.results[0]
    elif fault=='attribute':cmp.operation.attributes['unreviewed.meaning']=ir.UnitAttr.get()
    elif fault=='fastmath':
        cmp.operation.attributes['fastmath']=ir.Attribute.parse('#arith.fastmath<nnan>')
    else:raise ValueError(fault)
    module.operation.verify();path.write_text(str(module))
"""


def mutate(path, tools, fault):
    original = path.read_bytes()
    result = subprocess.run([str(tools.m2m_python), "-c", MUTATE, str(path), fault], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert path.read_bytes() != original


@pytest.mark.parametrize("fault", ["predicate", "input", "bits", "keep", "attribute", "fastmath"])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_actual_target_comparison_mutations_are_rechecked(tmp_path, lean_tools, fault, index_bits):
    stage, before, after = stage_for(tmp_path, lean_tools, source())
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)
    mutate(after, lean_tools, fault)
    result = check_typed_stage(stage, m2m_python=lean_tools.m2m_python, index_bits=index_bits)
    assert result["verdict"] == "abstracted", result.get("reason")
