"""Leading scalar constants may cross checked generic scopes without inventing captured facts."""

import json
import subprocess

import pytest
from test_lean_defined_conversions import certify, verify
from test_lean_integer_compare_order import PREDICATES, body_source, stage_for
from test_lean_integer_idempotent import certify as certify_conditional

from merlin.verify.lean.check import check_support, check_typed_stage


def float_source(fmt="f32", *, reduction=False):
    output = f"tensor<{fmt}>" if reduction else f"tensor<3x{fmt}>"
    return f"""func.func @forward(%data: tensor<3x{fmt}>, %initial: {output}, %keep: i32)
        -> ({output}, i32) {{
      %result = linalg.generic {{indexing_maps = [affine_map<(d0)->(d0)>,
          {"affine_map<(d0)->()>" if reduction else "affine_map<(d0)->(d0)>"}],
          iterator_types = ["{"reduction" if reduction else "parallel"}"]}}
          ins(%data : tensor<3x{fmt}>) outs(%initial : {output}) {{
        ^bb0(%x: {fmt}, %old: {fmt}):
          %cap = arith.constant 1.500000e+00 : {fmt}
          %first = arith.addf %x, %cap : {fmt}
          %mixed = arith.addf %first, %old : {fmt}
          linalg.yield %mixed : {fmt}
      }} -> {output}
      return %result, %keep : {output}, i32
    }}"""


@pytest.mark.parametrize("predicate", PREDICATES)
@pytest.mark.parametrize("reduction", [False, True])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_actual_constant_hoist_and_comparison_order_have_defined_chain(
    tmp_path, lean_tools, predicate, reduction, index_bits
):
    stage, before, after = stage_for(tmp_path, lean_tools, body_source(predicate, reduction=reduction))
    assert before.read_text().index("linalg.generic") < before.read_text().index("arith.constant")
    assert after.read_text().index("arith.constant") < after.read_text().index("linalg.generic")
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)


@pytest.mark.parametrize("predicate", ["eq", "slt", "uge"])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_actual_hoisted_scalar_remains_available_to_depth_two_capture(tmp_path, lean_tools, predicate, index_bits):
    stage, before, after = stage_for(tmp_path, lean_tools, body_source(predicate, nested=True))
    assert after.read_text().count("linalg.generic") == 2
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)


@pytest.mark.parametrize("fmt", ["f16", "bf16", "f32", "f64"])
@pytest.mark.parametrize("reduction", [False, True])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_actual_raw_float_constant_hoist_has_no_numeric_law_premise(tmp_path, lean_tools, fmt, reduction, index_bits):
    stage, before, after = stage_for(tmp_path, lean_tools, float_source(fmt, reduction=reduction))
    assert after.read_text().index("arith.constant") < after.read_text().index("linalg.generic")
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)


@pytest.mark.parametrize("predicate", ["eq", "slt"])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_hoisting_closed_constant_does_not_initialize_poison_inputs(tmp_path, lean_tools, predicate, index_bits):
    stage, before, _ = stage_for(tmp_path, lean_tools, body_source(predicate, dirty=True))
    certify_conditional(stage, before, lean_tools, tmp_path, defined=False, index_bits=index_bits)


MUTATE = """
from pathlib import Path
from torch_mlir import ir
import sys
with ir.Context():
    path,fault=Path(sys.argv[1]),sys.argv[2]
    m=ir.Module.parse(path.read_text());b=m.body.operations[0].regions[0].blocks[0]
    generic=next(o for o in b.operations if o.name=='linalg.generic')
    body=generic.regions[0].blocks[0]
    cmp=next(o for o in body.operations if o.name=='arith.cmpi')
    if fault=='bits':
        c=next(o for o in b.operations if o.name=='arith.constant')
        attr=ir.IntegerAttr(c.attributes['value']);c.operation.attributes['value']=ir.IntegerAttr.get(attr.type,6)
    elif fault=='predicate':
        attr=ir.IntegerAttr(cmp.attributes['predicate'])
        cmp.operation.attributes['predicate']=ir.IntegerAttr.get(attr.type,2)
    elif fault=='input':cmp.operation.operands[0]=body.arguments[1]
    elif fault=='keep':
        ret=list(b.operations)[-1]
        with ir.InsertionPoint(ret):
            c=ir.Operation.create('arith.constant',results=[ir.IntegerType.get_signless(32)],
                attributes={'value':ir.IntegerAttr.get(ir.IntegerType.get_signless(32),0)},loc=ir.Location.unknown())
        ret.operation.operands[-1]=c.results[0]
    elif fault=='attribute':cmp.operation.attributes['unreviewed.meaning']=ir.UnitAttr.get()
    elif fault=='fastmath':cmp.operation.attributes['fastmath']=ir.Attribute.parse('#arith.fastmath<nnan>')
    else:raise ValueError(fault)
    m.operation.verify();path.write_text(str(m))
"""


def mutate(path, tools, fault):
    original = path.read_bytes()
    result = subprocess.run([str(tools.m2m_python), "-c", MUTATE, str(path), fault], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert original != path.read_bytes()


@pytest.mark.parametrize("fault", ["bits", "predicate", "input", "keep", "attribute", "fastmath"])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_actual_hoisted_target_mutations_are_independently_rejected(tmp_path, lean_tools, fault, index_bits):
    stage, before, after = stage_for(tmp_path, lean_tools, body_source("slt"))
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)
    mutate(after, lean_tools, fault)
    result = check_typed_stage(stage, m2m_python=lean_tools.m2m_python, index_bits=index_bits)
    assert result["verdict"] == "abstracted", result.get("reason")


def test_unknown_source_constant_attributes_are_not_erased_by_proof(tmp_path, lean_tools):
    text = body_source().replace(
        "%cap = arith.constant 5 : i13",
        '%cap = "arith.constant"() {value = 5 : i13, unreviewed.meaning} : () -> i13',
    )
    stage, before, _ = stage_for(tmp_path, lean_tools, text)
    verify(before, lean_tools)
    assert (
        check_support(before, "forward", tmp_path / "support", m2m_python=lean_tools.m2m_python)["verdict"]
        == "abstracted"
    )
    assert check_typed_stage(stage, m2m_python=lean_tools.m2m_python)["verdict"] == "abstracted"


def manual_zero_stage(work, tools, *, reduction=False):
    stage = work / "stage"
    stage.mkdir(parents=True)
    before, after = stage / "before.mlir", stage / "after.mlir"
    before.write_text(body_source(reduction=reduction).replace("tensor<3xi13>", "tensor<0xi13>"))
    result = subprocess.run(
        [
            str(tools.m2m_python),
            "-c",
            "from pathlib import Path; from torch_mlir import ir; import sys\n"
            "with ir.Context():\n m=ir.Module.parse(Path(sys.argv[1]).read_text())\n"
            " b=m.body.operations[0].regions[0].blocks[0]\n"
            " g=next(o for o in b.operations if o.name=='linalg.generic')\n"
            " c=list(g.regions[0].blocks[0].operations)[0]; c.operation.move_before(g.operation)\n"
            " m.operation.verify(); Path(sys.argv[2]).write_text(str(m))",
            str(before),
            str(after),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert before.read_bytes() != after.read_bytes()
    verify(before, tools)
    (stage / "meta.json").write_text(json.dumps({"boundary": "mlir-verified-manual-zero-loop-constant-hoist"}))
    return stage, before, after


@pytest.mark.parametrize("reduction", [False, True])
def test_manual_zero_iteration_hoist_adds_no_failure(tmp_path, lean_tools, reduction):
    stage, before, _ = manual_zero_stage(tmp_path, lean_tools, reduction=reduction)
    certify(stage, before, lean_tools, tmp_path)
