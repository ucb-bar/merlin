"""Actual scalar bitcast identities use runtime typing and poison-exact origin facts."""

import json
import subprocess

import pytest
from test_lean_defined_conversions import certify, verify
from test_lean_integer_fold import canonicalize
from test_lean_integer_idempotent import certify as certify_conditional

from merlin.verify.lean.check import check_definedness, check_support, check_typed_stage

INTEGER_TYPES = [f"i{w}" for w in [1, 8, 13, 16, 32, 64, 128]]
PAIRS = [
    pair for i, f in [("i16", "f16"), ("i16", "bf16"), ("i32", "f32"), ("i64", "f64")] for pair in [(i, f), (f, i)]
]


def source(src="i32", mid="f32", *, escape=False, identity=False):
    result = "%first" if identity else "%result"
    returned = f"{result}, %first, %x, %keep" if escape and not identity else f"{result}, %x, %keep"
    types = f"{src}, {mid}, {src}, i32" if escape and not identity else f"{src}, {src}, i32"
    return f"""func.func @forward(%x: {src}, %other: {src}, %keep: i32) -> ({types}) {{
      %first = arith.bitcast %x : {src} to {mid}
      {"" if identity else f"%result = arith.bitcast %first : {mid} to {src}"}
      return {returned} : {types}
    }}"""


def body_source(src="i32", mid="f32", *, nested=False, dirty=False, accumulate=True):
    calculation = f"""%first = arith.bitcast %x : {src} to {mid}
          %result = arith.bitcast %first : {mid} to {src}
          linalg.yield %result : {src}"""
    if accumulate and not nested:
        if src.startswith("i"):
            calculation = calculation.replace(
                "linalg.yield %result", f"%mixed = arith.xori %result, %old : {src}\n          linalg.yield %mixed"
            )
        else:
            integer = f"i{16 if src == 'bf16' else src[1:]}"
            calculation = calculation.replace(
                f"linalg.yield %result : {src}",
                f"""%raw = arith.bitcast %result : {src} to {integer}
          %oldraw = arith.bitcast %old : {src} to {integer}
          %bits = arith.xori %raw, %oldraw : {integer}
          %mixed = arith.bitcast %bits : {integer} to {src}
          linalg.yield %mixed : {src}""",
            )
    if nested:
        calculation = f"""%fresh = tensor.splat %old : tensor<{src}>
          %inner = linalg.generic {{indexing_maps = [affine_map<()->()>], iterator_types = []}}
              outs(%fresh : tensor<{src}>) {{
            ^bb0(%previous: {src}):
              %first = arith.bitcast %x : {src} to {mid}
              %result = arith.bitcast %first : {mid} to {src}
              linalg.yield %result : {src}
          }} -> tensor<{src}>
          %value = tensor.extract %inner[] : tensor<{src}>
          linalg.yield %value : {src}"""
    return f"""func.func @forward(%data: tensor<3x{src}>, %initial: tensor<3x{src}>, %keep: i32)
        -> (tensor<3x{src}>, i32) {{
      {f"%empty = tensor.empty() : tensor<3x{src}>" if dirty else ""}
      %result = linalg.generic {{indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
        iterator_types = ["parallel"]}} ins({"%empty" if dirty else "%data"} : tensor<3x{src}>)
          outs(%initial : tensor<3x{src}>) {{
        ^bb0(%x: {src}, %old: {src}):
          {calculation}
      }} -> tensor<3x{src}>
      return %result, %keep : tensor<3x{src}>, i32
    }}"""


def stage_for(work, tools, text):
    stage, before, after = canonicalize(work, tools, text)
    (stage / "meta.json").write_text(json.dumps({"boundary": "scalar-bitcast-identity-and-roundtrip"}))
    return stage, before, after


@pytest.mark.parametrize("src", INTEGER_TYPES)
@pytest.mark.parametrize("index_bits", [32, 64])
def test_actual_integer_bitcast_identity_has_defined_chain(tmp_path, lean_tools, src, index_bits):
    stage, before, after = stage_for(tmp_path, lean_tools, source(src, src, identity=True))
    assert "arith.bitcast" not in after.read_text()
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)


@pytest.mark.parametrize("src,mid", PAIRS)
@pytest.mark.parametrize("escape", [False, True])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_actual_bitcast_roundtrip_retains_escaping_intermediate(tmp_path, lean_tools, src, mid, escape, index_bits):
    stage, before, after = stage_for(tmp_path, lean_tools, source(src, mid, escape=escape))
    assert after.read_text().count("arith.bitcast") == int(escape)
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)


@pytest.mark.parametrize("src,mid", PAIRS)
@pytest.mark.parametrize("nested", [False, True])
def test_actual_body_bitcast_cancellation_handles_captured_roots(tmp_path, lean_tools, src, mid, nested):
    stage, before, after = stage_for(tmp_path, lean_tools, body_source(src, mid, nested=nested))
    assert after.read_text().count("arith.bitcast") == (0 if nested or src.startswith("i") else 3)
    certify(stage, before, lean_tools, tmp_path)


@pytest.mark.parametrize("src,mid", [("i32", "f32"), ("f32", "i32")])
def test_poison_bitcast_roundtrip_stays_conditional(tmp_path, lean_tools, src, mid):
    stage, before, _ = stage_for(tmp_path, lean_tools, body_source(src, mid, dirty=True))
    certify_conditional(stage, before, lean_tools, tmp_path, defined=False)


@pytest.mark.parametrize("src,mid", [("f16", "f16"), ("bf16", "bf16"), ("f32", "f32"), ("f64", "f64"), ("f16", "bf16")])
def test_unmodelled_float_bitcast_is_still_refused(tmp_path, lean_tools, src, mid):
    stage, before, after = stage_for(tmp_path, lean_tools, source(src, mid, identity=src == mid))
    assert "arith.bitcast" not in after.read_text()
    verify(before, lean_tools)
    for checker, name in [(check_support, "support"), (check_definedness, "defined")]:
        result = checker(before, "forward", tmp_path / name, m2m_python=lean_tools.m2m_python)
        assert result["verdict"] == "abstracted", result.get("reason")
    assert check_typed_stage(stage, m2m_python=lean_tools.m2m_python)["verdict"] == "abstracted"


@pytest.mark.parametrize("src,mid", [("i32", "f32"), ("f32", "i32")])
def test_eliminating_identity_generic_at_return_has_a_defined_chain(tmp_path, lean_tools, src, mid):
    stage, before, after = stage_for(tmp_path, lean_tools, body_source(src, mid, accumulate=False))
    assert "linalg.generic" not in after.read_text()
    certify(stage, before, lean_tools, tmp_path)


MUTATE = """
from pathlib import Path
from torch_mlir import ir
import sys
with ir.Context():
    path,fault=Path(sys.argv[1]),sys.argv[2]
    m=ir.Module.parse(path.read_text());b=m.body.operations[0].regions[0].blocks[0]
    ret=list(b.operations)[-1]
    if fault=='result':ret.operation.operands[0]=b.arguments[1]
    elif fault=='keep':
        with ir.InsertionPoint(ret):
            c=ir.Operation.create('arith.constant',results=[ir.IntegerType.get_signless(32)],
                attributes={'value':ir.IntegerAttr.get(ir.IntegerType.get_signless(32),0)},loc=ir.Location.unknown())
        ret.operation.operands[-1]=c.results[0]
    else:
        cast=next(o for o in b.operations if o.name=='arith.bitcast')
        if fault=='intermediate':cast.operation.operands[0]=b.arguments[1]
        elif fault=='attribute':cast.operation.attributes['unreviewed.meaning']=ir.UnitAttr.get()
        elif fault=='nonNeg':cast.operation.attributes['nonNeg']=ir.UnitAttr.get()
        elif fault=='fastmath':cast.operation.attributes['fastmath']=ir.Attribute.parse('#arith.fastmath<nnan>')
        else:raise ValueError(fault)
    m.operation.verify();path.write_text(str(m))
"""


def mutate(path, tools, fault):
    original = path.read_bytes()
    result = subprocess.run([str(tools.m2m_python), "-c", MUTATE, str(path), fault], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert path.read_bytes() != original


@pytest.mark.parametrize("fault", ["result", "keep", "intermediate", "attribute", "nonNeg", "fastmath"])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_actual_target_bitcast_mutation_refuses(tmp_path, lean_tools, fault, index_bits):
    stage, before, after = stage_for(tmp_path, lean_tools, source(escape=True))
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)
    mutate(after, lean_tools, fault)
    result = check_typed_stage(stage, m2m_python=lean_tools.m2m_python, index_bits=index_bits)
    assert result["verdict"] == "abstracted", result.get("reason")
