"""Actual min/max canonicalization reorders operands without assuming initialized inputs."""

import json
import subprocess

import pytest
from test_lean_defined_conversions import certify, verify
from test_lean_integer_fold import canonicalize

from merlin.verify.lean.check import check_definedness, check_support, check_typed_stage, compose_chain

NAMES = ["minui", "maxui", "minsi", "maxsi"]


def source(name="minui", width=13, *, generic=False, escape=False):
    ty = f"i{width}"
    if not generic:
        result_types = f"{ty}, {ty}, i32" if escape else f"{ty}, i32"
        results = "%r, %input, %keep" if escape else "%r, %keep"
        return f"""func.func @forward(%input: {ty}, %keep: i32) -> ({result_types}) {{
          %dead = arith.addi %keep, %keep : i32
          %cap = arith.constant {1 if width == 1 else 5} : {ty}
          %r = arith.{name} %cap, %input : {ty}
          return {results} : {result_types}
        }}"""
    tensor = f"tensor<3x{ty}>"
    return f"""func.func @forward(%input: {tensor}, %initial: {tensor}, %keep: i32) -> ({tensor}, i32) {{
      %dead = arith.addi %keep, %keep : i32
      %cap = arith.constant 5 : {ty}
      %out = linalg.generic {{indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
        iterator_types = ["parallel"]}} ins(%input : {tensor}) outs(%initial : {tensor}) {{
        ^bb0(%x: {ty}, %old: {ty}):
          %r = arith.{name} %cap, %x : {ty}
          linalg.yield %r : {ty}
      }} -> {tensor}
      return %out, %keep : {tensor}, i32
    }}"""


def stage_for(tmp_path, tools, text):
    stage, before, after = canonicalize(tmp_path, tools, text)
    (stage / "meta.json").write_text(json.dumps({"boundary": "integer-minmax-commutation"}))
    return stage, before, after


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("width", [8, 13, 32, 64])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_actual_scalar_minmax_commutation_has_defined_chain(tmp_path, lean_tools, name, width, index_bits):
    stage, before, after = stage_for(tmp_path, lean_tools, source(name, width, escape=True))
    assert f"arith.{name}" in after.read_text()
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("index_bits", [32, 64])
def test_actual_generic_minmax_commutes_captured_constant(tmp_path, lean_tools, name, index_bits):
    stage, before, after = stage_for(tmp_path, lean_tools, source(name, generic=True))
    assert "linalg.generic" in after.read_text() and f"arith.{name}" in after.read_text()
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)


def manual_swap(tmp_path, tools, name):
    stage = tmp_path / "stage"
    stage.mkdir()
    before, after = stage / "before.mlir", stage / "after.mlir"
    text = source(name, 1, escape=True)
    before.write_text(text)
    after.write_text(text.replace(f"arith.{name} %cap, %input", f"arith.{name} %input, %cap"))
    verify(before, tools)
    verify(after, tools)
    (stage / "meta.json").write_text(json.dumps({"boundary": "mlir-verified-i1-commutation-control"}))
    return stage, before, after


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("index_bits", [32, 64])
def test_i1_manual_commutation_is_independently_accepted(tmp_path, lean_tools, name, index_bits):
    stage, before, _ = manual_swap(tmp_path, lean_tools, name)
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("index_bits", [32, 64])
def test_i1_actual_extreme_fold_distinguishes_neutral_and_absorbing_rules(tmp_path, lean_tools, name, index_bits):
    stage, before, after = stage_for(tmp_path, lean_tools, source(name, 1, escape=True))
    assert f"arith.{name}" not in after.read_text()
    for checker, directory in [(check_support, "support"), (check_definedness, "defined")]:
        result = checker(
            before, "forward", tmp_path / directory, index_bits=index_bits, m2m_python=lean_tools.m2m_python
        )
        assert result["verdict"] == "verified", result.get("reason")
    expected = "verified" if name in {"minui", "maxsi"} else "abstracted"
    result = check_typed_stage(stage, m2m_python=lean_tools.m2m_python, index_bits=index_bits)
    assert result["verdict"] == expected, result.get("reason")
    # Only neutral extrema preserve poison. The absorbing folds still fail the actual
    # boundary obligation, even though their initialized source execution is defined.
    result = compose_chain(
        [stage],
        tmp_path / "support/receipt.json",
        "forward",
        tmp_path / "chain.json",
        definedness=tmp_path / "defined/receipt.json",
        m2m_python=lean_tools.m2m_python,
    )
    assert result["verdict"] == expected, result.get("reason")


MUTATE = """
from pathlib import Path
from torch_mlir import ir
import sys
with ir.Context():
    path,fault=Path(sys.argv[1]),sys.argv[2]
    text=path.read_text()
    if fault in ['operation','signedness']:
        text=text.replace('arith.minui','arith.maxui' if fault=='operation' else 'arith.minsi',1)
    m=ir.Module.parse(text); block=m.body.operations[0].regions[0].blocks[0]
    if fault=='bits':
        c=next(o for o in block.operations if o.name=='arith.constant')
        a=ir.IntegerAttr(c.attributes['value'])
        c.operation.attributes['value']=ir.IntegerAttr.get(a.type,a.value+1)
    elif fault=='keep':
        ret=list(block.operations)[-1]
        with ir.InsertionPoint(ret):
            c=ir.Operation.create('arith.constant',results=[ir.IntegerType.get_signless(32)],
              attributes={'value':ir.IntegerAttr.get(ir.IntegerType.get_signless(32),0)},loc=ir.Location.unknown())
        ret.operation.operands[-1]=c.results[0]
    elif fault=='attribute':
        op=next(o for o in block.operations if o.name=='arith.minui')
        op.operation.attributes['unreviewed.meaning']=ir.UnitAttr.get()
    m.operation.verify(); path.write_text(str(m))
"""


def mutate(path, tools, fault):
    result = subprocess.run([str(tools.m2m_python), "-c", MUTATE, str(path), fault], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("fault", ["operation", "signedness", "bits", "keep", "attribute"])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_minmax_target_changes_are_rechecked(tmp_path, lean_tools, fault, index_bits):
    stage, before, after = stage_for(tmp_path, lean_tools, source(escape=True))
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)
    mutate(after, lean_tools, fault)
    result = check_typed_stage(stage, m2m_python=lean_tools.m2m_python, index_bits=index_bits)
    assert result["verdict"] == "abstracted", result.get("reason")
