"""Unsigned index clamping derives safe bounds from an actual initialized operand."""

import json
import subprocess

import pytest
from test_lean_defined_conversions import certify, verify
from test_lean_integer_fold import canonicalize

from merlin.verify.lean.check import check_definedness, check_support, check_typed_stage


def source(mode="constant", cap=7, *, reverse=False, size=8, extract=True, dirty=False, attributes=""):
    ty = f"tensor<{size}xi13>"
    bound_type = "index" if mode == "unknown" else "i3"
    if mode == "constant":
        producer = f"%cap = arith.constant {cap} : index"
    elif mode == "widened":
        producer = "%cap = arith.index_castui %bound : i3 to index"
    elif mode == "unknown":
        producer = ""
    else:
        raise ValueError(mode)
    bound = "%bound" if mode == "unknown" else "%cap"
    left, right = (bound, "%idx") if reverse else ("%idx", bound)
    result_type = "i13" if extract else "index"
    empty = f"%empty = tensor.empty() : {ty}" if dirty else ""
    read = f"%value = tensor.extract {'%empty' if dirty else '%data'}[%clamped] : {ty}" if extract else ""
    return f"""func.func @forward(%data: {ty}, %idx: index, %bound: {bound_type}, %keep: i32)
        -> ({result_type}, i32) {{
      %dead = arith.addi %keep, %keep : i32
      {producer}
      %clamped = "arith.minui"({left}, {right}) {attributes} : (index, index) -> index
      {empty}
      {read}
      return %{"value" if extract else "clamped"}, %keep : {result_type}, i32
    }}"""


def stage_for(tmp_path, tools, text):
    stage, before, after = canonicalize(tmp_path, tools, text)
    (stage / "meta.json").write_text(json.dumps({"boundary": "index-clamp-definedness"}))
    return stage, before, after


@pytest.mark.parametrize("mode,cap", [("constant", 3), ("constant", 7), ("widened", 7)])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_actual_clamped_tensor_read_has_a_defined_chain(tmp_path, lean_tools, mode, cap, reverse, index_bits):
    stage, before, after = stage_for(tmp_path, lean_tools, source(mode, cap, reverse=reverse))
    assert "arith.minui" in after.read_text()
    assert "arith.addi" in before.read_text() and "arith.addi" not in after.read_text()
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_minimum_without_bound_remains_initialized(tmp_path, lean_tools, reverse, index_bits):
    stage, before, _ = stage_for(tmp_path, lean_tools, source("unknown", reverse=reverse, extract=False))
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)


@pytest.mark.parametrize(
    "options",
    [{"cap": 8}, {"cap": 2147483648}, {"mode": "unknown"}, {"dirty": True}, {"size": 0}],
)
@pytest.mark.parametrize("index_bits", [32, 64])
def test_unsafe_or_unknown_read_bounds_still_refuse(tmp_path, lean_tools, options, index_bits):
    path = tmp_path / "source.mlir"
    path.write_text(source(**options))
    verify(path, lean_tools)
    result = check_support(
        path, "forward", tmp_path / "support", index_bits=index_bits, m2m_python=lean_tools.m2m_python
    )
    assert result["verdict"] == "verified", result.get("reason")
    result = check_definedness(
        path, "forward", tmp_path / "defined", index_bits=index_bits, m2m_python=lean_tools.m2m_python
    )
    assert result["verdict"] == "abstracted", result.get("reason")


@pytest.mark.parametrize("attribute", ["nonNeg", "isExact", "disjoint", "unreviewed.meaning"])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_clamp_promises_or_unknown_attributes_do_not_gain_definedness(tmp_path, lean_tools, attribute, index_bits):
    path = tmp_path / "source.mlir"
    path.write_text(source(attributes="{" + attribute + "}"))
    verify(path, lean_tools)
    result = check_definedness(
        path, "forward", tmp_path / "defined", index_bits=index_bits, m2m_python=lean_tools.m2m_python
    )
    assert result["verdict"] == "abstracted", result.get("reason")


MUTATE = """
from pathlib import Path
from torch_mlir import ir
import sys
with ir.Context():
    path, fault = Path(sys.argv[1]), sys.argv[2]
    m=ir.Module.parse(path.read_text()); block=m.body.operations[0].regions[0].blocks[0]
    minimum=next(o for o in block.operations if o.name=='arith.minui')
    extract=next(o for o in block.operations if o.name=='tensor.extract')
    if fault == 'constant_read':
        with ir.InsertionPoint(extract):
            zero=ir.Operation.create('arith.constant',results=[ir.IndexType.get()],
                attributes={'value':ir.IntegerAttr.get(ir.IndexType.get(),0)},loc=ir.Location.unknown())
        extract.operation.operands[1]=zero.results[0]
    elif fault == 'remove_clamp':
        extract.operation.operands[1]=block.arguments[1]
    elif fault == 'cap':
        constant=next(o for o in block.operations if o.name=='arith.constant')
        constant.operation.attributes['value']=ir.IntegerAttr.get(ir.IndexType.get(),8)
    else:
        minimum.operation.attributes['unreviewed.meaning']=ir.UnitAttr.get()
    m.operation.verify(); path.write_text(str(m))
"""


def mutate(path, tools, fault):
    result = subprocess.run([str(tools.m2m_python), "-c", MUTATE, str(path), fault], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("fault", ["constant_read", "remove_clamp", "cap", "attribute"])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_actual_clamp_mutations_are_rechecked(tmp_path, lean_tools, fault, index_bits):
    stage, before, after = stage_for(tmp_path, lean_tools, source())
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)
    mutate(after, lean_tools, fault)
    result = check_typed_stage(stage, m2m_python=lean_tools.m2m_python, index_bits=index_bits)
    assert result["verdict"] == "abstracted", result.get("reason")
    result = check_definedness(
        after, "forward", tmp_path / "mutated-defined", index_bits=index_bits, m2m_python=lean_tools.m2m_python
    )
    assert result["verdict"] == ("verified" if fault == "constant_read" else "abstracted"), result.get("reason")


@pytest.mark.parametrize("index_bits", [32, 64])
def test_zero_cap_source_is_defined_but_rewrite_remains_outside_family(tmp_path, lean_tools, index_bits):
    stage, before, after = stage_for(tmp_path, lean_tools, source(cap=0))
    assert "arith.minui" not in after.read_text()
    for checker, directory in [(check_support, "support"), (check_definedness, "defined")]:
        result = checker(
            before, "forward", tmp_path / directory, index_bits=index_bits, m2m_python=lean_tools.m2m_python
        )
        assert result["verdict"] == "verified", result.get("reason")
    result = check_typed_stage(stage, m2m_python=lean_tools.m2m_python, index_bits=index_bits)
    assert result["verdict"] == "abstracted", result.get("reason")


def captured_source(mode="constant"):
    text = source(mode)
    generic = """%value = linalg.generic {
      indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
      iterator_types = ["parallel"]} ins(%xs : tensor<3xi13>) outs(%initial : tensor<3xi13>) {
      ^bb0(%x: i13, %old: i13):
        %loaded = tensor.extract %data[%clamped] : tensor<8xi13>
        %sum = arith.addi %loaded, %x : i13
        linalg.yield %sum : i13
    } -> tensor<3xi13>"""
    text = text.replace(", %keep: i32)", ", %xs: tensor<3xi13>, %initial: tensor<3xi13>, %keep: i32)")
    text = text.replace("-> (i13, i32)", "-> (tensor<3xi13>, i32)")
    text = text.replace("%value = tensor.extract %data[%clamped] : tensor<8xi13>", generic)
    return text.replace("return %value, %keep : i13, i32", "return %value, %keep : tensor<3xi13>, i32")


@pytest.mark.parametrize("mode", ["constant", "widened"])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_clamp_range_survives_real_generic_capture(tmp_path, lean_tools, mode, index_bits):
    stage, before, _ = stage_for(tmp_path, lean_tools, captured_source(mode))
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)
