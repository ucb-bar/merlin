"""Actual widening/truncation cancellation with typed source-domain and full-chain checks."""

import subprocess

import pytest
from test_lean_integer_cast import cast_source
from test_lean_integer_cast_nested import certify
from test_lean_integer_cast_nested import nested_source as nested_cast_source
from test_lean_integer_cast_nested import source as body_cast_source
from test_lean_integer_fold import _verify_mlir, canonicalize

from merlin.verify.lean.check import check_definedness, check_support, check_typed_stage

WIDTHS = [(1, 2), (1, 128), (8, 13), (13, 32), (32, 64), (64, 128)]


def source(kind="extsi", widths=(8, 32), *, escape=False):
    width, wide = widths
    return cast_source(kind, "trunci", (width, wide, width), escape=escape)


def local_source(kind="extsi", *, depth=1, escape=False, reduction=False):
    if depth == 2:
        assert not escape and not reduction
        return nested_cast_source(kind, "trunci", (8, 32, 8))
    assert depth == 1
    return body_cast_source(kind, "trunci", (8, 32, 8), escape=escape, reduction=reduction)


def captured_source(kind="extsi"):
    return f"""func.func @forward(%x: i8, %keep: i8, %data: tensor<3xi8>, %init: tensor<3xi8>)
        -> (tensor<3xi8>, i8, i32) {{
      %mid = arith.{kind} %x : i8 to i32
      %roundtrip = arith.trunci %mid : i32 to i8
      %r = linalg.generic {{indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
          iterator_types = ["parallel"]}} ins(%data : tensor<3xi8>) outs(%init : tensor<3xi8>) {{
        ^bb0(%value: i8, %old: i8):
          %sum = arith.addi %value, %roundtrip : i8
          %mixed = arith.xori %sum, %old : i8
          linalg.yield %mixed : i8
      }} -> tensor<3xi8>
      return %r, %keep, %mid : tensor<3xi8>, i8, i32
    }}"""


def long_source(kind="extsi", *, escape=False):
    casts = []
    for width in range(9, 137):
        previous = "%x" if width == 9 else f"%v{width - 1}"
        casts.append(f"%v{width} = arith.{kind} {previous} : i{width - 1} to i{width}")
    outputs = ["%r", "%keep"] + (["%v13", "%v64"] if escape else [])
    types = ["i8", "i8"] + (["i13", "i64"] if escape else [])
    return f"""func.func @forward(%x: i8, %keep: i8) -> ({", ".join(types)}) {{
      {chr(10).join(casts)}
      %r = arith.trunci %v136 : i136 to i8
      return {", ".join(outputs)} : {", ".join(types)}
    }}"""


@pytest.mark.parametrize("kind", ["extui", "extsi"])
@pytest.mark.parametrize("widths", WIDTHS)
@pytest.mark.parametrize("escape", [False, True])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_actual_scalar_cast_roundtrips(tmp_path, lean_tools, kind, widths, escape, index_bits):
    stage, before, after = canonicalize(tmp_path, lean_tools, source(kind, widths, escape=escape))
    assert "arith.trunci" not in after.read_text()
    assert after.read_text().count(f"arith.{kind}") == int(escape)
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)


@pytest.mark.parametrize("kind", ["extui", "extsi"])
@pytest.mark.parametrize("escape,reduction", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_actual_body_roundtrips_preserve_escaping_producers(tmp_path, lean_tools, kind, escape, reduction, index_bits):
    stage, before, after = canonicalize(tmp_path, lean_tools, local_source(kind, escape=escape, reduction=reduction))
    assert "linalg.generic" in after.read_text() and "arith.trunci" not in after.read_text()
    assert after.read_text().count(f"arith.{kind}") == int(escape)
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)


@pytest.mark.parametrize("kind", ["extui", "extsi"])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_actual_depth_two_roundtrip_fresh_destinations(tmp_path, lean_tools, kind, index_bits):
    stage, before, after = canonicalize(tmp_path, lean_tools, local_source(kind, depth=2))
    assert after.read_text().count("linalg.generic") == 2
    assert "arith.trunci" not in after.read_text()
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)


@pytest.mark.parametrize("kind", ["extui", "extsi"])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_actual_alias_updates_captured_result_preserves_escaped_widening(tmp_path, lean_tools, kind, index_bits):
    stage, before, after = canonicalize(tmp_path, lean_tools, captured_source(kind))
    assert "linalg.generic" in after.read_text() and "arith.trunci" not in after.read_text()
    assert after.read_text().count(f"arith.{kind}") == 1
    certify(stage, before, lean_tools, tmp_path, index_bits=index_bits)


@pytest.mark.parametrize("kind", ["extui", "extsi"])
@pytest.mark.parametrize("escape", [False, True])
def test_actual_long_roundtrip_retains_only_escaping_widenings(tmp_path, lean_tools, kind, escape):
    stage, before, after = canonicalize(tmp_path, lean_tools, long_source(kind, escape=escape))
    assert "arith.trunci" not in after.read_text()
    assert after.read_text().count(f"arith.{kind}") == 2 * int(escape)
    certify(stage, before, lean_tools, tmp_path)


MUTATE = """
import sys
from pathlib import Path
from torch_mlir import ir
with ir.Context():
    path, fault = Path(sys.argv[1]), sys.argv[2]
    module = ir.Module.parse(path.read_text())
    block = module.body.operations[0].regions[0].blocks[0]
    returned = list(block.operations)[-1]
    if fault == 'wrong_alias':
        returned.operation.operands[0] = block.arguments[1]
    elif fault == 'escaping_input':
        cast = next(o for o in block.operations if o.name in ['arith.extsi','arith.extui'])
        cast.operation.operands[0] = block.arguments[1]
    elif fault == 'captured_alias':
        generic = next(o for o in block.operations if o.name == 'linalg.generic')
        addition = next(o for o in generic.regions[0].blocks[0].operations if o.name == 'arith.addi')
        addition.operation.operands[1] = block.arguments[1]
    elif fault in ['unknown', 'nonNeg']:
        cast = next(o for o in block.operations if o.name in ['arith.extsi','arith.extui'])
        cast.operation.attributes['unknown.numeric_meaning' if fault == 'unknown' else 'nonNeg'] = ir.UnitAttr.get()
    elif fault != 'signedness':
        raise ValueError(fault)
    text = str(module)
    if fault == 'signedness':
        text = text.replace('arith.extui', 'arith.extsi', 1)
    module = ir.Module.parse(text)
    module.operation.verify()
    path.write_text(str(module))
"""


def mutate(path, tools, fault):
    original = path.read_bytes()
    result = subprocess.run(
        [str(tools.m2m_python), "-c", MUTATE, str(path), fault], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert path.read_bytes() != original
    _verify_mlir(path, tools)


@pytest.mark.parametrize("fault", ["wrong_alias", "escaping_input", "signedness", "unknown", "nonNeg"])
def test_valid_wrong_roundtrip_targets_refuse(tmp_path, lean_tools, fault):
    stage, before, after = canonicalize(tmp_path, lean_tools, source("extui", escape=True))
    certify(stage, before, lean_tools, tmp_path)
    mutate(after, lean_tools, fault)
    result = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "abstracted", result.get("reason")


def test_valid_wrong_captured_alias_refuses(tmp_path, lean_tools):
    stage, before, after = canonicalize(tmp_path, lean_tools, captured_source())
    certify(stage, before, lean_tools, tmp_path)
    mutate(after, lean_tools, "captured_alias")
    result = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "abstracted", result.get("reason")


def test_source_unknown_cast_attributes_do_not_gain_strict_defined_chain(tmp_path, lean_tools):
    stage, before, _ = canonicalize(tmp_path, lean_tools, source("extui", escape=True))
    mutate(before, lean_tools, "unknown")
    for check, directory in [(check_support, "support"), (check_definedness, "defined")]:
        result = check(before, "forward", tmp_path / directory, m2m_python=lean_tools.m2m_python)
        assert result["verdict"] == "abstracted", result.get("reason")
