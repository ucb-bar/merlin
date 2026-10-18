"""Actual upstream rank-zero from_elements/extract folding and exact scalar aliases."""

import json
import subprocess

import pytest
from test_lean_reshape_compose import _canonicalize, _verify_mlir

from merlin.verify.lean.check import check_definedness, check_support, check_typed_stage, compose_chain


def _source(element="i32", *, escape=False, producer="from_elements"):
    tensor = f"tensor<{element}>"
    output_types = [element, element] + ([tensor] if escape else [])
    returned = ["%alias", "%x"] + (["%carrier"] if escape else [])
    return f"""func.func @forward(%x: {element}, %other: {element}) -> ({", ".join(output_types)}) {{
      %carrier = tensor.{producer} %x : {tensor}
      %alias = tensor.extract %carrier[] : {tensor}
      return {", ".join(returned)} : {", ".join(output_types)}
    }}"""


def _captured_source(element="i32", *, local=False):
    carrier = f"""%carrier = tensor.from_elements %x : tensor<{element}>
      %alias = tensor.extract %carrier[] : tensor<{element}>"""
    return f"""func.func @forward(%x: {element}, %other: {element}, %cond: i1,
        %payload: tensor<2x{element}>, %init: tensor<2x{element}>) -> tensor<2x{element}> {{
      {"" if local else carrier}
      %result = linalg.generic {{indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
          iterator_types = ["parallel"]}} ins(%payload : tensor<2x{element}>) outs(%init : tensor<2x{element}>) {{
        ^bb0(%p: {element}, %old: {element}):
          {carrier if local else ""}
          %selected = arith.select %cond, %alias, %p : {element}
          linalg.yield %selected : {element}
      }} -> tensor<2x{element}>
      return %result : tensor<2x{element}>
    }}"""


def _stage(tmp_path, compiler, source):
    stage, before, after = _canonicalize(tmp_path, compiler, source)
    (stage / "meta.json").write_text(json.dumps({"boundary": "scalar-carrier-extraction"}))
    return stage, before, after


@pytest.mark.parametrize("element", ["i1", "i13", "i32", "index", "bf16", "f32", "f64"])
@pytest.mark.parametrize("escape", [False, True])
def test_actual_scalar_carrier_fold(tmp_path, lean_tools, element, escape):
    stage, _, after = _stage(tmp_path, lean_tools.m2m_python, _source(element, escape=escape))
    assert "tensor.extract" not in after.read_text()
    assert after.read_text().count("tensor.from_elements") == int(escape)
    checked = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert checked["verdict"] == "verified", checked.get("reason")


@pytest.mark.parametrize("element", ["i13", "f32", "index"])
@pytest.mark.parametrize("index_bits", [32, 64])
@pytest.mark.parametrize("escape", [False, True])
def test_scalar_carrier_strict_chain(tmp_path, lean_tools, element, index_bits, escape):
    stage, before, after = _stage(tmp_path, lean_tools.m2m_python, _source(element, escape=escape))
    checked = check_typed_stage(stage, index_bits=index_bits, m2m_python=lean_tools.m2m_python)
    assert checked["verdict"] == "verified", checked.get("reason")
    for checker, name in [(check_support, "support"), (check_definedness, "defined")]:
        checked = checker(before, "forward", tmp_path / name, index_bits=index_bits, m2m_python=lean_tools.m2m_python)
        assert checked["verdict"] == "verified", checked.get("reason")

    def replay():
        return compose_chain(
            [stage],
            tmp_path / "support/receipt.json",
            "forward",
            tmp_path / "chain.json",
            definedness=tmp_path / "defined/receipt.json",
            m2m_python=lean_tools.m2m_python,
        )

    checked = replay()
    assert checked["verdict"] == "verified", checked.get("reason")
    assert checked["theorem"] == "MerlinVerify.Check.chain_typed_extended_defined_refines"
    after.write_text(after.read_text() + "\n// output bytes changed after receipt\n")
    assert replay()["verdict"] == "abstracted"


@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("element", ["i32", "f32"])
def test_scalar_carrier_in_structured_body(tmp_path, lean_tools, local, element):
    stage, before, after = _stage(tmp_path, lean_tools.m2m_python, _captured_source(element, local=local))
    assert "linalg.generic" in after.read_text()
    assert "tensor.extract" not in after.read_text()
    assert "tensor.from_elements" not in after.read_text()
    checked = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert checked["verdict"] == "verified", checked.get("reason")
    for checker, name in [(check_support, "support"), (check_definedness, "defined")]:
        checked = checker(before, "forward", tmp_path / name, m2m_python=lean_tools.m2m_python)
        assert checked["verdict"] == "verified", checked.get("reason")


MUTATE = """
import sys
from pathlib import Path
from torch_mlir import ir
with ir.Context():
    m=ir.Module.parse(Path(sys.argv[1]).read_text())
    block=m.body.operations[0].regions[0].blocks[0]
    if sys.argv[2]=='scalar':
        list(block.operations)[-1].operation.operands[0]=block.arguments[1]
    elif sys.argv[2]=='payload':
        carrier=next(op for op in block.operations if op.operation.name=='tensor.from_elements')
        carrier.operation.operands[0]=block.arguments[1]
    elif sys.argv[2]=='capture':
        generic=next(op for op in block.operations if op.operation.name=='linalg.generic')
        select=next(op for op in generic.regions[0].blocks[0].operations if op.operation.name=='arith.select')
        select.operation.operands[1]=block.arguments[1]
    else:
        carrier=next(op for op in block.operations if op.operation.name=='tensor.from_elements')
        carrier.operation.attributes['unreviewed.meaning']=ir.UnitAttr.get()
    m.operation.verify()
    Path(sys.argv[1]).write_text(str(m))
"""


@pytest.mark.parametrize("fault", ["scalar", "payload", "attribute", "capture"])
def test_scalar_carrier_valid_mlir_mutations(tmp_path, lean_tools, fault):
    source = _captured_source() if fault == "capture" else _source(escape=True)
    stage, _, after = _stage(tmp_path, lean_tools.m2m_python, source)
    assert check_typed_stage(stage, m2m_python=lean_tools.m2m_python)["verdict"] == "verified"
    changed = subprocess.run(
        [str(lean_tools.m2m_python), "-c", MUTATE, str(after), fault], capture_output=True, text=True, check=False
    )
    assert changed.returncode == 0, changed.stderr
    assert check_typed_stage(stage, m2m_python=lean_tools.m2m_python)["verdict"] == "abstracted"


def test_known_rankzero_splat_manual_alias_target(tmp_path, lean_tools):
    stage, _, after = _stage(tmp_path, lean_tools.m2m_python, _source("f32", producer="splat"))
    # This compiler version leaves the splat/extract pair unchanged. Its manually proposed alias
    # exercises the same exact semantic kernel without claiming an observed upstream fold.
    assert "tensor.extract" in after.read_text()
    after.write_text("func.func @forward(%x: f32, %other: f32) -> (f32, f32) { return %x, %x : f32, f32 }")
    _verify_mlir(after, lean_tools.m2m_python)
    checked = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert checked["verdict"] == "verified", checked.get("reason")


@pytest.mark.parametrize("shape", ["0x", "2x"])
def test_positive_rank_alias_requires_bounds_evidence(tmp_path, lean_tools, shape):
    source = f"""func.func @forward(%x: i32, %index: index) -> i32 {{
      %t = tensor.splat %x : tensor<{shape}i32>
      %v = tensor.extract %t[%index] : tensor<{shape}i32>
      return %v : i32
    }}"""
    stage, _, after = _stage(tmp_path, lean_tools.m2m_python, source)
    after.write_text("func.func @forward(%x: i32, %index: index) -> i32 { return %x : i32 }")
    _verify_mlir(after, lean_tools.m2m_python)
    assert check_typed_stage(stage, m2m_python=lean_tools.m2m_python)["verdict"] == "abstracted"


def _chain_source(length=8, element="f32", *, escape=False):
    operations = []
    value = "%x"
    for n in range(length):
        operations += [
            f"%c{n} = tensor.from_elements {value} : tensor<{element}>",
            f"%v{n} = tensor.extract %c{n}[] : tensor<{element}>",
        ]
        value = f"%v{n}"
    types = [element] + ([f"tensor<{element}>"] if escape else [])
    values = [value] + ([f"%c{length - 1}"] if escape else [])
    return f"""func.func @forward(%x: {element}, %other: {element}) -> ({", ".join(types)}) {{
      {" ".join(operations)}
      return {", ".join(values)} : {", ".join(types)}
    }}"""


@pytest.mark.parametrize("length,escape", [(8, False), (8, True), (64, False)])
def test_chained_scalar_carriers(tmp_path, lean_tools, length, escape):
    stage, _, after = _stage(tmp_path, lean_tools.m2m_python, _chain_source(length, escape=escape))
    assert "tensor.extract" not in after.read_text()
    assert after.read_text().count("tensor.from_elements") == int(escape)
    checked = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert checked["verdict"] == "verified", checked.get("reason")


def _mixed_source():
    return """func.func @forward(%x: i8, %values: tensor<2xi32>, %init: tensor<2xi32>) -> tensor<2xi32> {
      %first = arith.extui %x : i8 to i16
      %carrier = tensor.from_elements %first : tensor<i16>
      %alias = tensor.extract %carrier[] : tensor<i16>
      %wide = arith.extsi %alias : i16 to i32
      %result = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
          iterator_types = ["parallel"]} ins(%values : tensor<2xi32>) outs(%init : tensor<2xi32>) {
        ^bb0(%p: i32, %old: i32):
          %sum = arith.addi %p, %wide : i32
          %xor = arith.xori %sum, %p : i32
          linalg.yield %xor : i32
      } -> tensor<2xi32>
      return %result : tensor<2xi32>
    }"""
