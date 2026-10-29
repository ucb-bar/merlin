"""Actual rank-one from_elements selection, source-defined chains and checked scope limits."""

import subprocess

import pytest
from test_lean_reshape_compose import _canonicalize, _verify_mlir

from merlin.verify.lean.check import check_definedness, check_support, check_typed_stage, compose_chain


def _source(element="f32", *, size=3, coordinate=1, escape=False, repeated=False, dynamic=False):
    inputs = ["%x0", "%x1", "%x0"] if repeated else [f"%x{i}" for i in range(size)]
    args = [f"%x{i}: {element}" for i in range(max(size, 3))] + (["%position: index"] if dynamic else [])
    tensor = f"tensor<{size}x{element}>"
    types = [element] + ([tensor] if escape else [])
    values = ["%value"] + (["%carrier"] if escape else [])
    return f"""func.func @forward({", ".join(args)}) -> ({", ".join(types)}) {{
      %carrier = tensor.from_elements {", ".join(inputs)} : {tensor}
      {"" if dynamic else f"%position = arith.constant {coordinate} : index"}
      %value = tensor.extract %carrier[%position] : {tensor}
      return {", ".join(values)} : {", ".join(types)}
    }}"""


def _captured_source(element="f32", *, local=False, captured_tensor=False):
    carrier = f"""%carrier = tensor.from_elements %x0, {"%p" if local else "%x1"} : tensor<2x{element}>
      %position = arith.constant 1 : index
      %value = tensor.extract %carrier[%position] : tensor<2x{element}>"""
    if captured_tensor:
        outside = f"%carrier = tensor.from_elements %x0, %x1 : tensor<2x{element}>"
        inside = f"""%position = arith.constant 1 : index
          %value = tensor.extract %carrier[%position] : tensor<2x{element}>"""
    else:
        outside, inside = ("", carrier) if local else (carrier, "")
    return f"""func.func @forward(%x0: {element}, %x1: {element}, %condition: i1,
        %payload: tensor<2x{element}>, %init: tensor<2x{element}>) -> tensor<2x{element}> {{
      {outside}
      %result = linalg.generic {{indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
          iterator_types = ["parallel"]}} ins(%payload : tensor<2x{element}>) outs(%init : tensor<2x{element}>) {{
        ^bb0(%p: {element}, %old: {element}):
          {inside}
          %selected = arith.select %condition, %value, {"%x1" if local else "%p"} : {element}
          linalg.yield %selected : {element}
      }} -> tensor<2x{element}>
      return %result : tensor<2x{element}>
    }}"""


def _stage(tmp_path, compiler, source):
    stage, before, after = _canonicalize(tmp_path, compiler, source)
    (stage / "meta.json").write_text('{"boundary": "from-elements-selection"}')
    return stage, before, after


@pytest.mark.parametrize("element", ["i1", "i13", "i32", "index", "bf16", "f32", "f64"])
@pytest.mark.parametrize("escape", [False, True])
def test_actual_from_elements_selection(tmp_path, lean_tools, element, escape):
    stage, _, after = _stage(tmp_path, lean_tools.m2m_python, _source(element, escape=escape))
    assert "tensor.extract" not in after.read_text()
    assert after.read_text().count("tensor.from_elements") == int(escape)
    checked = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert checked["verdict"] == "verified", checked.get("reason")


@pytest.mark.parametrize("size,coordinate,repeated", [(1, 0, False), (4, 0, False), (4, 3, False), (3, 2, True)])
@pytest.mark.parametrize("escape", [False, True])
def test_constructor_size_and_selected_position(tmp_path, lean_tools, size, coordinate, repeated, escape):
    stage, _, after = _stage(
        tmp_path,
        lean_tools.m2m_python,
        _source("i32", size=size, coordinate=coordinate, repeated=repeated, escape=escape),
    )
    assert "tensor.extract" not in after.read_text()
    checked = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert checked["verdict"] == "verified", checked.get("reason")


@pytest.mark.parametrize("element", ["i13", "f32", "index"])
@pytest.mark.parametrize("index_bits", [32, 64])
@pytest.mark.parametrize("escape", [False, True])
def test_selection_strict_source_defined_chain(tmp_path, lean_tools, element, index_bits, escape):
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
    after.write_text(after.read_text() + "\n// changed boundary bytes\n")
    assert replay()["verdict"] == "abstracted"


@pytest.mark.parametrize("element", ["i32", "f32"])
@pytest.mark.parametrize("local", [False, True])
def test_selected_scalar_captures_and_local_body_producers(tmp_path, lean_tools, element, local):
    stage, before, after = _stage(tmp_path, lean_tools.m2m_python, _captured_source(element, local=local))
    assert "linalg.generic" in after.read_text()
    assert "tensor.extract" not in after.read_text()
    checked = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert checked["verdict"] == "verified", checked.get("reason")
    for checker, name in [(check_support, "support"), (check_definedness, "defined")]:
        checked = checker(before, "forward", tmp_path / name, m2m_python=lean_tools.m2m_python)
        assert checked["verdict"] == "verified", checked.get("reason")


MUTATE = """
import sys
from pathlib import Path
from torch_mlir import ir
with ir.Context(), ir.Location.unknown():
    m=ir.Module.parse(Path(sys.argv[1]).read_text())
    block=m.body.operations[0].regions[0].blocks[0]
    if sys.argv[2]=='coordinate':
        op=next(op for op in block.operations if op.operation.name=='arith.constant')
        op.operation.attributes['value']=ir.IntegerAttr.get(ir.IndexType.get(), 2)
    elif sys.argv[2]=='payload':
        op=next(op for op in block.operations if op.operation.name=='tensor.from_elements')
        op.operation.operands[1]=block.arguments[2]
    elif sys.argv[2]=='capture':
        generic=next(op for op in block.operations if op.operation.name=='linalg.generic')
        select=next(op for op in generic.regions[0].blocks[0].operations if op.operation.name=='arith.select')
        select.operation.operands[1]=block.arguments[0]
    elif sys.argv[2]=='attribute':
        op=next(op for op in block.operations if op.operation.name=='tensor.from_elements')
        op.operation.attributes['unknown.meaning']=ir.UnitAttr.get()
    else:
        list(block.operations)[-1].operation.operands[0]=block.arguments[0]
    m.operation.verify()
    Path(sys.argv[1]).write_text(str(m))
"""


@pytest.mark.parametrize("fault", ["coordinate", "payload", "return", "capture", "attribute"])
def test_selection_valid_mlir_mutations(tmp_path, lean_tools, fault):
    source = _captured_source() if fault == "capture" else _source(escape=True)
    stage, before, after = _stage(tmp_path, lean_tools.m2m_python, source)
    assert check_typed_stage(stage, m2m_python=lean_tools.m2m_python)["verdict"] == "verified"
    mutated = before if fault in {"coordinate", "payload", "attribute"} else after
    changed = subprocess.run(
        [str(lean_tools.m2m_python), "-c", MUTATE, str(mutated), fault], capture_output=True, text=True, check=False
    )
    assert changed.returncode == 0, changed.stderr
    assert check_typed_stage(stage, m2m_python=lean_tools.m2m_python)["verdict"] == "abstracted"


@pytest.mark.parametrize("size,coordinate,dynamic", [(0, 0, False), (3, -1, False), (3, 3, False), (3, 1, True)])
def test_missing_or_invalid_bounds_refuse_alias(tmp_path, lean_tools, size, coordinate, dynamic):
    stage = tmp_path / "stage"
    stage.mkdir()
    before, after = stage / "before.mlir", stage / "after.mlir"
    before.write_text(_source("i32", size=size, coordinate=coordinate, dynamic=dynamic))
    args = "%x0: i32, %x1: i32, %x2: i32" + (", %position: index" if dynamic else "")
    after.write_text(f"func.func @forward({args}) -> i32 {{ return %x1 : i32 }}")
    (stage / "meta.json").write_text('{"boundary": "from-elements-bounds-refusal"}')
    _verify_mlir(before, lean_tools.m2m_python)
    _verify_mlir(after, lean_tools.m2m_python)
    assert check_typed_stage(stage, m2m_python=lean_tools.m2m_python)["verdict"] == "abstracted"
    assert (
        check_definedness(before, "forward", tmp_path / "defined", m2m_python=lean_tools.m2m_python)["verdict"]
        == "abstracted"
    )


def test_captured_tensor_provenance_is_not_assumed_in_body(tmp_path, lean_tools):
    stage, before, after = _stage(tmp_path, lean_tools.m2m_python, _captured_source(captured_tensor=True))
    assert "tensor.extract" not in after.read_text()
    assert check_typed_stage(stage, m2m_python=lean_tools.m2m_python)["verdict"] == "abstracted"
    for checker, name in [(check_support, "support"), (check_definedness, "defined")]:
        checked = checker(before, "forward", tmp_path / name, m2m_python=lean_tools.m2m_python)
        assert checked["verdict"] == "verified", checked.get("reason")


def test_empty_constructor_source_is_defined(tmp_path, lean_tools):
    source = tmp_path / "empty.mlir"
    source.write_text(
        "func.func @forward() -> tensor<0xi32> { %t = tensor.from_elements : tensor<0xi32> return %t : tensor<0xi32> }"
    )
    _verify_mlir(source, lean_tools.m2m_python)
    for checker, name in [(check_support, "support"), (check_definedness, "defined")]:
        checked = checker(source, "forward", tmp_path / name, m2m_python=lean_tools.m2m_python)
        assert checked["verdict"] == "verified", checked.get("reason")


def _nested_source():
    return """func.func @forward(%seed: i32, %condition: i1,
        %payload: tensor<2xi32>, %init: tensor<2xi32>) -> tensor<2xi32> {
      %result = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
          iterator_types = ["parallel"]} ins(%payload : tensor<2xi32>) outs(%init : tensor<2xi32>) {
        ^outer(%p: i32, %old: i32):
          %fresh = tensor.splat %seed : tensor<i32>
          %inner = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],
              iterator_types = ["reduction"]} ins(%payload : tensor<2xi32>) outs(%fresh : tensor<i32>) {
            ^body(%q: i32, %acc: i32):
              %carrier = tensor.from_elements %p, %q : tensor<2xi32>
              %position = arith.constant 1 : index
              %value = tensor.extract %carrier[%position] : tensor<2xi32>
              %choice = arith.select %condition, %value, %p : i32
              %sum = arith.addi %acc, %choice : i32
              linalg.yield %sum : i32
          } -> tensor<i32>
          %scalar = tensor.extract %inner[] : tensor<i32>
          linalg.yield %scalar : i32
      } -> tensor<2xi32>
      return %result : tensor<2xi32>
    }"""


def test_nested_local_selection_has_defined_source(tmp_path, lean_tools):
    stage, before, after = _stage(tmp_path, lean_tools.m2m_python, _nested_source())
    assert after.read_text().count("linalg.generic") == 2
    assert "tensor.from_elements" not in after.read_text()
    assert check_typed_stage(stage, m2m_python=lean_tools.m2m_python)["verdict"] == "verified"
    for checker, name in [(check_support, "support"), (check_definedness, "defined")]:
        checked = checker(before, "forward", tmp_path / name, m2m_python=lean_tools.m2m_python)
        assert checked["verdict"] == "verified", checked.get("reason")


@pytest.mark.parametrize("index_bits", [32, 64])
def test_actual_selection_uses_decoded_index_width(tmp_path, lean_tools, index_bits):
    stage, before, after = _stage(tmp_path, lean_tools.m2m_python, _source("i32", coordinate=4294967297))
    assert "tensor.extract" not in after.read_text()
    expected = "verified" if index_bits == 32 else "abstracted"
    checked = check_typed_stage(stage, index_bits=index_bits, m2m_python=lean_tools.m2m_python)
    assert checked["verdict"] == expected, checked.get("reason")
    checked = check_definedness(
        before, "forward", tmp_path / "defined", index_bits=index_bits, m2m_python=lean_tools.m2m_python
    )
    assert checked["verdict"] == expected, checked.get("reason")
