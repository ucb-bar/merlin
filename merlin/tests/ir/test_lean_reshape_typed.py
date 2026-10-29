"""Typed, source-provenance mixed reshapes through actual upstream canonicalization."""

import subprocess

import pytest
from test_lean_reshape_compose import _canonicalize, _type, _verify_mlir
from test_lean_reshape_mixed import MUTATE

from merlin.verify.lean.check import check_definedness, check_support, check_typed_stage, compose_chain


def _source(shape=(2, 3, 4), element="i32", *, result="expand", escape=False):
    if result == "expand":
        a, b, c = shape
        shapes = [shape, (a * b, c), (a, b, c // 2, 2)]
        source_groups, output_groups = "[[0, 1], [2]]", "[[0, 1], [2, 3]]"
    else:
        a, b, c, d = shape
        shapes = [shape, (a * b * c * d,), (a * b, c * d)]
        source_groups, output_groups = "[[0, 1, 2, 3]]", "[[0, 1]]"
    types = [_type(s, element) for s in shapes]
    result_types = [types[2], types[0]] + ([types[1]] if escape else [])
    returns = "%out, %a" + (", %mid" if escape else "")
    return f"""func.func @forward(%a: {types[0]}, %alternate: {types[0]}, %other_result: {types[2]})
        -> ({", ".join(result_types)}) {{
      %mid = tensor.collapse_shape %a {source_groups} : {types[0]} into {types[1]}
      %out = tensor.expand_shape %mid {output_groups} output_shape [{", ".join(map(str, shapes[2]))}]
        : {types[1]} into {types[2]}
      return {returns} : {", ".join(result_types)}
    }}"""


CASES = [
    ((2, 3, 4), "i13", "expand"),
    ((0, 3, 4), "i8", "expand"),
    ((2, 1, 4), "f32", "expand"),
    ((2, 3, 4), "bf16", "expand"),
    ((2, 3, 2, 2), "i32", "collapse"),
]


@pytest.mark.parametrize("shape,element,result", CASES)
@pytest.mark.parametrize("escape", [False, True])
def test_actual_typed_mixed_reshape(tmp_path, lean_tools, shape, element, result, escape):
    stage, before, after = _canonicalize(
        tmp_path, lean_tools.m2m_python, _source(shape, element, result=result, escape=escape)
    )
    assert before.read_text().count("tensor.collapse_shape") == 1
    assert before.read_text().count("tensor.expand_shape") == 1
    assert after.read_text().count("tensor.expand_shape") == int(result == "expand")
    assert after.read_text().count("tensor.collapse_shape") == int(result == "collapse") + int(escape)
    checked = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert checked["verdict"] == "verified", checked.get("reason")


@pytest.mark.parametrize("index_bits", [32, 64])
@pytest.mark.parametrize("result,shape", [("expand", (2, 3, 4)), ("collapse", (2, 3, 2, 2))])
def test_typed_mixed_strict_chain(tmp_path, lean_tools, index_bits, result, shape):
    stage, before, after = _canonicalize(tmp_path, lean_tools.m2m_python, _source(shape, result=result, escape=True))
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
    assert checked["index_bits"] == index_bits
    after.write_text(after.read_text() + "\n// stale bytes\n")
    assert replay()["verdict"] == "abstracted"


@pytest.mark.parametrize("result,shape", [("expand", (2, 3, 4)), ("collapse", (2, 3, 2, 2))])
@pytest.mark.parametrize("fault", ["operand", "result", "escaping_input", "attribute"])
def test_typed_mixed_rejects_valid_mlir_mutations(tmp_path, lean_tools, result, shape, fault):
    stage, _, after = _canonicalize(tmp_path, lean_tools.m2m_python, _source(shape, result=result))
    assert check_typed_stage(stage, m2m_python=lean_tools.m2m_python)["verdict"] == "verified"
    old = after.read_bytes()
    changed = subprocess.run(
        [str(lean_tools.m2m_python), "-c", MUTATE, str(after), fault], capture_output=True, text=True, check=False
    )
    assert changed.returncode == 0, changed.stderr
    assert old != after.read_bytes()
    assert check_typed_stage(stage, m2m_python=lean_tools.m2m_python)["verdict"] == "abstracted"


def test_typed_mixed_rejects_changed_shape_metadata(tmp_path, lean_tools):
    stage, _, after = _canonicalize(tmp_path, lean_tools.m2m_python, _source())
    assert check_typed_stage(stage, m2m_python=lean_tools.m2m_python)["verdict"] == "verified"
    old = after.read_text()
    new = old.replace("output_shape [2, 3, 2, 2]", "output_shape [2, 3, 1, 4]")
    new = new.replace("tensor<2x3x2x2xi32>", "tensor<2x3x1x4xi32>")
    assert old != new
    after.write_text(new)
    _verify_mlir(after, lean_tools.m2m_python)
    assert check_typed_stage(stage, m2m_python=lean_tools.m2m_python)["verdict"] == "abstracted"


def test_typed_mixed_accepts_equivalent_unit_reassociation(tmp_path, lean_tools):
    source = """func.func @forward(%a: tensor<2x2x4xi32>) -> tensor<2x1x2x4xi32> {
      %mid = tensor.collapse_shape %a [[0,1],[2]] : tensor<2x2x4xi32> into tensor<4x4xi32>
      %out = tensor.expand_shape %mid [[0,1,2],[3]] output_shape [2,1,2,4]
        : tensor<4x4xi32> into tensor<2x1x2x4xi32>
      return %out : tensor<2x1x2x4xi32>
    }"""
    stage, _, after = _canonicalize(tmp_path, lean_tools.m2m_python, source)
    original = after.read_text()
    changed = original.replace("[[0], [1, 2], [3]]", "[[0, 1], [2], [3]]")
    if changed == original:
        changed = original.replace("[[0, 1], [2], [3]]", "[[0], [1, 2], [3]]")
    assert original != changed, original
    after.write_text(changed)
    _verify_mlir(after, lean_tools.m2m_python)
    checked = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert checked["verdict"] == "verified", checked.get("reason")


def test_typed_bare_input_cancellation_remains_unqualified(tmp_path, lean_tools):
    source = """func.func @forward(%a: tensor<2x3xi32>) -> tensor<2x3xi32> {
      %b = tensor.collapse_shape %a [[0,1]] : tensor<2x3xi32> into tensor<6xi32>
      %c = tensor.expand_shape %b [[0,1]] output_shape [2,3] : tensor<6xi32> into tensor<2x3xi32>
      return %c : tensor<2x3xi32>
    }"""
    stage, before, after = _canonicalize(tmp_path, lean_tools.m2m_python, source)
    assert "tensor." not in after.read_text()
    assert (
        check_support(before, "forward", tmp_path / "support", m2m_python=lean_tools.m2m_python)["verdict"]
        == "verified"
    )
    assert check_typed_stage(stage, m2m_python=lean_tools.m2m_python)["verdict"] == "abstracted"


def _capture_source():
    return """func.func @forward(%a: tensor<2x3x4xi32>, %alternate: tensor<2x3x4xi32>,
        %other_result: tensor<2x3x2x2xi32>, %init: tensor<i32>, %other_mid: tensor<6x4xi32>)
        -> (tensor<2x3x2x2xi32>, tensor<2x3x4xi32>, tensor<i32>) {
      %c0 = arith.constant 0 : index
      %mid = tensor.collapse_shape %a [[0,1],[2]] : tensor<2x3x4xi32> into tensor<6x4xi32>
      %out = tensor.expand_shape %mid [[0,1],[2,3]] output_shape [2,3,2,2]
        : tensor<6x4xi32> into tensor<2x3x2x2xi32>
      %captured = linalg.generic {indexing_maps = [affine_map<() -> ()>], iterator_types = []}
        outs(%init : tensor<i32>) {
        ^bb0(%old: i32):
          %read = tensor.extract %mid[%c0, %c0] : tensor<6x4xi32>
          linalg.yield %read : i32
      } -> tensor<i32>
      return %out, %a, %captured : tensor<2x3x2x2xi32>, tensor<2x3x4xi32>, tensor<i32>
    }"""


CAPTURE_MUTATE = """
import sys
from pathlib import Path
from torch_mlir import ir
with ir.Context():
    m = ir.Module.parse(Path(sys.argv[1]).read_text())
    block = m.body.operations[0].regions[0].blocks[0]
    generic = next(op for op in block.operations if op.operation.name == 'linalg.generic')
    extract = next(op for op in generic.regions[0].blocks[0].operations if op.operation.name == 'tensor.extract')
    extract.operation.operands[0] = block.arguments[4]
    m.operation.verify()
    Path(sys.argv[1]).write_text(str(m))
"""


def test_typed_mixed_preserves_nested_capture(tmp_path, lean_tools):
    stage, before, after = _canonicalize(tmp_path, lean_tools.m2m_python, _capture_source())
    assert after.read_text().count("tensor.collapse_shape") == 1
    assert after.read_text().count("tensor.expand_shape") == 1
    checked = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert checked["verdict"] == "verified", checked.get("reason")
    assert (
        check_support(before, "forward", tmp_path / "support", m2m_python=lean_tools.m2m_python)["verdict"]
        == "verified"
    )
    # Actual constant indices are proved in bounds for the captured rank-two tensor.
    assert (
        check_definedness(before, "forward", tmp_path / "defined", m2m_python=lean_tools.m2m_python)["verdict"]
        == "verified"
    )
    chain = compose_chain(
        [stage],
        tmp_path / "support/receipt.json",
        "forward",
        tmp_path / "chain.json",
        definedness=tmp_path / "defined/receipt.json",
        m2m_python=lean_tools.m2m_python,
    )
    assert chain["verdict"] == "verified", chain.get("reason")
    changed = subprocess.run(
        [str(lean_tools.m2m_python), "-c", CAPTURE_MUTATE, str(after)], capture_output=True, text=True, check=False
    )
    assert changed.returncode == 0, changed.stderr
    assert check_typed_stage(stage, m2m_python=lean_tools.m2m_python)["verdict"] == "abstracted"


def test_empty_collapse_first_pair_unchanged_upstream(tmp_path, lean_tools):
    stage, before, after = _canonicalize(tmp_path, lean_tools.m2m_python, _source((0, 3, 2, 2), result="collapse"))
    assert after.read_text().count("tensor.collapse_shape") == 1
    assert after.read_text().count("tensor.expand_shape") == 1
    checked = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert checked["verdict"] == "verified", checked.get("reason")
    assert (
        check_support(before, "forward", tmp_path / "support", m2m_python=lean_tools.m2m_python)["verdict"]
        == "verified"
    )


def test_typed_rank_zero_transition_stays_unsupported(tmp_path, lean_tools):
    source = """func.func @forward(%a: tensor<1x1xi32>) -> tensor<1x1x1xi32> {
      %b = tensor.collapse_shape %a [] : tensor<1x1xi32> into tensor<i32>
      %c = tensor.expand_shape %b [] output_shape [1,1,1] : tensor<i32> into tensor<1x1x1xi32>
      return %c : tensor<1x1x1xi32>
    }"""
    stage, before, _ = _canonicalize(tmp_path, lean_tools.m2m_python, source)
    assert check_typed_stage(stage, m2m_python=lean_tools.m2m_python)["verdict"] == "abstracted"
    assert (
        check_support(before, "forward", tmp_path / "support", m2m_python=lean_tools.m2m_python)["verdict"]
        == "abstracted"
    )


def test_typed_provenance_survives_unrelated_source_steps(tmp_path, lean_tools):
    source = """func.func @forward(%a: tensor<2x3x4xi32>, %control: i32)
        -> (tensor<2x3x2x2xi32>, i32) {
      %mid = tensor.collapse_shape %a [[0,1],[2]] : tensor<2x3x4xi32> into tensor<6x4xi32>
      %computed = arith.muli %control, %control : i32
      %out = tensor.expand_shape %mid [[0,1],[2,3]] output_shape [2,3,2,2]
        : tensor<6x4xi32> into tensor<2x3x2x2xi32>
      return %out, %computed : tensor<2x3x2x2xi32>, i32
    }"""
    stage, _, after = _canonicalize(tmp_path, lean_tools.m2m_python, source)
    assert "tensor.collapse_shape" not in after.read_text()
    checked = check_typed_stage(stage, m2m_python=lean_tools.m2m_python)
    assert checked["verdict"] == "verified", checked.get("reason")
