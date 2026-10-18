"""Qualify actual upstream expand/collapse composition and byte-bound strict chains."""

import subprocess

import pytest
from test_lean_reshape_compose import _canonicalize, _type, _verify_mlir

from merlin.verify.lean.check import check_definedness, check_stage, check_support, compose_chain


def _source(shape=(2, 3, 2, 2), element="i32", *, result="expand", escape=False):
    a, b, c, d = shape
    if result == "expand":
        shapes = [(a * b, c * d), (a, b, c, d), (a, b, c * d)]
        groups = ["[[0, 1], [2, 3]]", "[[0], [1], [2, 3]]"]
    else:
        shapes = [(a * b, c * d), (a, b, c * d), (a * b * c * d,)]
        groups = ["[[0, 1], [2]]", "[[0, 1, 2]]"]
    types = [_type(s, element) for s in shapes]
    output_shape = ", ".join(map(str, shapes[1]))
    returns = "%out, %a" + (", %mid" if escape else "")
    result_types = [types[2], types[0]] + ([types[1]] if escape else [])
    return f"""func.func @forward(%a: {types[0]}, %alternate: {types[0]}, %other_result: {types[2]})
        -> ({", ".join(result_types)}) {{
      %mid = tensor.expand_shape %a {groups[0]} output_shape [{output_shape}] : {types[0]} into {types[1]}
      %out = tensor.collapse_shape %mid {groups[1]} : {types[1]} into {types[2]}
      return {returns} : {", ".join(result_types)}
    }}"""


@pytest.mark.parametrize("result", ["collapse", "expand"])
@pytest.mark.parametrize(
    "shape,element", [((2, 3, 2, 2), "i13"), ((0, 3, 2, 2), "i8"), ((2, 3, 2, 2), "f32"), ((2, 3, 2, 2), "bf16")]
)
def test_actual_mixed_reshape(tmp_path, lean_tools, result, shape, element):
    stage, before, after = _canonicalize(tmp_path, lean_tools.m2m_python, _source(shape, element, result=result))
    assert before.read_text().count("tensor.expand_shape") == 1
    assert before.read_text().count("tensor.collapse_shape") == 1
    assert after.read_text().count("tensor.expand_shape") == int(result == "expand")
    assert after.read_text().count("tensor.collapse_shape") == int(result == "collapse")
    checked = check_stage(stage, m2m_python=lean_tools.m2m_python)
    assert checked["verdict"] == "verified", checked.get("reason")
    for checker, name in [(check_support, "support"), (check_definedness, "defined")]:
        checked = checker(before, "forward", tmp_path / name, m2m_python=lean_tools.m2m_python)
        assert checked["verdict"] == "verified", checked.get("reason")
    chain = compose_chain(
        [stage],
        tmp_path / "support/receipt.json",
        "forward",
        tmp_path / "chain.json",
        definedness=tmp_path / "defined/receipt.json",
        m2m_python=lean_tools.m2m_python,
    )
    assert chain["verdict"] == "verified", chain.get("reason")


@pytest.mark.parametrize("index_bits", [32, 64])
@pytest.mark.parametrize("result", ["collapse", "expand"])
def test_mixed_reshape_defined_chain_and_stale_output(tmp_path, lean_tools, index_bits, result):
    stage, before, after = _canonicalize(tmp_path, lean_tools.m2m_python, _source(result=result))
    assert check_stage(stage, m2m_python=lean_tools.m2m_python)["verdict"] == "verified"
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
    assert checked["theorem"] == "MerlinVerify.Check.chain_extended_defined_refines"
    assert checked["domain"].endswith(f"contexts with {index_bits}-bit index")
    after.write_text(after.read_text() + "\n// changed bytes after proof publication\n")
    assert replay()["verdict"] == "abstracted"


MUTATE = """
import sys
from pathlib import Path
from torch_mlir import ir
with ir.Context():
    m = ir.Module.parse(Path(sys.argv[1]).read_text())
    block = m.body.operations[0].regions[0].blocks[0]
    reshape = next(op for op in block.operations if op.operation.name.startswith('tensor.'))
    ret = list(block.operations)[-1]
    if sys.argv[2] == 'operand':
        reshape.operation.operands[0] = block.arguments[1]
    elif sys.argv[2] == 'result':
        ret.operation.operands[0] = block.arguments[2]
    elif sys.argv[2] == 'escaping_input':
        ret.operation.operands[1] = block.arguments[1]
    else:
        reshape.operation.attributes['unreviewed.meaning'] = ir.UnitAttr.get()
    m.operation.verify()
    Path(sys.argv[1]).write_text(str(m))
"""


@pytest.mark.parametrize("result", ["collapse", "expand"])
@pytest.mark.parametrize("fault", ["operand", "result", "escaping_input", "attribute"])
def test_mixed_reshape_rejects_verified_mlir_mutations(tmp_path, lean_tools, result, fault):
    stage, _, after = _canonicalize(tmp_path, lean_tools.m2m_python, _source(result=result))
    assert check_stage(stage, m2m_python=lean_tools.m2m_python)["verdict"] == "verified"
    original = after.read_bytes()
    changed = subprocess.run(
        [str(lean_tools.m2m_python), "-c", MUTATE, str(after), fault], capture_output=True, text=True, check=False
    )
    assert changed.returncode == 0, changed.stderr
    assert original != after.read_bytes()
    checked = check_stage(stage, m2m_python=lean_tools.m2m_python)
    assert checked["verdict"] == "abstracted", checked.get("reason")


def test_mixed_reshape_accepts_equivalent_unit_reassociation(tmp_path, lean_tools):
    stage, _, after = _canonicalize(tmp_path, lean_tools.m2m_python, _source((2, 1, 2, 2)))
    assert check_stage(stage, m2m_python=lean_tools.m2m_python)["verdict"] == "verified"
    original = after.read_text()
    changed = original.replace("[[0, 1], [2]]", "[[0], [1, 2]]")
    assert changed != original
    after.write_text(changed)
    _verify_mlir(after, lean_tools.m2m_python)
    # Both reassociations have the same checked runtime endpoints and row-major meaning.
    checked = check_stage(stage, m2m_python=lean_tools.m2m_python)
    assert checked["verdict"] == "verified", checked.get("reason")


@pytest.mark.parametrize("result", ["collapse", "expand"])
def test_mixed_escaping_intermediate_is_not_silently_dropped(tmp_path, lean_tools, result):
    stage, _, _ = _canonicalize(tmp_path, lean_tools.m2m_python, _source(result=result, escape=True))
    checked = check_stage(stage, m2m_python=lean_tools.m2m_python)
    assert checked["verdict"] == "abstracted", checked.get("reason")


def test_collapse_first_mixed_pair_still_requires_a_rule(tmp_path, lean_tools):
    source = """func.func @forward(%a: tensor<2x3x4xi32>) -> tensor<2x3x2x2xi32> {
      %b = tensor.collapse_shape %a [[0,1],[2]] : tensor<2x3x4xi32> into tensor<6x4xi32>
      %c = tensor.expand_shape %b [[0,1],[2,3]] output_shape [2,3,2,2] : tensor<6x4xi32> into tensor<2x3x2x2xi32>
      return %c : tensor<2x3x2x2xi32>
    }"""
    stage, _, after = _canonicalize(tmp_path, lean_tools.m2m_python, source)
    assert after.read_text().count("tensor.expand_shape") == 1
    assert "tensor.collapse_shape" not in after.read_text()
    checked = check_stage(stage, m2m_python=lean_tools.m2m_python)
    assert checked["verdict"] == "abstracted", checked.get("reason")


def test_mixed_rejects_changed_expanded_shape_metadata(tmp_path, lean_tools):
    stage, _, after = _canonicalize(tmp_path, lean_tools.m2m_python, _source())
    assert check_stage(stage, m2m_python=lean_tools.m2m_python)["verdict"] == "verified"
    original = after.read_text()
    changed = original.replace("output_shape [2, 3, 4]", "output_shape [3, 2, 4]")
    changed = changed.replace("tensor<2x3x4xi32>", "tensor<3x2x4xi32>")
    assert changed != original
    after.write_text(changed)
    _verify_mlir(after, lean_tools.m2m_python)
    checked = check_stage(stage, m2m_python=lean_tools.m2m_python)
    assert checked["verdict"] == "abstracted", checked.get("reason")


def test_mixed_rank_zero_transition_does_not_claim_coverage(tmp_path, lean_tools):
    source = """func.func @forward(%a: tensor<1xi32>) -> tensor<i32> {
      %b = tensor.expand_shape %a [[0,1]] output_shape [1,1] : tensor<1xi32> into tensor<1x1xi32>
      %c = tensor.collapse_shape %b [] : tensor<1x1xi32> into tensor<i32>
      return %c : tensor<i32>
    }"""
    stage, before, _ = _canonicalize(tmp_path, lean_tools.m2m_python, source)
    checked = check_stage(stage, m2m_python=lean_tools.m2m_python)
    assert checked["verdict"] == "abstracted", checked.get("reason")
    support = check_support(before, "forward", tmp_path / "support", m2m_python=lean_tools.m2m_python)
    assert support["verdict"] == "abstracted", support.get("reason")
    chain = compose_chain(
        [stage],
        tmp_path / "support/receipt.json",
        "forward",
        tmp_path / "chain.json",
        m2m_python=lean_tools.m2m_python,
    )
    assert chain["verdict"] == "abstracted", chain.get("reason")


def test_mixed_cancellation_to_bare_input_stays_unsupported(tmp_path, lean_tools):
    source = """func.func @forward(%a: tensor<6xi32>) -> tensor<6xi32> {
      %b = tensor.expand_shape %a [[0,1]] output_shape [2,3] : tensor<6xi32> into tensor<2x3xi32>
      %c = tensor.collapse_shape %b [[0,1]] : tensor<2x3xi32> into tensor<6xi32>
      return %c : tensor<6xi32>
    }"""
    stage, before, after = _canonicalize(tmp_path, lean_tools.m2m_python, source)
    assert "tensor." not in after.read_text()
    assert (
        check_support(before, "forward", tmp_path / "support", m2m_python=lean_tools.m2m_python)["verdict"]
        == "verified"
    )
    checked = check_stage(stage, m2m_python=lean_tools.m2m_python)
    assert checked["verdict"] == "abstracted", checked.get("reason")
