"""Check actual upstream reassociation composition and refuse unsupported SSA rewrites."""

import json
import subprocess

import pytest

from merlin.verify.lean.check import check_definedness, check_stage, check_support, compose_chain


def _type(shape, element):
    return "tensor<" + "".join(f"{d}x" for d in shape) + element + ">"


def _source(shape=(2, 3, 4), element="i32", *, expand=False, escape=False):
    a, b, c = shape
    shapes = [(a, b, c), (a * b, c), (a * b * c,)]
    groups = ["[[0, 1], [2]]", "[[0, 1]]"]
    if expand:
        shapes.reverse()
        groups.reverse()
    types = [_type(s, element) for s in shapes]
    name = "expand_shape" if expand else "collapse_shape"
    steps = []
    for i in range(2):
        output_shape = "output_shape [" + ", ".join(map(str, shapes[i + 1])) + "] " if expand else ""
        steps.append(
            f"%r{i} = tensor.{name} {'%a' if i == 0 else '%r0'} {groups[i]} "
            f"{output_shape}: {types[i]} into {types[i + 1]}"
        )
    results = ["%r1", "%a"] + (["%r0"] if escape else [])
    result_types = [types[2], types[0]] + ([types[1]] if escape else [])
    return f"""func.func @forward(%a: {types[0]}, %alternate: {types[0]}) -> ({", ".join(result_types)}) {{
      {chr(10).join(steps)}
      return {", ".join(results)} : {", ".join(result_types)}
    }}"""


def _canonicalize(tmp_path, compiler, source):
    stage = tmp_path / "stage"
    stage.mkdir()
    before, after = stage / "before.mlir", stage / "after.mlir"
    before.write_text(source)
    program = """
import sys
from pathlib import Path
from torch_mlir import ir, passmanager
with ir.Context():
    m = ir.Module.parse(Path(sys.argv[1]).read_text())
    m.operation.verify()
    passmanager.PassManager.parse('builtin.module(canonicalize)').run(m.operation)
    m.operation.verify()
    Path(sys.argv[2]).write_text(str(m))
"""
    result = subprocess.run(
        [str(compiler), "-c", program, str(before), str(after)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    (stage / "meta.json").write_text(json.dumps({"boundary": "reshape-composition"}))
    return stage, before, after


@pytest.mark.parametrize("expand", [False, True])
@pytest.mark.parametrize(
    "shape,element", [((2, 3, 4), "i13"), ((0, 3, 4), "i8"), ((2, 3, 4), "f32"), ((2, 3, 4), "bf16")]
)
def test_actual_reshape_composition(tmp_path, lean_tools, expand, shape, element):
    stage, before, after = _canonicalize(tmp_path, lean_tools.m2m_python, _source(shape, element, expand=expand))
    name = "tensor.expand_shape" if expand else "tensor.collapse_shape"
    assert before.read_text().count(name) == 2
    assert after.read_text().count(name) == 1
    receipt = check_stage(stage, m2m_python=lean_tools.m2m_python)
    assert receipt["verdict"] == "verified", receipt.get("reason")
    for checker, work in [(check_support, "support"), (check_definedness, "defined")]:
        result = checker(before, "forward", tmp_path / work, m2m_python=lean_tools.m2m_python)
        assert result["verdict"] == "verified", result.get("reason")
    chain = compose_chain(
        [stage],
        tmp_path / "support/receipt.json",
        "forward",
        tmp_path / "chain.json",
        definedness=tmp_path / "defined/receipt.json",
        m2m_python=lean_tools.m2m_python,
    )
    assert chain["verdict"] == "verified", chain.get("reason")
    after.write_text(after.read_text() + "\n// mutate bytes after verification\n")
    stale = compose_chain(
        [stage],
        tmp_path / "support/receipt.json",
        "forward",
        tmp_path / "stale.json",
        definedness=tmp_path / "defined/receipt.json",
        m2m_python=lean_tools.m2m_python,
    )
    assert stale["verdict"] != "verified"


@pytest.mark.parametrize("fault", ["operand", "escaping_input", "attribute"])
@pytest.mark.parametrize("expand", [False, True])
def test_reshape_rejects_verified_mlir_mutations(tmp_path, lean_tools, fault, expand):
    stage, _, after = _canonicalize(tmp_path, lean_tools.m2m_python, _source(expand=expand))
    assert check_stage(stage, m2m_python=lean_tools.m2m_python)["verdict"] == "verified"
    original = after.read_bytes()
    program = """
import sys
from pathlib import Path
from torch_mlir import ir
with ir.Context():
    m = ir.Module.parse(Path(sys.argv[1]).read_text())
    block = m.body.operations[0].regions[0].blocks[0]
    reshape = next(op for op in block.operations if op.operation.name.startswith('tensor.'))
    terminator = list(block.operations)[-1]
    if sys.argv[2] == 'operand':
        reshape.operation.operands[0] = block.arguments[1]
    elif sys.argv[2] == 'escaping_input':
        terminator.operation.operands[1] = block.arguments[1]
    else:
        reshape.operation.attributes['unreviewed.meaning'] = ir.UnitAttr.get()
    m.operation.verify()
    Path(sys.argv[1]).write_text(str(m))
"""
    result = subprocess.run(
        [str(lean_tools.m2m_python), "-c", program, str(after), fault], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert after.read_bytes() != original
    result = check_stage(stage, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "abstracted", result.get("reason")


@pytest.mark.parametrize("expand", [False, True])
def test_escaping_intermediate_requires_a_separate_rule(tmp_path, lean_tools, expand):
    stage, _, _ = _canonicalize(tmp_path, lean_tools.m2m_python, _source(expand=expand, escape=True))
    result = check_stage(stage, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "abstracted", result.get("reason")


def test_bare_input_cancellation_is_not_claimed(tmp_path, lean_tools):
    source = """func.func @forward(%a: tensor<6xi32>) -> tensor<6xi32> {
      %e = tensor.expand_shape %a [[0, 1]] output_shape [2, 3] : tensor<6xi32> into tensor<2x3xi32>
      %c = tensor.collapse_shape %e [[0, 1]] : tensor<2x3xi32> into tensor<6xi32>
      return %c : tensor<6xi32>
    }"""
    stage, _, after = _canonicalize(tmp_path, lean_tools.m2m_python, source)
    assert "tensor." not in after.read_text()
    result = check_stage(stage, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "abstracted", result.get("reason")


def test_legal_alternative_reassociation_is_explicitly_unproved(tmp_path, lean_tools):
    source = """func.func @forward(%a: tensor<2x1x2x2xi32>) -> tensor<2x4xi32> {
      %b = tensor.collapse_shape %a [[0,1],[2],[3]] : tensor<2x1x2x2xi32> into tensor<2x2x2xi32>
      %c = tensor.collapse_shape %b [[0],[1,2]] : tensor<2x2x2xi32> into tensor<2x4xi32>
      return %c : tensor<2x4xi32>
    }"""
    stage, _, after = _canonicalize(tmp_path, lean_tools.m2m_python, source)
    assert check_stage(stage, m2m_python=lean_tools.m2m_python)["verdict"] == "verified"
    original = after.read_text()
    changed = original.replace("[[0, 1], [2, 3]]", "[[0], [1, 2, 3]]")
    assert changed != original
    after.write_text(changed)
    _verify_mlir(after, lean_tools.m2m_python)
    # Unit dimensions permit an equivalent grouping, but this rule checks exact composition.
    result = check_stage(stage, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "abstracted", result.get("reason")


def _verify_mlir(path, compiler):
    result = subprocess.run(
        [
            str(compiler),
            "-c",
            """
import sys
from pathlib import Path
from torch_mlir import ir
with ir.Context():
    m = ir.Module.parse(Path(sys.argv[1]).read_text())
    m.operation.verify()
""",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_reshape_rejects_changed_expanded_shape_metadata(tmp_path, lean_tools):
    stage, _, after = _canonicalize(tmp_path, lean_tools.m2m_python, _source(expand=True))
    assert check_stage(stage, m2m_python=lean_tools.m2m_python)["verdict"] == "verified"
    original = after.read_text()
    changed = original.replace("output_shape [2, 3, 4]", "output_shape [3, 2, 4]")
    changed = changed.replace("tensor<2x3x4xi32>", "tensor<3x2x4xi32>")
    assert changed != original
    after.write_text(changed)
    _verify_mlir(after, lean_tools.m2m_python)
    result = check_stage(stage, m2m_python=lean_tools.m2m_python)
    assert result["verdict"] == "abstracted", result.get("reason")


def _rank_zero_source(expand):
    if expand:
        return """func.func @forward(%a: tensor<i32>) -> tensor<1x1xi32> {
          %b = tensor.expand_shape %a [] output_shape [1] : tensor<i32> into tensor<1xi32>
          %c = tensor.expand_shape %b [[0,1]] output_shape [1,1] : tensor<1xi32> into tensor<1x1xi32>
          return %c : tensor<1x1xi32>
        }"""
    return """func.func @forward(%a: tensor<1x1xi32>) -> tensor<i32> {
      %b = tensor.collapse_shape %a [[0,1]] : tensor<1x1xi32> into tensor<1xi32>
      %c = tensor.collapse_shape %b [] : tensor<1xi32> into tensor<i32>
      return %c : tensor<i32>
    }"""


@pytest.mark.parametrize("expand", [False, True])
def test_rank_zero_unit_transition_is_explicitly_unsupported(tmp_path, lean_tools, expand):
    # Upstream supports empty reassociations for all-unit rank-zero transitions. The current
    # semantics' group-coverage predicate rejects them; do not turn conditional proof into coverage.
    stage, before, _ = _canonicalize(tmp_path, lean_tools.m2m_python, _rank_zero_source(expand))
    support = check_support(before, "forward", tmp_path / "support", m2m_python=lean_tools.m2m_python)
    assert support["verdict"] == "abstracted", support.get("reason")
    result = check_stage(stage, m2m_python=lean_tools.m2m_python)
    # Composition can prove conditional refinement even when the source is stuck for this
    # declared domain. Only the separate source-support check can establish actual coverage.
    assert result["verdict"] == ("verified" if expand else "abstracted"), result.get("reason")
    chain = compose_chain(
        [stage],
        tmp_path / "support/receipt.json",
        "forward",
        tmp_path / "chain.json",
        m2m_python=lean_tools.m2m_python,
    )
    assert chain["verdict"] == "abstracted", chain.get("reason")
