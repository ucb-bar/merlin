"""Check comparison/selection semantics against MLIR and independent input-bit references."""

import json
import random
from types import SimpleNamespace

import pytest
from test_lean_control_definedness import PREDICATES, comparison_select_source
from test_lean_integer_fold import canonicalize

from merlin.verify.lean import differential as diff
from merlin.verify.lean.check import check_definedness, check_stage


def _predicates(a, b, width):
    sign, modulus = 1 << (width - 1), 1 << width
    x, y = a if a < sign else a - modulus, b if b < sign else b - modulus
    return [a == b, a != b, x < y, x <= y, x > y, x >= y, a < b, a <= b, a > b, a >= b]


@pytest.mark.parametrize("width", [1, 4, 8, 13, 32, 64])
def test_integer_comparisons_and_scalar_selection(tmp_path, fidelity_tools, width):
    ty, body, returned, types = f"i{width}", [], [], []
    for i, predicate in enumerate(PREDICATES):
        body.extend([f"%c{i} = arith.cmpi {predicate}, %x, %y : {ty}", f"%r{i} = arith.select %c{i}, %a, %b : {ty}"])
        returned.extend([f"%c{i}", f"%r{i}"])
        types.extend(["i1", ty])
    source = (
        f"func.func @forward(%x: {ty}, %y: {ty}, %a: {ty}, %b: {ty}) -> ({', '.join(types)}) {{\n"
        + "\n".join(body)
        + f"\nreturn {', '.join(returned)} : {', '.join(types)}\n}}"
    )
    mask = (1 << width) - 1
    if width <= 4:
        pairs = [(x, y) for x in range(mask + 1) for y in range(mask + 1)]
    else:
        edge = [0, 1, mask, mask >> 1, 1 << (width - 1)]
        rng = random.Random(10913 + width)
        pairs = [(x, y) for x in edge for y in edge] + [
            (rng.getrandbits(width), rng.getrandbits(width)) for _ in range(32)
        ]
    samples, expected = [], []
    for x, y in pairs:
        a, b = (x + 3) & mask, (y ^ mask) & mask
        samples.append([[x], [y], [a], [b]])
        expected.append(
            [
                value
                for condition in _predicates(x, y, width)
                for value in (str(int(condition)), str(a if condition else b))
            ]
        )
    diff.compare(source, samples, tmp_path, fidelity_tools, function="forward")
    assert json.loads((tmp_path / "native-results.json").read_text())["results"] == expected


@pytest.mark.parametrize("tensor_condition", [False, True])
@pytest.mark.parametrize("element,width", [("i13", 13), ("f32", 32), ("f64", 64)])
def test_tensor_selection_preserves_selected_bits(tmp_path, fidelity_tools, tensor_condition, element, width):
    ty = f"tensor<6x{element}>"
    condition_type = "tensor<6xi1>" if tensor_condition else "i1"
    signature = f"{condition_type}, {ty}" if tensor_condition else ty
    source = f"""func.func @forward(%c: {condition_type}, %a: {ty}, %b: {ty}) -> {ty} {{
      %r = arith.select %c, %a, %b : {signature}
      return %r : {ty}
    }}"""
    # Selection is a bit-preserving operation, including negative zero and NaN payloads.
    if element == "i13":
        lhs, rhs = [0, 1, 4095, 4096, 8190, 8191], [8191, 8190, 4096, 4095, 1, 0]
    else:
        exponent, fraction = (8, 23) if width == 32 else (11, 52)
        infinity = ((1 << exponent) - 1) << fraction
        lhs = [0, 1 << (width - 1), 1, infinity, infinity | 17, infinity | (1 << (fraction - 1)) | 31]
        rhs = list(reversed(lhs))
    conditions = [[0, 1, 0, 1, 1, 0], [1, 0, 1, 0, 0, 1]] if tensor_condition else [[0], [1]]
    samples = [[condition, lhs, rhs] for condition in conditions]
    expected = [
        [
            str(a if condition[i if tensor_condition else 0] else b)
            for i, (a, b) in enumerate(zip(lhs, rhs, strict=True))
        ]
        for condition in conditions
    ]
    diff.compare(source, samples, tmp_path, fidelity_tools, function="forward")
    assert json.loads((tmp_path / "native-results.json").read_text())["results"] == expected


def test_wrong_selected_branch_is_rejected_and_changes_native_output(tmp_path, fidelity_tools):
    tools = SimpleNamespace(m2m_python=fidelity_tools.compiler_python)
    stage, before, after = canonicalize(tmp_path, tools, comparison_select_source())
    receipt = check_stage(stage, m2m_python=tools.m2m_python)
    assert receipt["verdict"] == "verified", receipt.get("reason")
    receipt = check_definedness(before, "forward", tmp_path / "defined", m2m_python=tools.m2m_python)
    assert receipt["verdict"] == "verified", receipt.get("reason")
    mutate = """
import sys
from pathlib import Path
from torch_mlir import ir
with ir.Context():
    path = Path(sys.argv[1])
    module = ir.Module.parse(path.read_text())
    block = module.body.operations[0].regions[0].blocks[0]
    selected = next(op for op in block.operations if op.name == 'arith.select')
    lhs, rhs = selected.operands[1], selected.operands[2]
    selected.operands[1], selected.operands[2] = rhs, lhs
    module.operation.verify()
    path.write_text(str(module))
"""
    diff._run([str(tools.m2m_python), "-c", mutate, str(after)])
    assert check_stage(stage, m2m_python=tools.m2m_python)["verdict"] == "abstracted"
    samples = [[[8191], [1], [7], [13]], [[0], [8191], [17], [23]]]
    for label, program in [("source", before), ("wrong", after)]:
        diff.compare(program.read_text(), samples, tmp_path / label, fidelity_tools, function="forward")
    assert json.loads((tmp_path / "source/native-results.json").read_text())["results"] == [["1", "7"], ["0", "23"]]
    assert json.loads((tmp_path / "wrong/native-results.json").read_text())["results"] == [["1", "13"], ["0", "17"]]
