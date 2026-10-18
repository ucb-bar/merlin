"""Native actual hoists agree with independent modular cast/arith references; mutants differ."""

import json

import pytest
from test_lean_motion_independent import CASTS, _independent, _source
from test_lean_reshape_compose import _canonicalize, _verify_mlir

from merlin.verify.lean import differential as diff


def _cast(kind, value, source_width, target_width):
    value &= (1 << source_width) - 1
    if kind == "extsi" and value >= (1 << (source_width - 1)):
        value -= 1 << source_width
    return value & ((1 << target_width) - 1)


@pytest.mark.parametrize("index_bits", [32, 64])
@pytest.mark.parametrize("kind,source_width,target_width", CASTS)
def test_actual_hoisted_cast_matches_independent_bits(
    tmp_path, fidelity_tools, index_bits, kind, source_width, target_width
):
    _, before, after = _canonicalize(
        tmp_path, fidelity_tools.compiler_python, _source(kind, source_width, target_width)
    )
    values = [0, 1, (1 << (source_width - 1)), (1 << source_width) - 1]
    samples = [[[value], [7]] for value in values]
    expected = [[str(_cast(kind, value, source_width, target_width)), "22", "7"] for value in values]
    for label, path in [("source", before), ("target", after)]:
        work = tmp_path / label
        diff.compare(path.read_text(), samples, work, fidelity_tools, index_bits=index_bits)
        assert json.loads((work / "native-results.json").read_text())["results"] == expected


@pytest.mark.parametrize("operation", ["arith.muli", "arith.subi", "arith.andi", "arith.ori", "arith.xori"])
def test_two_input_expressions_match_independent_reference(tmp_path, fidelity_tools, operation):
    values = [(2, 3), (0, 7), (2147483648, 4294967295)]
    samples = [[[a], [b]] for a, b in values]
    fn = {
        "arith.muli": lambda a, b: a * b,
        "arith.subi": lambda a, b: a - b,
        "arith.andi": lambda a, b: a & b,
        "arith.ori": lambda a, b: a | b,
        "arith.xori": lambda a, b: a ^ b,
    }[operation]
    expected = [[str((a + b) & 4294967295), str(fn(a, b) & 4294967295), str(a)] for a, b in values]
    for label, reverse in [("source", False), ("target", True)]:
        work = tmp_path / label
        diff.compare(_independent(operation=operation, reverse=reverse), samples, work, fidelity_tools)
        assert json.loads((work / "native-results.json").read_text())["results"] == expected


@pytest.mark.parametrize("fault", ["signedness", "operand", "escaping-use", "constant"])
def test_valid_wrong_target_is_observable(tmp_path, fidelity_tools, fault):
    _, before, after = _canonicalize(tmp_path, fidelity_tools.compiler_python, _source())
    target = after.read_text()
    mutant, expected = {
        "signedness": (target.replace("arith.extsi", "arith.extui"), ["128", "22", "7"]),
        "operand": (target.replace("arith.extsi %arg0", "arith.extsi %arg1"), ["7", "22", "7"]),
        "escaping-use": (target.replace("return %0, %c22_i32", "return %c22_i32, %0"), ["22", "4294967168", "7"]),
        "constant": (target.replace("constant 22", "constant 23"), ["4294967168", "23", "7"]),
    }[fault]
    after.write_text(mutant)
    _verify_mlir(after, fidelity_tools.compiler_python)
    samples = [[[128], [7]]]
    for label, source, wanted in [
        ("source", before.read_text(), ["4294967168", "22", "7"]),
        ("mutant", mutant, expected),
    ]:
        work = tmp_path / label
        diff.compare(source, samples, work, fidelity_tools)
        assert json.loads((work / "native-results.json").read_text())["results"] == [wanted]


def test_valid_changed_dependency_is_observable(tmp_path, fidelity_tools):
    for label, source, expected in [
        ("source", _independent(dependency=True), ["5", "15", "2"]),
        ("mutant", _independent(reverse=True), ["5", "6", "2"]),
    ]:
        work = tmp_path / label
        diff.compare(source, [[[2], [3]]], work, fidelity_tools)
        assert json.loads((work / "native-results.json").read_text())["results"] == [expected]
