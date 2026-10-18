"""Exact scalar aliases and folded duplicate values match native MLIR and independent bits."""

import json

import pytest
from test_lean_constant_alias import FOLDS, captured_source, duplicate_source, local_source, stage_for
from test_lean_index_fold import NAMES
from test_lean_index_fold import source as fold_source

from merlin.verify.lean import differential as diff


def compare(program, samples, expected, work, tools, index_bits=64):
    diff.compare(program, samples, work, tools, function="forward", index_bits=index_bits)
    results = json.loads((work / "results.json").read_text())
    assert results["native"]["results"] == expected
    assert [[bit for value in case["values"] for bit in value["bits"]] for case in results["lean"]] == expected


@pytest.mark.parametrize(
    "element,literal,bits",
    [
        ("i1", "true", 1),
        ("i8", "-1", 255),
        ("i32", "-2147483648", 1 << 31),
        ("f32", "0x80000000", 1 << 31),
        ("f32", "0x7FC01234", 0x7FC01234),
        ("f64", "0x7FF8000000001234", 0x7FF8000000001234),
    ],
)
def test_duplicate_constant_native_exact_bits(tmp_path, fidelity_tools, element, literal, bits):
    source = duplicate_source(element, literal)
    _, _, target = stage_for(tmp_path, fidelity_tools.compiler_python, source)
    samples = [[[0], [17]], [[1], [-1]]]
    expected = [[str(bits), str(bits), str(keep)] for keep in [17, (1 << 32) - 1]]
    for name, program in [("source", source), ("target", target.read_text())]:
        compare(program, samples, expected, tmp_path / name, fidelity_tools)


@pytest.mark.parametrize("index_bits", [32, 64])
@pytest.mark.parametrize("name,a,b", FOLDS)
def test_fold_alias_native_independent_results(tmp_path, fidelity_tools, index_bits, name, a, b):
    source = fold_source(name, a, b, escape=True)
    _, _, target = stage_for(tmp_path, fidelity_tools.compiler_python, source)
    samples = [[[0], [17]], [[-1], [-1]]]
    expected = [[str(a & ((1 << index_bits) - 1))] * 2 + [str(keep)] for keep in [17, (1 << 32) - 1]]
    for side, program in [("source", source), ("target", target.read_text())]:
        compare(program, samples, expected, tmp_path / side, fidelity_tools, index_bits)


def test_wrapped_spelling_alias_native_32(tmp_path, fidelity_tools):
    source = fold_source("andi", -(1 << 63), 3, escape=True)
    _, _, target = stage_for(tmp_path, fidelity_tools.compiler_python, source)
    for name, program in [("source", source), ("target", target.read_text())]:
        compare(program, [[[0], [17]]], [["0", "0", "17"]], tmp_path / name, fidelity_tools, 32)


@pytest.mark.parametrize("index_bits", [32, 64])
def test_cyclic_duplicate_chain_native(tmp_path, fidelity_tools, index_bits):
    source = fold_source(length=128, cycle=True)
    _, _, target = stage_for(tmp_path, fidelity_tools.compiler_python, source)
    # Independent modular host arithmetic supplies the expected final bit pattern.
    mask, value = (1 << index_bits) - 1, 17
    for n in range(128):
        rhs = (137 * n + 5) & mask
        name = NAMES[n % len(NAMES)]
        value = {
            "addi": lambda: value + rhs,
            "subi": lambda: value - rhs,
            "muli": lambda: value * rhs,
            "andi": lambda: value & rhs,
            "ori": lambda: value | rhs,
            "xori": lambda: value ^ rhs,
        }[name]() & mask
    assert value == (-134 & mask)
    for name, program in [("source", source), ("target", target.read_text())]:
        compare(program, [[[0], [17]]], [[str(value), "17"]], tmp_path / name, fidelity_tools, index_bits)


def test_captured_alias_native(tmp_path, fidelity_tools):
    source = captured_source()
    _, _, target = stage_for(tmp_path, fidelity_tools.compiler_python, source)
    for name, program in [("source", source), ("target", target.read_text())]:
        compare(program, [[[0, 1, -7], [0, 0, 0]]], [["7", "8", "0", "7", "7"]], tmp_path / name, fidelity_tools)


def test_nested_local_alias_native_with_fresh_destinations(tmp_path, fidelity_tools):
    for alias in [False, True]:
        compare(local_source(alias=alias), [[[0, 1], [0, 0]]], [["6", "9"]], tmp_path / str(alias), fidelity_tools)
