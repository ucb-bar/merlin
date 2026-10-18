"""Exact scalar cleanup preserves native unknown payloads, captures, and fresh nested outputs."""

import pytest
from test_lean_constant_alias import stage_for
from test_lean_constant_alias_native import compare
from test_lean_constant_dead import captured_source, mixed_source, nested_source, source


@pytest.mark.parametrize(
    "element,literal,unknown",
    [
        ("i1", "true", 0),
        ("i8", "-1", 1),
        ("i32", "7", 0x80000000),
        ("f32", "0x7FC01234", 0x80000000),
        ("f64", "0x8000000000000000", 0x7FF8000000001234),
    ],
)
def test_dead_constant_native_exact_unknown_payload(tmp_path, fidelity_tools, element, literal, unknown):
    text = source(element, literal)
    _, _, after = stage_for(tmp_path, fidelity_tools.compiler_python, text)
    for side, program in [("source", text), ("target", after.read_text())]:
        compare(program, [[[unknown], [17]]], [[str(unknown), "17"]], tmp_path / side, fidelity_tools)


@pytest.mark.parametrize("index_bits", [32, 64])
def test_dead_index_native_actual_width(tmp_path, fidelity_tools, index_bits):
    text = source("index", "4294967296")
    _, _, after = stage_for(tmp_path, fidelity_tools.compiler_python, text)
    value = (1 << index_bits) - 1
    for side, program in [("source", text), ("target", after.read_text())]:
        compare(program, [[[value], [17]]], [[str(value), "17"]], tmp_path / side, fidelity_tools, index_bits)


@pytest.mark.parametrize("depth", [1, 2])
def test_nested_dead_constants_native_fresh_destinations(tmp_path, fidelity_tools, depth):
    text = nested_source(depth)
    _, _, after = stage_for(tmp_path, fidelity_tools.compiler_python, text)
    sample = [[1, 0xFFFFFFFF], [3, 7]]
    expected = [["4", "6"]] if depth == 1 else [["2", str(0xFFFFFFFE)]]
    for side, program in [("source", text), ("target", after.read_text())]:
        compare(program, [sample], expected, tmp_path / side, fidelity_tools)


def test_live_constant_native_capture(tmp_path, fidelity_tools):
    text = captured_source()
    _, _, after = stage_for(tmp_path, fidelity_tools.compiler_python, text)
    for side, program in [("source", text), ("target", after.read_text())]:
        compare(program, [[[1, 0xFFFFFFFF], [0, 0]]], [["8", "6"]], tmp_path / side, fidelity_tools)


def test_changed_live_constant_native_result(tmp_path, fidelity_tools):
    for side, text, value in [
        ("good", source(escape=True), 7),
        ("bad", source(escape=True).replace("constant 7", "constant 8"), 8),
    ]:
        compare(text, [[[0], [17]]], [[str(value), "17"]], tmp_path / side, fidelity_tools)


@pytest.mark.parametrize("index_bits", [32, 64])
def test_actual_mixed_fold_cleanup_motion_native_bits(tmp_path, fidelity_tools, index_bits):
    text = mixed_source()
    _, _, after = stage_for(tmp_path, fidelity_tools.compiler_python, text)
    samples = [[[value], [17]] for value in [0, 1, 127, 128, 255]]
    expected = [
        [str((value if value < 128 else value - 256) & 0xFFFFFFFF), "22", "17"] for value in [0, 1, 127, 128, 255]
    ]
    for side, program in [("source", text), ("target", after.read_text())]:
        compare(program, samples, expected, tmp_path / side, fidelity_tools, index_bits)
