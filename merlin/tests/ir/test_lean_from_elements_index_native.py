"""Computed constructor coordinates match native MLIR and independent scalar bit selection."""

import json

import pytest
from test_lean_from_elements_index import OPERATIONS, computed_source
from test_lean_from_elements_select import _stage

from merlin.verify.lean import differential as diff


def compare(program, samples, expected, work, tools, index_bits=64):
    diff.compare(program, samples, work, tools, function="forward", index_bits=index_bits)
    result = json.loads((work / "results.json").read_text())
    assert result["native"]["results"] == expected
    assert [[bit for value in case["values"] for bit in value["bits"]] for case in result["lean"]] == expected


@pytest.mark.parametrize("name,a,b", OPERATIONS)
@pytest.mark.parametrize("index_bits", [32, 64])
def test_computed_index_selects_exact_native_payload(tmp_path, fidelity_tools, name, a, b, index_bits):
    source = computed_source(name, a, b, escape=True)
    _, _, after = _stage(tmp_path, fidelity_tools.compiler_python, source)
    samples = [[[0], [0x7FC01234], [0x80000000]], [[1], [0x80000000], [0x7F800000]]]
    expected = [[str(sample[1][0])] + [str(x[0]) for x in sample] for sample in samples]
    for side, program in [("source", source), ("target", after.read_text())]:
        compare(program, samples, expected, tmp_path / side, fidelity_tools, index_bits)


def test_computed_index_f64_native_payload(tmp_path, fidelity_tools):
    source = computed_source("xori", 3, 2, element="f64", escape=True)
    _, _, after = _stage(tmp_path, fidelity_tools.compiler_python, source)
    sample = [[0x8000000000000000], [0x7FF8000000001234], [1]]
    expected = [[str(sample[1][0])] + [str(x[0]) for x in sample]]
    for side, program in [("source", source), ("target", after.read_text())]:
        compare(program, [sample], expected, tmp_path / side, fidelity_tools)


def test_wrapped_computed_index_native_32_only(tmp_path, fidelity_tools):
    # The 64-bit source coordinate is OOB; never execute it as a native fidelity witness.
    source = computed_source(a=1 << 32)
    _, _, after = _stage(tmp_path, fidelity_tools.compiler_python, source)
    for side, program in [("source", source), ("target", after.read_text())]:
        compare(program, [[[0], [0x7FC01234], [1]]], [[str(0x7FC01234)]], tmp_path / side, fidelity_tools, 32)


def test_selected_computed_value_native_capture(tmp_path, fidelity_tools):
    source = computed_source(capture=True)
    _, _, after = _stage(tmp_path, fidelity_tools.compiler_python, source)
    samples = [[[0], [0x7FC01234], [condition], [0, 0x80000000], [0, 0]] for condition in [0, 1]]
    expected = [["0", str(0x80000000)], [str(0x7FC01234)] * 2]
    for side, program in [("source", source), ("target", after.read_text())]:
        compare(program, samples, expected, tmp_path / side, fidelity_tools)


def test_changed_computed_coordinate_native_reference(tmp_path, fidelity_tools):
    # This source remains defined, but independently selects a different scalar than the target.
    source = computed_source(a=1)
    for side, program, selected in [("wrong-coordinate", source, 2), ("good-coordinate", computed_source(), 1)]:
        compare(
            program,
            [[[0], [0x7FC01234], [0x80000000]]],
            [[str([0, 0x7FC01234, 0x80000000][selected])]],
            tmp_path / side,
            fidelity_tools,
        )
