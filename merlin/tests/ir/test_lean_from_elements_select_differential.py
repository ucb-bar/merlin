"""Independent native bit-copy references for rank-one construction and bounded scalar selection."""

import json
import subprocess

import pytest
from test_lean_differential import _floats
from test_lean_from_elements_select import MUTATE, _captured_source, _nested_source, _source, _stage

from merlin.verify.lean import differential as diff


def _check(program, samples, expected, work, tools, *, index_bits=64):
    diff.compare(program, samples, work, tools, function="forward", index_bits=index_bits)
    results = json.loads((work / "results.json").read_text())
    assert results["native"]["results"] == expected
    assert [[bit for value in case["values"] for bit in value["bits"]] for case in results["lean"]] == expected


@pytest.mark.parametrize("element", ["i13", "i32", "f32", "f64"])
@pytest.mark.parametrize("escape", [False, True])
def test_from_elements_native_bits(tmp_path, fidelity_tools, element, escape):
    source = _source(element, escape=escape)
    _, _, after = _stage(tmp_path, fidelity_tools.compiler_python, source)
    width = int(element[1:])
    pool = (
        _floats(width) + [0x7FC01234 if width == 32 else 0x7FF8000000001234]
        if element.startswith("f")
        else [0, 1, (1 << (width - 1)) - 1, 1 << (width - 1), (1 << width) - 1]
    )
    samples = [[[pool[(n + j) % len(pool)]] for j in range(3)] for n in range(len(pool))]
    expected = [[str(sample[1][0])] + ([str(x[0]) for x in sample] if escape else []) for sample in samples]
    for name, program in [("source", source), ("target", after.read_text())]:
        _check(program, samples, expected, tmp_path / name, fidelity_tools)


@pytest.mark.parametrize("index_bits", [32, 64])
@pytest.mark.parametrize("escape", [False, True])
def test_from_elements_native_index_payloads(tmp_path, fidelity_tools, index_bits, escape):
    source = _source("index", escape=escape)
    _, _, after = _stage(tmp_path, fidelity_tools.compiler_python, source)
    pool = [0, 1, (1 << (index_bits - 1)) - 1, 1 << (index_bits - 1), (1 << index_bits) - 1]
    samples = [[[pool[(n + j) % len(pool)]] for j in range(3)] for n in range(len(pool))]
    expected = [[str(sample[1][0])] + ([str(x[0]) for x in sample] if escape else []) for sample in samples]
    for name, program in [("source", source), ("target", after.read_text())]:
        _check(program, samples, expected, tmp_path / name, fidelity_tools, index_bits=index_bits)


@pytest.mark.parametrize("size,coordinate,repeated", [(1, 0, False), (4, 0, False), (4, 3, False), (3, 2, True)])
def test_native_constructor_size_and_position(tmp_path, fidelity_tools, size, coordinate, repeated):
    source = _source("f32", size=size, coordinate=coordinate, repeated=repeated, escape=True)
    _, _, after = _stage(tmp_path, fidelity_tools.compiler_python, source)
    values = [0x7FC01234, 0x80000000, 1, 0x7F800000][: max(size, 3)]
    samples = [[[value] for value in values]]
    tensor = [values[0], values[1], values[0]] if repeated else values[:size]
    expected = [[str(tensor[coordinate])] + [str(value) for value in tensor]]
    for name, program in [("source", source), ("target", after.read_text())]:
        _check(program, samples, expected, tmp_path / name, fidelity_tools)


@pytest.mark.parametrize("element", ["i32", "f32", "f64"])
@pytest.mark.parametrize("local", [False, True])
def test_native_selected_scalar_capture_and_local_producer(tmp_path, fidelity_tools, element, local):
    source = _captured_source(element, local=local)
    _, _, after = _stage(tmp_path, fidelity_tools.compiler_python, source)
    width = int(element[1:])
    payload = [0, 1 << (width - 1)]
    x0 = 0x7FC01234 if element == "f32" else 0x7FF8000000001234 if element == "f64" else 0xFFFFFFFF
    x1 = (1 << width) - 1
    samples = [[[x0], [x1], [condition], payload, [0, 0]] for condition in [0, 1]]
    chosen = [[x1, x1], payload] if local else [payload, [x1, x1]]
    expected = [[str(x) for x in chosen[condition]] for condition in [0, 1]]
    for name, program in [("source", source), ("target", after.read_text())]:
        _check(program, samples, expected, tmp_path / name, fidelity_tools)


def test_native_empty_constructor(tmp_path, fidelity_tools):
    source = """func.func @forward(%x: i32) -> tensor<0xi32> {
      %t = tensor.from_elements : tensor<0xi32>
      return %t : tensor<0xi32>
    }"""
    _, _, after = _stage(tmp_path, fidelity_tools.compiler_python, source)
    for name, program in [("source", source), ("target", after.read_text())]:
        _check(program, [[[0]], [[1]]], [[], []], tmp_path / name, fidelity_tools)


def test_native_nested_local_selection(tmp_path, fidelity_tools):
    source = _nested_source()
    _, _, after = _stage(tmp_path, fidelity_tools.compiler_python, source)
    samples = [[[seed], [condition], [0xFFFFFFFF, 0x80000000], [0, 0]] for seed in [0, 7] for condition in [0, 1]]
    expected = [
        [str((sample[0][0] + (sum(sample[2]) if sample[1][0] else 2 * p)) & 0xFFFFFFFF) for p in sample[2]]
        for sample in samples
    ]
    # The inner destination is freshly initialized inside each outer iteration (S4 boundary).
    for name, program in [("source", source), ("target", after.read_text())]:
        _check(program, samples, expected, tmp_path / name, fidelity_tools)


def test_native_wrapped_index_selection_at_32_bits(tmp_path, fidelity_tools):
    source = _source("i32", coordinate=4294967297)
    _, _, after = _stage(tmp_path, fidelity_tools.compiler_python, source)
    samples = [[[0x80000000], [7], [99]], [[0], [0xFFFFFFFF], [1]]]
    expected = [["7"], [str(0xFFFFFFFF)]]
    for name, program in [("source", source), ("target", after.read_text())]:
        _check(program, samples, expected, tmp_path / name, fidelity_tools, index_bits=32)


@pytest.mark.parametrize("fault", ["coordinate", "payload", "capture"])
def test_native_selection_mutation_controls(tmp_path, fidelity_tools, fault):
    source = _captured_source() if fault == "capture" else _source()
    _, before, after = _stage(tmp_path, fidelity_tools.compiler_python, source)
    mutated = after if fault == "capture" else before
    good = mutated.read_text()
    changed = subprocess.run(
        [str(fidelity_tools.compiler_python), "-c", MUTATE, str(mutated), fault],
        capture_output=True,
        text=True,
        check=False,
    )
    assert changed.returncode == 0, changed.stderr
    if fault == "capture":
        samples = [[[0x80000000], [0x7FC01234], [1], [0, 0], [0, 0]]]
        expectations = [[str(0x7FC01234)] * 2], [[str(0x80000000)] * 2]
    else:
        samples = [[[0x80000000], [0x7FC01234], [1]]]
        expectations = [[str(0x7FC01234)]], [["1"]]
    for name, program, expected in [("good", good, expectations[0]), ("bad", mutated.read_text(), expectations[1])]:
        _check(program, samples, expected, tmp_path / name, fidelity_tools)
