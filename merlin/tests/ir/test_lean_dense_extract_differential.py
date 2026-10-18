"""Decoded dense scalar constants match MLIR bit for bit, independently of rewrite acceptance."""

import json
import subprocess

import pytest
from test_lean_dense_extract import MUTATE, _captured_source, _source, _stage

from merlin.verify.lean import differential as diff


def _check(program, samples, expected, work, tools, *, index_bits=64):
    diff.compare(program, samples, work, tools, function="forward", index_bits=index_bits)
    results = json.loads((work / "results.json").read_text())
    assert results["native"]["results"] == expected
    assert [[bit for value in case["values"] for bit in value["bits"]] for case in results["lean"]] == expected


@pytest.mark.parametrize(
    "element,literal,bits",
    [
        ("i1", "true", 1),
        ("i13", "-1", 8191),
        ("i32", "-2147483648", 0x80000000),
        ("f32", "0x80000000", 0x80000000),
        ("f32", "0x7FC01234", 0x7FC01234),
        ("f64", "0x7FF8000000001234", 0x7FF8000000001234),
        ("f64", "0x0000000000000001", 1),
    ],
)
@pytest.mark.parametrize("escape", [False, True])
def test_dense_scalar_native_bits(tmp_path, fidelity_tools, element, literal, bits, escape):
    source = _source(element, literal, escape=escape)
    _, _, after = _stage(tmp_path, fidelity_tools.compiler_python, source)
    samples = [[[0]], [[1]]]
    expected = [[str(bits)] * (1 + int(escape)) for _ in samples]
    for name, program in [("source", source), ("target", after.read_text())]:
        _check(program, samples, expected, tmp_path / name, fidelity_tools)


@pytest.mark.parametrize("index_bits", [32, 64])
@pytest.mark.parametrize("escape", [False, True])
def test_dense_index_native_width(tmp_path, fidelity_tools, index_bits, escape):
    source = _source("index", "-1", escape=escape)
    _, _, after = _stage(tmp_path, fidelity_tools.compiler_python, source)
    samples = [[[0]], [[1]], [[-1]]]
    expected = [[str((1 << index_bits) - 1)] * (1 + int(escape)) for _ in samples]
    for name, program in [("source", source), ("target", after.read_text())]:
        _check(program, samples, expected, tmp_path / name, fidelity_tools, index_bits=index_bits)


@pytest.mark.parametrize("local", [False, True])
def test_dense_scalar_native_structured_capture(tmp_path, fidelity_tools, local):
    # Local=True also observes scalar-constant hoisting, which the current rewrite checker
    # conservatively refuses. It is still an independent semantics/native fidelity witness.
    source = _captured_source(local=local)
    _, _, after = _stage(tmp_path, fidelity_tools.compiler_python, source)
    samples = [[[0x80000000], [condition], [0, 0x80000000], [0, 0]] for condition in [0, 1]]
    expected = [["0", "2147483648"], [str(0x7FC01234)] * 2]
    for name, program in [("source", source), ("target", after.read_text())]:
        _check(program, samples, expected, tmp_path / name, fidelity_tools)


def test_dense_scalar_native_mutation_control(tmp_path, fidelity_tools):
    _, _, after = _stage(tmp_path, fidelity_tools.compiler_python, _source())
    good = after.read_text()
    changed = subprocess.run(
        [str(fidelity_tools.compiler_python), "-c", MUTATE, str(after), "bits"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert changed.returncode == 0, changed.stderr
    for name, program, bits in [("good", good, 0x7FC01234), ("bad", after.read_text(), 0x3F800000)]:
        _check(program, [[[0]]], [[str(bits)]], tmp_path / name, fidelity_tools)
