"""Scalar-carrier aliases preserve exact native integer and floating bit patterns."""

import json
import subprocess

import pytest
from test_lean_differential import _floats
from test_lean_scalar_carrier import MUTATE, _captured_source, _source, _stage

from merlin.verify.lean import differential as diff


def _check(program, samples, expected, work, tools):
    diff.compare(program, samples, work, tools, function="forward")
    results = json.loads((work / "results.json").read_text())
    assert results["native"]["results"] == expected
    assert [[bit for value in case["values"] for bit in value["bits"]] for case in results["lean"]] == expected


@pytest.mark.parametrize("element", ["i13", "f32", "f64"])
@pytest.mark.parametrize("escape", [False, True])
def test_scalar_carrier_native_bits(tmp_path, fidelity_tools, element, escape):
    source = _source(element, escape=escape)
    _, _, after = _stage(tmp_path, fidelity_tools.compiler_python, source)
    width = int(element[1:])
    pool = (
        _floats(width) + [0x7FC01234 if width == 32 else 0x7FF8000000001234]
        if element.startswith("f")
        else [0, 1, (1 << (width - 1)) - 1, 1 << (width - 1), (1 << width) - 1]
    )
    samples = [[[x], [pool[(n + 1) % len(pool)]]] for n, x in enumerate(pool)]
    expected = [[str(sample[0][0])] * (3 if escape else 2) for sample in samples]
    for name, program in [("source", source), ("target", after.read_text())]:
        _check(program, samples, expected, tmp_path / name, fidelity_tools)


@pytest.mark.parametrize("element", ["i32", "f32", "f64"])
@pytest.mark.parametrize("local", [False, True])
def test_scalar_carrier_native_capture(tmp_path, fidelity_tools, element, local):
    source = _captured_source(element, local=local)
    _, _, after = _stage(tmp_path, fidelity_tools.compiler_python, source)
    width = int(element[1:])
    pool = (
        _floats(width) + [0x7FC01234 if width == 32 else 0x7FF8000000001234]
        if element.startswith("f")
        else [0, 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF]
    )
    samples = [
        [[x], [pool[(n + 1) % len(pool)]], [condition], [pool[(n + 2) % len(pool)], pool[(n + 3) % len(pool)]], [0, 0]]
        for n, x in enumerate(pool)
        for condition in [0, 1]
    ]
    expected = [[str(x) for x in ([sample[0][0]] * 2 if sample[2][0] else sample[3])] for sample in samples]
    for name, program in [("source", source), ("target", after.read_text())]:
        _check(program, samples, expected, tmp_path / name, fidelity_tools)


def test_scalar_carrier_native_wrong_capture_control(tmp_path, fidelity_tools):
    _, _, after = _stage(tmp_path, fidelity_tools.compiler_python, _captured_source("f32"))
    good = after.read_text()
    changed = subprocess.run(
        [str(fidelity_tools.compiler_python), "-c", MUTATE, str(after), "capture"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert changed.returncode == 0, changed.stderr
    samples = [[[0x7FC01234], [0x80000000], [1], [0, 0], [0, 0]]]
    for name, program, bit in [("good", good, "2143294004"), ("bad", after.read_text(), "2147483648")]:
        _check(program, samples, [[bit, bit]], tmp_path / name, fidelity_tools)


def test_rankzero_splat_manual_alias_native(tmp_path, fidelity_tools):
    source = _source("f32", producer="splat")
    target = "func.func @forward(%x: f32, %other: f32) -> (f32, f32) { return %x, %x : f32, f32 }"
    pool = _floats(32) + [0x7FC01234]
    samples = [[[x], [0]] for x in pool]
    expected = [[str(x), str(x)] for x in pool]
    for name, program in [("source", source), ("manual_target", target)]:
        _check(program, samples, expected, tmp_path / name, fidelity_tools)


def test_chained_scalar_carrier_native_bits(tmp_path, fidelity_tools):
    from test_lean_scalar_carrier import _chain_source

    source = _chain_source(8, escape=True)
    _, _, after = _stage(tmp_path, fidelity_tools.compiler_python, source)
    samples = [[[x], [0]] for x in _floats(32) + [0x7FC01234]]
    expected = [[str(sample[0][0])] * 2 for sample in samples]
    for name, program in [("source", source), ("target", after.read_text())]:
        _check(program, samples, expected, tmp_path / name, fidelity_tools)


def test_carrier_between_extending_casts_native(tmp_path, fidelity_tools):
    from test_lean_scalar_carrier import _mixed_source

    source = _mixed_source()
    _, _, after = _stage(tmp_path, fidelity_tools.compiler_python, source)
    assert "tensor.from_elements" not in after.read_text()
    assert "tensor.extract" not in after.read_text()
    assert "arith.extsi" not in after.read_text()
    assert "linalg.generic" in after.read_text()
    samples = [[[x], [0xFFFFFFFF, 0x80000000], [0, 0]] for x in [0, 1, 127, 128, 255]]
    expected = [[str(((p + sample[0][0]) & 0xFFFFFFFF) ^ p) for p in sample[1]] for sample in samples]
    for name, program in [("source", source), ("target", after.read_text())]:
        _check(program, samples, expected, tmp_path / name, fidelity_tools)
