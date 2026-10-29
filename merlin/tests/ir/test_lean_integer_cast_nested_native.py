"""Compare certified scalar body casts with native MLIR and independent modular arithmetic."""

import json
import random
from types import SimpleNamespace

import pytest
from test_lean_integer_cast import FAMILIES
from test_lean_integer_cast_nested import captured_source, certify, mutate, nested_source, source
from test_lean_integer_fold import canonicalize

from merlin.verify.lean import differential as diff
from merlin.verify.lean.check import check_typed_stage


def cast(value, kind, source_width, target_width):
    if kind == "extsi" and value & (1 << (source_width - 1)):
        value -= 1 << source_width
    return value % (1 << target_width)


def inputs(width):
    if width <= 8:
        return range(1 << width)
    rng = random.Random(825713 + width)
    return [0, 1, (1 << width) - 1, 1 << (width - 1), (1 << (width - 1)) - 1] + [
        rng.getrandbits(width) for _ in range(15)
    ]


def samples_and_expected(first, second, widths, *, escape=False, reduction=False):
    a, b, c = widths
    samples, expected = [], []
    for value in inputs(a):
        data = [(value + j * 37) % (1 << a) for j in range(3)]
        initial = [(value ^ (97 + j * 23)) % (1 << c) for j in range(1 if reduction else 3)]
        middle_initial = [(value ^ (33 + j * 11)) % (1 << b) for j in range(1 if reduction else 3)]
        keep = (value ^ 0xABCDEF01) % (1 << 32)
        middle = [cast(x, first, a, b) for x in data]
        wide = [cast(x, second, b, c) for x in middle]
        result = (
            [(sum(wide) + initial[0]) % (1 << c)]
            if reduction
            else [(x + old) % (1 << c) for x, old in zip(wide, initial, strict=True)]
        )
        if escape:
            result += (
                [(sum(middle) + middle_initial[0]) % (1 << b)]
                if reduction
                else [(x + old) % (1 << b) for x, old in zip(middle, middle_initial, strict=True)]
            )
        samples.append([data, initial, [keep]] + ([middle_initial] if escape else []))
        expected.append([*map(str, result), str(keep)])
    return samples, expected


def certify_source(tmp_path, tools, program):
    lean_tools = SimpleNamespace(m2m_python=tools.compiler_python)
    stage, before, after = canonicalize(tmp_path, lean_tools, program)
    certify(stage, before, lean_tools, tmp_path)
    return stage, before, after


def assert_execution(tmp_path, tools, before, after, samples, expected):
    outputs = []
    for label, program in [("before", before), ("after", after)]:
        work = tmp_path / label
        diff.compare(program.read_text(), samples, work, tools, function="forward")
        outputs.append(json.loads((work / "native-results.json").read_text())["results"])
    assert outputs == [expected, expected]


@pytest.mark.parametrize("first,second", FAMILIES)
@pytest.mark.parametrize("escape", [False, True])
@pytest.mark.parametrize("large", [False, True])
def test_defined_cast_chains_match_independent_native_reference(tmp_path, fidelity_tools, first, second, escape, large):
    widths = ((64, 13, 1) if large else (8, 2, 1)) if first == "trunci" else ((13, 32, 64) if large else (1, 2, 8))
    _, before, after = certify_source(tmp_path, fidelity_tools, source(first, second, widths, escape=escape))
    samples, expected = samples_and_expected(first, second, widths, escape=escape)
    assert_execution(tmp_path, fidelity_tools, before, after, samples, expected)


@pytest.mark.parametrize("first,second", FAMILIES)
def test_cast_composition_inside_initialized_reduction_matches_native(tmp_path, fidelity_tools, first, second):
    widths = (13, 8, 1) if first == "trunci" else (8, 13, 32)
    _, before, after = certify_source(
        tmp_path, fidelity_tools, source(first, second, widths, escape=True, reduction=True)
    )
    samples, expected = samples_and_expected(first, second, widths, escape=True, reduction=True)
    assert_execution(tmp_path, fidelity_tools, before, after, samples, expected)


@pytest.mark.parametrize("first,second", FAMILIES)
def test_nested_casts_match_independent_native_reference(tmp_path, fidelity_tools, first, second):
    widths = (13, 8, 1) if first == "trunci" else (8, 13, 32)
    _, before, after = certify_source(tmp_path, fidelity_tools, nested_source(first, second, widths))
    samples, expected = samples_and_expected(first, second, widths)
    assert_execution(tmp_path, fidelity_tools, before, after, samples, expected)


def test_native_long_cast_chain_has_certified_actual_boundary(tmp_path, fidelity_tools):
    _, before, after = certify_source(tmp_path, fidelity_tools, source(length=63))
    samples, expected = samples_and_expected("extui", "extui", (1, 13, 64))
    assert_execution(tmp_path, fidelity_tools, before, after, samples, expected)


@pytest.mark.parametrize("fault", ["input", "signedness", "keep"])
def test_wrong_native_cast_targets_are_observably_different_and_refused(tmp_path, fidelity_tools, fault):
    stage, before, after = certify_source(tmp_path, fidelity_tools, source("extui", "extui", escape=True))
    mutate(after, SimpleNamespace(m2m_python=fidelity_tools.compiler_python), fault)
    result = check_typed_stage(stage, m2m_python=fidelity_tools.compiler_python)
    assert result["verdict"] == "abstracted", result.get("reason")
    samples, expected = samples_and_expected("extui", "extui", (1, 8, 32), escape=True)
    outputs = []
    for label, program in [("before", before), ("wrong", after)]:
        work = tmp_path / label
        diff.compare(program.read_text(), samples, work, fidelity_tools, function="forward")
        outputs.append(json.loads((work / "native-results.json").read_text())["results"])
    assert outputs[0] == expected
    assert outputs[1] != expected


def test_captured_scalar_casts_use_actual_outer_runtime_bits(tmp_path, fidelity_tools):
    _, before, after = certify_source(tmp_path, fidelity_tools, captured_source())
    samples, expected = [], []
    for value in [0, 1]:
        for capture in [0, 1]:
            data, initial, keep = [value, 1 - value, value], [0, (1 << 32) - 1, 137], 827163
            samples.append([data, initial, [keep], [capture]])
            expected.append(
                [*map(str, [(x + capture + old) % (1 << 32) for x, old in zip(data, initial, strict=True)]), str(keep)]
            )
    assert_execution(tmp_path, fidelity_tools, before, after, samples, expected)
