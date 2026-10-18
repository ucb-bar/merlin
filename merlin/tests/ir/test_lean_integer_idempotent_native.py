"""Certify actual region aliases and compare initialized Lean/native values with a Python oracle."""

import json
import random
from types import SimpleNamespace

import pytest
from test_lean_integer_fold import canonicalize
from test_lean_integer_idempotent import NAMES, certify, mutate, nested_source, source

from merlin.verify.lean import differential as diff
from merlin.verify.lean.check import check_typed_stage


def values(width):
    if width <= 8:
        return range(1 << width)
    rng = random.Random(827163 + width)
    return [0, 1, (1 << width) - 1, 1 << (width - 1), (1 << (width - 1)) - 1] + [
        rng.getrandbits(width) for _ in range(23)
    ]


def samples_and_expected(width, reduction):
    modulus = 1 << width
    samples, expected = [], []
    for value in values(width):
        data = [(value + j * 37) % modulus for j in range(6)]
        initial = [(value ^ (97 + j * 23)) % modulus for j in range(1 if reduction else 6)]
        capture, keep = (value ^ 137) % modulus, (value ^ 0xABCDEF01) % (1 << 32)
        transformed = [(x ^ capture) + x for x in data]
        result = (
            [(initial[0] + sum(transformed)) % modulus]
            if reduction
            else [(a + b) % modulus for a, b in zip(initial, transformed, strict=True)]
        )
        samples.append([data, initial, [capture], [keep]])
        expected.append([*map(str, result), str(keep)])
    return samples, expected


def certify_source(tmp_path, tools, program):
    lean_tools = SimpleNamespace(m2m_python=tools.compiler_python)
    stage, before, after = canonicalize(tmp_path, lean_tools, program)
    certify(stage, before, lean_tools, tmp_path)
    return stage, before, after


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("width", [1, 8, 64])
def test_all_idempotent_families_have_defined_native_execution(tmp_path, fidelity_tools, name, width):
    _, before, after = certify_source(tmp_path, fidelity_tools, source(name, width=width))
    samples, expected = samples_and_expected(width, False)
    for label, program in [("before", before), ("after", after)]:
        work = tmp_path / label
        diff.compare(program.read_text(), samples, work, fidelity_tools, function="forward")
        assert json.loads((work / "native-results.json").read_text())["results"] == expected


@pytest.mark.parametrize("reduction", [False, True])
def test_all_six_alias_rules_compose_at_odd_width(tmp_path, fidelity_tools, reduction):
    _, before, after = certify_source(tmp_path, fidelity_tools, source(chain=True, reduction=reduction))
    samples, expected = samples_and_expected(13, reduction)
    for label, program in [("before", before), ("after", after)]:
        work = tmp_path / label
        diff.compare(program.read_text(), samples, work, fidelity_tools, function="forward")
        assert json.loads((work / "native-results.json").read_text())["results"] == expected


@pytest.mark.parametrize("name", ["andi", "maxsi"])
def test_outer_alias_is_captured_by_the_actual_inner_generic(tmp_path, fidelity_tools, name):
    _, before, after = certify_source(tmp_path, fidelity_tools, nested_source(name))
    samples, expected = [], []
    for value in values(13):
        data = [(value + j * 37) % 8192 for j in range(3)]
        initial = [(value ^ (j * 101 + 73)) % 8192 for j in range(3)]
        keep = (value ^ 0xABCDEFAA) % (1 << 32)
        samples.append([data, initial, [keep]])
        expected.append([*map(str, [(a + b) % 8192 for a, b in zip(data, initial, strict=True)]), str(keep)])
    for label, program in [("before", before), ("after", after)]:
        work = tmp_path / label
        diff.compare(program.read_text(), samples, work, fidelity_tools, function="forward")
        assert json.loads((work / "native-results.json").read_text())["results"] == expected


@pytest.mark.parametrize("fault", ["input", "escaping_keep"])
def test_independent_wrong_aliases_change_native_values(tmp_path, fidelity_tools, fault):
    stage, before, after = certify_source(tmp_path, fidelity_tools, source())
    mutate(after, SimpleNamespace(m2m_python=fidelity_tools.compiler_python), fault)
    result = check_typed_stage(stage, m2m_python=fidelity_tools.compiler_python)
    assert result["verdict"] == "abstracted", result.get("reason")
    samples, expected = samples_and_expected(13, False)
    samples, expected = samples[:3], expected[:3]
    outputs = []
    for label, program in [("before", before), ("wrong", after)]:
        work = tmp_path / label
        diff.compare(program.read_text(), samples, work, fidelity_tools, function="forward")
        outputs.append(json.loads((work / "native-results.json").read_text())["results"])
    assert outputs[0] == expected
    assert outputs[1] != expected
