"""Native evidence for certified body identities; exclude from binding-only CI."""

from types import SimpleNamespace

import pytest
from test_lean_integer_idempotent import certify
from test_lean_integer_idempotent_native import values
from test_lean_integer_neutral_bodies import NAMES, chain_source, mutate, nested_source, source, stage_for
from test_lean_minmax_commutative_native import compare_expected

from merlin.verify.lean.check import check_typed_stage


def cases(width, *, reduction=False, escape=False):
    mask = (1 << width) - 1
    samples, expected = [], []
    for i, value in enumerate(values(width)):
        data = [(value + j * 37) & mask for j in range(3)]
        initial = [(value ^ (j * 13 + 17)) & mask for j in range(1 if reduction else 3)]
        keep = (i * 131 + 37) % (1 << 32)
        if reduction:
            result = initial[0]
            for x in data:
                mixed = x ^ result
                result = (mixed + x) & mask if escape else mixed
            outputs = [result]
        else:
            mixed = [a ^ b for a, b in zip(data, initial, strict=True)]
            outputs = [(a + b) & mask for a, b in zip(data, mixed, strict=True)] if escape else mixed
        samples.append([data, initial, [keep]])
        expected.append([*outputs, keep])
    return samples, expected


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("width", [1, 13, 64])
def test_certified_neutral_native_matches_independent_identity_reference(tmp_path, fidelity_tools, name, width):
    tools = SimpleNamespace(m2m_python=fidelity_tools.compiler_python)
    stage, before, after = stage_for(tmp_path, tools, source(name, width))
    certify(stage, before, tools, tmp_path)
    samples, expected = cases(width)
    compare_expected(tmp_path, fidelity_tools, [("before", before), ("after", after)], samples, expected, 64)


@pytest.mark.parametrize("name", ["addi", "andi", "minsi"])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_certified_reverse_and_reduction_preserve_escaping_values(tmp_path, fidelity_tools, name, index_bits):
    tools = SimpleNamespace(m2m_python=fidelity_tools.compiler_python)
    stage, before, after = stage_for(tmp_path, tools, source(name, reverse=True, reduction=True, escape=True))
    certify(stage, before, tools, tmp_path, index_bits=index_bits)
    samples, expected = cases(13, reduction=True, escape=True)
    compare_expected(tmp_path, fidelity_tools, [("before", before), ("after", after)], samples, expected, index_bits)


def test_long_mixed_neutral_chain_has_certified_native_boundary(tmp_path, fidelity_tools):
    tools = SimpleNamespace(m2m_python=fidelity_tools.compiler_python)
    stage, before, after = stage_for(tmp_path, tools, chain_source())
    certify(stage, before, tools, tmp_path)
    samples, expected = cases(13)
    compare_expected(tmp_path, fidelity_tools, [("before", before), ("after", after)], samples, expected, 64)


@pytest.mark.parametrize("name", ["addi", "minsi"])
def test_exact_local_alias_updates_native_nested_capture(tmp_path, fidelity_tools, name):
    tools = SimpleNamespace(m2m_python=fidelity_tools.compiler_python)
    stage, before, after = stage_for(tmp_path, tools, nested_source(name))
    certify(stage, before, tools, tmp_path)
    samples, _ = cases(13)
    expected = [[*((a + b) % 8192 for a, b in zip(s[0], s[1], strict=True)), s[2][0]] for s in samples]
    compare_expected(tmp_path, fidelity_tools, [("before", before), ("after", after)], samples, expected, 64)


@pytest.mark.parametrize("fault", ["input", "keep", "constant", "direction"])
def test_rejected_alias_mutations_have_observable_native_values(tmp_path, fidelity_tools, fault):
    tools = SimpleNamespace(m2m_python=fidelity_tools.compiler_python)
    stage, before, after = stage_for(tmp_path, tools, source("subi"))
    certify(stage, before, tools, tmp_path)
    changed = before if fault in {"constant", "direction"} else after
    mutate(changed, tools, fault)
    result = check_typed_stage(stage, m2m_python=tools.m2m_python)
    assert result["verdict"] == "abstracted", result.get("reason")
    samples = [[[7, 4096, 8191], [3, 5, 17], [37]]]
    correct = [[7 ^ 3, 4096 ^ 5, 8191 ^ 17, 37]]
    if fault == "constant":
        corrupt = [[*((x - 1) % 8192 ^ y for x, y in zip(samples[0][0], samples[0][1], strict=True)), 37]]
    elif fault == "direction":
        corrupt = [[*((-x) % 8192 ^ y for x, y in zip(samples[0][0], samples[0][1], strict=True)), 37]]
    elif fault == "input":
        corrupt = [[0, 0, 0, 37]]
    else:
        corrupt = [[7 ^ 3, 4096 ^ 5, 8191 ^ 17, 0]]
    assert correct != corrupt
    before_expected = corrupt if changed == before else correct
    after_expected = corrupt if changed == after else correct
    compare_expected(tmp_path, fidelity_tools, [("before", before)], samples, before_expected, 64)
    compare_expected(tmp_path, fidelity_tools, [("after", after)], samples, after_expected, 64)
