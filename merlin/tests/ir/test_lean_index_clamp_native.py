"""Native unsigned-index clamp controls; exclude from binding-only CI."""

import json
from types import SimpleNamespace

import pytest
from test_lean_defined_conversions import certify
from test_lean_index_clamp import captured_source, mutate, source, stage_for

from merlin.verify.lean import differential as diff
from merlin.verify.lean.check import check_definedness, check_typed_stage


def compare_expected(tmp_path, tools, paths, samples, expected, index_bits):
    for name, path in paths:
        work = tmp_path / name
        diff.compare(path.read_text(), samples, work, tools, function="forward", index_bits=index_bits)
        actual = json.loads((work / "native-results.json").read_text())["results"]
        assert actual == [[str(value) for value in row] for row in expected]


def data_for(n):
    return [(937 * n + 53 * i + 11) % 8192 for i in range(8)]


@pytest.mark.parametrize("mode", ["constant", "widened"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_clamp_reads_match_unsigned_native_reference(tmp_path, fidelity_tools, mode, reverse, index_bits):
    tools = SimpleNamespace(m2m_python=fidelity_tools.compiler_python)
    stage, before, after = stage_for(tmp_path, tools, source(mode, reverse=reverse))
    certify(stage, before, tools, tmp_path, index_bits=index_bits)
    indices = [0, 1, 3, 7, 8, 1 << (index_bits - 1), (1 << index_bits) - 1]
    samples = [[data_for(n), [idx], [n % 8], [(131 * n) % (1 << 32)]] for n, idx in enumerate(indices)]
    expected = [[s[0][min(s[1][0], s[2][0] if mode == "widened" else 7)], s[3][0]] for s in samples]
    compare_expected(tmp_path, fidelity_tools, [("before", before), ("after", after)], samples, expected, index_bits)


@pytest.mark.parametrize("index_bits", [32, 64])
def test_initialized_minimum_can_return_high_bits_without_invented_read_bounds(tmp_path, fidelity_tools, index_bits):
    tools = SimpleNamespace(m2m_python=fidelity_tools.compiler_python)
    stage, before, after = stage_for(tmp_path, tools, source("unknown", extract=False))
    certify(stage, before, tools, tmp_path, index_bits=index_bits)
    mask, sign = (1 << index_bits) - 1, 1 << (index_bits - 1)
    pairs = [(0, mask), (mask, mask), (sign, mask), (mask, 7)]
    samples = [[data_for(n), [a], [b], [n]] for n, (a, b) in enumerate(pairs)]
    expected = [[min(a, b), n] for n, (a, b) in enumerate(pairs)]
    compare_expected(tmp_path, fidelity_tools, [("before", before), ("after", after)], samples, expected, index_bits)


@pytest.mark.parametrize("index_bits", [32, 64])
def test_refused_different_safe_read_changes_native_results(tmp_path, fidelity_tools, index_bits):
    tools = SimpleNamespace(m2m_python=fidelity_tools.compiler_python)
    stage, before, after = stage_for(tmp_path, tools, source())
    certify(stage, before, tools, tmp_path, index_bits=index_bits)
    mutate(after, tools, "constant_read")
    result = check_typed_stage(stage, m2m_python=tools.m2m_python, index_bits=index_bits)
    assert result["verdict"] == "abstracted", result.get("reason")
    result = check_definedness(
        after, "forward", tmp_path / "mutated-defined", m2m_python=tools.m2m_python, index_bits=index_bits
    )
    assert result["verdict"] == "verified", result.get("reason")
    samples = [[data_for(1), [9], [0], [37]]]
    compare_expected(tmp_path, fidelity_tools, [("before", before)], samples, [[data_for(1)[7], 37]], index_bits)
    compare_expected(tmp_path, fidelity_tools, [("after", after)], samples, [[data_for(1)[0], 37]], index_bits)


@pytest.mark.parametrize("index_bits", [32, 64])
def test_zero_clamp_is_defined_and_native_matching_while_rewrite_abstains(tmp_path, fidelity_tools, index_bits):
    tools = SimpleNamespace(m2m_python=fidelity_tools.compiler_python)
    stage, before, after = stage_for(tmp_path, tools, source(cap=0))
    result = check_definedness(
        before, "forward", tmp_path / "defined", m2m_python=tools.m2m_python, index_bits=index_bits
    )
    assert result["verdict"] == "verified", result.get("reason")
    result = check_typed_stage(stage, m2m_python=tools.m2m_python, index_bits=index_bits)
    assert result["verdict"] == "abstracted", result.get("reason")
    samples = [[data_for(2), [(1 << index_bits) - 1], [7], [13]]]
    compare_expected(
        tmp_path, fidelity_tools, [("before", before), ("after", after)], samples, [[data_for(2)[0], 13]], index_bits
    )


@pytest.mark.parametrize("index_bits", [32, 64])
def test_captured_clamped_index_matches_native_generic_execution(tmp_path, fidelity_tools, index_bits):
    tools = SimpleNamespace(m2m_python=fidelity_tools.compiler_python)
    stage, before, after = stage_for(tmp_path, tools, captured_source("widened"))
    certify(stage, before, tools, tmp_path, index_bits=index_bits)
    data, xs = data_for(3), [0, 8191, 4096]
    samples = [[data, [(1 << index_bits) - 1], [5], xs, [19, 23, 29], [37]]]
    expected = [[*((data[5] + x) % 8192 for x in xs), 37]]
    compare_expected(tmp_path, fidelity_tools, [("before", before), ("after", after)], samples, expected, index_bits)
