"""Native min/max commutation evidence; exclude this module from binding-only CI."""

import json
from types import SimpleNamespace

import pytest
from test_lean_defined_conversions import certify
from test_lean_minmax_commutative import NAMES, manual_swap, mutate, source, stage_for

from merlin.verify.lean import differential as diff
from merlin.verify.lean.check import check_definedness, check_typed_stage


def reference(name, x, y, width):
    mask, sign = (1 << width) - 1, 1 << (width - 1)
    x, y = x & mask, y & mask
    if name.endswith("si"):
        x, y = (x ^ sign) - sign, (y ^ sign) - sign
    return (min(x, y) if name.startswith("min") else max(x, y)) & mask


def compare_expected(tmp_path, tools, paths, samples, expected, index_bits):
    for name, path in paths:
        work = tmp_path / name
        diff.compare(path.read_text(), samples, work, tools, function="forward", index_bits=index_bits)
        actual = json.loads((work / "native-results.json").read_text())["results"]
        assert actual == [[str(value) for value in row] for row in expected]


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("width", [1, 13, 32, 64])
def test_certified_minmax_native_matches_independent_extrema_reference(tmp_path, fidelity_tools, name, width):
    tools = SimpleNamespace(m2m_python=fidelity_tools.compiler_python)
    stage, before, after = (
        manual_swap(tmp_path, tools, name)
        if width == 1
        else stage_for(tmp_path, tools, source(name, width, escape=True))
    )
    certify(stage, before, tools, tmp_path)
    mask, sign = (1 << width) - 1, 1 << (width - 1)
    values = list(dict.fromkeys([0, 1, min(5, mask), sign, sign - 1, mask]))
    samples = [[[value], [(131 * i) % (1 << 32)]] for i, value in enumerate(values)]
    expected = [
        [reference(name, 1 if width == 1 else 5, value, width), value, s[1][0]]
        for value, s in zip(values, samples, strict=True)
    ]
    compare_expected(tmp_path, fidelity_tools, [("before", before), ("after", after)], samples, expected, 64)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("index_bits", [32, 64])
def test_certified_minmax_native_preserves_captured_generic_inputs(tmp_path, fidelity_tools, name, index_bits):
    tools = SimpleNamespace(m2m_python=fidelity_tools.compiler_python)
    stage, before, after = stage_for(tmp_path, tools, source(name, generic=True))
    certify(stage, before, tools, tmp_path, index_bits=index_bits)
    samples = [[[0, 5, 8191], [13, 17, 19], [37]], [[4096, 4095, 6], [7, 11, 23], [(1 << 32) - 1]]]
    expected = [[*(reference(name, 5, x, 13) for x in s[0]), s[2][0]] for s in samples]
    compare_expected(tmp_path, fidelity_tools, [("before", before), ("after", after)], samples, expected, index_bits)


@pytest.mark.parametrize("name", NAMES)
def test_actual_i1_extreme_folding_native_tracks_exact_neutral_support(tmp_path, fidelity_tools, name):
    tools = SimpleNamespace(m2m_python=fidelity_tools.compiler_python)
    stage, before, after = stage_for(tmp_path, tools, source(name, 1, escape=True))
    result = check_definedness(before, "forward", tmp_path / "defined", m2m_python=tools.m2m_python)
    assert result["verdict"] == "verified", result.get("reason")
    result = check_typed_stage(stage, m2m_python=tools.m2m_python)
    expected_verdict = "verified" if name in {"minui", "maxsi"} else "abstracted"
    assert result["verdict"] == expected_verdict, result.get("reason")
    if expected_verdict == "verified":
        certify(stage, before, tools, tmp_path)
    samples = [[[0], [0]], [[1], [37]]]
    expected = [[reference(name, 1, s[0][0], 1), s[0][0], s[1][0]] for s in samples]
    compare_expected(tmp_path, fidelity_tools, [("before", before), ("after", after)], samples, expected, 64)


@pytest.mark.parametrize("fault", ["operation", "signedness"])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_refused_minmax_operation_mutation_changes_native_bits(tmp_path, fidelity_tools, fault, index_bits):
    tools = SimpleNamespace(m2m_python=fidelity_tools.compiler_python)
    stage, before, after = stage_for(tmp_path, tools, source(escape=True))
    certify(stage, before, tools, tmp_path, index_bits=index_bits)
    mutate(after, tools, fault)
    result = check_typed_stage(stage, m2m_python=tools.m2m_python, index_bits=index_bits)
    assert result["verdict"] == "abstracted", result.get("reason")
    value = 0 if fault == "operation" else 4096
    samples = [[[value], [37]]]
    correct = [[reference("minui", 5, value, 13), value, 37]]
    corrupted = [[reference("maxui" if fault == "operation" else "minsi", 5, value, 13), value, 37]]
    assert correct != corrupted
    compare_expected(tmp_path, fidelity_tools, [("before", before)], samples, correct, index_bits)
    compare_expected(tmp_path, fidelity_tools, [("after", after)], samples, corrupted, index_bits)
