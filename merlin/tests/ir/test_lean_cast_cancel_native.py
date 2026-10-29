"""Exact cast round trips match independent modular bits and native MLIR execution."""

import random
from types import SimpleNamespace

import pytest
from test_lean_cast_cancel import captured_source, local_source, long_source, mutate, source, two_capture_source
from test_lean_constant_alias_native import compare
from test_lean_integer_cast_nested import certify
from test_lean_integer_fold import canonicalize

from merlin.verify.lean.check import check_typed_stage


def stage_for(tmp_path, tools, text):
    selected = SimpleNamespace(m2m_python=tools.compiler_python)
    stage, before, after = canonicalize(tmp_path, selected, text)
    certify(stage, before, selected, tmp_path)
    return stage, before, after


def samples(width):
    if width <= 8:
        values = range(1 << width)
    else:
        rng = random.Random(948281 + width)
        values = [0, 1, (1 << width) - 1, 1 << (width - 1), (1 << (width - 1)) - 1]
        values += [rng.getrandbits(width) for _ in range(16)]
    return [[[x], [(x ^ 0x13579) & ((1 << width) - 1)]] for x in values]


def widened(kind, width, wide, value):
    if kind == "extsi" and value & (1 << (width - 1)):
        value -= 1 << width
    return value & ((1 << wide) - 1)


@pytest.mark.parametrize("kind", ["extui", "extsi"])
@pytest.mark.parametrize("widths", [(1, 8), (8, 32), (13, 64), (32, 64)])
@pytest.mark.parametrize("escape", [False, True])
def test_actual_roundtrip_native_exact_bits(tmp_path, fidelity_tools, kind, widths, escape):
    width, wide = widths
    _, before, after = stage_for(tmp_path, fidelity_tools, source(kind, widths, escape=escape))
    inputs = samples(width)
    expected = [
        [str(case[0][0]), str(case[1][0])] + ([str(widened(kind, width, wide, case[0][0]))] if escape else [])
        for case in inputs
    ]
    for side, program in [("before", before), ("after", after)]:
        compare(program.read_text(), inputs, expected, tmp_path / side, fidelity_tools)


@pytest.mark.parametrize("kind", ["extui", "extsi"])
@pytest.mark.parametrize("escape", [False, True])
def test_surviving_generic_roundtrip_native(tmp_path, fidelity_tools, kind, escape):
    _, before, after = stage_for(tmp_path, fidelity_tools, local_source(kind, escape=escape))
    values, initial, midinitial = [0, 128, 255], [7, 255, 1], [5, 0x80000000, 0xFFFFFFFF]
    inputs = [[values, initial, [17]] + ([midinitial] if escape else [])]
    expected = [str((x + old) & 255) for x, old in zip(values, initial)]
    if escape:
        expected += [str((widened(kind, 8, 32, x) + old) & 0xFFFFFFFF) for x, old in zip(values, midinitial)]
    expected += ["17"]
    for side, program in [("before", before), ("after", after)]:
        compare(program.read_text(), inputs, [expected], tmp_path / side, fidelity_tools)


@pytest.mark.parametrize("kind", ["extui", "extsi"])
def test_depth_two_native_uses_fresh_inner_destinations(tmp_path, fidelity_tools, kind):
    _, before, after = stage_for(tmp_path, fidelity_tools, local_source(kind, depth=2))
    inputs = [[[1, 128, 255], [3, 255, 1], [17]]]
    expected = [["4", "127", "0", "17"]]
    for side, program in [("before", before), ("after", after)]:
        compare(program.read_text(), inputs, expected, tmp_path / side, fidelity_tools)


@pytest.mark.parametrize("kind", ["extui", "extsi"])
def test_captured_roundtrip_native_preserves_escaped_widening(tmp_path, fidelity_tools, kind):
    _, before, after = stage_for(tmp_path, fidelity_tools, captured_source(kind))
    inputs = [[[128], [17], [0, 1, 255], [3, 7, 255]]]
    expected = [["131", "134", "128", "17", str(widened(kind, 8, 32, 128))]]
    for side, program in [("before", before), ("after", after)]:
        compare(program.read_text(), inputs, expected, tmp_path / side, fidelity_tools)


@pytest.mark.parametrize("kind", ["extui", "extsi"])
def test_parent_alias_two_capture_order_matches_native_bits(tmp_path, fidelity_tools, kind):
    _, before, after = stage_for(tmp_path, fidelity_tools, two_capture_source(kind))
    samples = [[[128], [17], [0, 1, 255], [3, 7, 255]]]
    expected = [["146", "149", "111"]]
    for side, program in [("before", before), ("after", after)]:
        compare(program.read_text(), samples, expected, tmp_path / side, fidelity_tools)


@pytest.mark.parametrize("kind", ["extui", "extsi"])
def test_long_widening_roundtrip_native(tmp_path, fidelity_tools, kind):
    _, before, after = stage_for(tmp_path, fidelity_tools, long_source(kind))
    inputs = samples(8)
    expected = [[str(case[0][0]), str(case[1][0])] for case in inputs]
    for side, program in [("before", before), ("after", after)]:
        compare(program.read_text(), inputs, expected, tmp_path / side, fidelity_tools)


@pytest.mark.parametrize("fault", ["wrong_alias", "escaping_input", "signedness"])
def test_valid_wrong_roundtrip_targets_change_native_results(tmp_path, fidelity_tools, fault):
    stage, before, after = stage_for(tmp_path, fidelity_tools, source("extui", escape=True))
    tools = SimpleNamespace(m2m_python=fidelity_tools.compiler_python)
    mutate(after, tools, fault)
    result = check_typed_stage(stage, m2m_python=tools.m2m_python)
    assert result["verdict"] == "abstracted", result.get("reason")
    inputs = [[[128], [17]]]
    wrong = {
        "wrong_alias": ["17", "17", "128"],
        "escaping_input": ["128", "17", "17"],
        "signedness": ["128", "17", str(0xFFFFFF80)],
    }[fault]
    compare(before.read_text(), inputs, [["128", "17", "128"]], tmp_path / "before", fidelity_tools)
    compare(after.read_text(), inputs, [wrong], tmp_path / "wrong", fidelity_tools)
