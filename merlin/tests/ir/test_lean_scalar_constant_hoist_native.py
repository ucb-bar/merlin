"""Native hoisting evidence with independent integer and exact finite-float references."""

import struct
from types import SimpleNamespace

import pytest
from test_lean_defined_conversions import certify
from test_lean_integer_compare_order import PREDICATES, body_source, stage_for
from test_lean_integer_compare_order_native import body_cases, reference
from test_lean_minmax_commutative_native import compare_expected
from test_lean_scalar_constant_hoist import float_source, manual_zero_stage, mutate

from merlin.verify.lean.check import check_typed_stage


def prepared(tmp_path, tools, text):
    selected = SimpleNamespace(m2m_python=tools.compiler_python)
    stage, before, after = stage_for(tmp_path, selected, text)
    certify(stage, before, selected, tmp_path)
    return stage, before, after


@pytest.mark.parametrize("predicate", PREDICATES)
@pytest.mark.parametrize("reduction", [False, True])
def test_actual_hoisting_native_matches_independent_integer_accumulator(tmp_path, fidelity_tools, predicate, reduction):
    _, before, after = prepared(tmp_path, fidelity_tools, body_source(predicate, reduction=reduction))
    samples, expected = body_cases(predicate, reduction=reduction)
    compare_expected(tmp_path, fidelity_tools, [("before", before), ("after", after)], samples, expected, 64)


@pytest.mark.parametrize("predicate", ["slt", "uge"])
def test_hoisted_constant_native_preserves_depth_two_capture(tmp_path, fidelity_tools, predicate):
    _, before, after = prepared(tmp_path, fidelity_tools, body_source(predicate, nested=True))
    samples, expected = body_cases(predicate, reduction=False)
    compare_expected(tmp_path, fidelity_tools, [("before", before), ("after", after)], samples, expected, 64)


def word(fmt, value):
    code = "f" if fmt == "f32" else "d"
    return int.from_bytes(struct.pack("<" + code, value), "little")


@pytest.mark.parametrize("fmt", ["f32", "f64"])
@pytest.mark.parametrize("reduction", [False, True])
def test_actual_float_constant_hoist_native_uses_exact_finite_samples(tmp_path, fidelity_tools, fmt, reduction):
    _, before, after = prepared(tmp_path, fidelity_tools, float_source(fmt, reduction=reduction))
    # Every selected input, addition and intermediate sum is a multiple of one half in [-8, 8].
    # All are exactly representable in both formats, so Python's wider operation introduces no
    # double-rounding assumption. This finite native check asserts no universal float identity.
    cases = [([-2.0, -1.5, -0.0], [0.0, 0.5, 1.0]), ([0.5, 1.0, 2.0], [-1.0, -0.5, -2.0])]
    samples, expected = [], []
    for n, (data, initial) in enumerate(cases):
        if reduction:
            accumulator = initial[0]
            for x in data:
                accumulator = (x + 1.5) + accumulator
            outputs = [accumulator]
            initial = initial[:1]
        else:
            outputs = [(x + 1.5) + old for x, old in zip(data, initial, strict=True)]
        samples.append([[word(fmt, x) for x in data], [word(fmt, x) for x in initial], [183771 + n]])
        expected.append([*(word(fmt, x) for x in outputs), 183771 + n])
    compare_expected(tmp_path, fidelity_tools, [("before", before), ("after", after)], samples, expected, 64)


@pytest.mark.parametrize("fault,x", [("bits", 6), ("predicate", 0), ("input", 7), ("keep", 7)])
def test_rejected_hoist_mutations_change_independent_native_values(tmp_path, fidelity_tools, fault, x):
    stage, before, after = prepared(tmp_path, fidelity_tools, body_source("slt"))
    mutate(after, SimpleNamespace(m2m_python=fidelity_tools.compiler_python), fault)
    assert check_typed_stage(stage, m2m_python=fidelity_tools.compiler_python)["verdict"] == "abstracted"
    samples = [[[x, x, x], [3, 3, 3], [37]]]
    correct = [[*[3 ^ reference("slt", 5, x, 13)] * 3, 37]]
    wrong_bit = {
        "bits": reference("slt", 6, x, 13),
        "predicate": reference("slt", x, 5, 13),
        "input": reference("slt", 5, 3, 13),
        "keep": reference("slt", 5, x, 13),
    }[fault]
    wrong = [[*[3 ^ wrong_bit] * 3, 0 if fault == "keep" else 37]]
    assert correct != wrong
    compare_expected(tmp_path, fidelity_tools, [("before", before)], samples, correct, 64)
    compare_expected(tmp_path, fidelity_tools, [("after", after)], samples, wrong, 64)


@pytest.mark.parametrize("reduction", [False, True])
def test_manual_zero_loop_hoist_native_preserves_empty_execution(tmp_path, fidelity_tools, reduction):
    selected = SimpleNamespace(m2m_python=fidelity_tools.compiler_python)
    stage, before, after = manual_zero_stage(tmp_path, selected, reduction=reduction)
    certify(stage, before, selected, tmp_path)
    initial = [17] if reduction else []
    samples = [[[], initial, [37]]]
    expected = [[*initial, 37]]
    compare_expected(tmp_path, fidelity_tools, [("before", before), ("after", after)], samples, expected, 64)
