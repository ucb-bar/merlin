"""Native comparison reversal evidence; exclude this module from binding-only CI."""

import random
from types import SimpleNamespace

import pytest
from test_lean_defined_conversions import certify
from test_lean_integer_compare_order import PREDICATES, body_source, manual_i1_stage, mutate, source, stage_for
from test_lean_minmax_commutative_native import compare_expected

from merlin.verify.lean.check import check_definedness, check_typed_stage


def reference(predicate, a, b, width):
    mask, sign = (1 << width) - 1, 1 << (width - 1)
    a, b = a & mask, b & mask
    if predicate.startswith("s"):
        a, b = (a ^ sign) - sign, (b ^ sign) - sign
    if predicate == "eq":
        return int(a == b)
    if predicate == "ne":
        return int(a != b)
    return int({"lt": a < b, "le": a <= b, "gt": a > b, "ge": a >= b}[predicate[1:]])


def values(width):
    if width <= 8:
        return list(range(1 << width))
    rng = random.Random(618391 + width)
    return list(
        dict.fromkeys(
            [0, 1, 4, 5, 6, (1 << (width - 1)) - 1, 1 << (width - 1), (1 << width) - 1]
            + [rng.getrandbits(width) for _ in range(16)]
        )
    )


def scalar_cases(predicate, width):
    samples, expected = [], []
    cap = 1 if width == 1 else 5
    for i, x in enumerate(values(width)):
        keep = (i * 131 + 37) % (1 << 32)
        samples.append([[x], [keep]])
        expected.append([reference(predicate, cap, x, width), x, keep])
    return samples, expected


def certify_source(tmp_path, tools, text, index_bits=64):
    selected = SimpleNamespace(m2m_python=tools.compiler_python)
    stage, before, after = stage_for(tmp_path, selected, text)
    certify(stage, before, selected, tmp_path, index_bits=index_bits)
    return stage, before, after


@pytest.mark.parametrize("predicate", PREDICATES)
@pytest.mark.parametrize("width", [8, 13, 64])
def test_certified_scalar_comparison_native_matches_integer_reference(tmp_path, fidelity_tools, predicate, width):
    _, before, after = certify_source(tmp_path, fidelity_tools, source(predicate, width))
    samples, expected = scalar_cases(predicate, width)
    compare_expected(tmp_path, fidelity_tools, [("before", before), ("after", after)], samples, expected, 64)


@pytest.mark.parametrize("predicate", ["eq", "slt", "ult"])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_certified_index_comparison_native_uses_signed_actual_width(tmp_path, fidelity_tools, predicate, index_bits):
    _, before, after = certify_source(tmp_path, fidelity_tools, source(predicate, "index"), index_bits)
    samples, expected = scalar_cases(predicate, index_bits)
    compare_expected(tmp_path, fidelity_tools, [("before", before), ("after", after)], samples, expected, index_bits)


@pytest.mark.parametrize("predicate", PREDICATES)
def test_manual_i1_predicate_swap_matches_native_exhaustive_inputs(tmp_path, fidelity_tools, predicate):
    selected = SimpleNamespace(m2m_python=fidelity_tools.compiler_python)
    stage, before, after = manual_i1_stage(tmp_path, selected, predicate)
    certify(stage, before, selected, tmp_path)
    samples, expected = scalar_cases(predicate, 1)
    compare_expected(tmp_path, fidelity_tools, [("before", before), ("after", after)], samples, expected, 64)


def body_cases(predicate, *, reduction):
    samples, expected = [], []
    for i, value in enumerate(values(13)):
        data = [(value + j * 37) % 8192 for j in range(3)]
        initial = [(value ^ (17 + j * 97)) % 8192 for j in range(1 if reduction else 3)]
        keep = (i * 131 + 37) % (1 << 32)
        if reduction:
            out = initial[0]
            for x in data:
                out ^= reference(predicate, 5, x, 13)
            outputs = [out]
        else:
            outputs = [old ^ reference(predicate, 5, x, 13) for x, old in zip(data, initial, strict=True)]
        samples.append([data, initial, [keep]])
        expected.append([*outputs, keep])
    return samples, expected


@pytest.mark.parametrize("predicate", PREDICATES)
def test_captured_comparison_inside_reduction_has_native_defined_chain(tmp_path, fidelity_tools, predicate):
    _, before, after = certify_source(tmp_path, fidelity_tools, body_source(predicate, reduction=True, captured=True))
    samples, expected = body_cases(predicate, reduction=True)
    compare_expected(tmp_path, fidelity_tools, [("before", before), ("after", after)], samples, expected, 64)


@pytest.mark.parametrize("predicate", ["slt", "uge"])
def test_depth_two_comparison_capture_matches_native_reference(tmp_path, fidelity_tools, predicate):
    _, before, after = certify_source(tmp_path, fidelity_tools, body_source(predicate, captured=True, nested=True))
    samples, expected = body_cases(predicate, reduction=False)
    compare_expected(tmp_path, fidelity_tools, [("before", before), ("after", after)], samples, expected, 64)


@pytest.mark.parametrize("fault,x", [("predicate", 0), ("input", 7), ("bits", 6), ("keep", 7)])
def test_rejected_comparison_mutations_have_observable_native_values(tmp_path, fidelity_tools, fault, x):
    stage, before, after = certify_source(tmp_path, fidelity_tools, source())
    mutate(after, SimpleNamespace(m2m_python=fidelity_tools.compiler_python), fault)
    result = check_typed_stage(stage, m2m_python=fidelity_tools.compiler_python)
    assert result["verdict"] == "abstracted", result.get("reason")
    samples = [[[x], [37]]]
    correct = [[reference("slt", 5, x, 13), x, 37]]
    wrong = {
        "predicate": [[reference("slt", x, 5, 13), x, 37]],
        "input": [[0, x, 37]],
        "bits": [[reference("sgt", x, 6, 13), x, 37]],
        "keep": [[reference("slt", 5, x, 13), x, 0]],
    }[fault]
    assert correct != wrong
    compare_expected(tmp_path, fidelity_tools, [("before", before)], samples, correct, 64)
    compare_expected(tmp_path, fidelity_tools, [("after", after)], samples, wrong, 64)


def test_reflexive_comparison_fold_matches_native_but_remains_a_separate_gap(tmp_path, fidelity_tools):
    selected = SimpleNamespace(m2m_python=fidelity_tools.compiler_python)
    text = source().replace("cmpi slt, %cap, %x", "cmpi slt, %x, %x")
    stage, before, after = stage_for(tmp_path, selected, text)
    result = check_definedness(before, "forward", tmp_path / "defined", m2m_python=selected.m2m_python)
    assert result["verdict"] == "verified", result.get("reason")
    result = check_typed_stage(stage, m2m_python=selected.m2m_python)
    assert result["verdict"] == "abstracted", result.get("reason")
    samples, _ = scalar_cases("slt", 13)
    expected = [[0, s[0][0], s[1][0]] for s in samples]
    compare_expected(tmp_path, fidelity_tools, [("before", before), ("after", after)], samples, expected, 64)
