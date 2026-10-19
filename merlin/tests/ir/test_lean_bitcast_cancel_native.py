"""Raw-bit native cancellation evidence; exclude this module from binding-only CI."""

from types import SimpleNamespace

import pytest
from test_lean_bitcast_cancel import body_source, mutate, source, stage_for
from test_lean_defined_conversions import certify
from test_lean_minmax_commutative_native import compare_expected
from test_lean_scalar_bitcast_native import values

from merlin.verify.lean.check import check_definedness, check_support, check_typed_stage

PAIRS = [("i32", "f32"), ("f32", "i32"), ("i64", "f64"), ("f64", "i64")]


def prepared(tmp_path, tools, text):
    selected = SimpleNamespace(m2m_python=tools.compiler_python)
    stage, before, after = stage_for(tmp_path, selected, text)
    certify(stage, before, selected, tmp_path)
    return stage, before, after


def scalar_samples(width, *, escape=False):
    samples, expected = [], []
    mask = (1 << width) - 1
    for n, value in enumerate(values(width)):
        keep = 183771 + n
        samples.append([[value], [value ^ mask], [keep]])
        # These are raw scalar ABI words, including float NaNs and both signed zeroes.
        expected.append([value, *([value] if escape else []), value, keep])
    return samples, expected


@pytest.mark.parametrize("width", [1, 8, 13, 32, 64])
def test_certified_integer_identity_native_preserves_all_observed_bits(tmp_path, fidelity_tools, width):
    ty = f"i{width}"
    _, before, after = prepared(tmp_path, fidelity_tools, source(ty, ty, identity=True))
    samples, expected = scalar_samples(width)
    compare_expected(tmp_path, fidelity_tools, [("before", before), ("after", after)], samples, expected, 64)


@pytest.mark.parametrize("src,mid", PAIRS)
@pytest.mark.parametrize("escape", [False, True])
def test_certified_cross_kind_roundtrip_native_preserves_nan_payloads(tmp_path, fidelity_tools, src, mid, escape):
    _, before, after = prepared(tmp_path, fidelity_tools, source(src, mid, escape=escape))
    samples, expected = scalar_samples(int(src[1:]), escape=escape)
    compare_expected(tmp_path, fidelity_tools, [("before", before), ("after", after)], samples, expected, 64)


@pytest.mark.parametrize("src,mid", PAIRS)
@pytest.mark.parametrize("nested", [False, True])
def test_body_roundtrip_native_preserves_captured_payload_bits(tmp_path, fidelity_tools, src, mid, nested):
    _, before, after = prepared(tmp_path, fidelity_tools, body_source(src, mid, nested=nested))
    data = values(int(src[1:]))
    samples, expected = [], []
    for n, value in enumerate(data):
        inputs = [value, data[(n + 1) % len(data)], data[(n + 2) % len(data)]]
        initial = [data[(n + 3 + j) % len(data)] for j in range(3)]
        samples.append([inputs, initial, [183771 + n]])
        output = inputs if nested else [x ^ old for x, old in zip(inputs, initial, strict=True)]
        expected.append([*output, 183771 + n])
    compare_expected(tmp_path, fidelity_tools, [("before", before), ("after", after)], samples, expected, 64)


@pytest.mark.parametrize("fault", ["result", "keep", "intermediate"])
def test_rejected_bitcast_mutation_changes_independent_native_reference(tmp_path, fidelity_tools, fault):
    stage, before, after = prepared(tmp_path, fidelity_tools, source(escape=True))
    mutate(after, SimpleNamespace(m2m_python=fidelity_tools.compiler_python), fault)
    assert check_typed_stage(stage, m2m_python=fidelity_tools.compiler_python)["verdict"] == "abstracted"
    samples = [[[0x7FC12345], [0xFFC54321], [37]]]
    correct = [[0x7FC12345, 0x7FC12345, 0x7FC12345, 37]]
    wrong = {
        "result": [[0xFFC54321, 0x7FC12345, 0x7FC12345, 37]],
        "intermediate": [[0x7FC12345, 0xFFC54321, 0x7FC12345, 37]],
        "keep": [[0x7FC12345, 0x7FC12345, 0x7FC12345, 0]],
    }[fault]
    assert correct != wrong
    compare_expected(tmp_path, fidelity_tools, [("before", before)], samples, correct, 64)
    compare_expected(tmp_path, fidelity_tools, [("after", after)], samples, wrong, 64)


@pytest.mark.parametrize("src,mid", [("i32", "f32"), ("f32", "i32")])
def test_pure_identity_generic_native_matches_while_alias_rule_is_missing(tmp_path, fidelity_tools, src, mid):
    selected = SimpleNamespace(m2m_python=fidelity_tools.compiler_python)
    stage, before, after = stage_for(tmp_path, selected, body_source(src, mid, accumulate=False))
    assert "linalg.generic" not in after.read_text()
    for checker, name in [(check_support, "support"), (check_definedness, "defined")]:
        result = checker(before, "forward", tmp_path / name, m2m_python=selected.m2m_python)
        assert result["verdict"] == "verified", result.get("reason")
    assert check_typed_stage(stage, m2m_python=selected.m2m_python)["verdict"] == "abstracted"
    data = values(32)
    samples = [[[x, x ^ 0xFFFFFFFF, x], [17, 37, 59], [183771 + n]] for n, x in enumerate(data)]
    expected = [[*sample[0], sample[2][0]] for sample in samples]
    compare_expected(tmp_path, fidelity_tools, [("before", before), ("after", after)], samples, expected, 64)
