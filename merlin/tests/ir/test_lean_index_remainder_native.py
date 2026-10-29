"""Native initialized remainder/bounds controls; exclude from binding-only CI."""

from types import SimpleNamespace

import pytest
from test_lean_defined_conversions import certify
from test_lean_index_fold_native import compare_expected
from test_lean_index_remainder import body_source, captured_source, mutate, source, stage_for

from merlin.verify.lean.check import check_definedness, check_typed_stage


def data_for(n, size=8):
    return [(937 * n + 53 * i + 11) % 8192 for i in range(size)]


@pytest.mark.parametrize("mode,divisor,size", [("constant", 3, 3), ("constant", 7, 7), ("widened", 3, 4)])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_certified_remainder_reads_match_independent_unsigned_reference(
    tmp_path, fidelity_tools, mode, divisor, size, index_bits
):
    tools = SimpleNamespace(m2m_python=fidelity_tools.compiler_python)
    stage, before, after = stage_for(tmp_path, tools, source(mode, divisor, size=size))
    certify(stage, before, tools, tmp_path, index_bits=index_bits)
    mask, sign = (1 << index_bits) - 1, 1 << (index_bits - 1)
    indices = [0, 1, 2, 3, 7, sign, mask]
    samples = [[data_for(n, size), [idx], [n % 4], [(131 * n) % (1 << 32)]] for n, idx in enumerate(indices)]
    expected = [[s[0][s[1][0] % (s[2][0] + 1 if mode == "widened" else divisor)], s[3][0]] for s in samples]
    compare_expected(tmp_path, fidelity_tools, [("before", before), ("after", after)], samples, expected, index_bits)


@pytest.mark.parametrize("index_bits", [32, 64])
def test_refused_different_safe_remainder_read_is_observable(tmp_path, fidelity_tools, index_bits):
    tools = SimpleNamespace(m2m_python=fidelity_tools.compiler_python)
    stage, before, after = stage_for(tmp_path, tools, source())
    certify(stage, before, tools, tmp_path, index_bits=index_bits)
    mutate(after, tools, "constant_read")
    result = check_typed_stage(stage, m2m_python=tools.m2m_python, index_bits=index_bits)
    assert result["verdict"] == "abstracted", result.get("reason")
    result = check_definedness(
        after, "forward", tmp_path / "target-defined", m2m_python=tools.m2m_python, index_bits=index_bits
    )
    assert result["verdict"] == "verified", result.get("reason")
    samples = [[data_for(1), [5], [0], [37]]]
    compare_expected(tmp_path, fidelity_tools, [("before", before)], samples, [[data_for(1)[2], 37]], index_bits)
    compare_expected(tmp_path, fidelity_tools, [("after", after)], samples, [[data_for(1)[0], 37]], index_bits)


@pytest.mark.parametrize("index_bits", [32, 64])
def test_refused_signed_remainder_changes_high_bit_scalar_results(tmp_path, fidelity_tools, index_bits):
    tools = SimpleNamespace(m2m_python=fidelity_tools.compiler_python)
    stage, before, after = stage_for(tmp_path, tools, source(extract=False))
    certify(stage, before, tools, tmp_path, index_bits=index_bits)
    mutate(after, tools, "signedness")
    result = check_typed_stage(stage, m2m_python=tools.m2m_python, index_bits=index_bits)
    assert result["verdict"] == "abstracted", result.get("reason")
    mask, sign = (1 << index_bits) - 1, 1 << (index_bits - 1)
    samples = [[data_for(n), [idx], [0], [n]] for n, idx in enumerate([0, sign - 1, sign, mask])]
    original = [[s[1][0] % 3, s[3][0]] for s in samples]
    signed = []
    for sample in samples:
        integer = sample[1][0] if sample[1][0] < sign else sample[1][0] - (1 << index_bits)
        rem = (abs(integer) % 3) * (-1 if integer < 0 else 1)
        signed.append([rem & mask, sample[3][0]])
    assert original != signed
    compare_expected(tmp_path, fidelity_tools, [("before", before)], samples, original, index_bits)
    compare_expected(tmp_path, fidelity_tools, [("after", after)], samples, signed, index_bits)


@pytest.mark.parametrize("index_bits", [32, 64])
def test_captured_remainder_bound_matches_native_execution(tmp_path, fidelity_tools, index_bits):
    tools = SimpleNamespace(m2m_python=fidelity_tools.compiler_python)
    stage, before, after = stage_for(tmp_path, tools, captured_source())
    certify(stage, before, tools, tmp_path, index_bits=index_bits)
    mask = (1 << index_bits) - 1
    data, xs = data_for(3, 3), [0, 8191, 4096]
    samples = [[data, [mask], [0], xs, [19, 23, 29], [37]]]
    expected = [[*((data[mask % 3] + x) % 8192 for x in xs), 37]]
    compare_expected(tmp_path, fidelity_tools, [("before", before), ("after", after)], samples, expected, index_bits)


@pytest.mark.parametrize("index_bits", [32, 64])
def test_remainder_inside_generic_matches_native_execution(tmp_path, fidelity_tools, index_bits):
    tools = SimpleNamespace(m2m_python=fidelity_tools.compiler_python)
    stage, before, after = stage_for(tmp_path, tools, body_source())
    certify(stage, before, tools, tmp_path, index_bits=index_bits)
    indices = [0, 1 << (index_bits - 1), (1 << index_bits) - 1]
    data = data_for(2, 3)
    samples = [[data, indices, [19, 23, 29], [37]]]
    expected = [[*(data[x % 3] for x in indices), 37]]
    compare_expected(tmp_path, fidelity_tools, [("before", before), ("after", after)], samples, expected, index_bits)
