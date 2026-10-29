"""Independent native values for defined dead-empty and insertion-alias boundaries."""

from types import SimpleNamespace

import pytest
from test_lean_integer_fold import canonicalize
from test_lean_integer_idempotent import certify
from test_lean_minmax_commutative_native import compare_expected
from test_lean_tensor_empty_dead import dead_source, mutate, source

from merlin.verify.lean.check import check_typed_stage


def samples_and_expected(*, depth=1, keep_empty=False):
    data, initial = [7, 4096, 8191], [3, 5, 17]
    samples = [[data, initial, [271]]]
    values = [x ^ old for x, old in zip(data, initial, strict=True)]
    if depth == 1:
        values = [(v + x * (2 if keep_empty else 1)) % 8192 for v, x in zip(values, data, strict=True)]
    return samples, [[*values, 271]]


@pytest.mark.parametrize("depth,keep_empty", [(1, False), (2, False), (1, True)])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_actual_nested_empty_cleanup_matches_independent_values(
    tmp_path, fidelity_tools, depth, keep_empty, index_bits
):
    tools = SimpleNamespace(m2m_python=fidelity_tools.compiler_python)
    stage, before, after = canonicalize(tmp_path, tools, source(depth=depth, keep_empty=keep_empty))
    certify(stage, before, tools, tmp_path, index_bits=index_bits)
    samples, expected = samples_and_expected(depth=depth, keep_empty=keep_empty)
    compare_expected(tmp_path, fidelity_tools, [("before", before), ("after", after)], samples, expected, index_bits)


@pytest.mark.parametrize("shape", [(), (2, 0, 3), (2, 3)])
def test_dead_poison_tensor_never_changes_live_scalar_bits(tmp_path, fidelity_tools, shape):
    tools = SimpleNamespace(m2m_python=fidelity_tools.compiler_python)
    stage, before, after = canonicalize(tmp_path, tools, dead_source(shape))
    certify(stage, before, tools, tmp_path)
    pool = [0, 1, 4095, 4096, 8191]
    compare_expected(
        tmp_path, fidelity_tools, [("before", before), ("after", after)], [[[x]] for x in pool], [[x] for x in pool], 64
    )


@pytest.mark.parametrize("fault", ["payload", "return"])
def test_rejected_mutation_has_different_native_values(tmp_path, fidelity_tools, fault):
    tools = SimpleNamespace(m2m_python=fidelity_tools.compiler_python)
    stage, before, after = canonicalize(tmp_path, tools, source())
    certify(stage, before, tools, tmp_path)
    mutate(after, tools, fault)
    result = check_typed_stage(stage, m2m_python=tools.m2m_python)
    assert result["verdict"] == "abstracted", result.get("reason")
    samples, expected = samples_and_expected()
    corrupt = [[*(samples[0][0] if fault == "payload" else samples[0][1]), 271]]
    assert expected != corrupt
    compare_expected(tmp_path / "source", fidelity_tools, [("before", before)], samples, expected, 64)
    compare_expected(tmp_path / "target", fidelity_tools, [("after", after)], samples, corrupt, 64)
