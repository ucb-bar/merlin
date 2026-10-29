"""Independent exact bit references for whole-function identity loop elimination."""

from math import prod
from types import SimpleNamespace

import pytest
from test_lean_differential import _floats
from test_lean_identity_return import mutate, source
from test_lean_integer_fold import canonicalize
from test_lean_integer_idempotent import certify
from test_lean_minmax_commutative_native import compare_expected

from merlin.verify.lean.check import check_typed_stage


def values(shape, element, index_bits):
    width = index_bits if element == "index" else int(element[1:])
    if element.startswith("f"):
        pool = _floats(width) + [0x7FC01234 if width == 32 else 0x7FF8000000001234]
    else:
        pool = [0, 1, (1 << (width - 1)) - 1, 1 << (width - 1), (1 << width) - 1]
    samples = [
        [
            [pool[(n + i) % len(pool)] for i in range(prod(shape))],
            [pool[(n + i + 2) % len(pool)] for i in range(prod(shape))],
            [37],
        ]
        for n in range(len(pool))
    ]
    expected = [[*s[0], 37, *s[0], *s[1], *s[0]] for s in samples]
    return samples, expected


@pytest.mark.parametrize("shape", [(), (0,), (1,), (2, 3), (1, 2, 1)])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_identity_return_preserves_all_observable_integer_bits(tmp_path, fidelity_tools, shape, index_bits):
    tools = SimpleNamespace(m2m_python=fidelity_tools.compiler_python)
    stage, before, after = canonicalize(tmp_path, tools, source(shape))
    certify(stage, before, tools, tmp_path, index_bits=index_bits)
    samples, expected = values(shape, "i13", index_bits)
    compare_expected(tmp_path, fidelity_tools, [("before", before), ("after", after)], samples, expected, index_bits)


@pytest.mark.parametrize("element", ["i1", "i64", "f32", "f64", "index"])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_identity_return_preserves_float_payloads_and_actual_index_width(tmp_path, fidelity_tools, element, index_bits):
    tools = SimpleNamespace(m2m_python=fidelity_tools.compiler_python)
    stage, before, after = canonicalize(tmp_path, tools, source(element=element))
    certify(stage, before, tools, tmp_path, index_bits=index_bits)
    samples, expected = values((2, 2), element, index_bits)
    compare_expected(tmp_path, fidelity_tools, [("before", before), ("after", after)], samples, expected, index_bits)


@pytest.mark.parametrize("fault", ["return", "yield", "transpose"])
def test_rejected_wrong_values_have_independent_native_witnesses(tmp_path, fidelity_tools, fault):
    tools = SimpleNamespace(m2m_python=fidelity_tools.compiler_python)
    stage, before, after = canonicalize(tmp_path, tools, source())
    certify(stage, before, tools, tmp_path)
    changed = after if fault == "return" else before
    mutate(changed, tools, fault)
    result = check_typed_stage(stage, m2m_python=tools.m2m_python)
    assert result["verdict"] == "abstracted", result.get("reason")
    data, initial = [1, 7, 4096, 8191], [9, 11, 13, 17]
    samples = [[data, initial, [37]]]
    correct = [[*data, 37, *data, *initial, *data]]
    selected = initial if fault in {"return", "yield"} else [data[i] for i in [0, 2, 1, 3]]
    last = data if fault == "return" else selected
    corrupt = [[*selected, 37, *data, *initial, *last]]
    assert correct != corrupt
    for side, path in [("before", before), ("after", after)]:
        compare_expected(
            tmp_path / side, fidelity_tools, [(side, path)], samples, corrupt if path == changed else correct, 64
        )
