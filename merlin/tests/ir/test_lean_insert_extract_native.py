"""Independent exact native references for verified tensor insertion selection."""

from math import prod
from types import SimpleNamespace

import pytest
from test_lean_differential import _floats
from test_lean_insert_extract import body_source, mutate, source
from test_lean_integer_fold import canonicalize
from test_lean_integer_idempotent import certify
from test_lean_minmax_commutative_native import compare_expected

from merlin.verify.lean.check import check_typed_stage


def cases(shape, coords, escape):
    size = prod(shape)
    offset = 0
    for coordinate, extent in zip(coords, shape, strict=True):
        offset = offset * extent + coordinate
    samples, expected = [], []
    for payload in [0, 1, 4095, 4096, 8191]:
        initial = [(payload + 17 * i + 37) % 8192 for i in range(size)]
        samples.append([[payload], initial, [271]])
        updated = initial.copy()
        updated[offset] = payload
        expected.append([payload, 271, *(updated if escape else [])])
    return samples, expected


@pytest.mark.parametrize("shape,coords", [((), ()), ((1,), (0,)), ((3,), (1,)), ((2, 3), (1, 2))])
@pytest.mark.parametrize("escape", [False, True])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_insert_extract_matches_independent_point_update(tmp_path, fidelity_tools, shape, coords, escape, index_bits):
    tools = SimpleNamespace(m2m_python=fidelity_tools.compiler_python)
    stage, before, after = canonicalize(tmp_path, tools, source(shape, coords, escape=escape))
    certify(stage, before, tools, tmp_path, index_bits=index_bits)
    samples, expected = cases(shape, coords, escape)
    compare_expected(tmp_path, fidelity_tools, [("before", before), ("after", after)], samples, expected, index_bits)


@pytest.mark.parametrize("shape,coords", [((3,), (1,)), ((2, 3), (1, 2))])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_computed_closed_coordinates_match_native(tmp_path, fidelity_tools, shape, coords, index_bits):
    tools = SimpleNamespace(m2m_python=fidelity_tools.compiler_python)
    stage, before, after = canonicalize(tmp_path, tools, source(shape, coords, computed=True, escape=True))
    certify(stage, before, tools, tmp_path, index_bits=index_bits)
    samples, expected = cases(shape, coords, True)
    compare_expected(tmp_path, fidelity_tools, [("before", before), ("after", after)], samples, expected, index_bits)


@pytest.mark.parametrize("element", ["f32", "f64", "index"])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_exact_float_payloads_and_actual_width_index_payloads(tmp_path, fidelity_tools, element, index_bits):
    tools = SimpleNamespace(m2m_python=fidelity_tools.compiler_python)
    stage, before, after = canonicalize(tmp_path, tools, source(element=element, escape=True))
    certify(stage, before, tools, tmp_path, index_bits=index_bits)
    if element.startswith("f"):
        width = int(element[1:])
        pool = _floats(width) + [0x7FC01234 if width == 32 else 0x7FF8000000001234]
    else:
        pool = [0, 1, (1 << (index_bits - 1)) - 1, 1 << (index_bits - 1), (1 << index_bits) - 1]
    samples = [[[x], [pool[(i + j + 1) % len(pool)] for j in range(3)], [271]] for i, x in enumerate(pool)]
    expected = [[s[0][0], 271, s[1][0], s[0][0], s[1][2]] for s in samples]
    compare_expected(tmp_path, fidelity_tools, [("before", before), ("after", after)], samples, expected, index_bits)


@pytest.mark.parametrize("depth", [1, 2])
def test_nested_alias_uses_fresh_native_destination(tmp_path, fidelity_tools, depth):
    tools = SimpleNamespace(m2m_python=fidelity_tools.compiler_python)
    stage, before, after = canonicalize(tmp_path, tools, body_source(depth=depth))
    certify(stage, before, tools, tmp_path)
    data, initial = [7, 4096, 8191], [3, 5, 17]
    samples = [[data, initial, [271]]]
    values = data if depth == 1 else [(a + b) % 8192 for a, b in zip(data, initial, strict=True)]
    expected = [[*(a ^ b for a, b in zip(values, initial, strict=True)), 271]]
    compare_expected(tmp_path, fidelity_tools, [("before", before), ("after", after)], samples, expected, 64)


def test_dirty_rankzero_destination_is_completely_overwritten(tmp_path, fidelity_tools):
    tools = SimpleNamespace(m2m_python=fidelity_tools.compiler_python)
    stage, before, after = canonicalize(tmp_path, tools, source((), (), dirty=True, escape=True))
    certify(stage, before, tools, tmp_path)
    samples, expected = cases((), (), True)
    compare_expected(tmp_path, fidelity_tools, [("before", before), ("after", after)], samples, expected, 64)


@pytest.mark.parametrize("fault", ["selected", "keep", "tensor", "coordinate", "payload"])
def test_rejected_wrong_point_has_independent_observable_native_values(tmp_path, fidelity_tools, fault):
    tools = SimpleNamespace(m2m_python=fidelity_tools.compiler_python)
    stage, before, after = canonicalize(tmp_path, tools, source(escape=True))
    certify(stage, before, tools, tmp_path)
    changed = before if fault in {"coordinate", "payload"} else after
    mutate(changed, tools, fault)
    result = check_typed_stage(stage, m2m_python=tools.m2m_python)
    assert result["verdict"] == "abstracted", result.get("reason")
    samples = [[[4096], [11, 13, 17], [271]]]
    correct = [[4096, 271, 11, 4096, 17]]
    corrupt = {
        "selected": [[0, 271, 11, 4096, 17]],
        "keep": [[4096, 0, 11, 4096, 17]],
        "tensor": [[4096, 271, 11, 13, 17]],
        "coordinate": [[11, 271, 11, 4096, 17]],
        "payload": [[7, 271, 11, 7, 17]],
    }[fault]
    assert correct != corrupt
    for side, path in [("before", before), ("after", after)]:
        compare_expected(
            tmp_path / side, fidelity_tools, [(side, path)], samples, corrupt if path == changed else correct, 64
        )
