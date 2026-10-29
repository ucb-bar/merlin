"""Native partial-cast boundaries; exclude this module from binding-only CI."""

from types import SimpleNamespace

import pytest
from test_lean_integer_cast import mutate
from test_lean_integer_cast_nested import certify, nested_source
from test_lean_integer_cast_nested import source as body_source
from test_lean_integer_cast_nested_native import inputs, samples_and_expected
from test_lean_integer_cast_partial import captured_source, source
from test_lean_integer_fold import canonicalize
from test_lean_minmax_commutative_native import compare_expected

from merlin.verify.lean.check import check_typed_stage

# Internal widths above64 retain binding coverage; the separately reviewed native adapter
# must admit arbitrary internal integers before those programs can add native evidence.
WIDTHS = [(1, 32, 8), (8, 32, 13), (13, 64, 32), (8, 32, 1), (13, 32, 8), (32, 64, 13)]


def reference(value, kind, a, c):
    value &= (1 << a) - 1
    if kind == "extsi" and value & (1 << (a - 1)):
        value -= 1 << a
    return value & ((1 << c) - 1)


def stage_for(tmp_path, tools, text, index_bits=64):
    selected = SimpleNamespace(m2m_python=tools.compiler_python)
    stage, before, after = canonicalize(tmp_path, selected, text)
    certify(stage, before, selected, tmp_path, index_bits=index_bits)
    return stage, before, after


@pytest.mark.parametrize("kind", ["extui", "extsi"])
@pytest.mark.parametrize("widths", WIDTHS)
@pytest.mark.parametrize("escape", [False, True])
def test_certified_partial_scalar_casts_match_exact_integer_reference(tmp_path, fidelity_tools, kind, widths, escape):
    a, b, c = widths
    _, before, after = stage_for(tmp_path, fidelity_tools, source(kind, widths, escape=escape))
    samples, expected = [], []
    for value in inputs(a):
        keep = (value ^ 37) & ((1 << a) - 1)
        samples.append([[value], [keep]])
        expected.append([reference(value, kind, a, c), keep] + ([reference(value, kind, a, b)] if escape else []))
    compare_expected(tmp_path, fidelity_tools, [("before", before), ("after", after)], samples, expected, 64)


@pytest.mark.parametrize("kind", ["extui", "extsi"])
@pytest.mark.parametrize("widths", [(8, 32, 13), (13, 32, 8)])
@pytest.mark.parametrize("escape,reduction", [(True, False), (False, True)])
def test_body_partial_casts_preserve_accumulators_and_escaping_values(
    tmp_path, fidelity_tools, kind, widths, escape, reduction
):
    _, before, after = stage_for(
        tmp_path, fidelity_tools, body_source(kind, "trunci", widths, escape=escape, reduction=reduction)
    )
    samples, expected = samples_and_expected(kind, "trunci", widths, escape=escape, reduction=reduction)
    expected = [[int(value) for value in row] for row in expected]
    compare_expected(tmp_path, fidelity_tools, [("before", before), ("after", after)], samples, expected, 64)


@pytest.mark.parametrize("kind", ["extui", "extsi"])
def test_partial_casts_in_depth_two_generic_match_native(tmp_path, fidelity_tools, kind):
    widths = (8, 32, 13)
    _, before, after = stage_for(tmp_path, fidelity_tools, nested_source(kind, "trunci", widths))
    samples, expected = samples_and_expected(kind, "trunci", widths)
    expected = [[int(value) for value in row] for row in expected]
    compare_expected(tmp_path, fidelity_tools, [("before", before), ("after", after)], samples, expected, 64)


@pytest.mark.parametrize("kind", ["extui", "extsi"])
def test_captured_root_value_has_independent_native_reference(tmp_path, fidelity_tools, kind):
    _, before, after = stage_for(tmp_path, fidelity_tools, captured_source(kind))
    samples = [[[0, 127, 255], [1, 17, 4096], [37], [128]], [[128, 1, 255], [31, 8191, 7], [131], [255]]]
    expected = [
        [
            *(
                (reference(x, kind, 8, 13) + reference(s[3][0], kind, 8, 13) + old) % 8192
                for x, old in zip(s[0], s[1], strict=True)
            ),
            s[2][0],
        ]
        for s in samples
    ]
    compare_expected(tmp_path, fidelity_tools, [("before", before), ("after", after)], samples, expected, 64)


@pytest.mark.parametrize("fault", ["input", "signedness"])
def test_rejected_partial_cast_mutations_change_initialized_native_values(tmp_path, fidelity_tools, fault):
    stage, before, after = stage_for(tmp_path, fidelity_tools, source())
    mutate(after, SimpleNamespace(m2m_python=fidelity_tools.compiler_python), fault)
    result = check_typed_stage(stage, m2m_python=fidelity_tools.compiler_python)
    assert result["verdict"] == "abstracted", result.get("reason")
    samples = [[[128], [37]]]
    correct = [[128, 37]]
    corrupt = [[37, 37]] if fault == "input" else [[8064, 37]]
    compare_expected(tmp_path, fidelity_tools, [("before", before)], samples, correct, 64)
    compare_expected(tmp_path, fidelity_tools, [("after", after)], samples, corrupt, 64)
