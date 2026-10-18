"""Native scalar index comparison controls; exclude this module from binding-only CI."""

import json
from types import SimpleNamespace

import pytest
from test_lean_defined_conversions import certify
from test_lean_index_compare import PREDICATES, mutate, source, stage_for

from merlin.verify.lean import differential as diff
from merlin.verify.lean.check import check_typed_stage


def reference(predicate, a, b, width):
    modulus = 1 << width
    a, b = a % modulus, b % modulus
    signed_a = a - modulus if a & (modulus >> 1) else a
    signed_b = b - modulus if b & (modulus >> 1) else b
    return int(
        {
            "eq": a == b,
            "ne": a != b,
            "slt": signed_a < signed_b,
            "sle": signed_a <= signed_b,
            "sgt": signed_a > signed_b,
            "sge": signed_a >= signed_b,
            "ult": a < b,
            "ule": a <= b,
            "ugt": a > b,
            "uge": a >= b,
        }[predicate]
    )


def compare_expected(tmp_path, tools, paths, samples, expected, index_bits):
    for name, path in paths:
        work = tmp_path / name
        diff.compare(path.read_text(), samples, work, tools, function="forward", index_bits=index_bits)
        actual = json.loads((work / "native-results.json").read_text())["results"]
        assert actual == [[str(value) for value in row] for row in expected]


@pytest.mark.parametrize("predicate", PREDICATES)
@pytest.mark.parametrize("index_bits", [32, 64])
def test_certified_index_comparisons_match_independent_signedness_reference(
    tmp_path, fidelity_tools, predicate, index_bits
):
    tools = SimpleNamespace(m2m_python=fidelity_tools.compiler_python)
    stage, before, after = stage_for(tmp_path, tools, source(predicate, -7, 5, escape=True))
    certify(stage, before, tools, tmp_path, index_bits=index_bits)
    samples = [[[0], [0], [0]], [[1], [(1 << index_bits) - 1], [(1 << 32) - 1]]]
    result = reference(predicate, -7, 5, index_bits)
    expected = [[result, (1 << index_bits) - 7, s[2][0]] for s in samples]
    compare_expected(tmp_path, fidelity_tools, [("before", before), ("after", after)], samples, expected, index_bits)


@pytest.mark.parametrize("fault", ["value", "predicate"])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_refused_bool_or_predicate_mutations_are_observable_natively(tmp_path, fidelity_tools, fault, index_bits):
    tools = SimpleNamespace(m2m_python=fidelity_tools.compiler_python)
    stage, before, after = stage_for(tmp_path, tools, source())
    certify(stage, before, tools, tmp_path, index_bits=index_bits)
    mutated = before if fault == "predicate" else after
    mutate(mutated, tools, fault)
    result = check_typed_stage(stage, m2m_python=tools.m2m_python, index_bits=index_bits)
    assert result["verdict"] == "abstracted", result.get("reason")
    samples = [[[1], [0], [37]]]
    before_result = [[1 if fault == "predicate" else 0, 37]]
    after_result = [[1 if fault == "value" else 0, 37]]
    compare_expected(tmp_path, fidelity_tools, [("before", before)], samples, before_result, index_bits)
    compare_expected(tmp_path, fidelity_tools, [("after", after)], samples, after_result, index_bits)
