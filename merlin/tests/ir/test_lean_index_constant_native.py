"""Exact wrapped index spelling controls; exclude this module from binding-only CI."""

import json
from types import SimpleNamespace

import pytest
from test_lean_defined_conversions import certify
from test_lean_index_fold import mutate, source, stage_for
from test_lean_index_fold_native import compare_expected, reference

from merlin.verify.lean import differential as diff
from merlin.verify.lean.check import check_typed_stage

CASES = [(name, False, False) for name in ["addi", "andi", "ori", "xori"]] + [
    (name, True, capture) for name in ["addi", "ori", "xori"] for capture in [False, True]
]


@pytest.mark.parametrize("name,escape,capture", CASES)
@pytest.mark.parametrize("index_bits", [32, 64])
def test_certified_wrapped_spelling_matches_independent_native_reference(
    tmp_path, fidelity_tools, name, index_bits, escape, capture
):
    tools = SimpleNamespace(m2m_python=fidelity_tools.compiler_python)
    stage, before, after = stage_for(tmp_path, tools, source(name, -(1 << 63), 3, escape=escape, capture=capture))
    certify(stage, before, tools, tmp_path, index_bits=index_bits)
    mask = (1 << index_bits) - 1
    samples = [[[0], [0]], [[mask], [(1 << 32) - 1]]]
    folded = reference(name, -(1 << 63), 3, index_bits)
    escaped = (-(1 << 63)) & mask
    if capture:
        samples = [s + [[0, 1 << (index_bits - 1), mask], [13, 17, 19]] for s in samples]
        expected = [[*((x + folded) & mask for x in s[2]), escaped, s[1][0]] for s in samples]
    else:
        expected = [[folded] + ([escaped] if escape else []) + [s[1][0]] for s in samples]
    compare_expected(tmp_path, fidelity_tools, [("before", before), ("after", after)], samples, expected, index_bits)


@pytest.mark.parametrize("index_bits", [32, 64])
def test_corrupted_wrapped_literal_is_observable_natively(tmp_path, fidelity_tools, index_bits):
    tools = SimpleNamespace(m2m_python=fidelity_tools.compiler_python)
    stage, before, after = stage_for(tmp_path, tools, source("addi", -(1 << 63), 3, escape=True))
    certify(stage, before, tools, tmp_path, index_bits=index_bits)
    mutate(after, tools, "value")
    result = check_typed_stage(stage, m2m_python=tools.m2m_python, index_bits=index_bits)
    assert result["verdict"] == "abstracted", result.get("reason")
    # Corrupting either live constant changes the returned row. Execute both programs only
    # after the MLIR verifier accepted the mutation; source-definedness remains separately checked.
    samples = [[[0], [37]]]
    outputs = []
    for label, path in [("before", before), ("after", after)]:
        diff.compare(
            path.read_text(), samples, tmp_path / label, fidelity_tools, function="forward", index_bits=index_bits
        )
        outputs.append(json.loads((tmp_path / label / "native-results.json").read_text())["results"])
    assert outputs[0] != outputs[1]
