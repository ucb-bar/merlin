"""Compare mixed real-pass boundaries with independent bit arithmetic and tensor references."""

import json
from types import SimpleNamespace

import pytest
from test_lean_shared_normalize import CASTS, MUTATE, certify

from merlin.verify.lean import differential as diff
from merlin.verify.lean.check import check_typed_stage


def _cast(value, kind, src, dst):
    if kind == "extsi" and value >= 1 << (src - 1):
        value -= 1 << src
    return value % (1 << dst)


def _samples(casts, empty=False):
    width = casts[2]
    count, body_count = (0, 0) if empty else (24, 3)
    integers = [0, 1, (1 << width) - 1, 1 << (width - 1)]
    return [
        [
            [integer],
            [(i * 137 + 0xFFFFFFF0) % (1 << 32) for i in range(count)],
            [value for value in [0, 0xFFFFFFFF, 8][:body_count]],
            [10 if n % 2 == 0 else 0xFFFFFFFD],
            [i + 100 for i in range(count)],
            [41 + i for i in range(body_count)],
            [i + 200 for i in range(count)],
        ]
        for n, integer in enumerate(integers)
    ]


def _expected(sample, casts, *, compact=False, constant=7, wrong_cast=False):
    first, second, a, b, c = casts
    x, shape, body, keep, _, _, _ = sample
    if wrong_cast:
        cast = _cast(x[0], "extsi", a, c)
    else:
        cast = _cast(_cast(x[0], first, a, b), second, b, c)
    signed_keep = keep[0] if keep[0] < 1 << 31 else keep[0] - (1 << 32)
    mapped = []
    for value in body:
        signed = value if value < 1 << 31 else value - (1 << 32)
        extra = keep[0] if compact else 0
        mapped.append((value + constant + extra) % (1 << 32) if signed < signed_keep else value)
    return [str(cast), *map(str, shape), *map(str, mapped), *map(str, shape * 2), str(keep[0])]


@pytest.mark.parametrize("casts", CASTS)
@pytest.mark.parametrize("empty", [False, True])
def test_mixed_boundary_matches_native_reference(tmp_path, fidelity_tools, casts, empty):
    tools = SimpleNamespace(m2m_python=fidelity_tools.compiler_python)
    _, before, after = certify(tmp_path, tools, casts, empty=empty)
    samples = _samples(casts, empty)
    expected = [_expected(sample, casts) for sample in samples]
    for label, program in [("source", before), ("target", after)]:
        work = tmp_path / label
        diff.compare(program.read_text(), samples, work, fidelity_tools, function="forward")
        assert json.loads((work / "native-results.json").read_text())["results"] == expected


@pytest.mark.parametrize("empty", [False, True])
def test_actual_compaction_and_canonicalization_match_native_reference(tmp_path, fidelity_tools, empty):
    tools = SimpleNamespace(m2m_python=fidelity_tools.compiler_python)
    _, before, after = certify(tmp_path, tools, compact=True, empty=empty)
    samples = _samples(CASTS[2], empty)
    expected = [_expected(sample, CASTS[2], compact=True) for sample in samples]
    for label, program in [("source", before), ("compacted", tmp_path / "compacted.mlir"), ("target", after)]:
        work = tmp_path / label
        diff.compare(program.read_text(), samples, work, fidelity_tools, function="forward")
        assert json.loads((work / "native-results.json").read_text())["results"] == expected


@pytest.mark.parametrize("fault", ["cast_signedness", "body_constant"])
def test_mixed_boundary_has_observable_wrong_output_controls(tmp_path, fidelity_tools, fault):
    tools = SimpleNamespace(m2m_python=fidelity_tools.compiler_python)
    stage, before, after = certify(tmp_path, tools)
    diff._run([str(tools.m2m_python), "-c", MUTATE, str(after), fault])
    assert check_typed_stage(stage, m2m_python=tools.m2m_python)["verdict"] == "abstracted"
    samples = _samples(CASTS[2])
    expected = [_expected(sample, CASTS[2]) for sample in samples]
    wrong = [
        _expected(
            sample, CASTS[2], constant=11 if fault == "body_constant" else 7, wrong_cast=fault == "cast_signedness"
        )
        for sample in samples
    ]
    assert expected != wrong
    for label, program, reference in [("source", before, expected), ("wrong", after, wrong)]:
        work = tmp_path / label
        diff.compare(program.read_text(), samples, work, fidelity_tools, function="forward")
        assert json.loads((work / "native-results.json").read_text())["results"] == reference
