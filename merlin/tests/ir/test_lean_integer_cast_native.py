"""Certify actual cast compositions and compare Lean/native bits with independent integer arithmetic."""

import json
import random
from types import SimpleNamespace

import pytest
from test_lean_integer_cast import FAMILIES, cast_source, long_source, mutate
from test_lean_integer_fold import canonicalize

from merlin.verify.lean import differential as diff
from merlin.verify.lean.check import check_definedness, check_support, check_typed_stage, compose_chain


def certify(tmp_path, tools, source):
    stage, before, after = canonicalize(tmp_path, SimpleNamespace(m2m_python=tools.compiler_python), source)
    result = check_typed_stage(stage, m2m_python=tools.compiler_python)
    assert result["verdict"] == "verified", result.get("reason")
    for check, directory in [(check_support, "support"), (check_definedness, "defined")]:
        result = check(before, "forward", tmp_path / directory, m2m_python=tools.compiler_python)
        assert result["verdict"] == "verified", result.get("reason")
    result = compose_chain(
        [stage],
        tmp_path / "support/receipt.json",
        "forward",
        tmp_path / "chain.json",
        m2m_python=tools.compiler_python,
        definedness=tmp_path / "defined/receipt.json",
    )
    assert result["verdict"] == "verified", result.get("reason")
    return stage, before, after


def values(width):
    if width <= 8:
        return list(range(1 << width))
    rng = random.Random(71263 + width)
    return [0, 1, (1 << width) - 1, 1 << (width - 1), (1 << (width - 1)) - 1] + [
        rng.getrandbits(width) for _ in range(24)
    ]


def reference(kind, source_width, target_width, bits):
    if kind == "extsi" and bits & (1 << (source_width - 1)):
        bits -= 1 << source_width
    return bits % (1 << target_width)


@pytest.mark.parametrize("first,second", FAMILIES)
@pytest.mark.parametrize("width_case", range(3))
@pytest.mark.parametrize("escape", [False, True])
def test_certified_casts_match_independent_native_reference(
    tmp_path, fidelity_tools, first, second, width_case, escape
):
    widths = [(8, 2, 1), (32, 13, 8), (64, 32, 13)] if first == "trunci" else [(1, 2, 8), (8, 13, 32), (13, 32, 64)]
    a, b, c = widths[width_case]
    _, before, after = certify(tmp_path, fidelity_tools, cast_source(first, second, (a, b, c), escape=escape))
    samples = [[[x], [(x ^ 0x13579) % (1 << a)]] for x in values(a)]
    expected = []
    for sample in samples:
        x, keep = sample[0][0], sample[1][0]
        mid = reference(first, a, b, x)
        result = [reference(second, b, c, mid), keep] + ([mid] if escape else [])
        expected.append(list(map(str, result)))
    for name, program in [("before", before), ("after", after)]:
        work = tmp_path / name
        diff.compare(program.read_text(), samples, work, fidelity_tools, function="forward")
        assert json.loads((work / "native-results.json").read_text())["results"] == expected


@pytest.mark.parametrize("escape", [False, True])
def test_long_certified_cast_chain_native(tmp_path, fidelity_tools, escape):
    _, before, after = certify(tmp_path, fidelity_tools, long_source(escape=escape))
    samples = [[[x], [keep]] for x in [0, 1] for keep in [0, 1]]
    expected = [[str(s[0][0]), str(s[1][0])] + ([str(s[0][0])] * 2 if escape else []) for s in samples]
    for name, program in [("before", before), ("after", after)]:
        work = tmp_path / name
        diff.compare(program.read_text(), samples, work, fidelity_tools, function="forward")
        assert json.loads((work / "native-results.json").read_text())["results"] == expected


@pytest.mark.parametrize("fault", ["input", "escaping_input", "signedness"])
def test_wrong_cast_boundaries_change_native_bits(tmp_path, fidelity_tools, fault):
    stage, before, after = certify(tmp_path, fidelity_tools, cast_source("extui", "extui", escape=True))
    mutate(after, SimpleNamespace(m2m_python=fidelity_tools.compiler_python), fault)
    result = check_typed_stage(stage, m2m_python=fidelity_tools.compiler_python)
    assert result["verdict"] == "abstracted", result.get("reason")
    outputs = []
    for name, program in [("before", before), ("wrong", after)]:
        work = tmp_path / name
        diff.compare(program.read_text(), [[[1], [0]]], work, fidelity_tools, function="forward")
        outputs.append(json.loads((work / "native-results.json").read_text())["results"])
    assert outputs[0] == [["1", "0", "1"]]
    assert outputs[1] != outputs[0]


def test_computed_cast_input_native(tmp_path, fidelity_tools):
    from test_lean_integer_cast import computed_input_source

    _, before, after = certify(tmp_path, fidelity_tools, computed_input_source())
    samples = [[[x], [(x * 117 + 31) % 256]] for x in range(256)]
    expected = [[str(s[0][0] ^ s[1][0])] * 2 + [str(s[1][0])] for s in samples]
    for name, program in [("before", before), ("after", after)]:
        work = tmp_path / name
        diff.compare(program.read_text(), samples, work, fidelity_tools, function="forward")
        assert json.loads((work / "native-results.json").read_text())["results"] == expected


@pytest.mark.parametrize("incorrect_kind", ["extui", "extsi"])
def test_sign_then_zero_has_no_direct_root_cast(tmp_path, fidelity_tools, incorrect_kind):
    stage, before, after = certify(tmp_path, fidelity_tools, cast_source("extsi", "extui", (1, 8, 32)))
    after.write_text(
        "func.func @forward(%x: i1, %keep: i1) -> (i32, i1) {\n"
        f"%r = arith.{incorrect_kind} %x : i1 to i32\nreturn %r, %keep : i32, i1\n}}"
    )
    result = check_typed_stage(stage, m2m_python=fidelity_tools.compiler_python)
    assert result["verdict"] == "abstracted", result.get("reason")
    results = []
    for name, program in [("before", before), ("wrong", after)]:
        work = tmp_path / name
        diff.compare(program.read_text(), [[[1], [0]]], work, fidelity_tools, function="forward")
        results.append(json.loads((work / "native-results.json").read_text())["results"])
    assert results[0] == [["255", "0"]]
    assert results[1] != results[0]
