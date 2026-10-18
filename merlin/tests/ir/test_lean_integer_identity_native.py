"""Certify real input-dependent integer rewrites and compare native execution with identity."""

import json
import random
from types import SimpleNamespace

import pytest
from test_lean_integer_fold import canonicalize
from test_lean_integer_identity import IDENTITIES, MUTATE, identity_source

from merlin.verify.lean import differential as diff
from merlin.verify.lean.check import check_definedness, check_stage, check_support, compose_chain


def _certify(tmp_path, tools, source):
    stage, before, after = canonicalize(tmp_path, SimpleNamespace(m2m_python=tools.compiler_python), source)
    result = check_stage(stage, m2m_python=tools.compiler_python)
    assert result["verdict"] == "verified", result.get("reason")
    for checker, directory in [(check_support, "support"), (check_definedness, "defined")]:
        result = checker(before, "forward", tmp_path / directory, m2m_python=tools.compiler_python)
        assert result["verdict"] == "verified", result.get("reason")
    chain = compose_chain(
        [stage],
        tmp_path / "support/receipt.json",
        "forward",
        tmp_path / "chain.json",
        m2m_python=tools.compiler_python,
        definedness=tmp_path / "defined/receipt.json",
    )
    assert chain["verdict"] == "verified", chain.get("reason")
    return stage, before, after


def _values(width):
    if width <= 8:
        return list(range(1 << width))
    rng = random.Random(19613 + width)
    return [0, 1, (1 << width) - 1, 1 << (width - 1), (1 << (width - 1)) - 1] + [
        rng.getrandbits(width) for _ in range(32)
    ]


@pytest.mark.parametrize("width", [1, 8, 13, 32, 64])
def test_all_integer_identity_sites_have_defined_native_execution(tmp_path, fidelity_tools, width):
    ty = f"i{width}"
    body, results = [], []
    for i, (name, neutral, reverse) in enumerate(IDENTITIES):
        operands = f"%c{i}, %x" if reverse else f"%x, %c{i}"
        body.extend(
            [
                f"%c{i} = arith.constant {neutral} : {ty}",
                f"%r{i} = arith.{name} {operands} : {ty}",
            ]
        )
        results.append(f"%r{i}")
    results.append("%keep")
    types = [ty] * len(IDENTITIES) + ["i32"]
    source = (
        f"func.func @forward(%x: {ty}, %keep: i32) -> ({', '.join(types)}) {{\n"
        + "\n".join(body)
        + f"\nreturn {', '.join(results)} : {', '.join(types)}\n}}"
    )
    _, before, after = _certify(tmp_path, fidelity_tools, source)
    assert "arith." not in after.read_text()
    samples = [[[x], [(x ^ 0xABCDEF01) & ((1 << 32) - 1)]] for x in _values(width)]
    expected = [[str(sample[0][0])] * len(IDENTITIES) + [str(sample[1][0])] for sample in samples]
    for name, program in [("before", before), ("after", after)]:
        work = tmp_path / name
        diff.compare(program.read_text(), samples, work, fidelity_tools, function="forward")
        assert json.loads((work / "native-results.json").read_text())["results"] == expected


def test_retained_neutral_constant_native_results(tmp_path, fidelity_tools):
    _, before, after = _certify(tmp_path, fidelity_tools, identity_source("andi", -1, True, escape=True))
    samples = [[[value], [(value + 37) % 8192]] for value in _values(13)]
    expected = [[str(sample[0][0]), str(sample[1][0]), "8191"] for sample in samples]
    for name, program in [("before", before), ("after", after)]:
        work = tmp_path / name
        diff.compare(program.read_text(), samples, work, fidelity_tools, function="forward")
        assert json.loads((work / "native-results.json").read_text())["results"] == expected


def test_wrong_identity_alias_changes_native_results(tmp_path, fidelity_tools):
    stage, before, after = _certify(tmp_path, fidelity_tools, identity_source("muli", 1, True))
    diff._run([str(fidelity_tools.compiler_python), "-c", MUTATE, str(after), "result_alias"])
    result = check_stage(stage, m2m_python=fidelity_tools.compiler_python)
    assert result["verdict"] == "abstracted", result.get("reason")
    samples = [[[7], [13]]]
    for name, program in [("original", before), ("mutated", after)]:
        diff.compare(program.read_text(), samples, tmp_path / name, fidelity_tools, function="forward")
    original = json.loads((tmp_path / "original/native-results.json").read_text())["results"]
    mutated = json.loads((tmp_path / "mutated/native-results.json").read_text())["results"]
    assert original == [["7", "13"]] and mutated == [["13", "13"]]


def test_integer_identity_and_mixed_reshape_compose_in_native_execution(tmp_path, fidelity_tools):
    from test_lean_shared_rule_chain import combined_source

    _, before, after = _certify(tmp_path, fidelity_tools, combined_source(escape_constant=True))
    samples = [
        [[integer], [(i * 137 + integer) % (1 << 32) for i in range(24)], [keep]]
        for integer, keep in [(0, 73), (8191, 0xFFFFFFFF), (4096, 0)]
    ]
    expected = [[str(s[0][0]), *map(str, s[1] * 2), str(s[2][0]), "0"] for s in samples]
    for name, program in [("before", before), ("after", after)]:
        work = tmp_path / name
        diff.compare(program.read_text(), samples, work, fidelity_tools, function="forward")
        assert json.loads((work / "native-results.json").read_text())["results"] == expected
