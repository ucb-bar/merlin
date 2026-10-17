"""Independent arithmetic references and native execution qualify closed integer folding.

This module requires native MLIR execution and belongs outside binding-only qualification.
All divisors, signed divisions, shifts and cast widths are chosen to be defined independently.
"""

import json
from types import SimpleNamespace

import pytest
from test_lean_integer_fold import BINARY, MUTATE, binary_source, canonicalize

from merlin.verify.lean import differential as diff
from merlin.verify.lean.check import check_definedness, check_stage, check_support, compose_chain


def _signed(value, width):
    sign = 1 << (width - 1)
    return ((value & ((1 << width) - 1)) ^ sign) - sign


def _reference(name, a, b, width):
    mask = (1 << width) - 1
    a, b = a & mask, b & mask
    sa, sb = _signed(a, width), _signed(b, width)
    if name == "addi":
        value = a + b
    elif name == "subi":
        value = a - b
    elif name == "muli":
        value = a * b
    elif name == "andi":
        value = a & b
    elif name == "ori":
        value = a | b
    elif name == "xori":
        value = a ^ b
    elif name == "maxsi":
        value = max(sa, sb)
    elif name == "minsi":
        value = min(sa, sb)
    elif name == "maxui":
        value = max(a, b)
    elif name == "minui":
        value = min(a, b)
    elif name == "divui":
        value = a // b
    elif name in {"divsi", "remsi"}:
        quotient = (abs(sa) // abs(sb)) * (-1 if (sa < 0) != (sb < 0) else 1)
        value = quotient if name == "divsi" else sa - quotient * sb
    elif name == "remui":
        value = a % b
    elif name == "ceildivui":
        value = (a + b - 1) // b
    elif name == "ceildivsi":
        value = -((-sa) // sb)
    elif name == "floordivsi":
        value = sa // sb
    elif name == "shli":
        value = a << b
    elif name == "shrui":
        value = a >> b
    elif name == "shrsi":
        value = sa >> b
    else:
        raise AssertionError(f"missing independent reference for {name}")
    return value & mask


def _arithmetic_program(width):
    ty, mask = f"i{width}", (1 << width) - 1
    names = BINARY[:10] if width == 1 else BINARY
    pairs = [(1, 0)] if width == 1 else [(0, 1), (1, 1), (mask, 3), (1 << (width - 1), 3), (mask - 1, 2)]
    body, results, expected = [], [], []
    for a, b in pairs:
        for name in names:
            i = len(results)
            body.extend(
                [
                    f"%a{i} = arith.constant {a} : {ty}",
                    f"%b{i} = arith.constant {b} : {ty}",
                    f"%r{i} = arith.{name} %a{i}, %b{i} : {ty}",
                ]
            )
            results.append(f"%r{i}")
            expected.append(str(_reference(name, a, b, width)))
    types = [ty] * len(results) + ["i32"]
    results.append("%keep")
    source = (
        f"func.func @forward(%keep: i32) -> ({', '.join(types)}) {{\n"
        + "\n".join(body)
        + f"\nreturn {', '.join(results)} : {', '.join(types)}\n}}"
    )
    return source, expected


@pytest.mark.parametrize("width", [1, 8, 13, 32, 64])
def test_integer_folds_native_with_independent_reference(tmp_path, fidelity_tools, width):
    source, expected = _arithmetic_program(width)
    _, before, after = canonicalize(tmp_path, SimpleNamespace(m2m_python=fidelity_tools.compiler_python), source)
    samples = [[[keep]] for keep in [0, 1 << 31, (1 << 32) - 1]]
    for name, path in [("before", before), ("after", after)]:
        work = tmp_path / name
        diff.compare(path.read_text(), samples, work, fidelity_tools, function="forward")
        native = json.loads((work / "native-results.json").read_text())["results"]
        assert native == [expected + [str(sample[0][0])] for sample in samples]


@pytest.mark.parametrize("source_width,target_width", [(1, 8), (4, 13), (8, 32), (13, 64), (32, 64)])
def test_cast_folds_native_with_independent_reference(tmp_path, fidelity_tools, source_width, target_width):
    small, large = f"i{source_width}", f"i{target_width}"
    body, results, types, expected = [], [], [], []
    edges = [0, 1, 1 << (source_width - 1), (1 << source_width) - 1]
    for i, value in enumerate(edges):
        body.extend(
            [
                f"%a{i} = arith.constant {value} : {small}",
                f"%s{i} = arith.extsi %a{i} : {small} to {large}",
                f"%u{i} = arith.extui %a{i} : {small} to {large}",
                f"%t{i} = arith.trunci %s{i} : {large} to {small}",
            ]
        )
        results.extend([f"%s{i}", f"%u{i}", f"%t{i}"])
        types.extend([large, large, small])
        expected.extend([str(_signed(value, source_width) % (1 << target_width)), str(value), str(value)])
    source = (
        f"func.func @forward() -> ({', '.join(types)}) {{\n"
        + "\n".join(body)
        + f"\nreturn {', '.join(results)} : {', '.join(types)}\n}}"
    )
    _, before, after = canonicalize(tmp_path, SimpleNamespace(m2m_python=fidelity_tools.compiler_python), source)
    for name, path in [("before", before), ("after", after)]:
        work = tmp_path / name
        diff.compare(path.read_text(), [[]], work, fidelity_tools, function="forward")
        native = json.loads((work / "native-results.json").read_text())["results"]
        assert native == [expected]


def test_wrong_fold_constant_changes_native_results(tmp_path, fidelity_tools):
    stage, before, after = canonicalize(
        tmp_path, SimpleNamespace(m2m_python=fidelity_tools.compiler_python), binary_source("addi")
    )
    assert check_stage(stage, m2m_python=fidelity_tools.compiler_python)["verdict"] == "verified"
    diff._run([str(fidelity_tools.compiler_python), "-c", MUTATE, str(after), "bits"])
    refused = check_stage(stage, m2m_python=fidelity_tools.compiler_python)
    assert refused["verdict"] == "abstracted", refused.get("reason")
    samples = [[[19]]]
    for name, path in [("original", before), ("mutated", after)]:
        diff.compare(path.read_text(), samples, tmp_path / name, fidelity_tools, function="forward")
    original = json.loads((tmp_path / "original/native-results.json").read_text())["results"]
    mutated = json.loads((tmp_path / "mutated/native-results.json").read_text())["results"]
    assert original == [["10", "19"]] and mutated == [["11", "19"]]


def test_long_integer_fold_certified_chain_and_native_execution(tmp_path, fidelity_tools):
    constants = [37 * (i + 1) for i in range(64)]
    body = ["%seed = arith.constant -17 : i13"]
    previous = "%seed"
    for i, value in enumerate(constants):
        body.extend(
            [
                f"%c{i} = arith.constant {value} : i13",
                f"%r{i} = arith.addi {previous}, %c{i} : i13",
            ]
        )
        previous = f"%r{i}"
    source = (
        "func.func @forward(%keep: i32) -> (i13, i32) {\n"
        + "\n".join(body)
        + f"\nreturn {previous}, %keep : i13, i32\n}}"
    )
    stage, before, after = canonicalize(tmp_path, SimpleNamespace(m2m_python=fidelity_tools.compiler_python), source)
    assert before.read_text().count("arith.addi") == 64
    assert after.read_text().count("arith.constant") == 1
    assert "arith.addi" not in after.read_text()
    boundary = check_stage(stage, m2m_python=fidelity_tools.compiler_python)
    assert boundary["verdict"] == "verified", boundary.get("reason")
    for checker, name in [(check_support, "support"), (check_definedness, "defined")]:
        checked = checker(before, "forward", tmp_path / name, m2m_python=fidelity_tools.compiler_python)
        assert checked["verdict"] == "verified", checked.get("reason")
    chain = compose_chain(
        [stage],
        tmp_path / "support/receipt.json",
        "forward",
        tmp_path / "chain.json",
        m2m_python=fidelity_tools.compiler_python,
        definedness=tmp_path / "defined/receipt.json",
    )
    assert chain["verdict"] == "verified", chain.get("reason")
    assert chain["theorem"] == "MerlinVerify.Check.chain_defined_refines"

    samples = [[[keep]] for keep in [0, 1 << 31, (1 << 32) - 1]]
    expected = str((-17 + sum(constants)) % 8192)
    for name, path in [("before", before), ("after", after)]:
        work = tmp_path / name
        diff.compare(path.read_text(), samples, work, fidelity_tools, function="forward")
        native = json.loads((work / "native-results.json").read_text())["results"]
        assert native == [[expected, str(sample[0][0])] for sample in samples]
