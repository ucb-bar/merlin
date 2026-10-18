"""Native index arithmetic evidence and an explicit retained S6 disagreement.

Requires native MLIR execution; exclude this module from binding-only CI.
"""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_lean_defined_conversions import certify
from test_lean_index_fold import NAMES, mutate, source, stage_for, wrapped_compare

from merlin.verify.lean import differential as diff
from merlin.verify.lean.check import check_definedness, check_support, check_typed_stage


def reference(name, a, b, width):
    mask = (1 << width) - 1
    a, b = a & mask, b & mask
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
    else:
        raise AssertionError(f"missing independent index reference: {name}")
    return value & mask


def certified(tmp_path, tools, text, index_bits):
    stage, before, after = stage_for(tmp_path, SimpleNamespace(m2m_python=tools.compiler_python), text)
    certify(stage, before, SimpleNamespace(m2m_python=tools.compiler_python), tmp_path, index_bits=index_bits)
    return stage, before, after


def compare_expected(tmp_path, tools, paths, samples, expected, index_bits):
    for name, path in paths:
        work = tmp_path / name
        diff.compare(path.read_text(), samples, work, tools, function="forward", index_bits=index_bits)
        actual = json.loads((work / "native-results.json").read_text())["results"]
        assert actual == [[str(value) for value in row] for row in expected]


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("index_bits", [32, 64])
def test_certified_index_folds_match_modular_native_reference(tmp_path, fidelity_tools, name, index_bits):
    a, b = 2147483646, 5
    _, before, after = certified(tmp_path, fidelity_tools, source(name, a, b, escape=True), index_bits)
    mask = (1 << index_bits) - 1
    samples = [[[value & mask], [keep]] for value, keep in [(0, 0), (mask, 1 << 31), (17, (1 << 32) - 1)]]
    result = reference(name, a, b, index_bits)
    expected = [[result, a, sample[1][0]] for sample in samples]
    compare_expected(tmp_path, fidelity_tools, [("before", before), ("after", after)], samples, expected, index_bits)


@pytest.mark.parametrize("index_bits", [32, 64])
def test_certified_folded_index_captures_match_native_reference(tmp_path, fidelity_tools, index_bits):
    _, before, after = certified(tmp_path, fidelity_tools, source("xori", capture=True, escape=True), index_bits)
    mask = (1 << index_bits) - 1
    samples = [
        [[0], [0], [0, 1, mask], [13, 17, 19]],
        [[mask], [(1 << 32) - 1], [mask - 1, 1 << (index_bits - 1), 31], [7, 11, 23]],
    ]
    constant = reference("xori", 17, 5, index_bits)
    expected = [[*((value + constant) & mask for value in s[2]), 17, s[1][0]] for s in samples]
    compare_expected(tmp_path, fidelity_tools, [("before", before), ("after", after)], samples, expected, index_bits)


@pytest.mark.parametrize("index_bits", [32, 64])
def test_certified_long_index_chain_matches_native_reference(tmp_path, fidelity_tools, index_bits):
    _, before, after = certified(tmp_path, fidelity_tools, source(length=128), index_bits)
    assert after.read_text().count("arith.constant") == 1
    result = 17
    for n in range(128):
        result = reference("addi", result, 137 * n + 5, index_bits)
    samples = [[[0], [0]], [[(1 << index_bits) - 1], [(1 << 32) - 1]]]
    expected = [[result, sample[1][0]] for sample in samples]
    compare_expected(tmp_path, fidelity_tools, [("before", before), ("after", after)], samples, expected, index_bits)


@pytest.mark.parametrize("fault", ["value", "input"])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_refused_index_mutations_change_native_results(tmp_path, fidelity_tools, fault, index_bits):
    stage, before, after = certified(tmp_path, fidelity_tools, source(), index_bits)
    mutate(after, SimpleNamespace(m2m_python=fidelity_tools.compiler_python), fault)
    result = check_typed_stage(stage, m2m_python=fidelity_tools.compiler_python, index_bits=index_bits)
    assert result["verdict"] == "abstracted", result.get("reason")
    samples = [[[91], [37]]]
    original = [[22, 37]]
    corrupted = [[23 if fault == "value" else 91, 37]]
    compare_expected(tmp_path, fidelity_tools, [("original", before)], samples, original, index_bits)
    compare_expected(tmp_path, fidelity_tools, [("mutated", after)], samples, corrupted, index_bits)


def tool_identity(tools):
    result = diff._run(
        [str(tools.compiler_python), "-c", "from torch_mlir._mlir_libs import _mlir; print(_mlir.__file__)"]
    )
    library = Path(result.stdout.strip())
    return {
        "checker_sha256": hashlib.sha256(tools.lean_exe.read_bytes()).hexdigest(),
        "native_library": str(library),
        "native_library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "llvm_source_revision": "unknown; package version is not a source revision",
    }


@pytest.mark.parametrize("explicit_layout,index_bits", [(False, 32), (False, 64), (True, 32)])
def test_s6_index_wrap_comparison_disagreement_is_retained(tmp_path, fidelity_tools, explicit_layout, index_bits):
    tools = SimpleNamespace(m2m_python=fidelity_tools.compiler_python)
    stage, before, after = stage_for(tmp_path, tools, wrapped_compare(explicit_layout=explicit_layout))
    checks = {}
    for checker, name in [(check_support, "support"), (check_definedness, "defined")]:
        result = checker(before, "forward", tmp_path / name, m2m_python=tools.m2m_python, index_bits=index_bits)
        assert result["verdict"] == "verified", result.get("reason")
        checks[name] = result["verdict"]
    boundary = check_typed_stage(stage, m2m_python=tools.m2m_python, index_bits=index_bits)
    assert boundary["verdict"] == ("verified" if index_bits == 64 else "abstracted"), boundary.get("reason")
    if index_bits == 64:
        certify(stage, before, tools, tmp_path, index_bits=index_bits)
    samples = [[[0]], [[(1 << 32) - 1]]]
    if index_bits == 32:
        with pytest.raises(AssertionError, match="Lean bits=1, native MLIR bits=0"):
            diff.compare(
                before.read_text(), samples, tmp_path / "before", fidelity_tools, function="forward", index_bits=32
            )
    else:
        diff.compare(
            before.read_text(), samples, tmp_path / "before", fidelity_tools, function="forward", index_bits=64
        )
    diff.compare(
        after.read_text(), samples, tmp_path / "after", fidelity_tools, function="forward", index_bits=index_bits
    )
    evidence = {name: json.loads((tmp_path / name / "results.json").read_text()) for name in ["before", "after"]}
    for name, result in evidence.items():
        expected_compare = "1" if name == "before" and index_bits == 32 else "0"
        assert [sample["values"][1]["bits"][0] for sample in result["lean"]] == [expected_compare] * 2
        assert result["native"]["results"] == [["2147483648", "0", str(s[0][0])] for s in samples]
    manifest = {
        "discrepancy": "S6",
        "classification": "index32 model/native or compiler-configuration discrepancy; cause unproved",
        "explicit_dlti32": explicit_layout,
        "index_bits": index_bits,
        "source_checks": checks,
        "typed_boundary": boundary["verdict"],
        "comparison_validation": "actual checked-width bits must agree with the target bool constant",
        "torch_mlir_package_version": evidence["before"]["native"]["mlir_revision"],
        "legacy_mlir_revision_field": "package version, not a proven LLVM source revision",
        "tool_identity": tool_identity(fidelity_tools),
        "diagnostic_references": [
            "https://mlir.llvm.org/docs/DataLayout/",
            "https://mlir.llvm.org/docs/Dialects/IndexOps/",
        ],
        "evidence": evidence,
    }
    (tmp_path / "s6-manifest.json").write_text(json.dumps(manifest, indent=2))


@pytest.mark.parametrize("index_bits", [32, 64])
def test_cyclic_index_chain_matches_native_but_retains_alias_abstention(tmp_path, fidelity_tools, index_bits):
    tools = SimpleNamespace(m2m_python=fidelity_tools.compiler_python)
    stage, before, after = stage_for(tmp_path, tools, source(length=128, cycle=True))
    boundary = check_typed_stage(stage, m2m_python=tools.m2m_python, index_bits=index_bits)
    assert boundary["verdict"] == "abstracted", boundary.get("reason")
    result = 17
    for n in range(128):
        result = reference(NAMES[n % len(NAMES)], result, 137 * n + 5, index_bits)
    samples = [[[0], [37]]]
    compare_expected(
        tmp_path, fidelity_tools, [("before", before), ("after", after)], samples, [[result, 37]], index_bits
    )
