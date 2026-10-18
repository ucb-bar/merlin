"""Independent modular-sum references for nested reshape chains with fresh destinations."""

import json
import subprocess

import pytest
from test_lean_reshape_compose import _canonicalize
from test_lean_reshape_nested import MUTATE, _source

from merlin.verify.lean import differential as diff


def _expected(sample, *, escape=False, root=0, captured=0):
    mask = (1 << 32) - 1
    total = sum(sample[root]) + sample[3][0]
    if escape:
        total += sum(sample[captured]) + sample[3][0]
    return [str((total + x) & mask) for x in sample[1]] + [str(x) for x in sample[0]]


@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("escape", [False, True])
def test_nested_reshape_native_reference(tmp_path, fidelity_tools, depth, escape):
    source = _source(depth, escape=escape)
    _, _, after = _canonicalize(tmp_path, fidelity_tools.compiler_python, source)
    pool = [0, 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF]
    samples = [
        [[pool[(n + i + arg) % len(pool)] for i in range(count)] for arg, count in enumerate([24, 2, 24, 1, 24])]
        for n in range(len(pool))
    ]
    expected = [_expected(sample, escape=escape) for sample in samples]
    for name, program in [("source", source), ("target", after.read_text())]:
        work = tmp_path / name
        diff.compare(program, samples, work, fidelity_tools, function="forward")
        results = json.loads((work / "results.json").read_text())
        assert results["native"]["results"] == expected
        assert [[bit for value in case["values"] for bit in value["bits"]] for case in results["lean"]] == expected


@pytest.mark.parametrize("fault,escape", [("root", False), ("capture", True)])
def test_nested_reshape_native_wrong_capture_controls(tmp_path, fidelity_tools, fault, escape):
    _, _, after = _canonicalize(tmp_path, fidelity_tools.compiler_python, _source(2, escape=escape))
    good = after.read_text()
    changed = subprocess.run(
        [str(fidelity_tools.compiler_python), "-c", MUTATE, str(after), fault],
        capture_output=True,
        text=True,
        check=False,
    )
    assert changed.returncode == 0, changed.stderr
    sample = [list(range(24)), [5, 9], list(range(100, 124)), [7], list(range(200, 224))]
    for name, program, root, captured in [
        ("good", good, 0, 0),
        ("bad", after.read_text(), 2 if fault == "root" else 0, 4 if fault == "capture" else 0),
    ]:
        work = tmp_path / name
        diff.compare(program, [sample], work, fidelity_tools, function="forward")
        expected = [_expected(sample, escape=escape, root=root, captured=captured)]
        assert json.loads((work / "native-results.json").read_text())["results"] == expected
