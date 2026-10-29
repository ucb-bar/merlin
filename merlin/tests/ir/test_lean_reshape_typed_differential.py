"""Independent native row-major references for source-typed mixed reshape composition."""

import json
import math
import subprocess

import pytest
from test_lean_differential import _floats
from test_lean_reshape_compose import _canonicalize
from test_lean_reshape_mixed import MUTATE
from test_lean_reshape_typed import _source

from merlin.verify.lean import differential as diff


@pytest.mark.parametrize(
    "shape,element,result,escape",
    [
        ((2, 3, 4), "i13", "expand", False),
        ((0, 3, 4), "i8", "expand", True),
        ((2, 1, 4), "f32", "expand", True),
        ((2, 3, 4), "f64", "expand", False),
        ((2, 3, 2, 2), "i32", "collapse", True),
        ((2, 3, 2, 2), "f32", "collapse", False),
    ],
)
def test_typed_mixed_matches_independent_native_reference(tmp_path, fidelity_tools, shape, element, result, escape):
    source = _source(shape, element, result=result, escape=escape)
    _, _, after = _canonicalize(tmp_path, fidelity_tools.compiler_python, source)
    count, width = math.prod(shape), int(element[1:])
    pool = (
        _floats(width) + [0x7FC01234 if width == 32 else 0x7FF8000000001234]
        if element.startswith("f")
        else [0, 1, (1 << (width - 1)) - 1, 1 << (width - 1), (1 << width) - 1]
    )
    samples = [
        [[pool[(offset + i + arg) % len(pool)] for i in range(count)] for arg in range(3)]
        for offset in range(len(pool))
    ]
    expected = [[str(x) for x in sample[0] * (3 if escape else 2)] for sample in samples]
    for name, program in [("source", source), ("target", after.read_text())]:
        work = tmp_path / name
        diff.compare(program, samples, work, fidelity_tools, function="forward")
        results = json.loads((work / "results.json").read_text())
        assert results["native"]["results"] == expected
        assert [[bit for value in case["values"] for bit in value["bits"]] for case in results["lean"]] == expected


@pytest.mark.parametrize("result,shape", [("expand", (2, 3, 4)), ("collapse", (2, 3, 2, 2))])
def test_typed_mixed_native_wrong_operand_control(tmp_path, fidelity_tools, result, shape):
    _, _, after = _canonicalize(tmp_path, fidelity_tools.compiler_python, _source(shape, result=result))
    good = after.read_text()
    mutated = subprocess.run(
        [str(fidelity_tools.compiler_python), "-c", MUTATE, str(after), "operand"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert mutated.returncode == 0, mutated.stderr
    bad = after.read_text()
    assert good != bad
    sample = [[list(range(24)), list(range(100, 124)), list(range(200, 224))]]
    for name, program, expected in [
        ("good", good, sample[0][0] * 2),
        ("bad", bad, sample[0][1] + sample[0][0]),
    ]:
        work = tmp_path / name
        diff.compare(program, sample, work, fidelity_tools, function="forward")
        assert json.loads((work / "native-results.json").read_text())["results"] == [[str(x) for x in expected]]


def test_typed_mixed_native_capture_and_wrong_capture_control(tmp_path, fidelity_tools):
    from test_lean_reshape_typed import CAPTURE_MUTATE, _capture_source

    source = _capture_source()
    _, _, after = _canonicalize(tmp_path, fidelity_tools.compiler_python, source)
    good = after.read_text()
    changed = subprocess.run(
        [str(fidelity_tools.compiler_python), "-c", CAPTURE_MUTATE, str(after)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert changed.returncode == 0, changed.stderr
    sample = [[list(range(24)), list(range(100, 124)), list(range(200, 224)), [7], list(range(300, 324))]]
    for name, program, capture in [
        ("source", source, 0),
        ("target", good, 0),
        ("wrong_capture", after.read_text(), 300),
    ]:
        work = tmp_path / name
        diff.compare(program, sample, work, fidelity_tools, function="forward")
        expected = [[str(x) for x in sample[0][0] * 2 + [capture]]]
        assert json.loads((work / "native-results.json").read_text())["results"] == expected
