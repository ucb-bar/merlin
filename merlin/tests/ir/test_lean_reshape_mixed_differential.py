"""Independent native evidence for supported mixed expand/collapse compositions."""

import json
import math
import subprocess

import pytest
from test_lean_differential import _floats
from test_lean_reshape_compose import _canonicalize
from test_lean_reshape_mixed import MUTATE, _source

from merlin.verify.lean import differential as diff


@pytest.mark.parametrize("result", ["collapse", "expand"])
@pytest.mark.parametrize(
    "shape,element", [((2, 3, 2, 2), "i13"), ((0, 3, 2, 2), "i8"), ((2, 3, 2, 2), "f32"), ((2, 3, 2, 2), "f64")]
)
def test_mixed_reshape_matches_native_bits(tmp_path, fidelity_tools, result, shape, element):
    source = _source(shape, element, result=result)
    _, _, after = _canonicalize(tmp_path, fidelity_tools.compiler_python, source)
    count, width = math.prod(shape), int(element[1:])
    if element.startswith("f"):
        pool = _floats(width) + [0x7FC01234 if width == 32 else 0x7FF8000000001234]
    else:
        pool = [0, 1, (1 << (width - 1)) - 1, 1 << (width - 1), (1 << width) - 1]
    samples = [
        [[pool[(offset + i + arg) % len(pool)] for i in range(count)] for arg in range(3)]
        for offset in range(len(pool))
    ]
    # Every reshape reads the same row-major stream. The second returned value is the original
    # input; neither unused alternative operand may influence either result.
    expected = [[str(x) for x in sample[0] * 2] for sample in samples]
    for name, program in [("source", source), ("target", after.read_text())]:
        work = tmp_path / name
        diff.compare(program, samples, work, fidelity_tools, function="forward")
        answer = json.loads((work / "results.json").read_text())
        assert answer["native"]["results"] == expected
        assert [[bit for value in case["values"] for bit in value["bits"]] for case in answer["lean"]] == expected


@pytest.mark.parametrize("result", ["collapse", "expand"])
def test_native_mixed_wrong_input_control(tmp_path, fidelity_tools, result):
    source = _source(result=result)
    _, _, after = _canonicalize(tmp_path, fidelity_tools.compiler_python, source)
    good = after.read_text()
    changed = subprocess.run(
        [str(fidelity_tools.compiler_python), "-c", MUTATE, str(after), "operand"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert changed.returncode == 0, changed.stderr
    bad = after.read_text()
    assert good != bad
    samples = [[[i for i in range(24)], [100 + i for i in range(24)], [200 + i for i in range(24)]]]
    for name, program, expected in [
        ("good", good, samples[0][0] * 2),
        ("bad", bad, samples[0][1] + samples[0][0]),
    ]:
        work = tmp_path / name
        diff.compare(program, samples, work, fidelity_tools, function="forward")
        assert json.loads((work / "native-results.json").read_text())["results"] == [[str(x) for x in expected]]
