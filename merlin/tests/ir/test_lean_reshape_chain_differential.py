"""Long-chain native evidence uses an independent flattened row-major bit reference."""

import json
import math
import subprocess

import pytest
from test_lean_differential import _floats
from test_lean_reshape_chain import _source
from test_lean_reshape_compose import _canonicalize
from test_lean_reshape_mixed import MUTATE

from merlin.verify.lean import differential as diff


@pytest.mark.parametrize(
    "cycles,shape,element,escape",
    [
        (1, (2, 3, 2, 2), "i13", False),
        (4, (2, 3, 2, 2), "f32", True),
        (12, (2, 3, 2, 2), "f64", False),
        (3, (0, 3, 2, 2), "i8", True),
        (3, (2, 1, 2, 2), "f32", True),
    ],
)
def test_transitive_native_reference(tmp_path, fidelity_tools, cycles, shape, element, escape):
    source = _source(cycles, shape, element, escape=escape)
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
    copies = 2 * cycles + 3 if escape else 2
    expected = [[str(x) for x in sample[0] * copies] for sample in samples]
    for name, program in [("source", source), ("target", after.read_text())]:
        work = tmp_path / name
        diff.compare(program, samples, work, fidelity_tools, function="forward")
        results = json.loads((work / "results.json").read_text())
        assert results["native"]["results"] == expected
        assert [[bit for value in case["values"] for bit in value["bits"]] for case in results["lean"]] == expected


def test_transitive_native_wrong_operand_control(tmp_path, fidelity_tools):
    _, _, after = _canonicalize(tmp_path, fidelity_tools.compiler_python, _source(4))
    good = after.read_text()
    changed = subprocess.run(
        [str(fidelity_tools.compiler_python), "-c", MUTATE, str(after), "operand"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert changed.returncode == 0, changed.stderr
    samples = [[list(range(24)), list(range(100, 124)), list(range(200, 224))]]
    for name, program, expected in [
        ("good", good, samples[0][0] * 2),
        ("bad", after.read_text(), samples[0][1] + samples[0][0]),
    ]:
        work = tmp_path / name
        diff.compare(program, samples, work, fidelity_tools, function="forward")
        assert json.loads((work / "native-results.json").read_text())["results"] == [[str(x) for x in expected]]
