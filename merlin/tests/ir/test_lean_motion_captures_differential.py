"""Independent native references for motion with recursively captured live results.

Nested destinations are fresh inside every outer iteration. This does not qualify
S4 captured-destination reuse or operation motion across a region-bearing barrier.
"""

import json

import pytest
from test_lean_motion_captures import _adjacent_hoist, _body_motion, _captured, _mutant
from test_lean_reshape_compose import _canonicalize, _verify_mlir

from merlin.verify.lean import differential as diff

MASK = (1 << 32) - 1


def _samples(kind):
    if kind == "reduction":
        return [[[x], [1, 4], [5]] for x in (0, 128, 255)]
    return [[[x], [3, 7]] for x in (0, 128, 255)]


def _reference(samples, kind, fault=None):
    result = []
    for sample in samples:
        x = sample[0][0]
        cast = x if x < 128 else x - 256
        delta = 2 * cast if fault == "capture" else cast + 22
        if kind == "reduction":
            values = [sample[2][0] + sum(sample[1]) + len(sample[1]) * delta]
        else:
            values = [delta if fault == "yield" else old + delta for old in sample[1]]
        if fault == "escaped-tensor":
            values = sample[1]
        first = 22 if fault == "escaped-scalar" else cast
        result.append([str(first & MASK), "22", *[str(value & MASK) for value in values]])
    return result


@pytest.mark.parametrize("bits", [32, 64])
@pytest.mark.parametrize("kind", ["parallel", "nested", "reduction"])
def test_native_actual_motion_captures_match_modular_reference(tmp_path, fidelity_tools, bits, kind):
    _, before, after = _canonicalize(tmp_path, fidelity_tools.compiler_python, _captured(kind))
    samples = _samples(kind)
    expected = _reference(samples, kind)
    for label, path in (("source", before), ("target", after)):
        work = tmp_path / label
        diff.compare(path.read_text(), samples, work, fidelity_tools, index_bits=bits)
        assert json.loads((work / "native-results.json").read_bytes())["results"] == expected


@pytest.mark.parametrize("fault", ["capture", "yield", "escaped-scalar", "escaped-tensor"])
def test_valid_capture_and_escape_mutants_have_observed_wrong_outputs(tmp_path, fidelity_tools, fault):
    source = _captured("nested")
    mutant = _mutant(_adjacent_hoist(source), fault)
    mutant_path = tmp_path / "mutant.mlir"
    mutant_path.write_text(mutant)
    _verify_mlir(mutant_path, fidelity_tools.compiler_python)
    samples = _samples("nested")
    expected = _reference(samples, "nested")
    changed = _reference(samples, "nested", fault)
    assert expected != changed
    for label, program, wanted in (("source", source, expected), ("mutant", mutant, changed)):
        work = tmp_path / label
        diff.compare(program, samples, work, fidelity_tools)
        assert json.loads((work / "native-results.json").read_bytes())["results"] == wanted


def test_inner_operation_reorder_gap_is_empirically_equal_on_samples(tmp_path, fidelity_tools):
    samples = [[[3], [2, 7]], [[4294967295], [2147483648, 0]]]
    expected = [[str((old + capture[0] + old * capture[0]) & MASK) for old in dest] for capture, dest in samples]
    for label, reverse in (("source", False), ("target", True)):
        work = tmp_path / label
        diff.compare(_body_motion(reverse), samples, work, fidelity_tools)
        assert json.loads((work / "native-results.json").read_bytes())["results"] == expected
