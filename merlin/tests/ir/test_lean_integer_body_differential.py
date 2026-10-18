"""Native qualification of integer body ordering with independent modular references.

Requires the host runner; excluded from binding-only CI. Nested destinations are fresh on every
outer iteration, retaining the separately tracked S4 captured-destination exclusion.
"""

import json
import subprocess

import pytest
from test_lean_differential import _tensor_samples
from test_lean_integer_body import MUTATE, NAMES, NESTED_MUTATE, _composed_stage, _nested, _source, _stage

from merlin.verify.lean import differential as diff


def _apply(name, left, right, width):
    mask = (1 << width) - 1
    operations = {
        "addi": lambda: left + right,
        "muli": lambda: left * right,
        "andi": lambda: left & right,
        "ori": lambda: left | right,
        "xori": lambda: left ^ right,
    }
    return operations[name]() & mask


def _reference(samples, name, options):
    width = options.get("width", 32)
    combine = "xori" if name == "addi" else "addi"
    result = []
    for data, initial, _other, _capture in samples:
        if options.get("reduction"):
            acc = initial[0]
            for value in data:
                acc = _apply(combine, _apply(name, 7, value, width), acc, width)
            values = [acc]
        else:
            values = [_apply(name, 7, value, width) for value in data]
            if options.get("read_old"):
                values = [_apply(combine, value, old, width) for value, old in zip(values, initial, strict=True)]
        if options.get("escape"):
            values += data
        result.append([str(value) for value in values])
    return result


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("options", [{"width": 13, "read_old": True, "escape": True}, {"reduction": True}])
def test_integer_body_native_modular_reference(tmp_path, fidelity_tools, name, options):
    source = _source(name, **options)
    stage = _stage(tmp_path, source, fidelity_tools)
    samples = _tensor_samples(source, tmp_path, fidelity_tools)
    expected = _reference(samples, name, options)
    for side in ["before", "after"]:
        work = tmp_path / side
        diff.compare((stage / f"{side}.mlir").read_text(), samples, work, fidelity_tools)
        assert json.loads((work / "native-results.json").read_text())["results"] == expected


@pytest.mark.parametrize("name,reduction", [("addi", False), ("muli", True), ("xori", True)])
def test_nested_integer_body_native_reference(tmp_path, fidelity_tools, name, reduction):
    source = _nested(name, reduction=reduction)
    stage = _stage(tmp_path, source, fidelity_tools)
    samples = _tensor_samples(source, tmp_path, fidelity_tools)
    combine = "xori" if name == "addi" else "addi"
    expected = []
    for data, _initial, seed in samples:
        values = []
        for value in data:
            acc = seed[0]
            for _ in range(2 if reduction else 1):
                acc = _apply(combine, _apply(name, 7, value, 32), acc, 32)
            values.append(str(acc))
        expected.append(values)
    for side in ["before", "after"]:
        work = tmp_path / side
        diff.compare((stage / f"{side}.mlir").read_text(), samples, work, fidelity_tools)
        assert json.loads((work / "native-results.json").read_text())["results"] == expected


def test_integer_body_native_detects_wrong_constant(tmp_path, fidelity_tools):
    stage = _stage(tmp_path, _source(), fidelity_tools)
    after = stage / "after.mlir"
    subprocess.run(
        [str(fidelity_tools.compiler_python), "-c", MUTATE, str(after), "constant"],
        check=True,
        capture_output=True,
        text=True,
    )
    samples = [[[1, 2, 3, 4, 5, 6], [0] * 6, [13] * 6, [21]]]
    for side in ["before", "after"]:
        diff.compare((stage / f"{side}.mlir").read_text(), samples, tmp_path / side, fidelity_tools)
    assert json.loads((tmp_path / "before/native-results.json").read_text())["results"] == [
        ["8", "9", "10", "11", "12", "13"]
    ]
    assert json.loads((tmp_path / "after/native-results.json").read_text())["results"] == [
        ["9", "10", "11", "12", "13", "14"]
    ]


@pytest.mark.parametrize("options", [{"width": 64}, {"width": 8, "dirty": True}, {"shape": (0, 3)}, {"shape": ()}])
def test_integer_body_native_shapes_widths_and_dirty_outputs(tmp_path, fidelity_tools, options):
    source = _source("xori", **options)
    stage = _stage(tmp_path, source, fidelity_tools)
    samples = _tensor_samples(source, tmp_path, fidelity_tools)
    expected = _reference(samples, "xori", options)
    for side in ["before", "after"]:
        work = tmp_path / side
        diff.compare((stage / f"{side}.mlir").read_text(), samples, work, fidelity_tools)
        assert json.loads((work / "native-results.json").read_text())["results"] == expected


def test_composed_integer_body_native_reference(tmp_path, fidelity_tools):
    stage = _composed_stage(tmp_path, fidelity_tools)
    samples = _tensor_samples((stage / "before.mlir").read_text(), tmp_path, fidelity_tools)
    expected = [
        [str(((value + 7) & ((1 << 32) - 1)) ^ capture[0]) for value in data]
        for data, _initial, _other, capture in samples
    ]
    for side in ["before", "after"]:
        work = tmp_path / side
        diff.compare((stage / f"{side}.mlir").read_text(), samples, work, fidelity_tools)
        assert json.loads((work / "native-results.json").read_text())["results"] == expected


@pytest.mark.parametrize("fault,expected", [("capture", ["47", "61"]), ("yield", ["5", "5"])])
def test_nested_integer_reduction_mutants_are_observable(tmp_path, fidelity_tools, fault, expected):
    stage = _stage(tmp_path, _nested("muli", reduction=True), fidelity_tools)
    subprocess.run(
        [str(fidelity_tools.compiler_python), "-c", NESTED_MUTATE, str(stage / "after.mlir"), fault],
        check=True,
        capture_output=True,
        text=True,
    )
    samples = [[[1, 2], [3, 4], [5]]]
    for side, reference in [("before", ["19", "33"]), ("after", expected)]:
        work = tmp_path / side
        diff.compare((stage / f"{side}.mlir").read_text(), samples, work, fidelity_tools)
        assert json.loads((work / "native-results.json").read_text())["results"] == [reference]
