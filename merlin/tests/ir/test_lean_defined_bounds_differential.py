"""Native bounds qualification on initialized inputs with independent row-major references.

Unsafe/OOB/poison cases belong to refusal tests and are never sent to the native runner. Nested
splat fixtures allocate fresh per-outer-iteration destinations, retaining the tracked S4 exclusion.
"""

import json
import subprocess

import pytest
from test_lean_defined_bounds import CAPTURE_CASES, MUTATE, _captured_source, _reshape_source, _source
from test_lean_differential import _tensor_samples
from test_lean_reshape_compose import _canonicalize
from test_lean_reshape_typed import CAPTURE_MUTATE
from test_lean_splat_nested import _stage as _splat_stage

from merlin.verify.lean import differential as diff


def _position(shape, coordinates):
    result = 0
    for size, coordinate in zip(shape, coordinates, strict=True):
        assert 0 <= coordinate < size
        result = result * size + coordinate
    return result


@pytest.mark.parametrize("index_bits", [32, 64])
@pytest.mark.parametrize(
    "shape,indices,element",
    [
        ((2, 3), (1, 2), "i32"),
        ((1,), (0,), "f32"),
        ((2, 1, 3), (1, 0, 2), "i13"),
        ((), (), "f64"),
        ((2,), (1,), "index"),
    ],
)
def test_extraction_matches_independent_row_major_reference(
    tmp_path, fidelity_tools, index_bits, shape, indices, element
):
    source = _source(shape, indices, element)
    samples = _tensor_samples(source, tmp_path, fidelity_tools)
    if element.startswith("f"):
        width = int(element[1:])
        nan = 0x7FC01234 if width == 32 else 0x7FF8000000001234
        for bits in [0, 1 << (width - 1), nan, (1 << width) - 1]:
            samples.append([[bits] * len(samples[0][0])])
    mask = (1 << index_bits) - 1
    expected = [
        [str(data[_position(shape, indices)] & mask if element == "index" else data[_position(shape, indices)])]
        for (data,) in samples
    ]
    diff.compare(source, samples, tmp_path / "native", fidelity_tools, index_bits=index_bits)
    assert json.loads((tmp_path / "native/native-results.json").read_text())["results"] == expected


@pytest.mark.parametrize("n", [4294967297, 4294967296, -4294967295])
def test_index32_wraparound_is_observed_independently(tmp_path, fidelity_tools, n):
    source = _source((2,), (n,))
    # This finite input is safe only at index width32: the actual index is0 or1 after wrapping.
    wrapped = n % (1 << 32)
    assert wrapped in (0, 1)
    samples = [[[13, 29]]]
    diff.compare(source, samples, tmp_path / "native", fidelity_tools, index_bits=32)
    assert json.loads((tmp_path / "native/native-results.json").read_text())["results"] == [
        [str(samples[0][0][wrapped])]
    ]


@pytest.mark.parametrize("index_bits", [32, 64])
@pytest.mark.parametrize("shape,coordinates", CAPTURE_CASES)
def test_nested_capture_nonzero_coordinates_native(tmp_path, fidelity_tools, index_bits, shape, coordinates):
    source = _captured_source(shape, coordinates, multiple=True)
    stage, count = _splat_stage(tmp_path, source)
    assert count == 2
    samples = _tensor_samples(source, tmp_path, fidelity_tools)
    position = _position(shape, coordinates)
    expected = []
    for payloads, old, data, _unused_init, seed, alternate in samples:
        expected.append(
            [
                str((data[position] + value + previous + seed[0] + alternate[0]) % (1 << 32))
                for value, previous in zip(payloads, old, strict=True)
            ]
        )
    for side in ["before", "after"]:
        work = tmp_path / side
        diff.compare((stage / f"{side}.mlir").read_text(), samples, work, fidelity_tools, index_bits=index_bits)
        assert json.loads((work / "native-results.json").read_text())["results"] == expected


def test_nested_changed_inbounds_coordinate_is_observable(tmp_path, fidelity_tools):
    stage, _ = _splat_stage(tmp_path, _captured_source())
    subprocess.run(
        [str(fidelity_tools.compiler_python), "-c", MUTATE, str(stage / "after.mlir"), "0"],
        check=True,
        capture_output=True,
        text=True,
    )
    samples = [[[1, 2], [3, 4], [10, 20, 30, 40, 50, 60], [0] * 6, [7], [0]]]
    for side, expected in [("before", ["71", "73"]), ("after", ["41", "43"])]:
        work = tmp_path / side
        diff.compare((stage / f"{side}.mlir").read_text(), samples, work, fidelity_tools)
        assert json.loads((work / "native-results.json").read_text())["results"] == [expected]


def test_reshape_changed_capture_at_nonzero_coordinate_is_observable(tmp_path, fidelity_tools):
    source = _reshape_source()
    stage, _, after = _canonicalize(tmp_path, fidelity_tools.compiler_python, source)
    original = after.read_text()
    subprocess.run(
        [str(fidelity_tools.compiler_python), "-c", CAPTURE_MUTATE, str(after)],
        check=True,
        capture_output=True,
        text=True,
    )
    sample = [[list(range(24)), list(range(100, 124)), list(range(200, 224)), [7], list(range(300, 324))]]
    for side, program, value in [("before", source, 5), ("after", original, 5), ("mutant", after.read_text(), 305)]:
        work = tmp_path / side
        diff.compare(program, sample, work, fidelity_tools, function="forward")
        assert json.loads((work / "native-results.json").read_text())["results"] == [
            [str(x) for x in sample[0][0] * 2 + [value]]
        ]
