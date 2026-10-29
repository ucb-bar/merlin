"""Native unsigned-widening extraction references; signed negative/OOB reads never execute."""

import json

import pytest
from test_lean_index_cast_bounds import _captured_cast, _source
from test_lean_splat_nested import _stage as _splat_stage

from merlin.verify.lean import differential as diff


@pytest.mark.parametrize("index_bits", [32, 64])
@pytest.mark.parametrize("width", [1, 3, 8])
def test_all_fixed_width_inputs_index_unsigned_range(tmp_path, fidelity_tools, index_bits, width):
    data = [13 + 17 * value for value in range(1 << width)]
    # Exhaust every i1/i3 input; i8 covers both sides of the signed boundary plus largest value.
    values = list(range(1 << width)) if width < 8 else [0, 1, 127, 128, 254, 255]
    samples = [[data, [value]] for value in values]
    diff.compare(_source(width, len(data)), samples, tmp_path / "native", fidelity_tools, index_bits=index_bits)
    assert json.loads((tmp_path / "native/native-results.json").read_text())["results"] == [
        [str(data[value])] for value in values
    ]


@pytest.mark.parametrize("index_bits", [32, 64])
def test_signed_unsigned_cast_results_differ_observably(tmp_path, fidelity_tools, index_bits):
    for signed, expected in [(False, 255), (True, (1 << index_bits) - 1)]:
        work = tmp_path / str(signed)
        diff.compare(
            _source(extract=False, signed=signed), [[[0] * 256, [255]]], work, fidelity_tools, index_bits=index_bits
        )
        assert json.loads((work / "native-results.json").read_text())["results"] == [[str(expected)]]


@pytest.mark.parametrize("index_bits", [32, 64])
def test_captured_unsigned_index_native(tmp_path, fidelity_tools, index_bits):
    stage, count = _splat_stage(tmp_path, _captured_cast())
    assert count == 1
    data = [value * 3 for value in range(256)]
    sample = [[[1, 2], [3, 4], data, [0] * 256, [7], [0], [128]]]
    for side in ["before", "after"]:
        work = tmp_path / side
        diff.compare((stage / f"{side}.mlir").read_text(), sample, work, fidelity_tools, index_bits=index_bits)
        assert json.loads((work / "native-results.json").read_text())["results"] == [["395", "397"]]
