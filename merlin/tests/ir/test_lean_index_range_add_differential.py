"""Native interval-addition comparisons with independent indexing references and safe mutants."""

import json

import pytest
from test_lean_index_range_add import _captured_offset, _source
from test_lean_reshape_compose import _verify_mlir
from test_lean_splat_nested import _stage as _splat_stage

from merlin.verify.lean import differential as diff


@pytest.mark.parametrize("index_bits", [32, 64])
@pytest.mark.parametrize("width,offset", [(1, 0), (1, 1), (3, 1), (8, 1), (8, 7)])
def test_range_addition_matches_independent_reference(tmp_path, fidelity_tools, index_bits, width, offset):
    data = [13 + 17 * value for value in range((1 << width) + offset)]
    values = list(range(1 << width)) if width < 8 else [0, 1, 127, 128, 254, 255]
    samples = [[data, [value]] for value in values]
    diff.compare(_source(width, len(data), offset), samples, tmp_path / "native", fidelity_tools, index_bits=index_bits)
    assert json.loads((tmp_path / "native/native-results.json").read_text())["results"] == [
        [str(data[value + offset])] for value in values
    ]


@pytest.mark.parametrize("index_bits", [32, 64])
def test_captured_offset_and_wrong_inbounds_offset_native(tmp_path, fidelity_tools, index_bits):
    stage, _ = _splat_stage(tmp_path, _captured_offset())
    after = stage / "after.mlir"
    original = after.read_text()
    after.write_text(original.replace("arith.constant 1 : index", "arith.constant 0 : index", 1))
    _verify_mlir(after, fidelity_tools.compiler_python)
    data = [value * 3 for value in range(257)]
    sample = [[[1, 2], [3, 4], data, [0] * 257, [7], [0], [128]]]
    for label, source, expected in [
        ("source", (stage / "before.mlir").read_text(), ["398", "400"]),
        ("target", original, ["398", "400"]),
        ("mutant", after.read_text(), ["395", "397"]),
    ]:
        work = tmp_path / label
        diff.compare(source, sample, work, fidelity_tools, index_bits=index_bits)
        assert json.loads((work / "native-results.json").read_text())["results"] == [expected]
