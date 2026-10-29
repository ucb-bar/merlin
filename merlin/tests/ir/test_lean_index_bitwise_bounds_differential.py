"""Independent native bitwise references, including unknown index inputs and safe changed targets."""

import json

import pytest
from test_lean_index_bitwise_bounds import CASES, _captured_bitwise, _source
from test_lean_reshape_compose import _verify_mlir
from test_lean_splat_nested import _stage as _splat_stage

from merlin.verify.lean import differential as diff


def _apply(kind, lhs, rhs, width):
    return {"andi": lambda: lhs & rhs, "ori": lambda: lhs | rhs, "xori": lambda: lhs ^ rhs}[kind]() & ((1 << width) - 1)


@pytest.mark.parametrize("index_bits", [32, 64])
@pytest.mark.parametrize("kind,lhs,rhs,size", CASES)
def test_bitwise_indices_match_independent_bits(tmp_path, fidelity_tools, index_bits, kind, lhs, rhs, size):
    coordinate = _apply(kind, lhs, rhs, index_bits)
    assert 0 <= coordinate < size
    data = [13 + 17 * n for n in range(size)]
    diff.compare(_source(kind, lhs, rhs, size), [[data]], tmp_path / "native", fidelity_tools, index_bits=index_bits)
    assert json.loads((tmp_path / "native/native-results.json").read_text())["results"] == [[str(data[coordinate])]]


@pytest.mark.parametrize("index_bits", [32, 64])
@pytest.mark.parametrize("kind", ["andi", "ori", "xori"])
def test_unknown_bitwise_index_inputs_preserve_initialized_bits(tmp_path, fidelity_tools, index_bits, kind):
    values = [0, 1, (1 << (index_bits - 1)), (1 << index_bits) - 1]
    samples = [[[0, 0], [value]] for value in values]
    diff.compare(
        _source(kind, dynamic=True, extract=False), samples, tmp_path / "native", fidelity_tools, index_bits=index_bits
    )
    assert json.loads((tmp_path / "native/native-results.json").read_text())["results"] == [
        [str(_apply(kind, value, 1, index_bits))] for value in values
    ]


@pytest.mark.parametrize("kind,lhs,rhs,size", [("andi", 4294967296, 4294967296, 1), ("ori", 4294967296, 1, 2)])
def test_bitwise_index32_wrap_native(tmp_path, fidelity_tools, kind, lhs, rhs, size):
    coordinate = _apply(kind, lhs, rhs, 32)
    data = [13 + 17 * n for n in range(size)]
    diff.compare(_source(kind, lhs, rhs, size), [[data]], tmp_path / "native", fidelity_tools, index_bits=32)
    assert json.loads((tmp_path / "native/native-results.json").read_text())["results"] == [[str(data[coordinate])]]


@pytest.mark.parametrize("index_bits", [32, 64])
def test_changed_bitwise_capture_is_observable(tmp_path, fidelity_tools, index_bits):
    stage, _ = _splat_stage(tmp_path, _captured_bitwise())
    after = stage / "after.mlir"
    original = after.read_text()
    after.write_text(original.replace("arith.xori", "arith.andi", 1))
    _verify_mlir(after, fidelity_tools.compiler_python)
    sample = [[[1, 2], [3, 4], [10, 20, 30, 40, 50, 60], [0] * 6, [7], [0]]]
    for label, source, expected in [
        ("source", (stage / "before.mlir").read_text(), ["71", "73"]),
        ("target", original, ["71", "73"]),
        ("mutant", after.read_text(), ["61", "63"]),
    ]:
        work = tmp_path / label
        diff.compare(source, sample, work, fidelity_tools, index_bits=index_bits)
        assert json.loads((work / "native-results.json").read_text())["results"] == [expected]
