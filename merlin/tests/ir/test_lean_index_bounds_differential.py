"""Independent native references for safely bounded modular index arithmetic."""

import json

import pytest
from test_lean_index_bounds import SAFE, _captured_arithmetic, _source
from test_lean_reshape_compose import _verify_mlir
from test_lean_splat_nested import _stage as _splat_stage

from merlin.verify.lean import differential as diff


def _result(kind, lhs, rhs, width):
    result = {"addi": lambda: lhs + rhs, "subi": lambda: lhs - rhs, "muli": lambda: lhs * rhs}[kind]()
    result %= 1 << width
    return result if result < 1 << (width - 1) else result - (1 << width)


@pytest.mark.parametrize("index_bits", [32, 64])
@pytest.mark.parametrize("kind,lhs,rhs,size", SAFE)
def test_index_arithmetic_matches_independent_reference(tmp_path, fidelity_tools, index_bits, kind, lhs, rhs, size):
    coordinate = _result(kind, lhs, rhs, index_bits)
    assert 0 <= coordinate < size
    data = [11 + 17 * i for i in range(size)]
    diff.compare(_source(kind, lhs, rhs, size), [[data]], tmp_path / "native", fidelity_tools, index_bits=index_bits)
    assert json.loads((tmp_path / "native/native-results.json").read_text())["results"] == [[str(data[coordinate])]]


@pytest.mark.parametrize("kind,lhs,rhs", [("muli", 2147483647, 2147483647), ("addi", 4294967296, 1)])
def test_index32_arithmetic_wrap_is_observed(tmp_path, fidelity_tools, kind, lhs, rhs):
    coordinate = _result(kind, lhs, rhs, 32)
    assert coordinate == 1
    diff.compare(_source(kind, lhs, rhs, 2), [[[13, 29]]], tmp_path / "native", fidelity_tools, index_bits=32)
    assert json.loads((tmp_path / "native/native-results.json").read_text())["results"] == [["29"]]


@pytest.mark.parametrize("index_bits", [32, 64])
def test_arithmetic_captured_index_native(tmp_path, fidelity_tools, index_bits):
    stage, count = _splat_stage(tmp_path, _captured_arithmetic())
    assert count == 1
    sample = [[[1, 2], [3, 4], [10, 20, 30, 40, 50, 60], [0] * 6, [7], [0]]]
    for side in ["before", "after"]:
        work = tmp_path / side
        diff.compare((stage / f"{side}.mlir").read_text(), sample, work, fidelity_tools, index_bits=index_bits)
        assert json.loads((work / "native-results.json").read_text())["results"] == [["71", "73"]]


def test_wrong_inbounds_arithmetic_capture_is_observable(tmp_path, fidelity_tools):
    stage, _ = _splat_stage(tmp_path, _captured_arithmetic())
    after = stage / "after.mlir"
    after.write_text(after.read_text().replace("arith.constant 1 : index", "arith.constant 0 : index", 1))
    _verify_mlir(after, fidelity_tools.compiler_python)
    sample = [[[1, 2], [3, 4], [10, 20, 30, 40, 50, 60], [0] * 6, [7], [0]]]
    for side, expected in [("before", ["71", "73"]), ("after", ["41", "43"])]:
        work = tmp_path / side
        diff.compare((stage / f"{side}.mlir").read_text(), sample, work, fidelity_tools)
        assert json.loads((work / "native-results.json").read_text())["results"] == [expected]
