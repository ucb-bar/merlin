"""Regressions for the target-neutral xDSL capture parser bridge."""
from __future__ import annotations

import hashlib
import io
import sys
from pathlib import Path

from xdsl.printer import Printer


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
sys.path.insert(0, str(ROOT / "submission"))

from mlir_oot.frontend import (  # noqa: E402
    normalize_xdsl_parser_compat,
    parse_verified,
)


def test_multi_result_normalization_preserves_ordered_result_types() -> None:
    source = """builtin.module {
  func.func @pair(%x: tensor<1x2xi64>, %i0: tensor<1xi64>, %i1: tensor<1xi64>) -> (tensor<1xi64>, tensor<1xi64>) {
    %y0, %y1 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%x : tensor<1x2xi64>) outs(%i0, %i1 : tensor<1xi64>, tensor<1xi64>) {
    ^bb0(%v: i64, %a0: i64, %a1: i64):
      linalg.yield %v, %v : i64, i64
    } -> (tensor<1xi64>, tensor<1xi64>)
    return %y0, %y1 : tensor<1xi64>, tensor<1xi64>
  }
}
"""
    normalized, rewrites = normalize_xdsl_parser_compat(source)
    assert rewrites == 1
    assert "} -> tensor<1xi64>, tensor<1xi64>" in normalized
    assert "outs(%i0, %i1 : tensor<1xi64>, tensor<1xi64>)" in normalized
    assert normalize_xdsl_parser_compat(normalized) == (normalized, 0)

    workload = parse_verified(source)
    generic = next(op for op in workload.module.walk() if op.name == "linalg.generic")
    assert [str(result.type) for result in generic.results] == [
        "tensor<1xi64>",
        "tensor<1xi64>",
    ]
    stream = io.StringIO()
    Printer(stream=stream).print_op(workload.module)
    # xDSL's own printer recreates the tuple wrapper.  Feeding that output
    # through the same bridge must therefore be a stable parser round trip.
    reparsed = parse_verified(stream.getvalue())
    reparsed_generic = next(
        op for op in reparsed.module.walk() if op.name == "linalg.generic"
    )
    assert [str(result.type) for result in reparsed_generic.results] == [
        "tensor<1xi64>",
        "tensor<1xi64>",
    ]


def test_full_smolvla_capture_parses_without_mutating_the_capture() -> None:
    capture = REPO / "out/artifacts/recaptures/smolvla_fp32_consistent/model.mlir"
    before = capture.read_bytes()
    source = before.decode("utf-8")
    normalized, rewrites = normalize_xdsl_parser_compat(source)
    assert rewrites == 8
    assert len(source) - len(normalized) == 16

    workload = parse_verified(source)
    assert workload.grammar == "linalg-on-tensors"
    assert len(workload.tensors) == 513
    assert len(workload.outputs) == 1
    assert workload.ops[0]["op"] == "model"
    assert len(workload.ops[0]["payload"]) == 8570
    assert hashlib.sha256(capture.read_bytes()).digest() == hashlib.sha256(before).digest()
