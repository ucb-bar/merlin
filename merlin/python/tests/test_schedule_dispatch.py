"""Multicore partitioning of the dispatch DAG (``merlin-partition-dispatches``).

Pure graph scheduling over the dispatch program — runs everywhere xDSL is present (no
toolchain needed). Correctness is structural: the schedule is dependency-safe iff every
edge crosses a barrier upward, which (on a single-writer dataflow program) makes
whole-level parallel execution equivalent to the serial run.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from merlin.xdsl_dialects import _common

pytestmark = pytest.mark.skipif(not _common.HAS_XDSL, reason="xDSL not installed")

REPO = Path(__file__).resolve().parents[3]


def _prog(text):
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.xdsl_dialects.lowering.dispatch_program import (
        lower_model_to_dispatch_program)

    _, prog = lower_model_to_dispatch_program(parse_mlir_text(text))
    return prog


DIAMOND = """
builtin.module {
  func.func @forward(%x: tensor<4x4xf32>, %a: tensor<4x4xf32>, %b: tensor<4x4xf32>)
      -> tensor<4x4xf32> {
    %e = tensor.empty() : tensor<4x4xf32>
    %c = arith.constant 0.0 : f32
    %f0 = linalg.fill ins(%c : f32) outs(%e : tensor<4x4xf32>) -> tensor<4x4xf32>
    %p = linalg.matmul ins(%x, %a : tensor<4x4xf32>, tensor<4x4xf32>)
         outs(%f0 : tensor<4x4xf32>) -> tensor<4x4xf32>
    %f1 = linalg.fill ins(%c : f32) outs(%e : tensor<4x4xf32>) -> tensor<4x4xf32>
    %q = linalg.matmul ins(%x, %b : tensor<4x4xf32>, tensor<4x4xf32>)
         outs(%f1 : tensor<4x4xf32>) -> tensor<4x4xf32>
    %f2 = linalg.fill ins(%c : f32) outs(%e : tensor<4x4xf32>) -> tensor<4x4xf32>
    %r = linalg.matmul ins(%p, %q : tensor<4x4xf32>, tensor<4x4xf32>)
         outs(%f2 : tensor<4x4xf32>) -> tensor<4x4xf32>
    func.return %r : tensor<4x4xf32>
  }
}
"""

# a, b, c sequential — no parallelism
CHAIN = """
builtin.module {
  func.func @forward(%x: tensor<4x4xf32>, %a: tensor<4x4xf32>, %b: tensor<4x4xf32>)
      -> tensor<4x4xf32> {
    %e = tensor.empty() : tensor<4x4xf32>
    %c = arith.constant 0.0 : f32
    %f0 = linalg.fill ins(%c : f32) outs(%e : tensor<4x4xf32>) -> tensor<4x4xf32>
    %p = linalg.matmul ins(%x, %a : tensor<4x4xf32>, tensor<4x4xf32>)
         outs(%f0 : tensor<4x4xf32>) -> tensor<4x4xf32>
    %f1 = linalg.fill ins(%c : f32) outs(%e : tensor<4x4xf32>) -> tensor<4x4xf32>
    %q = linalg.matmul ins(%p, %b : tensor<4x4xf32>, tensor<4x4xf32>)
         outs(%f1 : tensor<4x4xf32>) -> tensor<4x4xf32>
    func.return %q : tensor<4x4xf32>
  }
}
"""


def test_diamond_exposes_parallelism():
    from merlin.xdsl_dialects.lowering.schedule_dispatch import partition_dispatches, validate

    prog = _prog(DIAMOND)
    pr = partition_dispatches(prog, n_harts=2)
    assert validate(prog, pr.schedule) == []
    assert pr.schedule.depth == 2                 # {two independent matmuls}, {combiner}
    assert pr.schedule.max_width == 2
    assert pr.stats["speedup"] > 1.0              # the two matmuls run concurrently


def test_emit_schedule_c_header():
    """The partition emits a C dispatch table the multicore runtime consumes."""
    from merlin.xdsl_dialects.lowering.schedule_dispatch import (emit_schedule_c,
                                                                 partition_dispatches)

    prog = _prog(DIAMOND)
    pr = partition_dispatches(prog, n_harts=4)
    hdr = emit_schedule_c(prog, pr.schedule)
    assert "merlin_dispatch_t" in hdr
    assert "MERLIN_SCHEDULE_N 3" in hdr
    assert "MERLIN_SCHEDULE_LEVELS 2" in hdr
    # the level-1 combiner consumes the two level-0 outputs: one buffer index appears
    # both as a level-0 out_buf and in the level-1 in_buf list.
    assert hdr.count("forward$kernel_") == 3
    assert ', 1, 0, {' in hdr            # one dispatch at level 1 on hart 0


def test_chain_has_no_parallelism():
    from merlin.xdsl_dialects.lowering.schedule_dispatch import partition_dispatches, validate

    prog = _prog(CHAIN)
    pr = partition_dispatches(prog, n_harts=4)
    assert validate(prog, pr.schedule) == []
    assert pr.schedule.depth == 2                 # strictly sequential
    assert pr.schedule.max_width == 1
    assert pr.stats["speedup"] == 1.0             # nothing to overlap


def test_more_harts_never_slower():
    from merlin.xdsl_dialects.lowering.schedule_dispatch import partition_dispatches

    prog = _prog(DIAMOND)
    spans = [partition_dispatches(prog, n_harts=h).schedule.makespan for h in (1, 2, 4)]
    assert spans[0] >= spans[1] >= spans[2]
    # 1 hart makespan equals the serial cost (no parallelism possible)
    assert spans[0] == partition_dispatches(prog, n_harts=1).schedule.serial_cost


@pytest.mark.skipif(not (REPO / "artifacts/recaptures/small_consistent/model.mlir").is_file(),
                    reason="small_llama capture not present")
def test_real_model_schedule_is_valid_and_parallel():
    from merlin.frontends.linalg_mlir import parse_mlir_file
    from merlin.xdsl_dialects.lowering.dispatch_program import (
        lower_model_to_dispatch_program)
    from merlin.xdsl_dialects.lowering.schedule_dispatch import partition_dispatches, validate

    _, prog = lower_model_to_dispatch_program(
        parse_mlir_file(REPO / "artifacts/recaptures/small_consistent/model.mlir"))
    pr = partition_dispatches(prog, n_harts=4)
    assert validate(prog, pr.schedule) == []      # dependency-safe across barriers
    assert pr.schedule.max_width > 1              # real intra-layer parallelism exists
    assert pr.stats["makespan"] < pr.stats["serial_cost"]   # 4 harts beat 1
    assert pr.stats["critical_path_cost"] <= pr.stats["makespan"]  # CP is the lower bound
