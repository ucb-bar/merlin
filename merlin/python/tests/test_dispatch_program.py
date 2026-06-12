"""The dispatch program (``merlin-emit-dispatch-program``) the runtime consumes.

Flattens the outlined driver into an ordered DAG of kernel dispatches + view nodes, with
SSA buffer dependencies. Structural + DAG-validity checks run everywhere xDSL is present;
the real-model checks gate on the captured artifacts.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from merlin.xdsl_dialects import _common

pytestmark = pytest.mark.skipif(not _common.HAS_XDSL, reason="xDSL not installed")

REPO = Path(__file__).resolve().parents[3]

CHAIN = """
builtin.module {
  func.func @forward(%w: tensor<8x6xf32>, %x: tensor<4x8xf32>, %z: tensor<6x5xf32>)
      -> tensor<4x5xf32> {
    %e0 = tensor.empty() : tensor<4x6xf32>
    %c0 = arith.constant 0.0 : f32
    %f0 = linalg.fill ins(%c0 : f32) outs(%e0 : tensor<4x6xf32>) -> tensor<4x6xf32>
    %y0 = linalg.matmul ins(%x, %w : tensor<4x8xf32>, tensor<8x6xf32>)
          outs(%f0 : tensor<4x6xf32>) -> tensor<4x6xf32>
    %e1 = tensor.empty() : tensor<4x5xf32>
    %c1 = arith.constant 0.0 : f32
    %f1 = linalg.fill ins(%c1 : f32) outs(%e1 : tensor<4x5xf32>) -> tensor<4x5xf32>
    %y1 = linalg.matmul ins(%y0, %z : tensor<4x6xf32>, tensor<6x5xf32>)
          outs(%f1 : tensor<4x5xf32>) -> tensor<4x5xf32>
    func.return %y1 : tensor<4x5xf32>
  }
}
"""


def _program(text, prune=True):
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.xdsl_dialects.lowering.dispatch_program import (
        lower_model_to_dispatch_program)

    _, prog = lower_model_to_dispatch_program(parse_mlir_text(text), prune=prune)
    return prog


def test_program_is_a_valid_dag_of_dispatches():
    from merlin.xdsl_dialects.lowering.dispatch_program import verify_program

    prog = _program(CHAIN)
    assert prog.n_dispatches == 2
    assert verify_program(prog) == []
    # chained: kernel_1 consumes kernel_0's output buffer.
    d0, d1 = [n for n in prog.nodes if n.kind == "dispatch"]
    assert d0.outputs[0] in d1.inputs
    assert prog.results == d1.outputs


def test_args_are_bound_with_indices():
    prog = _program(CHAIN)
    arg_buffers = [b for b in prog.buffers.values() if b.kind == "arg"]
    assert sorted(b.arg_index for b in arg_buffers) == [0, 1, 2]
    assert all(b.shape and b.dtype == "f32" for b in arg_buffers)


def test_prune_removes_dead_accumulator_copies():
    """Without prune the dead cloned fills appear as view nodes; pruning drops them."""
    full = _program(CHAIN, prune=False)
    pruned = _program(CHAIN, prune=True)
    assert any(n.op == "linalg.fill" for n in full.nodes)
    assert all(n.kind == "dispatch" for n in pruned.nodes)
    assert len(pruned.nodes) == 2


def test_program_is_json_serializable():
    prog = _program(CHAIN)
    blob = json.dumps(prog.to_dict())
    back = json.loads(blob)
    assert back["entry"] == "forward"
    assert len(back["nodes"]) == 2
    assert set(back["buffers"]) >= set(back["results"])


@pytest.mark.skipif(not (REPO / "output/small_consistent/model.mlir").is_file(),
                    reason="small_llama capture not present")
def test_program_on_real_small_llama():
    from merlin.frontends.linalg_mlir import parse_mlir_file
    from merlin.xdsl_dialects.lowering.dispatch_program import (
        lower_model_to_dispatch_program, verify_program)

    m = parse_mlir_file(REPO / "output/small_consistent/model.mlir")
    _, prog = lower_model_to_dispatch_program(m)
    assert verify_program(prog) == []           # a well-formed DAG over real buffers
    matmuls = [n for n in prog.nodes
               if n.kind == "dispatch" and n.prov.get("prov.op") == "matmul"]
    assert len(matmuls) == 15
    json.dumps(prog.to_dict())                   # serializes for the runtime
