"""The dispatch program (``merlin-emit-dispatch-program``) the runtime consumes.

Flattens the outlined driver into an ordered DAG of kernel dispatches + view nodes, with
SSA buffer dependencies. Structural + DAG-validity checks run everywhere xDSL is present;
the real-model checks gate on the captured artifacts.
"""
from __future__ import annotations
from merlin.common.paths import repo_root, merlin_dir

import json
from pathlib import Path

import pytest

from merlin.xdsl_dialects import _common

pytestmark = pytest.mark.skipif(not _common.HAS_XDSL, reason="xDSL not installed")

REPO = repo_root()

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


# A prov-tagged chain: each matmul carries the model-layer provenance model2MLIR stamps.
PROV_CHAIN = """
builtin.module {
  func.func @forward(%w: tensor<8x6xf32>, %x: tensor<4x8xf32>, %z: tensor<6x5xf32>)
      -> tensor<4x5xf32> {
    %e0 = tensor.empty() : tensor<4x6xf32>
    %c0 = arith.constant 0.0 : f32
    %f0 = linalg.fill ins(%c0 : f32) outs(%e0 : tensor<4x6xf32>) -> tensor<4x6xf32>
    %y0 = linalg.matmul {prov.op = "matmul", prov.fqn = "blocks.0.attn.q", prov.region_id = "matmul_0"}
          ins(%x, %w : tensor<4x8xf32>, tensor<8x6xf32>)
          outs(%f0 : tensor<4x6xf32>) -> tensor<4x6xf32>
    %e1 = tensor.empty() : tensor<4x5xf32>
    %c1 = arith.constant 0.0 : f32
    %f1 = linalg.fill ins(%c1 : f32) outs(%e1 : tensor<4x5xf32>) -> tensor<4x5xf32>
    %y1 = linalg.matmul {prov.op = "matmul", prov.fqn = "blocks.0.mlp.g", prov.region_id = "matmul_1"}
          ins(%y0, %z : tensor<4x6xf32>, tensor<6x5xf32>)
          outs(%f1 : tensor<4x5xf32>) -> tensor<4x5xf32>
    func.return %y1 : tensor<4x5xf32>
  }
}
"""


def test_dispatch_symbol_encodes_region_id():
    """The region_id rides INTO the dispatch symbol (the thread that survives to the ELF), while the
    ``$kernel_<idx>`` marker stays intact so driver-vs-kernel detection is unaffected."""
    from merlin.xdsl_dialects.lowering.outline import outline_dispatches, region_id_of_symbol
    from merlin.frontends.linalg_mlir import parse_mlir_text

    r = outline_dispatches(parse_mlir_text(PROV_CHAIN))
    syms = [d.symbol for d in r.dispatches]
    assert syms == ["forward$kernel_0__rmatmul_0", "forward$kernel_1__rmatmul_1"]
    assert all("$kernel_" in s for s in syms)                       # marker preserved
    assert [region_id_of_symbol(s) for s in syms] == ["matmul_0", "matmul_1"]


def test_untagged_symbol_is_backward_compatible():
    """No prov.region_id (pre-provenance capture) -> the symbol is byte-identical to before."""
    from merlin.xdsl_dialects.lowering.outline import outline_dispatches, region_id_of_symbol
    from merlin.frontends.linalg_mlir import parse_mlir_text

    r = outline_dispatches(parse_mlir_text(CHAIN))
    assert [d.symbol for d in r.dispatches] == ["forward$kernel_0", "forward$kernel_1"]
    assert all(region_id_of_symbol(d.symbol) is None for d in r.dispatches)


def test_prov_flows_to_dispatch_program_nodes():
    prog = _program(PROV_CHAIN)
    disp = [n for n in prog.nodes if n.kind == "dispatch"]
    assert [n.prov.get("prov.region_id") for n in disp] == ["matmul_0", "matmul_1"]
    assert [n.prov.get("prov.fqn") for n in disp] == ["blocks.0.attn.q", "blocks.0.mlp.g"]


# --- C8: selective section slicing (compile whole, profile a section) -------------------------

def test_slice_single_region_is_standalone_and_smaller():
    from merlin.xdsl_dialects.lowering.arena_plan import plan_arena
    from merlin.xdsl_dialects.lowering.dispatch_program import slice_program, verify_program

    prog = _program(PROV_CHAIN)
    s0 = slice_program(prog, {"matmul_0"}, entry_suffix="$r_matmul_0")
    assert verify_program(s0) == []                       # a valid standalone DAG
    assert s0.n_dispatches == 1
    assert s0.entry == "forward$r_matmul_0"
    # matmul_0's output leaves the slice (matmul_1 consumed it) -> it is the slice's result.
    d0 = next(n for n in s0.nodes if n.kind == "dispatch")
    assert s0.results == d0.outputs
    # its inputs (model x, w) are boundary args, fed from region-boundary tensors.
    assert all(s0.buffers[b].kind == "arg" for b in d0.inputs)
    # a section arena is no larger than the whole-model arena.
    assert plan_arena(s0).arena_bytes <= plan_arena(prog).arena_bytes


def test_slice_midgraph_region_reclassifies_upstream_output_as_arg():
    from merlin.xdsl_dialects.lowering.dispatch_program import slice_program

    prog = _program(PROV_CHAIN)
    s1 = slice_program(prog, {"matmul_1"})
    d1 = next(n for n in s1.nodes if n.kind == "dispatch")
    # matmul_1 consumes matmul_0's output (produced OUTSIDE this slice) -> that buffer is a boundary arg.
    boundary = [b for b in d1.inputs if s1.buffers[b].kind == "arg"]
    assert len(boundary) == len(d1.inputs)                # both inputs enter from the boundary
    assert s1.results == list(prog.results)               # matmul_1 output is the model output


def test_slice_combined_regions_and_bad_id():
    import pytest as _pt

    from merlin.xdsl_dialects.lowering.dispatch_program import slice_program, verify_program

    prog = _program(PROV_CHAIN)
    both = slice_program(prog, {"matmul_0", "matmul_1"})   # combined = one sub-program
    assert both.n_dispatches == 2 and verify_program(both) == []
    assert both.results == list(prog.results)
    with _pt.raises(ValueError):
        slice_program(prog, {"nonexistent_region"})


@pytest.mark.skipif(not (REPO / "out/artifacts/recaptures/small_consistent/model.mlir").is_file(),
                    reason="small_llama capture not present")
def test_program_on_real_small_llama():
    from merlin.frontends.linalg_mlir import parse_mlir_file
    from merlin.xdsl_dialects.lowering.dispatch_program import (
        lower_model_to_dispatch_program, verify_program)

    m = parse_mlir_file(REPO / "out/artifacts/recaptures/small_consistent/model.mlir")
    _, prog = lower_model_to_dispatch_program(m)
    assert verify_program(prog) == []           # a well-formed DAG over real buffers
    matmuls = [n for n in prog.nodes
               if n.kind == "dispatch" and n.prov.get("prov.op") == "matmul"]
    assert len(matmuls) == 15
    json.dumps(prog.to_dict())                   # serializes for the runtime
