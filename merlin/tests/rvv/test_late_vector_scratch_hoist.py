"""Vector-lowering scratch must not allocate once per contraction iteration.

``convert-vector-to-scf`` creates fixed-size memref scratch after the pipeline's original buffer
hoisting stage.  If that scratch remains inside a reduction loop, LLVM gives every iteration a new
stack address and reclaims none of them until ``forward`` returns.  A single ordinary contraction
then exhausts the process stack even though its static frame is small.
"""
from __future__ import annotations

import pytest

from merlin.llvmlower import pipeline as P
from merlin.llvmlower.passes_xdsl import preprocess_text_textual
from merlin.llvmlower.toolchain import available as toolchain_available


_needs_toolchain = pytest.mark.skipif(
    not toolchain_available(), reason="m2m lowering toolchain unavailable")


_INT8_MATMUL = """
module {
  func.func @forward(%a: tensor<1x512xi8>, %b: tensor<512x64xi8>) -> tensor<1x64xi32> {
    %zero = arith.constant 0 : i32
    %empty = tensor.empty() : tensor<1x64xi32>
    %init = linalg.fill ins(%zero : i32) outs(%empty : tensor<1x64xi32>)
      -> tensor<1x64xi32>
    %result = linalg.matmul ins(%a, %b : tensor<1x512xi8>, tensor<512x64xi8>)
      outs(%init : tensor<1x64xi32>) -> tensor<1x64xi32>
    return %result : tensor<1x64xi32>
  }
}
"""


def _function_cfg(llvm_ir: str, symbol: str) -> tuple[dict[str, list[str]], dict[str, set[str]]]:
    """Return block lines and branch edges for one textual LLVM function."""
    blocks: dict[str, list[str]] = {}
    edges: dict[str, set[str]] = {}
    current = "entry"
    inside = False
    for line in llvm_ir.splitlines():
        stripped = line.strip()
        if not inside:
            if stripped.startswith("define ") and f"@{symbol}(" in stripped:
                inside = True
                blocks[current] = []
            continue
        if stripped == "}":
            break
        # LLVM basic-block labels are unindented.  Instructions (including their type labels) are
        # indented, so this does not depend on numeric SSA/block names.
        if line and not line[0].isspace() and stripped.endswith(":"):
            current = stripped[:-1]
            blocks.setdefault(current, [])
            continue
        if line and not line[0].isspace() and ":" in stripped:
            head, _, tail = stripped.partition(":")
            if tail.lstrip().startswith("; preds"):
                current = head
                blocks.setdefault(current, [])
                continue
        blocks[current].append(stripped)

    for block, lines in blocks.items():
        edges[block] = set()
        for line in lines:
            instruction = line.partition(";")[0].strip()
            if not (instruction.startswith("br ") or instruction.startswith("switch ")):
                continue
            for piece in instruction.split("label %")[1:]:
                target = piece.split(",", 1)[0].split(None, 1)[0]
                edges[block].add(target.strip('"'))
    return blocks, edges


def _cyclic_blocks(edges: dict[str, set[str]]) -> set[str]:
    """Blocks reachable from themselves by at least one CFG edge."""
    cyclic: set[str] = set()
    for origin in edges:
        pending = list(edges[origin])
        seen: set[str] = set()
        while pending:
            block = pending.pop()
            if block == origin:
                cyclic.add(origin)
                break
            if block in seen:
                continue
            seen.add(block)
            pending.extend(edges.get(block, ()))
    return cyclic


def test_late_hoist_runs_after_vector_to_scf_and_before_cfg_lowering():
    pipeline = P.build_rvv_pipeline("/schedule", features=frozenset())
    stage = "func.func(buffer-hoisting,buffer-loop-hoisting)"
    vector_to_scf = pipeline.index("convert-vector-to-scf")
    late_hoist = pipeline.index(stage, vector_to_scf)
    cfg_lowering = pipeline.index("convert-scf-to-cf")
    assert pipeline.count(stage) == 2
    assert vector_to_scf < late_hoist < cfg_lowering


@_needs_toolchain
def test_vector_transfer_scratch_is_not_allocated_in_a_loop(tmp_path):
    upstream, _ = preprocess_text_textual(_INT8_MATMUL)
    llvm_ir = P.lower_to_llvm_ir(upstream, workdir=tmp_path, vectorize=True)
    blocks, edges = _function_cfg(llvm_ir, "forward")
    cyclic = _cyclic_blocks(edges)
    offenders = {
        block: [line for line in lines if " = alloca " in line]
        for block, lines in blocks.items()
        if block in cyclic and any(" = alloca " in line for line in lines)
    }
    assert not offenders, f"loop-local stack scratch grows once per iteration: {offenders}"
