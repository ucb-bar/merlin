from __future__ import annotations

import pytest

from merlin.compiler.dialects import _common

pytestmark = pytest.mark.skipif(not _common.HAS_XDSL, reason="xDSL not installed")
LOOP_CAPTURE = "\nbuiltin.module {\n  func.func @forward(%x: tensor<64x64xf32>) -> tensor<64x64xf32> {\n    %c0 = arith.constant 0 : index\n    %c1 = arith.constant 1 : index\n    %c2 = arith.constant 2 : index\n    %t0 = func.call @forward$kernel_0(%x) : (tensor<64x64xf32>) -> tensor<64x64xf32>\n    %t1 = func.call @forward$kernel_1(%x) : (tensor<64x64xf32>) -> tensor<64x64xf32>\n    %t2 = func.call @forward$kernel_2(%t1) : (tensor<64x64xf32>) -> tensor<64x64xf32>\n    %lp = scf.for %i = %c0 to %c2 step %c1 iter_args(%acc = %t2) -> (tensor<64x64xf32>) {\n      %s = linalg.add ins(%acc, %t0 : tensor<64x64xf32>, tensor<64x64xf32>)\n           outs(%acc : tensor<64x64xf32>) -> tensor<64x64xf32>\n      scf.yield %s : tensor<64x64xf32>\n    }\n    func.return %lp : tensor<64x64xf32>\n  }\n  func.func private @forward$kernel_0(%p: tensor<64x64xf32>) -> tensor<64x64xf32> {\n    func.return %p : tensor<64x64xf32> }\n  func.func private @forward$kernel_1(%p: tensor<64x64xf32>) -> tensor<64x64xf32> {\n    func.return %p : tensor<64x64xf32> }\n  func.func private @forward$kernel_2(%p: tensor<64x64xf32>) -> tensor<64x64xf32> {\n    func.return %p : tensor<64x64xf32> }\n}\n"


def _build(text: str, n_dispatches: int, *, tagged: bool = False):
    """Flatten ``text`` against a dispatch table of ``n_dispatches`` entries, as the outliner hands it
    over. Built directly (rather than by running the outliner) because the outliner only walks the
    driver's TOP-LEVEL ops -- which is exactly the blind spot under test."""
    from merlin.compiler.dialects.lowering.dispatch_program import build_dispatch_program
    from merlin.compiler.dialects.lowering.outline import DispatchInfo, OutlineResult
    from merlin.compiler.frontend.linalg_mlir import parse_mlir_text

    module = parse_mlir_text(text)
    functions = {op.sym_name.data: op for op in module.body.block.ops if op.name == "func.func"}
    suffix = "__rmatmul_{}" if tagged else ""
    table = [
        DispatchInfo(
            index=i,
            symbol=f"forward$kernel_{i}" + (suffix.format(i) if tagged else ""),
            root_op="linalg.matmul",
            n_operands=1,
            result_types=[str(value) for value in functions[symbol].function_type.outputs]
            if (symbol := (f"forward$kernel_{i}" + (suffix.format(i) if tagged else ""))) in functions
            else [],
            prov={"prov.region_id": f"matmul_{i}"} if tagged else {},
        )
        for i in range(n_dispatches)
    ]
    return build_dispatch_program(OutlineResult(module=module, dispatches=table))


def test_a_value_only_the_loop_body_reads_is_recorded_as_an_input():
    """`%t0` is used inside the scf.for body and is not an operand of the loop. Before the fix it was
    absent from the node's inputs, so nothing downstream knew the loop reads it."""
    prog = _build(LOOP_CAPTURE, 3)
    loop = next((n for n in prog.nodes if n.op == "scf.for"))
    producer = next((n for n in prog.nodes if n.op == "forward$kernel_0"))
    captured = producer.outputs[0]
    assert captured in loop.inputs, (captured, loop.inputs)
    assert loop.captures == [captured], loop.captures
    assert loop.regions == 1
    assert len(loop.captures) == 1


def test_the_planner_does_not_reuse_bytes_a_loop_is_still_reading():
    """The consequence, checked where it bites. Before the fix `last_use` for the captured buffer was
    node 3 (its definition) though the loop reads it at node 6, and the planner seated the NEXT
    intermediate at the same offset: two simultaneously-live buffers sharing bytes."""
    from merlin.compiler.dialects.lowering.arena_plan import _live_ranges, plan_arena

    prog = _build(LOOP_CAPTURE, 3)
    loop_idx = next((i for i, n in enumerate(prog.nodes) if n.op == "scf.for"))
    captured = next((n for n in prog.nodes if n.op == "forward$kernel_0")).outputs[0]
    _, last_use = _live_ranges(prog)
    assert last_use[captured] == loop_idx, (captured, last_use[captured], loop_idx)
    plan = plan_arena(prog)
    first_def, last = _live_ranges(prog)
    placed = [(b, plan.offsets[b], plan.sizes[b]) for b in plan.offsets]
    for i, (bi, oi, si) in enumerate(placed):
        for bj, oj, sj in placed[i + 1 :]:
            overlaps = oi < oj + sj and oj < oi + si
            together = first_def[bi] <= last[bj] and first_def[bj] <= last[bi]
            assert not (overlaps and together), (bi, bj, oi, si, oj, sj)
