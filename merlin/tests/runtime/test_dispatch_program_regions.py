"""Region-carrying driver ops: the two ways the dispatch program used to be SILENTLY WRONG.

Both defects share a signature that makes them worse than a crash: the emitted program stays a
self-consistent DAG, ``verify_program`` returns ``[]``, and nothing downstream can tell.

1. **Under-recorded reads.** ``op.operands`` on an ``scf.for`` is lb/ub/step/iter_args and nothing
   else, so a tensor the loop BODY reads out of the enclosing scope was absent from ``node.inputs``.
   ``arena_plan._live_ranges`` reads liveness from ``node.inputs`` alone, so that buffer's live range
   ended before its last read and the planner handed its bytes to the next buffer. Measured on the
   program below before the fix: ``last_use[b4] == 3`` (the loop reads it at node 6), and ``b4`` and
   ``b5`` were both placed at offset 0 while both were live — a 32768-byte arena where 49152 is
   needed. Wrong numbers, on the shapes that happen to collide, with no diagnostic anywhere.

2. **A kernel call hidden in a region.** The top-level ``block.ops`` walk that pairs driver calls with
   the outlined dispatch table cannot see a nested ``func.call``, so every LATER top-level call took
   the wrong table entry. Measured on the program below before the fix: the THIRD call in the IR
   (``forward$kernel_2__rmatmul_2``) was emitted as ``forward$kernel_1__rmatmul_1`` carrying
   ``prov.region_id == "matmul_1"`` — wrong symbol AND wrong provenance — and the dispatch iterator
   was left holding an unconsumed entry.
"""
from __future__ import annotations

import pytest

from merlin.xdsl_dialects import _common

pytestmark = pytest.mark.skipif(not _common.HAS_XDSL, reason="xDSL not installed")

# A driver whose scf.for body reads %t0 (produced by kernel_0) without %t0 being an operand of the
# loop: it is neither an iter_arg nor a bound, so it appears nowhere in `scf.for`'s operand list.
LOOP_CAPTURE = """
builtin.module {
  func.func @forward(%x: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %t0 = func.call @forward$kernel_0(%x) : (tensor<64x64xf32>) -> tensor<64x64xf32>
    %t1 = func.call @forward$kernel_1(%x) : (tensor<64x64xf32>) -> tensor<64x64xf32>
    %t2 = func.call @forward$kernel_2(%t1) : (tensor<64x64xf32>) -> tensor<64x64xf32>
    %lp = scf.for %i = %c0 to %c2 step %c1 iter_args(%acc = %t2) -> (tensor<64x64xf32>) {
      %s = linalg.add ins(%acc, %t0 : tensor<64x64xf32>, tensor<64x64xf32>)
           outs(%acc : tensor<64x64xf32>) -> tensor<64x64xf32>
      scf.yield %s : tensor<64x64xf32>
    }
    func.return %lp : tensor<64x64xf32>
  }
  func.func private @forward$kernel_0(%p: tensor<64x64xf32>) -> tensor<64x64xf32> {
    func.return %p : tensor<64x64xf32> }
  func.func private @forward$kernel_1(%p: tensor<64x64xf32>) -> tensor<64x64xf32> {
    func.return %p : tensor<64x64xf32> }
  func.func private @forward$kernel_2(%p: tensor<64x64xf32>) -> tensor<64x64xf32> {
    func.return %p : tensor<64x64xf32> }
}
"""

# The same shape, but with the second kernel call INSIDE the loop region.
NESTED_CALL = """
builtin.module {
  func.func @forward(%x: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %a = func.call @forward$kernel_0__rmatmul_0(%x) : (tensor<4x4xf32>) -> tensor<4x4xf32>
    %b = scf.for %i = %c0 to %c2 step %c1 iter_args(%acc = %a) -> (tensor<4x4xf32>) {
      %n = func.call @forward$kernel_1__rmatmul_1(%acc) : (tensor<4x4xf32>) -> tensor<4x4xf32>
      scf.yield %n : tensor<4x4xf32>
    }
    %c = func.call @forward$kernel_2__rmatmul_2(%b) : (tensor<4x4xf32>) -> tensor<4x4xf32>
    func.return %c : tensor<4x4xf32>
  }
  func.func private @forward$kernel_0__rmatmul_0(%p: tensor<4x4xf32>) -> tensor<4x4xf32> {
    func.return %p : tensor<4x4xf32> }
  func.func private @forward$kernel_1__rmatmul_1(%p: tensor<4x4xf32>) -> tensor<4x4xf32> {
    func.return %p : tensor<4x4xf32> }
  func.func private @forward$kernel_2__rmatmul_2(%p: tensor<4x4xf32>) -> tensor<4x4xf32> {
    func.return %p : tensor<4x4xf32> }
}
"""

# A flat driver: no regions anywhere. The regression anchor -- nothing about it may change.
FLAT = """
builtin.module {
  func.func @forward(%x: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %t0 = func.call @forward$kernel_0(%x) : (tensor<64x64xf32>) -> tensor<64x64xf32>
    %t1 = func.call @forward$kernel_1(%t0) : (tensor<64x64xf32>) -> tensor<64x64xf32>
    func.return %t1 : tensor<64x64xf32>
  }
  func.func private @forward$kernel_0(%p: tensor<64x64xf32>) -> tensor<64x64xf32> {
    func.return %p : tensor<64x64xf32> }
  func.func private @forward$kernel_1(%p: tensor<64x64xf32>) -> tensor<64x64xf32> {
    func.return %p : tensor<64x64xf32> }
}
"""


def _build(text: str, n_dispatches: int, *, tagged: bool = False):
    """Flatten ``text`` against a dispatch table of ``n_dispatches`` entries, as the outliner hands it
    over. Built directly (rather than by running the outliner) because the outliner only walks the
    driver's TOP-LEVEL ops -- which is exactly the blind spot under test."""
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.xdsl_dialects.lowering.dispatch_program import build_dispatch_program
    from merlin.xdsl_dialects.lowering.outline import DispatchInfo, OutlineResult

    suffix = "__rmatmul_{}" if tagged else ""
    table = [
        DispatchInfo(index=i, symbol=f"forward$kernel_{i}" + (suffix.format(i) if tagged else ""),
                     root_op="linalg.matmul", n_operands=1,
                     result_types=["tensor<64x64xf32>"],
                     prov={"prov.region_id": f"matmul_{i}"} if tagged else {})
        for i in range(n_dispatches)]
    return build_dispatch_program(OutlineResult(module=parse_mlir_text(text), dispatches=table))


# ---- 1. under-recorded reads ---------------------------------------------------------------

def test_a_value_only_the_loop_body_reads_is_recorded_as_an_input():
    """`%t0` is used inside the scf.for body and is not an operand of the loop. Before the fix it was
    absent from the node's inputs, so nothing downstream knew the loop reads it."""
    prog = _build(LOOP_CAPTURE, 3)
    loop = next(n for n in prog.nodes if n.op == "scf.for")
    producer = next(n for n in prog.nodes if n.op == "forward$kernel_0")
    captured = producer.outputs[0]
    assert captured in loop.inputs, (captured, loop.inputs)
    assert loop.captures == [captured], loop.captures
    assert loop.regions == 1
    # the induction variable and the iter_args body argument are block args -> internal, not captures
    assert len(loop.captures) == 1


def test_the_planner_does_not_reuse_bytes_a_loop_is_still_reading():
    """The consequence, checked where it bites. Before the fix `last_use` for the captured buffer was
    node 3 (its definition) though the loop reads it at node 6, and the planner seated the NEXT
    intermediate at the same offset: two simultaneously-live buffers sharing bytes."""
    from merlin.xdsl_dialects.lowering.arena_plan import _live_ranges, plan_arena

    prog = _build(LOOP_CAPTURE, 3)
    loop_idx = next(i for i, n in enumerate(prog.nodes) if n.op == "scf.for")
    captured = next(n for n in prog.nodes if n.op == "forward$kernel_0").outputs[0]
    _, last_use = _live_ranges(prog)
    assert last_use[captured] == loop_idx, (captured, last_use[captured], loop_idx)

    plan = plan_arena(prog)
    # every pair of buffers that is live at the same time must be byte-disjoint
    first_def, last = _live_ranges(prog)
    placed = [(b, plan.offsets[b], plan.sizes[b]) for b in plan.offsets]
    for i, (bi, oi, si) in enumerate(placed):
        for bj, oj, sj in placed[i + 1:]:
            overlaps = oi < oj + sj and oj < oi + si
            together = (first_def[bi] <= last[bj] and first_def[bj] <= last[bi])
            assert not (overlaps and together), (bi, bj, oi, si, oj, sj)


def test_the_planner_refuses_a_region_node_whose_captures_were_never_computed():
    """Fail closed. A hand-built or older-writer program can carry a region node with no capture set;
    the flat program gives the planner no way to recover one (the captured value IS defined earlier,
    it is simply never listed), so it must refuse rather than assume the operand list is complete."""
    from merlin.xdsl_dialects.lowering.arena_plan import ArenaPlanError, plan_arena
    from merlin.xdsl_dialects.lowering.dispatch_program import Buffer, DispatchProgram, Node

    bufs = {b.id: b for b in (
        Buffer(id="a", shape=[8, 8], dtype="f32", kind="arg", arg_index=0),
        Buffer(id="t", shape=[8, 8], dtype="f32", kind="intermediate"),
        Buffer(id="r", shape=[8, 8], dtype="f32", kind="intermediate"))}
    nodes = [Node(kind="dispatch", op="k0", inputs=["a"], outputs=["t"]),
             Node(kind="view", op="scf.for", inputs=["t"], outputs=["r"], regions=1)]
    prog = DispatchProgram(entry="forward", args=[0], buffers=bufs, nodes=nodes, results=["r"])
    with pytest.raises(ArenaPlanError) as e:
        plan_arena(prog)
    assert "region" in str(e.value) and "capture" in str(e.value)

    # ...and a capture recorded but kept OUT of inputs is refused too: liveness reads inputs.
    nodes[1] = Node(kind="view", op="scf.for", inputs=[], outputs=["r"], regions=1, captures=["t"])
    with pytest.raises(ArenaPlanError) as e2:
        plan_arena(prog)
    assert "absent from its inputs" in str(e2.value)

    # the same node with the capture folded into inputs plans fine
    nodes[1] = Node(kind="view", op="scf.for", inputs=["t"], outputs=["r"], regions=1,
                    captures=["t"])
    assert plan_arena(prog).arena_bytes > 0


# ---- 2. a kernel call hidden in a region ---------------------------------------------------

def test_a_kernel_call_inside_a_region_is_refused_not_misattributed():
    """Before the fix this returned a program in which the third IR call carried the SECOND kernel's
    symbol and its prov.region_id, and verify_program reported no problem at all."""
    from merlin.xdsl_dialects.lowering.outline import OutlineError

    with pytest.raises(OutlineError) as e:
        _build(NESTED_CALL, 3, tagged=True)
    msg = str(e.value)
    assert "forward$kernel_1__rmatmul_1" in msg      # names the call it found
    assert "scf.for" in msg


def test_an_unconsumed_dispatch_entry_is_refused():
    """The other half of the same desync: a driver that makes fewer top-level calls than the table has
    entries leaves the iterator un-exhausted, which is the only place the mismatch is visible."""
    from merlin.xdsl_dialects.lowering.outline import OutlineError

    with pytest.raises(OutlineError) as e:
        _build(FLAT, 3)                               # driver makes 2 calls, table lists 3
    assert "desynchronised" in str(e.value)
    assert "forward$kernel_2" in str(e.value)


def test_more_driver_calls_than_dispatch_entries_is_refused():
    from merlin.xdsl_dialects.lowering.outline import OutlineError

    with pytest.raises(OutlineError) as e:
        _build(FLAT, 1)                               # driver makes 2 calls, table lists 1
    assert "exhausted" in str(e.value)


# ---- 3. the flat path is untouched ---------------------------------------------------------

def test_a_flat_driver_is_unchanged_and_plans_as_before():
    """No region anywhere -> regions == 0, an empty (not unknown) capture set, and a plan."""
    from merlin.xdsl_dialects.lowering.arena_plan import plan_arena
    from merlin.xdsl_dialects.lowering.dispatch_program import verify_program

    prog = _build(FLAT, 2)
    assert verify_program(prog) == []
    assert prog.n_dispatches == 2
    assert all(n.regions == 0 and n.captures == [] for n in prog.nodes)
    plan = plan_arena(prog)
    # one intermediate (t0); t1 is the result and stays out of the arena
    assert plan.stats["n_intermediate_buffers"] == 1
    assert plan.arena_bytes == 64 * 64 * 4
