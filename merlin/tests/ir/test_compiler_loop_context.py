"""Actual sibling-loop segments lift through partial prefixes and full entries."""

import json

import pytest

from merlin.transforms.certificates import ir_text, parse_ir
from merlin.transforms.unrolling import unroll_ordered_loops
from merlin.verify import compiler_loop_intake as intake


def source(*, effects=True):
    declarations = "func.func private @start(i32)->i32" if effects else ""
    prefix = "%z=func.call @start(%x):(i32)->i32" if effects else "%z=arith.addi %x,%y:i32"
    return f"""builtin.module {{
    {declarations}
    func.func @forward(%x:i32,%y:i32)->i32 {{
      {prefix}
      %lb=arith.constant 0:index
      %small=arith.constant 2:index
      %large=arith.constant 5:index
      %st=arith.constant 1:index
      %a=scf.for %i=%lb to %small step %st iter_args(%c=%z)->(i32) {{
        %n=arith.addi %c,%y:i32
        scf.yield %n:i32
      }}
      %b=scf.for %j=%lb to %large step %st iter_args(%d=%a)->(i32) {{
        %n=arith.addi %d,%y:i32
        scf.yield %n:i32
      }}
      func.return %b:i32
    }}
    func.func @untouched(%x:i32)->i32 {{func.return %x:i32}}
    }}"""


def pair(*, effects=True):
    module = parse_ir(source(effects=effects))
    before = ir_text(module)
    unroll_ordered_loops(module, (2,), index_bits=64)
    return before, ir_text(module)


def test_complete_actual_context_reconstruction_has_no_producer_replay(monkeypatch):
    buffers = pair()
    monkeypatch.setattr(
        "merlin.transforms.unrolling.unroll_ordered_loops", lambda *a, **k: pytest.fail("producer replay")
    )
    prepared = intake.prepare(buffers, kinds=(intake.UNROLL,))
    assert intake.refresh(prepared) == prepared
    capture = json.loads(prepared.capture_json)
    assert capture["mode"] == "actual-loop-context"
    assert len(capture["stages"]) == 2
    assert len(capture["projection"]["functions"]) == 3
    assert len(prepared.roots) == 178
    assert prepared.actual_target == buffers[-1]
    assert "Merlin.GraphStatePipelineContext" in prepared.query
    assert "Context.segment_factor" in prepared.query
    assert "namespace Merlin.CompilerLoopContextStep0.SourceAtomic" in prepared.query
    assert "namespace Merlin.CompilerLoopContextStep1.TargetAtomic" in prepared.query
    assert "theorem nestedAccepted0" in prepared.query


@pytest.mark.parametrize("mutation", (["prefix", "suffix", "frame"])[:1])
def test_complete_actual_prefix_suffix_and_module_frame_are_required(mutation):
    from xdsl.dialects import func
    from xdsl.rewriter import Rewriter

    before, after = pair()
    module = parse_ir(after)
    functions = [op for op in module.body.block.ops if type(op) is func.FuncOp and op.body.blocks]
    forward, untouched = functions
    if mutation == "prefix":
        call = next(op for op in forward.walk() if type(op) is func.CallOp)
        Rewriter.replace_op(call, [], [forward.body.block.args[0]])
    elif mutation == "suffix":
        ret = forward.body.block.last_op
        ret.operands = (forward.body.block.args[0],)
    else:
        module.body.block.erase_op(untouched)
    module.verify()
    with pytest.raises(intake.Unsupported):
        intake.prepare((before, ir_text(module)), kinds=(intake.UNROLL,))


def bulk_pair():
    module = parse_ir(source().replace("%small=arith.constant 2:index", "%small=arith.constant 6:index"))
    before = ir_text(module)
    unroll_ordered_loops(module, (2,), index_bits=64)
    return before, ir_text(module)


def test_bulk_sibling_context_uses_actual_target_and_undelivered_scaffold(monkeypatch):
    buffers = bulk_pair()
    monkeypatch.setattr(
        "merlin.transforms.unrolling.unroll_ordered_loops", lambda *a, **k: pytest.fail("producer replay")
    )
    prepared = intake.prepare(buffers, kinds=(intake.UNROLL,))
    assert intake.refresh(prepared) == prepared
    assert len(prepared.roots) == 216
    capture = json.loads(prepared.capture_json)
    assert [row["local"]["mode"] for row in capture["stages"]] == ["bulk", "tail"]
    assert capture["stages"][0]["local"]["auxiliary"] != buffers[-1]
    assert prepared.actual_target == buffers[-1]
