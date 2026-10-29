"""Actual sibling-loop segments lift through partial prefixes and full entries."""

import json
from pathlib import Path

import pytest

from merlin.transforms.certificates import ir_text, parse_ir
from merlin.transforms.unrolling import unroll_ordered_loops
from merlin.verify import compiler_loop_intake as intake
from merlin.verify import loop_transform_gate as gate


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


def test_context_kernel_covers_partial_prefix_failures_and_complete_module(tmp_path, pass_gate_tools):
    buffers = pair()
    prepared = intake.prepare(buffers, kinds=(intake.UNROLL,))
    token = gate.check(
        prepared, lean=Path(pass_gate_tools.sdk_bin) / "lean", destination=tmp_path / "check", timeout=300
    )
    delivered = []
    gate.deliver(token, prepared, delivered.append)
    assert delivered == [buffers[-1]]
    report = json.loads(Path(token.report).read_text())
    statement = report["statements"][intake.ROOT]
    assert "PreservesEntries" in statement
    assert "control0" not in statement and "run :" not in statement
    assert set(report["audit"]["axioms_used"]) <= gate.STD3
    assert report["relation"] == "all defined entries equal returned values/errors/terminal world"


@pytest.mark.parametrize("mutation", ["tail-upper", "tail-step", "tail-body"])
def test_actual_target_controls_and_body_cannot_hide_behind_wrapper(tmp_path, pass_gate_tools, mutation):
    from xdsl.dialects import arith, scf
    from xdsl.rewriter import Rewriter

    before, after = pair()
    module = parse_ir(after)
    loops = [op for op in module.walk() if type(op) is scf.ForOp]
    bulk, tail = loops
    if mutation == "tail-upper":
        tail.operands = (tail.lb, bulk.ub, tail.step, *tail.iter_args)
    elif mutation == "tail-step":
        tail.operands = (tail.lb, tail.ub, bulk.step, *tail.iter_args)
    else:
        addition = next(op for op in tail.walk() if type(op) is arith.AddiOp and str(op.result.type) == "i32")
        Rewriter.replace_op(addition, arith.SubiOp(addition.lhs, addition.rhs))
    module.verify()
    prepared = intake.prepare((before, ir_text(module)), kinds=(intake.UNROLL,))
    with pytest.raises(gate.NotEstablished):
        gate.check(prepared, lean=Path(pass_gate_tools.sdk_bin) / "lean", destination=tmp_path / mutation, timeout=300)


@pytest.mark.parametrize("mutation", ["prefix", "suffix", "frame"])
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


def test_real_compiler_context_gate_and_native_results(tmp_path, pass_gate_tools):
    import ctypes

    from test_compiler_formal_transforms import shared_library

    from merlin.llvmlower import lower
    from merlin.verify import compiler_transforms, pass_validation
    from merlin.verify.speculative_gate import SpeculativeGate

    text = source(effects=False)
    options = compiler_transforms.Options(unroll_factor=2)
    baseline = lower.lower_model(text, tmp_path / "baseline", targets=(), formal_transforms=options)
    worker = SpeculativeGate(tmp_path / "gate", job_timeout=1200)
    selected = pass_validation.PassValidation(worker, pass_gate_tools, frozenset({intake.UNROLL}))
    checked = lower.lower_model(
        text, tmp_path / "checked", targets=(), formal_transforms=options, pass_validation=selected
    )
    assert checked.ll_path.read_bytes() == baseline.ll_path.read_bytes()
    assert len(worker.verdicts) == 1 and worker.verdicts[0]["verdict"] == "accepted"
    (tmp_path / "checked-context").mkdir()
    library = shared_library(checked, tmp_path / "checked-context")
    library.forward.argtypes = (ctypes.c_int32, ctypes.c_int32)
    library.forward.restype = ctypes.c_int32
    for x, y in ((3, 5), (2**31 - 4, 17), (-(2**31), -23)):
        assert library.forward(x, y) == ctypes.c_int32(x + 8 * y).value
    library.untouched.argtypes = (ctypes.c_int32,)
    library.untouched.restype = ctypes.c_int32
    assert library.untouched(-29) == -29


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


def test_bulk_and_tail_context_kernel_covers_both_actual_boundaries(tmp_path, pass_gate_tools):
    buffers = bulk_pair()
    prepared = intake.prepare(buffers, kinds=(intake.UNROLL,))
    token = gate.check(
        prepared, lean=Path(pass_gate_tools.sdk_bin) / "lean", destination=tmp_path / "check", timeout=300
    )
    delivered = []
    gate.deliver(token, prepared, delivered.append)
    assert delivered == [buffers[-1]]


def test_bulk_context_mutation_is_a_kernel_refusal(tmp_path, pass_gate_tools):
    from xdsl.dialects import arith
    from xdsl.rewriter import Rewriter

    before, after = bulk_pair()
    module = parse_ir(after)
    addition = next(op for op in module.walk() if type(op) is arith.AddiOp and str(op.result.type) == "i32")
    Rewriter.replace_op(addition, arith.SubiOp(addition.lhs, addition.rhs))
    prepared = intake.prepare((before, ir_text(module)), kinds=(intake.UNROLL,))
    with pytest.raises(gate.NotEstablished):
        gate.check(prepared, lean=Path(pass_gate_tools.sdk_bin) / "lean", destination=tmp_path / "bad", timeout=300)
