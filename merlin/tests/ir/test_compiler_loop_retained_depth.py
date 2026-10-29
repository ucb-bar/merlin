"""Retained deeper loops keep strict actual admission and checked auxiliary captures."""

from pathlib import Path

import pytest
from test_compiler_loop_selected_depth import source
from xdsl.dialects import arith, builtin, func, scf
from xdsl.rewriter import Rewriter

from merlin.transforms.certificates import ir_text, parse_ir
from merlin.transforms.unrolling import unroll_ordered_loops
from merlin.verify import compiler_loop_intake as intake
from merlin.verify import lean_graph_state_scalar_capture as capture
from merlin.verify import loop_transform_gate as gate


def retained_source(*, effects=True):
    module = parse_ir(source(3, effects=effects))
    function = next(op for op in module.body.block.ops if type(op) is func.FuncOp and op.sym_name.data == "forward")
    outer, middle, inner = [op for op in function.walk() if type(op) is scf.ForOp]
    two = arith.ConstantOp.from_int_and_width(2, builtin.IndexType())
    function.body.block.insert_op_before(two, outer)
    middle.operands = (middle.lb, inner.ub, middle.step, *middle.iter_args)
    inner.operands = (inner.lb, function.body.block.args[0], two.result, *inner.iter_args)
    module.verify()
    return ir_text(module)


def retained_pair():
    module = parse_ir(retained_source())
    before = ir_text(module)
    unroll_ordered_loops(module, (1, 2), index_bits=64)
    return before, ir_text(module)


@pytest.mark.parametrize("value", [-1, 0])
@pytest.mark.parametrize("auxiliary", [False, True])
def test_known_nonpositive_stride_stays_refused_in_every_projection_mode(value, auxiliary):
    module = parse_ir(retained_source())
    outer, _, inner = [op for op in module.walk() if type(op) is scf.ForOp]
    bad = arith.ConstantOp.from_int_and_width(value, builtin.IndexType())
    outer.parent_block().insert_op_before(bad, outer)
    inner.operands = (inner.lb, inner.ub, bad.result, *inner.iter_args)
    with pytest.raises(ValueError, match="positive constant stride"):
        capture.project(module, module, _allow_captured_stride=auxiliary)


def test_symbolic_actual_retained_stride_stays_outside_admitted_compiler_profile():
    module = parse_ir(retained_source())
    function = next(op for op in module.body.block.ops if type(op) is func.FuncOp and op.sym_name.data == "forward")
    inner = [op for op in module.walk() if type(op) is scf.ForOp][-1]
    inner.operands = (inner.lb, inner.ub, function.body.block.args[0], *inner.iter_args)
    before = ir_text(module)
    unroll_ordered_loops(module, (1, 2), index_bits=64)
    with pytest.raises(intake.Unsupported, match="positive constant stride"):
        intake.prepare((before, ir_text(module)), kinds=(intake.UNROLL,))


@pytest.mark.parametrize("mutation", ["callee", "stride"])
def test_raw_retained_callee_and_stride_are_closed_scope_kernel_refusals(tmp_path, pass_gate_tools, mutation):
    before, after = retained_pair()
    module = parse_ir(after)
    outer = next(op for op in module.walk() if type(op) is scf.ForOp)
    bulk = next(op for op in outer.body.block.ops if type(op) is scf.ForOp)
    retained = next(op for op in bulk.body.block.ops if type(op) is scf.ForOp)
    if mutation == "callee":
        call = next(op for op in retained.walk() if type(op) is func.CallOp)
        Rewriter.replace_op(call, func.CallOp("altered", list(call.arguments), [value.type for value in call.results]))
    else:
        one = next(
            op
            for op in module.walk()
            if type(op) is arith.ConstantOp and type(op.result.type) is builtin.IndexType and op.value.value.data == 1
        )
        retained.operands = (retained.lb, retained.ub, one.result, *retained.iter_args)
    module.verify()
    prepared = intake.prepare((before, ir_text(module)), kinds=(intake.UNROLL,))
    with pytest.raises(gate.NotEstablished):
        gate.check(prepared, lean=Path(pass_gate_tools.sdk_bin) / "lean", destination=tmp_path / "bad", timeout=300)
