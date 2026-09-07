"""Dynamic host work must count loop iterations, including nesting and exits."""
from xdsl.dialects import llvm
from xdsl.dialects.builtin import IntegerAttr, ModuleOp, i32, i64
from xdsl.ir import Block, Region
import pytest

from merlin.perf.host_cfg_activity import analyze_host_cfg_activity


def function(*, outer=3, inner=4, dynamic_bound=False, claimed_bytes=16):
    entry = Block(arg_types=[llvm.LLVMPointerType(), i64])
    region = Region(entry)

    def constant(value):
        op = llvm.ConstantOp(IntegerAttr(value, i64), i64)
        entry.add_op(op)
        return op.results[0]

    zero, one, out_limit, inner_limit = [constant(x) for x in (0, 1, outer, inner)]
    allocation = llvm.AllocaOp(inner_limit, i32)
    allocation.attributes["merlin.host_storage_bytes"] = IntegerAttr(claimed_bytes, i64)
    entry.add_op(allocation)
    head, body, end = Block(arg_types=[i64]), Block(), Block()
    inside, work, latch = Block(arg_types=[i64]), Block(), Block()
    for block in (head, body, end, inside, work, latch):
        region.add_block(block)
    entry.add_op(llvm.BrOp(head, zero))
    comparison = llvm.ICmpOp(head.args[0], entry.args[1] if dynamic_bound else out_limit, IntegerAttr(2, i64))
    head.add_ops([comparison, llvm.CondBrOp(comparison, body, [], end, [])])
    body.add_op(llvm.BrOp(inside, zero))
    comparison = llvm.ICmpOp(inside.args[0], inner_limit, IntegerAttr(2, i64))
    inside.add_ops([comparison, llvm.CondBrOp(comparison, work, [], latch, [])])
    value = llvm.LoadOp(entry.args[0], i32)
    work.add_ops([value, llvm.StoreOp(value.results[0], entry.args[0])])
    update = llvm.AddOp(inside.args[0], one)
    work.add_ops([update, llvm.BrOp(inside, update)])
    update = llvm.AddOp(head.args[0], one)
    latch.add_ops([update, llvm.BrOp(head, update)])
    end.add_op(llvm.ReturnOp())
    fn = llvm.FuncOp("probe", llvm.LLVMFunctionType([llvm.LLVMPointerType(), i64]), body=region)
    ModuleOp([fn]).verify()
    return fn


def test_nested_counts_header_exit_checks_and_memory_payload():
    report = analyze_host_cfg_activity(function())
    assert report["status"] == "derived", report["problems"]
    assert report["loop_count"] == 2
    assert report["static_operations"]["load"] == 1
    assert report["dynamic_operations"]["load"] == 12
    assert report["dynamic_operations"]["comparison"] == 4 + 3 * 5
    assert report["load_payload_bytes"] == report["store_payload_bytes"] == 48
    assert report["static_allocation_payload_bytes"] == 16
    assert report["allocations"][0]["annotation_status"] == "matches"
    assert report["cpu_cycles"] is report["dram_bytes"] is None


def test_zero_trip_outer_loop_does_not_execute_inner_header_or_payload():
    report = analyze_host_cfg_activity(function(outer=0))
    assert report["status"] == "derived"
    assert report["dynamic_operations"]["comparison"] == 1
    assert report["dynamic_operations"]["load"] == 0
    assert report["load_payload_bytes"] == report["store_payload_bytes"] == 0


def test_runtime_bound_preserves_unknown_instead_of_one_iteration():
    report = analyze_host_cfg_activity(function(dynamic_bound=True))
    assert report["status"] == "UNKNOWN"
    assert report["dynamic_operations"] is None
    assert report["load_payload_bytes"] is None
    assert report["static_allocation_payload_bytes"] == 16
    assert all(row["execution_count"] is None for row in report["blocks"])


def test_allocation_annotation_cannot_change_derived_bytes():
    report = analyze_host_cfg_activity(function(claimed_bytes=0))
    assert report["status"] == "UNKNOWN"
    assert report["static_allocation_payload_bytes"] == 16
    assert report["allocations"][0]["annotation_status"] == "mismatch"


def test_hotspot_joins_exact_alloca_identity_to_access_payload_and_task():
    fn = function()
    allocation = next(op for op in fn.walk() if op.name == "llvm.alloca")
    allocation.attributes["merlin.global_task"] = IntegerAttr(7, i64)
    for op in fn.walk():
        if op.name == "llvm.load":
            op.operands = [allocation.results[0]]
            op.attributes["merlin.global_task"] = IntegerAttr(8, i64)
        elif op.name == "llvm.store":
            op.operands = [op.operands[0], allocation.results[0]]
            op.attributes["merlin.global_task"] = IntegerAttr(9, i64)
    report = analyze_host_cfg_activity(fn)
    allocation_row = report["top_allocations_by_static_payload"][0]
    buffer_row = report["top_buffers_by_scalar_memory_payload"][0]
    assert allocation_row == buffer_row
    assert allocation_row["buffer_root"] == f"alloca:{list(fn.walk()).index(allocation)}"
    assert allocation_row["allocation_task"] == "7"
    assert allocation_row["access_tasks"] == ["8", "9"]
    assert allocation_row["static_allocation_payload_bytes"] == 16
    assert allocation_row["load_payload_bytes"] == allocation_row["store_payload_bytes"] == 48
    assert allocation_row["source_operation_attribution"] == "UNKNOWN"


def test_unknown_dynamic_payload_does_not_hide_static_allocation_hotspot():
    report = analyze_host_cfg_activity(function(dynamic_bound=True))
    allocation = report["top_allocations_by_static_payload"][0]
    assert allocation["static_allocation_payload_bytes"] == 16
    assert allocation["load_payload_bytes"] is allocation["store_payload_bytes"] is None
    assert allocation["allocation_task"] == "unowned"
    assert report["top_buffers_by_scalar_memory_payload"][0]["buffer_root"] == "arg:0"
    assert report["top_buffers_by_scalar_memory_payload"][0]["allocation_operation_index"] is None


def carried_function(*, trips=9, induction_first=True, coupled_step=False, coupled_bound=False,
                     forwarded=False, wrong_forwarding=False):
    entry = Block(arg_types=[llvm.LLVMPointerType(), i64])
    head, body, end = Block(arg_types=[i64, i64]), Block(arg_types=[i64, i64] if forwarded else []), Block()
    region = Region([entry, head, body, end])
    zero = llvm.ConstantOp(IntegerAttr(0, i64), i64)
    one = llvm.ConstantOp(IntegerAttr(1, i64), i64)
    bound = llvm.ConstantOp(IntegerAttr(trips, i64), i64)
    entry.add_ops([zero, one, bound])
    initial = [zero.results[0], entry.args[1]]
    if not induction_first:
        initial.reverse()
    entry.add_op(llvm.BrOp(head, *initial))
    iv, accumulator = head.args if induction_first else tuple(reversed(head.args))
    compare = llvm.ICmpOp(iv, accumulator if coupled_bound else bound.results[0], IntegerAttr(2, i64))
    body_args = [zero.results[0] if wrong_forwarding else iv, accumulator] if forwarded else []
    head.add_ops([compare, llvm.CondBrOp(compare, body, body_args, end, [])])
    body_iv, body_accumulator = body.args if forwarded else (iv, accumulator)
    value = llvm.LoadOp(entry.args[0], i64)
    accum_update = llvm.AddOp(body_accumulator, value.results[0])
    update = llvm.AddOp(body_iv, body_accumulator if coupled_step else one.results[0])
    incoming = [update.results[0], accum_update.results[0]]
    if not induction_first:
        incoming.reverse()
    body.add_ops([value, accum_update, update, llvm.BrOp(head, *incoming)])
    end.add_ops([llvm.StoreOp(accumulator, entry.args[0]), llvm.ReturnOp()])
    fn = llvm.FuncOp("probe", llvm.LLVMFunctionType([llvm.LLVMPointerType(), i64]), body=region)
    ModuleOp([fn]).verify()
    return fn


@pytest.mark.parametrize("induction_first", [True, False])
@pytest.mark.parametrize("trips", [0, 1, 9])
def test_carried_accumulator_does_not_hide_proven_constant_trip_count(induction_first, trips):
    report = analyze_host_cfg_activity(carried_function(trips=trips, induction_first=induction_first))
    assert report["status"] == "derived", report["problems"]
    assert report["dynamic_operations"]["load"] == trips
    assert report["dynamic_operations"]["store"] == 1
    assert report["dynamic_operations"]["comparison"] == trips + 1
    assert report["load_payload_bytes"] == trips * 8
    assert report["store_payload_bytes"] == 8
    assert report["cpu_cycles"] is report["dram_bytes"] is None


@pytest.mark.parametrize("coupling", ["coupled_step", "coupled_bound"])
def test_recurrence_dependent_induction_is_not_treated_as_constant(coupling):
    report = analyze_host_cfg_activity(carried_function(**{coupling: True}))
    assert report["status"] == "UNKNOWN"
    assert report["dynamic_operations"] is report["load_payload_bytes"] is None


@pytest.mark.parametrize('induction_first', [True, False])
def test_body_block_argument_forwarding_preserves_proven_loop_count(induction_first):
    report = analyze_host_cfg_activity(carried_function(forwarded=True, induction_first=induction_first))
    assert report['status'] == 'derived', report['problems']
    assert report['dynamic_operations']['load'] == 9
    assert report['load_payload_bytes'] == 72


def test_forwarding_different_value_does_not_prove_induction_update():
    report = analyze_host_cfg_activity(carried_function(forwarded=True, wrong_forwarding=True))
    assert report['status'] == 'UNKNOWN'
    assert report['dynamic_operations'] is None
