"""Counted completion ownership and actual SSA callback consumption, without target latency guesses."""
from dataclasses import replace

import pytest
from test_scf_commands import ExecuteIndices
from xdsl.dialects import builtin, llvm, scf

from merlin.compile.scheduling.llvm_program import FnBuilderIndexAdapter
from merlin.compile.scheduling.scf_commands import (
    emit_scf_commands,
    materialize_scf_commands,
    unroll_small_command_loops,
    verify_scf_commands,
)
from merlin.llvmlower.host_builder import FnBuilder
from merlin.sched.ir import Call, Const, Kernel, Loop, Stage, Var, check_structure, parse_kernel


class ExecuteCarried(ExecuteIndices):
    def loop_carried(self, low, high, step, initial, body):
        carried = initial
        for index in range(low, high, step):
            carried = body(index, carried)
        return carried


def pipeline(trips=3, *, hazard=False, nested=False):
    a = Stage("private", 0, 4, writes=True)
    b = Stage("private", 4, 4, writes=True)
    body = (
        Call("compute", (("tile", Var("i") * 2), ("slot", Const(0))), produces="ca", awaits=("a",),
             stages=(replace(a, writes=False),)),
        Call("compute", (("tile", Var("i") * 2 + 1), ("slot", Const(1))), produces="cb", awaits=("b",),
             stages=(replace(b, writes=False),)),
        Call("fetch", (("tile", Var("i") * 2 + 2), ("slot", Const(0))), produces="na",
             awaits=() if hazard else ("ca",), stages=(a,)),
        Call("fetch", (("tile", Var("i") * 2 + 3), ("slot", Const(1))), produces="nb",
             awaits=("cb",), stages=(b,)),
    )
    loop = Loop("i", trips, body, (("a", "start_a"), ("b", "start_b")), ("na", "nb"), ("end_a", "end_b"))
    if nested:
        # The outer loop transports the inner loop's real results; this does
        # not assert that a result finished merely because the child returned.
        child = replace(loop, iter_args=(("a", "outer_a"), ("b", "outer_b")))
        loop = Loop("outer", 1, (child,), (("outer_a", "start_a"), ("outer_b", "start_b")),
                    ("end_a", "end_b"), ("end_a", "end_b"))
    return Kernel("stream", (), (
        Call("fetch", (("tile", Const(0)), ("slot", Const(0))), produces="start_a", stages=(a,)),
        Call("fetch", (("tile", Const(1)), ("slot", Const(1))), produces="start_b", stages=(b,)),
        loop,
        Call("finish", (("tile", Const(trips * 2)), ("slot", Const(0))), awaits=("end_a",),
             stages=(replace(a, writes=False),)),
        Call("finish", (("tile", Const(trips * 2 + 1)), ("slot", Const(1))), awaits=("end_b",),
             stages=(replace(b, writes=False),)),
    ))


def execute(module):
    storage, pending, outputs, trace = {}, {}, {}, []
    counter = 0

    def lower(name, args, metadata):
        nonlocal counter
        tile, slot = args["tile"], args["slot"]
        trace.append((name, tile, slot))
        if name == "finish":
            outputs[tile] = storage[slot] * 3
            return None
        counter += 1
        if name == "fetch":
            pending[counter] = lambda: storage.__setitem__(slot, tile + 5)
        else:
            pending[counter] = lambda: outputs.__setitem__(tile, storage[slot] * 3)
        return counter

    def wait(handles):
        for handle in handles:
            action = pending.pop(handle, None)
            if action is not None:
                action()

    emit_scf_commands(module, ExecuteCarried(), lower, await_events=wait)
    assert not pending
    return outputs, trace


@pytest.mark.parametrize("trips,nested", [(1, False), (3, False), (7, False), (5, True)])
def test_startup_steady_drain_and_nested_real_loop_results(trips, nested):
    kernel = pipeline(trips, nested=nested)
    assert not check_structure(kernel)
    assert parse_kernel(kernel.text()) == kernel
    module = materialize_scf_commands(kernel)
    assert any(type(op) is scf.ForOp and len(op.iter_args) == 2 for op in module.walk())
    result, _ = execute(module)
    assert result == {i: (i + 5) * 3 for i in range(2 * (trips + 1))}


def test_actual_xdsl_unroll_threads_event_phi_edges_and_alpha_names():
    module = materialize_scf_commands(pipeline(2))
    actual = unroll_small_command_loops(module)
    assert not any(type(op) is scf.ForOp for op in actual.walk())
    assert execute(actual) == execute(module)


def test_reentry_hazard_not_visible_in_completed_startup_is_rejected():
    stage = Stage("private", 0, 4, writes=True)
    kernel = Kernel("reentry", (), (
        Call("seed", (), produces="seed"), Call("wait", (), awaits=("seed",)),
        Loop("i", 3, (Call("clobber", (), stages=(stage,)), Call("wait", (), awaits=("incoming",)),
                      Call("launch", (), produces="next", stages=(stage,))),
             (("incoming", "seed"),), ("next",), ("last",)),
        Call("wait", (), awaits=("last",)),
    ))
    with pytest.raises(ValueError, match="private rows reused"):
        materialize_scf_commands(kernel)


def test_war_hazard_and_missing_final_drain_refuse_before_target_callbacks():
    with pytest.raises(ValueError, match="private rows reused"):
        materialize_scf_commands(pipeline(hazard=True))
    kernel = replace(pipeline(), body=pipeline().body[:-2])
    with pytest.raises(ValueError, match="every launched event"):
        materialize_scf_commands(kernel)


def test_large_periodic_handle_permutation_is_compact_not_trip_enumerated():
    kernel = Kernel("rotate", (), (
        Call("a", (), produces="a", stages=(Stage("x", 0, 1, writes=True),)),
        Call("b", (), produces="b", stages=(Stage("x", 1, 1, writes=True),)),
        Loop("i", 1_000_001, (), (("x", "a"), ("y", "b")), ("y", "x"), ("last_x", "last_y")),
        Call("wait", (), awaits=("last_x", "last_y")),
    ))
    report = verify_scf_commands(materialize_scf_commands(kernel))
    assert report["loop_carried_completion"]["state_transfer_operations"] < 32


def test_lexical_capture_stale_initial_and_aliased_yield_refuse():
    k = pipeline()
    carried = k.body[2]
    bad = replace(carried, yields=("na", "na"))
    assert check_structure(replace(k, body=(*k.body[:2], bad, *k.body[3:])))
    bad = replace(carried, body=(*carried.body, Call("illegal", (), awaits=("start_a",))))
    assert check_structure(replace(k, body=(*k.body[:2], bad, *k.body[3:])))
    assert check_structure(replace(k, body=(*k.body, Call("stale", (), awaits=("start_a",)))))


def test_missing_actual_carried_builder_refuses_before_any_target_call():
    calls = []
    with pytest.raises(ValueError, match="actual loop input/yield/result"):
        emit_scf_commands(materialize_scf_commands(pipeline()), ExecuteIndices(),
                          lambda *args: calls.append(args), await_events=lambda values: None)
    assert not calls


def test_actual_llvm_cfg_consumes_carried_event_handles_and_completion_calls():
    builder = FnBuilder([])
    adapter = FnBuilderIndexAdapter(builder)

    def lower(name, args, metadata):
        operands = [adapter.value(v) for v in args.values()]
        result_types = [builtin.i64] if metadata.produces else []
        op = builder.add(llvm.CallOp("target_" + name, *operands,
                                    return_type=result_types[0] if result_types else None))
        return op.results[0] if metadata.produces else None

    def await_events(handles):
        for handle in handles:
            builder.add(llvm.CallOp("target_complete", handle))

    emit_scf_commands(materialize_scf_commands(pipeline()), adapter, lower, await_events=await_events)
    function = llvm.FuncOp("schedule", llvm.LLVMFunctionType([]), body=builder.finish())
    declarations = [llvm.FuncOp("target_" + name, llvm.LLVMFunctionType(inputs, output),
                               linkage=llvm.LinkageAttr("external")) for name, inputs, output in (
        ("fetch", [builtin.i64, builtin.i64], builtin.i64),
        ("compute", [builtin.i64, builtin.i64], builtin.i64),
        ("finish", [builtin.i64, builtin.i64], None),
        ("complete", [builtin.i64], None),
    )]
    builtin.ModuleOp([*declarations, function]).verify()
    assert any(len(block.args) == 3 for block in function.body.blocks)
    assert any(len(block.args) == 2 for block in function.body.blocks)
    assert sum(type(op) is llvm.CallOp and op.callee.string_value() == "target_complete"
               for op in function.walk()) == 6


def test_old_synchronous_text_and_digest_are_unchanged():
    kernel = Kernel("old", (), (Loop("i", 3, (Call("step", (("index", Var("i")),)),)),))
    assert kernel.text() == "kernel old()\n  for i in 0..3:\n    step(index=i)\n"
    assert parse_kernel(kernel.text()) == kernel


@pytest.mark.parametrize("trips,action,factor", [(7, "split", 2), (7, "unroll", 2), (2, "unroll", 2),
                                               (9, "split", 4), (9, "unroll", 4)])
def test_ordered_ir_split_unroll_tail_and_alpha_renamed_completion_edges(trips, action, factor):
    from merlin.compile.scheduling.loop_schedule import LoopTransform, propose_loop_transforms

    source = pipeline(trips)
    result = propose_loop_transforms(source, (LoopTransform(action, ("i",), factor=factor),))
    assert result.digest() != source.digest()
    assert parse_kernel(result.text()) == result
    assert execute(materialize_scf_commands(result)) == execute(materialize_scf_commands(source))


def test_nested_ir_partial_unroll_and_partition_thread_enclosing_yields():
    from merlin.compile.scheduling.loop_schedule import LoopTransform, propose_loop_transforms

    source = pipeline(7, nested=True)
    result = propose_loop_transforms(source, (
        LoopTransform("partition", ("outer", "i"), cuts=(2, 5), names=("first", "middle", "last")),
        LoopTransform("unroll", ("outer", "middle"), factor=2),
    ))
    assert parse_kernel(result.text()) == result
    assert execute(materialize_scf_commands(result)) == execute(materialize_scf_commands(source))


def test_partial_unroll_preserves_original_select_tail_operand():
    from merlin.compile.scheduling.loop_schedule import LoopTransform, propose_loop_transforms
    from merlin.sched.ir.expr import select

    source = pipeline(7)
    carried = source.body[2]
    # A final-iteration equality must become the correct quotient/remainder
    # condition in the bulk and the correct local predicate in the tail.
    body = tuple(replace(call, args=(*call.args, ("width", select(Var("i"), 6, 1, 4)))) for call in carried.body)
    source = replace(source, body=(*source.body[:2], replace(carried, body=body), *source.body[3:]))
    result = propose_loop_transforms(source, (LoopTransform("unroll", ("i",), factor=2),))
    from merlin.sched.ir.kernel import concretize, instances

    assert [(c.instr, concretize(c, env)) for c, env in instances(result)] == [
        (c.instr, concretize(c, env)) for c, env in instances(source)]
    assert execute(materialize_scf_commands(result)) == execute(materialize_scf_commands(source))


def test_dependency_changing_carried_transform_refuses_and_low_level_split_is_not_erased():
    from merlin.compile.scheduling.loop_schedule import LoopTransform, propose_loop_transforms
    from merlin.sched.primitives import Cursor, NotApplicable, divide_loop

    source = pipeline(4)
    result = divide_loop(source, Cursor(loops=("i",)), 2)
    assert result.body[2].iter_args
    assert execute(materialize_scf_commands(result)) == execute(materialize_scf_commands(source))
    for request in (LoopTransform("fission", ("i",), cut=2), LoopTransform("reorder", ("i",)),
                    LoopTransform("hoist", ("i",), index=0)):
        with pytest.raises(NotApplicable, match="dependency transport"):
            propose_loop_transforms(source, (request,))


def test_mutated_actual_ssa_edge_and_untyped_loop_marker_refuse():
    import json

    from merlin.compile.scheduling.scf_carried_events import CARRIED_EVENTS
    from merlin.compile.scheduling.scf_commands import SCHEMA

    module = materialize_scf_commands(pipeline())
    loop = next(op for op in module.walk() if type(op) is scf.ForOp)
    loop.attributes[CARRIED_EVENTS] = builtin.UnitAttr()
    with pytest.raises(ValueError, match="malformed lexical"):
        verify_scf_commands(module)
    module = materialize_scf_commands(pipeline())
    loop = next(op for op in module.walk() if type(op) is scf.ForOp)
    call = next(op for op in loop.body.block.ops if op.name == "func.call")
    data = json.loads(call.attributes[SCHEMA].data)
    data["awaits"] = ["b"]  # The actual SSA operand still refers to a.
    call.attributes[SCHEMA] = builtin.StringAttr(json.dumps(data))
    with pytest.raises(ValueError, match="actual SSA|lexical binding"):
        verify_scf_commands(module)


def test_nested_complete_unroll_propagates_actual_final_event_to_outer_yield():
    from merlin.compile.scheduling.loop_schedule import LoopTransform, propose_loop_transforms

    source = pipeline(3, nested=True)
    result = propose_loop_transforms(source, (LoopTransform("unroll", ("outer", "i"), factor=3),))
    assert result.body[2].yields != source.body[2].yields
    assert execute(materialize_scf_commands(result)) == execute(materialize_scf_commands(source))


def test_old_sync_checker_cannot_silently_flatten_lexical_event_ownership():
    from types import SimpleNamespace

    from merlin.sched.check.sync import check_sync

    report = check_sync(pipeline(), SimpleNamespace(hazard_resolution="explicit", units=()))
    assert not report.decidable
    assert "SCF event-edge" in report.reason


@pytest.mark.parametrize("bounds", [(0, 2, 6), (1, 2, 7), (0, 5, 2, 7), (0, 2, 2, 7)])
def test_direct_partition_helper_cannot_drop_duplicate_or_reorder_original_trips(bounds):
    from merlin.sched.primitives import NotApplicable
    from merlin.sched.primitives.event_loops import partition_event_loop

    with pytest.raises(NotApplicable, match="exact complete increasing"):
        partition_event_loop(pipeline(7), ("i",), bounds, tuple(f"part{i}" for i in range(len(bounds)-1)))


def test_legacy_completed_token_capture_keeps_original_nonlexical_mode():
    # The previous bridge allowed a completed outer token to be awaited again
    # within a loop. A program without explicit carries keeps that old scope;
    # opting into lexical ownership must not silently change an old program.
    source = Kernel("legacy", (), (
        Call("seed", (), produces="ready"), Call("wait", (), awaits=("ready",)),
        Loop("i", 3, (Call("wait", (), awaits=("ready",)),)),
    ))
    module = materialize_scf_commands(source)
    assert "loop_carried_completion" not in verify_scf_commands(module)
    waited = []
    emit_scf_commands(module, ExecuteIndices(), lambda *_: 1, await_events=lambda handles: waited.extend(handles))
    assert waited == [1, 1, 1, 1]
