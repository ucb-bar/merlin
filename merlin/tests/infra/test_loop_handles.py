from dataclasses import replace

import pytest

from merlin.sched.ir import Call, Kernel, Loop, Ptr, TensorArg
from merlin.sched.ir.expr import Const, Select, Var, evaluate
from merlin.sched.primitives import Cursor, PrimitiveError, unroll
from merlin.sched.primitives.loop_handles import locate_loop, loop_handles, transform_loop, uniquify_loop_binders


def source():
    def leaf(offset):
        return Loop("i", 5, (Call("work", (("address", Ptr("a", Var("i") + offset)),
            ("tail", Select("i", 4, Const(1), Const(4)))),
            sets=(("pitch", Var("i") + 1),), assumes=(("mode", Const(1)),)),))
    return Kernel("pair", (TensorArg("a", (32,), "i8", "read"),), (leaf(0), leaf(10)))


def trace(kernel):
    out = []
    def walk(body, env):
        for op in body:
            if type(op) is Loop:
                for i in range(op.extent):
                    walk(op.body, {**env, op.var: i})
            else:
                values = tuple((key, evaluate(value.offset if type(value) is Ptr else value, env))
                               for key, value in op.args)
                out.append((op.instr, values, tuple((k, evaluate(v, env)) for k, v in op.sets)))
    walk(kernel.body, {})
    return out


def test_same_named_siblings_select_exact_second_occurrence_and_compose_tail():
    original = source()
    with pytest.raises(PrimitiveError, match="sibling"):
        unroll(original, Cursor(loops=("i",)))
    result = transform_loop(original, locate_loop(original, (1,)), action="split", factor=3)
    assert trace(result.kernel) == trace(original)
    assert len(result.replacement_loops) == 2  # counted bulk and ordered tail
    tail = result.replacement_loops[-1]
    final = transform_loop(result.kernel, tail, action="unroll", factor=2)
    assert trace(final.kernel) == trace(original)
    assert original.body[0].var == "i"


def test_stale_forged_and_nonloop_handles_refuse():
    original = source()
    handle = locate_loop(original, (1,))
    with pytest.raises(PrimitiveError, match="stale"):
        handle.resolve(replace(original, name="different"))
    with pytest.raises(PrimitiveError, match="content"):
        replace(handle, loop=original.body[0]).resolve(original)
    with pytest.raises(PrimitiveError, match="occurrence"):
        locate_loop(original, (0, 0))
    with pytest.raises(PrimitiveError, match="only"):
        transform_loop(original, handle, action="reorder")


def test_nested_hygiene_preserves_pointer_select_and_state_values():
    original = replace(source(), body=(Loop("outer", 2, source().body),))
    actual = uniquify_loop_binders(original)
    assert trace(actual.kernel) == trace(original)
    assert len({h.loop.var for h in loop_handles(actual.kernel)}) == 3
    assert len(actual.bindings) == 3


def test_carried_event_names_are_not_integer_binder_names():
    original = Kernel("carried", (), (Call("start", (), produces="ready"),
        Loop("i", 3, (Call("step", (("offset", Var("i")),), produces="next", awaits=("current",)),),
             iter_args=(("current", "ready"),), yields=("next",), results=("done",)),
        Call("drain", (), awaits=("done",))))
    actual = uniquify_loop_binders(original).kernel
    assert actual.body[1].iter_args == original.body[1].iter_args
    assert actual.body[1].body[0].awaits == ("current",)
    result = transform_loop(original, locate_loop(original, (1,)), action="split", factor=2)
    assert result.kernel.body[-1].awaits == ("done",)


def test_selected_occurrence_reaches_actual_scf_and_llvm_target_callback():
    from xdsl.dialects import builtin, llvm

    from merlin.compile.scheduling.llvm_program import FnBuilderIndexAdapter
    from merlin.compile.scheduling.scf_commands import emit_scf_commands, materialize_scf_commands
    from merlin.llvmlower.host_builder import FnBuilder

    # No target law is invented: this neutral target callback is an external
    # synchronous command interface, not a latency or hardware qualification.
    original = Kernel("commands", (), tuple(Loop("i", 5, (
        Call("command", (("index", Var("i") + offset),)),)) for offset in (0, 10)))
    selected = transform_loop(original, locate_loop(original, (1,)), action="unroll", factor=3)
    scf = materialize_scf_commands(selected.kernel)
    builder = FnBuilder([])
    index = FnBuilderIndexAdapter(builder)
    def lower(name, values, metadata):
        assert name == "command" and not metadata.produces
        builder.add(llvm.CallOp("target_command", index.value(values["index"])))
    emit_scf_commands(scf, index, lower)
    function = llvm.FuncOp("schedule", llvm.LLVMFunctionType([]), body=builder.finish())
    builtin.ModuleOp([llvm.FuncOp("target_command", llvm.LLVMFunctionType([builtin.i64]),
                                 linkage=llvm.LinkageAttr("external")), function]).verify()
    # First sibling remains counted; second becomes three commands per bulk
    # iteration and its ordered two-instance tail. Actual LLVM contains five
    # command sites, rather than the two-site original program.
    assert sum(type(op) is llvm.CallOp for op in function.walk()) == 5
    assert trace(selected.kernel) == trace(original)
