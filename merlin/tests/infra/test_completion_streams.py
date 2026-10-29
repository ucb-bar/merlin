"""Counted independent streams: real recurrence fusion, no native ticket claim."""

from dataclasses import replace

import pytest
from test_scf_carried_events import ExecuteCarried
from xdsl.dialects import builtin, llvm

from merlin.compile.scheduling.completion_streams import (
    CompletionEffectEnvelope,
    CompletionEffectsAdapter,
    fuse_completion_streams,
)
from merlin.compile.scheduling.configuration import ConfigurationEffects
from merlin.compile.scheduling.llvm_program import FnBuilderIndexAdapter
from merlin.compile.scheduling.scf_commands import emit_scf_commands, materialize_scf_commands
from merlin.llvmlower.host_builder import FnBuilder
from merlin.sched.ir import Call, Const, Kernel, Loop, Stage, Var
from merlin.sched.primitives import NotApplicable


def source(trips=3, *, nested=False):
    def stream(role, variable, local, passive, initial, passive_initial, output, passive_output):
        memory = Stage("private", role * 16, 16, writes=True)
        body = (
            Call(
                "consume",
                (("stream", Const(role)), ("tile", Var(variable))),
                awaits=(local,),
                stages=(replace(memory, writes=False),),
            ),
            Call("fetch", (("stream", Const(role)), ("tile", Var(variable) + 1)), produces="next", stages=(memory,)),
        )
        if nested:
            # One completed consumer followed by an actual counted inner issue
            # is nested while all outstanding handles remain explicit inputs.
            # A child result cannot implicitly escape. Keep this source closed
            # by carrying an already completed current event through the child.
            body = (
                body[0],
                Loop(
                    "inner" + str(role),
                    1,
                    (body[1],),
                    (("in", local), ("pass", passive)),
                    ("next", "pass"),
                    ("loaded", "through"),
                ),
            )
            yielded = "loaded"
            passive_yield = "through"
        else:
            yielded = "next"
        if not nested:
            passive_yield = passive
        return Loop(
            variable,
            trips,
            body,
            ((local, initial), (passive, passive_initial)),
            (yielded, passive_yield),
            (output, passive_output),
        )

    return Kernel(
        "independent",
        (),
        (
            Call(
                "fetch",
                (("stream", Const(0)), ("tile", Const(0))),
                produces="seed_a",
                stages=(Stage("private", 0, 16, writes=True),),
            ),
            Call(
                "fetch",
                (("stream", Const(1)), ("tile", Const(0))),
                produces="seed_b",
                stages=(Stage("private", 16, 16, writes=True),),
            ),
            stream(0, "i", "a", "b", "seed_a", "seed_b", "done_a", "start_b"),
            stream(1, "j", "b", "a", "start_b", "done_a", "done_b", "final_a"),
            Call(
                "finish",
                (("stream", Const(0)), ("tile", Const(trips))),
                awaits=("final_a",),
                stages=(Stage("private", 0, 16),),
            ),
            Call(
                "finish",
                (("stream", Const(1)), ("tile", Const(trips))),
                awaits=("done_b",),
                stages=(Stage("private", 16, 16),),
            ),
        ),
    )


def effects(call, domains):
    role = call.arg("stream").value
    # This neutral test's real callback touches just its own source/output
    # arrays and declared private cell; these cover every counted instance.
    extent = next(n for name, n in domains if name in ("i", "j"))
    return CompletionEffectEnvelope(
        (
            *call.stages,
            Stage("output" + str(role), 0, extent + 1, writes=call.instr != "fetch"),
            Stage("input" + str(role), 0, extent + 1),
        ),
        ConfigurationEffects(ordered=True),
        True,
        "none",
    )


ADAPTER = CompletionEffectsAdapter(
    effects, "neutral-deferred-array-callback-v1", "exact-integer", "distinct-backing-arrays"
)


def execute(kernel):
    memory, pending, outputs, trace = {}, {}, {}, []
    token = 0

    def lower(instruction, args, metadata):
        nonlocal token
        role, tile = args["stream"], args["tile"]
        trace.append((instruction, role, tile))
        if instruction == "fetch":
            token += 1
            pending[token] = lambda: memory.__setitem__(role, 100 * role + tile)
            return token
        outputs[(role, tile)] = memory[role] * 3

    def await_events(handles):
        for handle in handles:
            action = pending.pop(handle, None)
            if action:
                action()

    emit_scf_commands(materialize_scf_commands(kernel), ExecuteCarried(), lower, await_events=await_events)
    assert not pending
    return outputs, trace


@pytest.mark.parametrize("trips,nested", [(1, False), (3, False), (17, False), (5, True)])
def test_real_fusion_changes_order_and_keeps_both_source_subsequences(trips, nested):
    original = source(trips, nested=nested)
    result = fuse_completion_streams(original, ("i",), ADAPTER)
    result.revalidate(original, ADAPTER)
    before, trace_before = execute(original)
    after, trace_after = execute(result.kernel)
    assert before == after == {(role, i): 3 * (role * 100 + i) for role in (0, 1) for i in range(trips + 1)}
    for role in (0, 1):
        assert [x for x in trace_before if x[1] == role] == [x for x in trace_after if x[1] == role]
    if trips > 1:
        assert trace_before != trace_after
    assert result.report["static_commutation_pairs"] == 4
    assert len(result.kernel.body) == len(original.body) - 1


def test_large_counted_proof_does_not_enumerate_calls():
    result = fuse_completion_streams(source(1_000_001), ("i",), ADAPTER)
    assert result.report["counted_extent"] == 1_000_001
    assert result.report["static_commutation_pairs"] == 4


def test_memory_configuration_unknown_and_observer_effects_refuse():
    original = source()

    def bad(call, domains, kind):
        effect = effects(call, domains)
        if kind == "alias":
            return replace(effect, memory=(Stage("same", 0, 16, writes=True),))
        if kind == "unknown":
            return None
        if kind == "observer":
            return replace(effect, observers="unknown")
        if kind == "state":
            return replace(effect, configuration=ConfigurationEffects(writes=(("mode", 1),), ordered=True))
        return replace(effect, complete=False)

    for kind in ("alias", "unknown", "observer", "state", "incomplete"):
        adapter = replace(ADAPTER, project=lambda call, domains, kind=kind: bad(call, domains, kind))
        with pytest.raises(NotApplicable):
            fuse_completion_streams(original, ("i",), adapter)


def test_changed_flow_or_actual_proposal_cannot_reuse_custody():
    original = source()
    result = fuse_completion_streams(original, ("i",), ADAPTER)
    with pytest.raises(ValueError, match="changed"):
        replace(result, kernel=original).revalidate(original, ADAPTER)
    second = original.body[3]
    changed = replace(original, body=(*original.body[:3], replace(second, extent=4), *original.body[4:]))
    with pytest.raises(NotApplicable, match="equal counted"):
        fuse_completion_streams(changed, ("i",), ADAPTER)


def test_real_llvm_callback_consumes_fused_recurrence_phi_edges():
    original = source(3)
    actual = fuse_completion_streams(original, ("i",), ADAPTER).kernel
    builder = FnBuilder([])
    bridge = FnBuilderIndexAdapter(builder)

    def lower(name, args, metadata):
        call = builder.add(
            llvm.CallOp(
                name, *(bridge.value(v) for v in args.values()), return_type=builtin.i64 if metadata.produces else None
            )
        )
        return call.results[0] if metadata.produces else None

    def wait(handles):
        for h in handles:
            builder.add(llvm.CallOp("complete", h))

    emit_scf_commands(materialize_scf_commands(actual), bridge, lower, await_events=wait)
    function = llvm.FuncOp("fused", llvm.LLVMFunctionType([]), body=builder.finish())
    declarations = [
        llvm.FuncOp(name, llvm.LLVMFunctionType([builtin.i64] * arity, result), linkage=llvm.LinkageAttr("external"))
        for name, arity, result in (
            ("fetch", 2, builtin.i64),
            ("consume", 2, None),
            ("finish", 2, None),
            ("complete", 1, None),
        )
    ]
    builtin.ModuleOp([*declarations, function]).verify()
    assert sum(len(b.args) == 3 for b in function.body.blocks) == 1


def test_tail_predicates_and_enclosing_counted_composition_preserve_each_actual_operand():
    from merlin.sched.ir.expr import select
    from merlin.sched.ir.kernel import concretize, instances
    from merlin.sched.primitives import _map_calls

    original = source(17)
    loops = []
    for op in original.body:
        if type(op) is Loop:
            body = _map_calls(
                op.body, lambda call: replace(call, args=(*call.args, ("rows", select(Var(op.var), 16, 1, 16))))
            )
            op = replace(op, body=body)
        loops.append(op)
    original = replace(original, body=(Loop("batch", 5, tuple(loops)),))
    actual = fuse_completion_streams(original, ("batch", "i"), ADAPTER).kernel

    def calls(kernel, role):
        return [(c.instr, concretize(c, env)) for c, env in instances(kernel) if c.arg("stream") == Const(role)]

    for role in (0, 1):
        assert calls(original, role) == calls(actual, role)
    assert execute(original)[0] == execute(actual)[0]


def test_possible_cross_stream_event_observation_and_bad_domain_refuse():
    original = source()
    first = original.body[2]
    # Awaiting the other stream, even with no private row access, makes it
    # active and destroys the required independent recurrence decomposition.
    first = replace(first, body=(*first.body, Call("observe", (), awaits=("b",))))
    changed = replace(original, body=(*original.body[:2], first, *original.body[3:]))
    with pytest.raises(NotApplicable, match="active in exactly one"):
        fuse_completion_streams(changed, ("i",), ADAPTER)

    # Different persistent fields may coexist, while a write against the
    # other stream's read still cannot commute merely because values match.
    def state(call, domains):
        effect = effects(call, domains)
        cfg = (
            ConfigurationEffects(writes=(("scale", 1),), ordered=True)
            if call.arg("stream") == Const(0)
            else ConfigurationEffects(reads=(("scale", 1),), ordered=True)
        )
        return replace(effect, configuration=cfg)

    with pytest.raises(NotApplicable, match="persistent state"):
        fuse_completion_streams(original, ("i",), replace(ADAPTER, project=state))


def test_existing_persistent_state_contract_composes_only_with_explicit_opt_in():
    original = source()
    loops = []
    for op in original.body:
        if type(op) is Loop:
            calls = []
            for call in op.body:
                key = "stream" + str(call.arg("stream").value) + ".scale"
                call = (
                    replace(call, sets=((key, Const(1)),))
                    if call.instr == "consume"
                    else replace(call, assumes=((key, Const(1)),))
                )
                calls.append(call)
            op = replace(op, body=tuple(calls))
        loops.append(op)
    original = replace(original, body=tuple(loops))

    def project(call, domains):
        config = ConfigurationEffects(
            writes=tuple((k, v.value) for k, v in call.sets),
            reads=tuple((k, v.value) for k, v in call.assumes),
            ordered=True,
        )
        return replace(effects(call, domains), configuration=config)

    adapter = replace(ADAPTER, project=project)
    with pytest.raises(ValueError, match="persistent|annotation"):
        fuse_completion_streams(original, ("i",), adapter)
    result = fuse_completion_streams(original, ("i",), adapter, allow_persistent_state=True)
    result.revalidate(original, adapter)
    materialize_scf_commands(result.kernel, allow_persistent_state=True).verify()


def test_effect_projection_cannot_mutate_the_original_program():
    original = source()

    def bad(call, domains):
        effect = effects(call, domains)
        object.__setattr__(call, "instr", "rewritten_by_projector")
        return effect

    with pytest.raises(ValueError, match="mutated its original source"):
        fuse_completion_streams(original, ("i",), replace(ADAPTER, project=bad))
