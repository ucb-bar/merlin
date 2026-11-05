"""Declared scalar state survives real SCF transforms; no target state law inferred."""

import json

import pytest
from test_scf_commands import ExecuteIndices, async_kernel
from xdsl.dialects import arith, builtin, func, scf

from merlin.compile.scheduling import scf_commands as C
from merlin.sched.ir import Call, Const, Kernel, Loop, Var


def configure(value, **kwargs):
    return Call("configure", (), sets=(("mode", Const(value) if type(value) is int else value),), **kwargs)


def consume(value, **kwargs):
    return Call("consume", (), assumes=(("mode", Const(value) if type(value) is int else value),), **kwargs)


def materialize(*statements):
    return C.materialize_scf_commands(Kernel("state", (), statements), allow_persistent_state=True)


def test_explicit_opt_in_is_required_at_every_public_bridge_boundary():
    kernel = Kernel("state", (), (configure(Const(1)), consume(1)))
    with pytest.raises(ValueError, match="persistent configuration"):
        C.materialize_scf_commands(kernel)
    module = C.materialize_scf_commands(kernel, allow_persistent_state=True)
    for action in (
        lambda: C.verify_scf_commands(module),
        lambda: C.unroll_small_command_loops(module),
        lambda: C.emit_scf_commands(module, ExecuteIndices(), lambda *args: pytest.fail("callback reached")),
        lambda: C.group_scf_command_loops(module, lambda op: True, max_issues=2, contract_sha256="a" * 64),
    ):
        with pytest.raises(ValueError, match="explicit opt-in"):
            action()
    for flag in (None, 1, "yes"):
        with pytest.raises(ValueError, match="explicit Boolean"):
            C.materialize_scf_commands(kernel, allow_persistent_state=flag)


def test_unannotated_materialization_is_byte_identical_when_opted_in():
    kernel = Kernel("plain", (), (Loop("i", 2, (Call("step", (("i", Var("i")),)),)),))
    assert str(C.materialize_scf_commands(kernel)) == str(
        C.materialize_scf_commands(kernel, allow_persistent_state=True)
    )


def test_materialize_print_parse_unroll_and_actual_callbacks_preserve_typed_state():
    from xdsl.context import Context
    from xdsl.parser import Parser

    kernel = Kernel("nested", (), (
        configure(Const(-7)),
        Loop("m", 3, (Loop("n", 2, (
            Call("read", (("index", Var("m") * 2 + Var("n")),), assumes=(("mode", Const(-7)),)),
            configure(9), consume(Const(9)), configure(Const(-7)),
        )),)),
        consume(-7),
    ))
    original = C.materialize_scf_commands(kernel, allow_persistent_state=True)
    transformed = C.unroll_small_command_loops(original, allow_persistent_state=True)
    assert sum(type(op) is scf.ForOp for op in original.walk()) == 2
    assert sum(type(op) is scf.ForOp for op in transformed.walk()) == 1
    context = Context()
    for dialect in (builtin.Builtin, arith.Arith, func.Func, scf.Scf):
        context.load_dialect(dialect)
    parsed = Parser(context, str(transformed)).parse_module()
    assert C.verify_scf_commands(parsed, allow_persistent_state=True) == C.verify_scf_commands(
        transformed, allow_persistent_state=True,
    )
    traces = []
    for module in (original, parsed):
        live, trace = {}, []

        def lower(name, operands, metadata):
            for key, value in metadata.assumes:
                assert live[key] == (value.value if type(value) is Const else value)
            for key, value in metadata.sets:
                live[key] = value.value if type(value) is Const else value
            trace.append((name, operands, metadata.sets, metadata.assumes))

        report = C.emit_scf_commands(module, ExecuteIndices(), lower, allow_persistent_state=True)
        assert report["declared_persistent_state"]["target_state_visibility_and_clobber_law"] == "NOT_ESTABLISHED"
        assert live == {"mode": -7}
        assert [operands["index"] for name, operands, _, _ in trace if name == "read"] == list(range(6))
        traces.append(trace)
    assert traces[0] == traces[1]
    assert traces[0][0][2] == (("mode", Const(-7)),)
    assert traces[0][1][3] == (("mode", Const(-7)),)


@pytest.mark.parametrize("extent", [2, 1_000_000_000])
def test_first_iteration_legal_does_not_authorize_bad_reentry(extent):
    materialize(configure(1), Loop("i", 1, (consume(1), configure(2))))
    with pytest.raises(ValueError, match="loop reentry"):
        materialize(configure(1), Loop("i", extent, (consume(1), configure(2))))


def test_nested_loop_tail_is_visible_to_outer_reentry_and_following_calls():
    with pytest.raises(ValueError, match="loop reentry"):
        materialize(configure(1), Loop("m", 2, (consume(1), Loop("n", 5, (configure(2),)))))
    module = materialize(configure(1), Loop("m", 1, (consume(1), Loop("n", 5, (configure(2),)))), consume(2))
    report = C.verify_scf_commands(module, allow_persistent_state=True)
    assert report["declared_persistent_state"]["final_keys"] == ["mode"]


def test_deep_huge_loops_use_static_summaries_not_nested_iteration_unfolding(monkeypatch):
    statements = (consume(1), configure(Const(1)))
    for depth in range(28):
        statements = (Loop(f"i{depth}", 1_000_000_000, statements),)
    calls = 0
    original = C._command

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(C, "_command", counted)
    module = materialize(configure(1), *statements)
    assert sum(type(op) is func.CallOp for op in module.walk()) == 3
    assert calls == 6  # syntax/event walk + one constant-transfer summary walk


def test_zero_trip_state_has_no_reads_or_writes_even_with_dead_contradiction():
    module = materialize(configure(1), Loop("i", 1, (configure(2),)), consume(2))
    loop = next(op for op in module.walk() if type(op) is scf.ForOp)
    loop.ub.owner.properties["value"] = builtin.IntegerAttr(0, builtin.i64)
    dead = next(op for op in loop.body.block.ops if type(op) is func.CallOp)
    dead.attributes[C.PERSISTENT_STATE] = builtin.StringAttr(json.dumps({
        "sets": [["mode", {"kind": "const", "value": 2}]],
        "assumes": [["uninitialized", {"kind": "const", "value": 42}]],
    }))
    last = [op for op in module.walk() if type(op) is func.CallOp][-1]
    last.attributes[C.PERSISTENT_STATE] = builtin.StringAttr(json.dumps({
        "sets": [], "assumes": [["mode", {"kind": "const", "value": 1}]],
    }))
    C.verify_scf_commands(module, allow_persistent_state=True)
    calls = []
    C.emit_scf_commands(module, ExecuteIndices(), lambda name, args, metadata: calls.append(name),
                        allow_persistent_state=True)
    assert calls == ["configure", "consume"]
    # Dead state still needs syntactically closed scalar declarations.
    data = json.loads(dead.attributes[C.PERSISTENT_STATE].data)
    data["sets"][0][1]["value"] = True
    dead.attributes[C.PERSISTENT_STATE] = builtin.StringAttr(json.dumps(data))
    with pytest.raises(ValueError, match="scalar type"):
        C.verify_scf_commands(module, allow_persistent_state=True)


@pytest.mark.parametrize("statements,message", [
    ((consume(1),), "no earlier setter"),
    ((Call("read_then_set", (), sets=(("mode", Const(1)),), assumes=(("mode", Const(1)),)),), "no earlier setter"),
    ((configure(1), consume(2)), "differs"),
])
def test_unknown_initial_state_and_read_before_write_refuse(statements, message):
    with pytest.raises(ValueError, match=message):
        materialize(*statements)


@pytest.mark.parametrize("value", [True, 1.0, "1", None, Var("i"), Const(1 << 63), 1 << 63, -(1 << 63) - 1])
def test_ambiguous_dynamic_or_out_of_domain_state_refuses(value):
    with pytest.raises(ValueError):
        materialize(Loop("i", 2, (configure(value),)))


@pytest.mark.parametrize(
    "pairs", [(("x", Const(1)), ("x", Const(1))), (("", Const(1)),), ((1, Const(1)),), [("x", Const(1))]]
)
def test_state_keys_and_payload_pair_shapes_are_closed(pairs):
    with pytest.raises(ValueError, match="unique named"):
        materialize(Call("setup", (), sets=pairs))


@pytest.mark.parametrize("payload", [
    '{"sets":[],"sets":[],"assumes":[]}',
    '{"sets":[],"assumes":[],"unknown":0}',
    '{"sets":[["x",{"kind":"integer","value":0.0}]],"assumes":[]}',
    '{"sets":[["x",{"kind":"dynamic","value":0}]],"assumes":[]}',
])
def test_changed_or_ambiguous_actual_ssa_metadata_refuses_before_callbacks(payload):
    module = materialize(configure(1), consume(1))
    command = next(op for op in module.walk() if type(op) is func.CallOp)
    command.attributes[C.PERSISTENT_STATE] = builtin.StringAttr(payload)
    with pytest.raises(ValueError):
        C.emit_scf_commands(module, ExecuteIndices(), lambda *args: pytest.fail("callback reached"),
                            allow_persistent_state=True)


def test_state_does_not_relax_existing_event_or_staging_requirements():
    from dataclasses import replace

    kernel = async_kernel(hazard=True)
    loop = kernel.body[0]
    kernel = replace(kernel, body=(configure(1), replace(loop, body=tuple(
        replace(call, assumes=(("mode", Const(1)),)) for call in loop.body
    ))))
    with pytest.raises(ValueError, match="private rows reused"):
        C.materialize_scf_commands(kernel, allow_persistent_state=True)
    module = C.materialize_scf_commands(async_kernel(), allow_persistent_state=True)
    with pytest.raises(ValueError, match="actual target completion"):
        C.emit_scf_commands(module, ExecuteIndices(), lambda *args: None, allow_persistent_state=True)


def test_issue_group_composition_keeps_state_and_real_completion_hooks():
    module = materialize(configure(1), Loop("i", 7, (consume(1), configure(1))))
    grouped = C.group_scf_command_loops(module, lambda op: True, max_issues=2, contract_sha256="a" * 64,
                                      allow_persistent_state=True)
    actual = C.unroll_small_command_loops(grouped, preserve_issue_groups=True, allow_persistent_state=True)
    names, groups = [], []
    C.emit_scf_commands(actual, ExecuteIndices(), lambda name, args, metadata: names.append((name, metadata)),
                        enter_issue_group=lambda op: groups.append("enter"),
                        exit_issue_group=lambda op: groups.append("complete"), allow_persistent_state=True)
    assert len(names) == 15 and groups == ["enter", "complete"] * 4
    assert all(metadata.assumes == (("mode", Const(1)),) for name, metadata in names if name == "consume")


def test_only_existing_typed_constants_are_admitted_and_both_i64_endpoints_survive():
    with pytest.raises(ValueError, match="closed signed-i64"):
        materialize(Call("raw_python_integer", (), sets=(("mode", 1),)))
    module = materialize(configure(-(1 << 63)), consume(-(1 << 63)),
                         configure((1 << 63) - 1), consume((1 << 63) - 1))
    seen = []
    C.emit_scf_commands(module, ExecuteIndices(), lambda name, args, metadata: seen.append(metadata),
                        allow_persistent_state=True)
    assert seen[0].sets == (("mode", Const(-(1 << 63))),)
    assert seen[-1].assumes == (("mode", Const((1 << 63) - 1)),)


def test_actual_scf_reentry_count_uses_low_high_step_not_just_unit_bounds():
    module = materialize(configure(1), Loop("i", 1, (consume(1), configure(2))))
    loop = next(op for op in module.walk() if type(op) is scf.ForOp)
    loop.ub.owner.properties["value"] = builtin.IntegerAttr(3, builtin.i64)
    loop.step.owner.properties["value"] = builtin.IntegerAttr(2, builtin.i64)
    with pytest.raises(ValueError, match="loop reentry"):
        C.verify_scf_commands(module, allow_persistent_state=True)


def test_partial_state_updates_preserve_other_keys_through_nested_loops():
    module = materialize(
        Call("setup", (), sets=(("a", Const(1)), ("b", Const(2)))),
        Loop("i", 5, (
            Call("read_pair", (), assumes=(("a", Const(1)), ("b", Const(2)))),
            Loop("j", 7, (Call("update_b", (), sets=(("b", Const(2)),)),)),
        )),
        Call("read_a", (), assumes=(("a", Const(1)),)),
    )
    report = C.verify_scf_commands(module, allow_persistent_state=True)
    assert report["declared_persistent_state"]["final_keys"] == ["a", "b"]


def test_persistent_payload_on_noncommand_and_lost_setter_refuse_before_emission():
    module = materialize(configure(1), consume(1))
    calls = [op for op in module.walk() if type(op) is func.CallOp]
    payload = calls[0].attributes.pop(C.PERSISTENT_STATE)
    with pytest.raises(ValueError, match="no earlier setter"):
        C.emit_scf_commands(module, ExecuteIndices(), lambda *args: pytest.fail("callback reached"),
                            allow_persistent_state=True)
    calls[0].attributes[C.PERSISTENT_STATE] = payload
    entry = next(op for op in module.walk() if type(op) is func.FuncOp and op.body.blocks)
    entry.body.block.last_op.attributes[C.PERSISTENT_STATE] = payload
    with pytest.raises(ValueError, match="actual command calls only"):
        C.verify_scf_commands(module, allow_persistent_state=True)
