"""Real SCF transformations feed actual index/target command callbacks."""

from dataclasses import replace

import pytest
from xdsl.dialects import arith, builtin, func, scf

from merlin.compile.scheduling.scf_commands import (
    SCHEMA,
    emit_scf_commands,
    materialize_scf_commands,
    unroll_small_command_loops,
    verify_scf_commands,
)
from merlin.sched.ir import Call, Kernel, Loop, Stage, Var, evaluate, instances, select


class ExecuteIndices:
    """Small test-only control interpreter; no library tensor implementation."""

    def op(self, name, a, b):
        return a + b if name == "add" else a * b

    def cmp(self, name, a, b):
        assert name == "eq"
        return a == b

    def select(self, test, a, b):
        return a if test else b

    def loop(self, low, high, step, body):
        for index in range(low, high, step):
            body(index)


def record(module):
    calls = []
    emit_scf_commands(module, ExecuteIndices(), lambda name, args, meta: calls.append((name, args)))
    return calls


def test_existing_scf_unroller_changes_actual_ir_but_preserves_every_index_and_tail():
    kernel = Kernel(
        "tail",
        (),
        (
            Loop(
                "m",
                3,
                (
                    Loop(
                        "k",
                        2,
                        (
                            Call(
                                "consume",
                                (
                                    ("offset", Var("m") * 7 + Var("k") * 3),
                                    ("width", select("k", 1, 1, 3)),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    original = materialize_scf_commands(kernel)
    transformed = unroll_small_command_loops(original)
    assert sum(type(op) is scf.ForOp for op in original.walk()) == 2
    assert sum(type(op) is scf.ForOp for op in transformed.walk()) == 1
    expected = [(c.instr, {n: evaluate(v, env) for n, v in c.args}) for c, env in instances(kernel)]
    assert record(original) == record(transformed) == expected
    assert verify_scf_commands(original)["command_ir_sha256"] != verify_scf_commands(transformed)["command_ir_sha256"]


def async_kernel(*, hazard=False, wait=True):
    first = Stage("private", 0, 4, writes=True)
    other = Stage("private", 0 if hazard else 4, 4, writes=True)
    operations = [
        Call("fetch", (("source", Var("i")),), "load", "input_ready", stages=(first,)),
        Call("independent", (), "compute", stages=(other,)),
    ]
    if wait:
        operations.append(
            Call("consume", (), "compute", awaits=("input_ready",), stages=(replace(first, writes=False),))
        )
    return Kernel("overlap", (), (Loop("i", 2, tuple(operations)),))


def test_event_ssa_and_private_lifetimes_survive_actual_unrolling_and_execution():
    original = materialize_scf_commands(async_kernel())
    transformed = unroll_small_command_loops(original)
    assert any(v.type.name == "runtime.event" for op in transformed.walk() for v in op.results)
    for module in (original, transformed):
        device = {"ready": False, "independent": 0, "consumed": []}

        def lower(name, args, metadata):
            assert metadata.unit in ("load", "compute")
            assert metadata.stages
            if name == "fetch":
                device["ready"] = False
                return args["source"]
            if name == "independent":
                assert not device["ready"]
                device["independent"] += 1
            else:
                assert device["ready"]
                device["consumed"].append(device["value"])

        def complete(handles):
            assert len(handles) == 1
            device.update(ready=True, value=handles[0])

        emit_scf_commands(module, ExecuteIndices(), lower, await_events=complete)
        assert device["consumed"] == [0, 1] and device["independent"] == 2


@pytest.mark.parametrize(
    "options,message", [({"hazard": True}, "private rows reused"), ({"wait": False}, "every launched event")]
)
def test_bad_async_lifetime_refuses_before_target_callbacks(options, message):
    with pytest.raises(ValueError, match=message):
        materialize_scf_commands(async_kernel(**options))


def test_event_callbacks_cannot_be_omitted_or_return_no_handle():
    module = materialize_scf_commands(async_kernel())
    calls = []
    with pytest.raises(ValueError, match="actual target completion"):
        emit_scf_commands(module, ExecuteIndices(), lambda *a: calls.append(a))
    assert not calls
    with pytest.raises(ValueError, match="did not return"):
        emit_scf_commands(module, ExecuteIndices(), lambda *a: None, await_events=lambda h: None)


@pytest.mark.parametrize("mutation", ["unknown_op", "wrong_event", "persistent", "overflow", "event_escape"])
def test_unknown_semantics_and_missing_dependencies_refuse(mutation):
    if mutation == "persistent":
        kernel = Kernel("bad", (), (Call("set", (), sets=(("state", 1),)),))
    elif mutation == "overflow":
        kernel = Kernel("bad", (), (Loop("i", 3, (Call("step", (("value", Var("i") * (1 << 62)),)),)),))
    elif mutation == "event_escape":
        kernel = Kernel(
            "bad", (), (Call("launch", (), produces="event"), Loop("i", 2, (Call("wait", (), awaits=("event",)),)))
        )
    else:
        module = materialize_scf_commands(async_kernel())
        if mutation == "unknown_op":
            block = next(op for op in module.walk() if type(op) is func.FuncOp and op.body.blocks).body.block
            block.insert_op_before(arith.SubiOp(block.first_op.result, block.first_op.result), block.last_op)
        else:
            consumer = next(
                op
                for op in module.walk()
                if type(op) is func.CallOp and op.arguments and op.arguments[-1].type.name == "runtime.event"
            )
            import json

            data = json.loads(consumer.attributes[SCHEMA].data)
            data["awaits"] = ["other"]
            consumer.attributes[SCHEMA] = builtin.StringAttr(json.dumps(data))
        with pytest.raises(ValueError):
            verify_scf_commands(module)
        return
    with pytest.raises(ValueError):
        materialize_scf_commands(kernel)


def test_huge_loop_stays_counted_and_budget_refusal_preserves_original():
    kernel = Kernel("huge", (), (Loop("i", 1_000_000, (Call("step", (("index", Var("i")),)),)),))
    module = materialize_scf_commands(kernel)
    original = verify_scf_commands(module)
    result = unroll_small_command_loops(module)
    assert sum(type(op) is func.CallOp for op in result.walk()) == 1
    assert sum(type(op) is scf.ForOp for op in result.walk()) == 1
    with pytest.raises(ValueError, match="operation budget"):
        unroll_small_command_loops(module, max_operations=1)
    assert verify_scf_commands(module) == original


@pytest.mark.parametrize("shape", [(1, 1, 1), (3, 5, 7), (9, 13, 7), (8, 8, 8)])
@pytest.mark.parametrize("prefetch", [False, True])
def test_periodic_dense_uses_exact_original_macro_operands_after_scf_transform(shape, prefetch):
    from merlin.compile.scheduling.llvm_program import emit_index_program
    from merlin.compile.scheduling.periodic_dense import build_periodic_staged_dense, verify_periodic_staged_dense
    from merlin.compile.scheduling.staged_dense import StagedDenseContract, StagedDenseProblem

    contract = StagedDenseContract(
        "selected-test-hardware",
        "selected-test-software",
        "signed-ordered",
        "conditional-test-adapter",
        4,
        16,
        4,
        8,
        32,
        32,
        "read_at_compute_consume",
        "data_commands_preserve_pending_descriptor",
        "raw_accumulator_identity",
        True,
        True,
        False,
    )
    problem = StagedDenseProblem(*shape)
    kernel = build_periodic_staged_dense(problem, contract, prefetch=prefetch)
    verify_periodic_staged_dense(kernel, problem, contract, prefetch=prefetch)
    expected = []
    emit_index_program(kernel, ExecuteIndices(), lambda name, args: expected.append((name, args)))
    actual = unroll_small_command_loops(materialize_scf_commands(kernel))
    assert record(actual) == expected


def test_print_parse_preserves_real_event_edges_and_full_stage_payload():
    from xdsl.context import Context
    from xdsl.parser import Parser

    from merlin.xdsl_dialects import runtime

    module = unroll_small_command_loops(materialize_scf_commands(async_kernel()))
    context = Context()
    for dialect in (builtin.Builtin, arith.Arith, func.Func, scf.Scf, runtime.get_dialect()):
        context.load_dialect(dialect)
    parsed = Parser(context, str(module)).parse_module()
    assert verify_scf_commands(parsed) == verify_scf_commands(module)


@pytest.mark.parametrize("extent,batch", [(2, 2), (3, 2), (8, 4), (9, 4), (1_000_000, 4)])
def test_issue_groups_partition_actual_control_without_expanding_runtime_iterations(extent, batch):
    from merlin.compile.scheduling.scf_commands import ISSUE_GROUP, group_scf_command_loops

    kernel = Kernel("queued", (), (Loop("i", extent, (Call("paired_compute", (("index", Var("i")),)),)),))
    module = materialize_scf_commands(kernel)
    grouped = group_scf_command_loops(module, lambda op: True, max_issues=batch, contract_sha256="a" * 64)
    assert sum(type(op) is func.CallOp for op in grouped.walk()) <= 2
    assert sum(ISSUE_GROUP in op.attributes for op in grouped.walk()) <= 2
    if extent < 100:
        observed, boundaries = [], []
        emit_scf_commands(
            grouped,
            ExecuteIndices(),
            lambda name, args, meta: observed.append(args["index"]),
            enter_issue_group=lambda op: boundaries.append("enter"),
            exit_issue_group=lambda op: boundaries.append("complete"),
        )
        assert observed == list(range(extent))
        assert boundaries == [item for _ in range((extent + batch - 1) // batch) for item in ("enter", "complete")]


def test_queue_completion_is_real_and_missing_hooks_never_erase_group_semantics():
    from merlin.compile.scheduling.scf_commands import group_scf_command_loops

    kernel = Kernel("queued", (), (Loop("i", 7, (Call("paired_compute", (("value", Var("i") + 1),)),)),))
    grouped = group_scf_command_loops(
        materialize_scf_commands(kernel), lambda op: True, max_issues=2, contract_sha256="a" * 64
    )
    calls = []
    with pytest.raises(ValueError, match="actual target entry/completion"):
        emit_scf_commands(grouped, ExecuteIndices(), lambda *a: calls.append(a))
    assert not calls
    state = {"active": False, "pending": [], "acc": 0, "drains": 0}

    def enter(op):
        assert not state["active"] and not state["pending"]
        state["active"] = True

    def lower(name, args, meta):
        assert state["active"]
        state["pending"].append(args["value"])
        assert len(state["pending"]) <= 2

    def complete(op):
        assert state["active"]
        for value in state["pending"]:
            state["acc"] += value
        state.update(active=False, pending=[], drains=state["drains"] + 1)

    emit_scf_commands(grouped, ExecuteIndices(), lower, enter_issue_group=enter, exit_issue_group=complete)
    assert state == {"active": False, "pending": [], "acc": 28, "drains": 4}
    with pytest.raises(ValueError, match="do not erase target region"):
        unroll_small_command_loops(grouped)


@pytest.mark.parametrize("control", ["missing_contract", "selector_mutation", "malformed_marker"])
def test_issue_group_selection_and_contract_cannot_drift(control):
    from merlin.compile.scheduling.scf_commands import ISSUE_GROUP, group_scf_command_loops

    kernel = Kernel("queued", (), (Loop("i", 3, (Call("compute", (("i", Var("i")),)),)),))
    module = materialize_scf_commands(kernel)
    if control == "missing_contract":
        with pytest.raises(ValueError, match="selected contract"):
            group_scf_command_loops(module, lambda op: True, max_issues=2, contract_sha256="unknown")
    elif control == "selector_mutation":

        def bad_selector(op):
            module.attributes[SCHEMA] = builtin.StringAttr("changed")
            return True

        with pytest.raises(ValueError, match="selection changed"):
            group_scf_command_loops(module, bad_selector, max_issues=2, contract_sha256="a" * 64)
    else:
        grouped = group_scf_command_loops(module, lambda op: True, max_issues=2, contract_sha256="a" * 64)
        loop = next(op for op in grouped.walk() if ISSUE_GROUP in op.attributes)
        loop.attributes[ISSUE_GROUP] = builtin.DictionaryAttr(
            {"max_issues": builtin.IntegerAttr(2, builtin.i64), "contract_sha256": builtin.StringAttr("unknown")}
        )
        with pytest.raises(ValueError, match="malformed"):
            verify_scf_commands(grouped)


@pytest.mark.parametrize("order", [("group", "unroll"), ("unroll", "group")])
def test_real_transform_orders_preserve_commands_and_completion_regions(order):
    import json

    from merlin.compile.scheduling.scf_commands import group_scf_command_loops

    kernel = Kernel("ordered", (), (
        Loop("load", 2, (Call("fetch", (("i", Var("load")),)),)),
        Loop("compute", 4, (Call("paired_compute", (("i", Var("compute")),)),)),
    ))
    original = materialize_scf_commands(kernel)
    actual = original

    def compute_loop(loop):
        return any(type(op) is func.CallOp and
                   json.loads(op.attributes[SCHEMA].data)["instruction"] == "paired_compute"
                   for op in loop.body.block.ops)

    for transform in order:
        before = verify_scf_commands(actual)["command_ir_sha256"]
        if transform == "group":
            actual = group_scf_command_loops(actual, compute_loop, max_issues=2, contract_sha256="a" * 64)
        else:
            actual = unroll_small_command_loops(actual, preserve_issue_groups=True)
        assert verify_scf_commands(actual)["command_ir_sha256"] != before
    calls, completions = [], []
    emit_scf_commands(actual, ExecuteIndices(), lambda name, args, meta: calls.append((name, args)),
                      enter_issue_group=lambda op: completions.append("enter"),
                      exit_issue_group=lambda op: completions.append("complete"))
    assert calls == record(original)
    assert completions == ["enter", "complete"] * 2
    assert verify_scf_commands(original)["command_ir_sha256"] == verify_scf_commands(
        materialize_scf_commands(kernel))["command_ir_sha256"]
