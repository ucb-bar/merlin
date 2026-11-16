"""Event-driven completion stays event-driven through counted-loop emission."""

import shutil
import subprocess
from dataclasses import replace

import pytest
from xdsl.dialects.builtin import IntegerAttr, ModuleOp, StringAttr, i64

from merlin.compile.scheduling.completion_schedule import apply_completion_plan, make_completion_plan
from merlin.compile.scheduling.static_schedule import schedule_fixed_block
from merlin.sched.codegen import emit_c_function
from merlin.sched.ir import Call, Const, Kernel, Loop
from merlin.sched.isa.model import InstrDef, InstructionSet, IsaError, Operand
from merlin.xdsl_dialects import _common, schedule


def source():
    return Kernel(
        "async_loop",
        (),
        (
            Loop(
                "i",
                3,
                (
                    Call("launch", (("value", Const(7)),), unit="queue", produces="ready"),
                    Call("consume", (), awaits=("ready",)),
                ),
            ),
        ),
    )


def waiter(call):
    return Call("join", (), awaits=(call.produces,))


def test_wait_remains_inside_counted_loop_and_reaches_executed_c(tmp_path):
    kernel = source()
    plan = make_completion_plan(kernel)
    module = _common.roundtrip(ModuleOp([plan]), schedule.get_dialect())
    actual = apply_completion_plan(kernel, module.body.block.first_op, wait=waiter)
    assert isinstance(actual.body[0], Loop) and actual.body[0].extent == 3
    assert actual.body[0].body == (kernel.body[0].body[0], waiter(kernel.body[0].body[0]), kernel.body[0].body[1])
    isa = InstructionSet(
        "queue-test",
        {
            "launch": InstrDef("launch", (Operand("value", "int"),), lambda a: f"launch({a[0]});"),
            "join": InstrDef("join", (), lambda a: "/* explicit completion join */"),
            "consume": InstrDef("consume", (), lambda a: "consume();"),
        },
        {},
        "finish();",
    )
    with pytest.raises(IsaError, match="nothing can render"):
        emit_c_function(actual, isa, symbol="run")

    class Speller:
        def issue_on(self, call):
            return "/* selected queue */"

        def on_produce(self, call):
            return "/* launch owns the completion token */"

        def on_await(self, call):
            return "join();"

    emitted = emit_c_function(actual, isa, symbol="run", spell=Speller())
    assert "for (" in emitted and "join();" in emitted
    if shutil.which("cc") is None:
        pytest.skip("C compiler unavailable")
    # The external engine exposes no completion cycle. Only join publishes it.
    c = (
        """#include <stdint.h>
#include <stdlib.h>
static int pending, visible, count;
void launch(int v) { if(pending) abort(); pending=v; }
void join(void) { if(pending) {visible=pending; pending=0;} }
void consume(void) { if(pending || visible!=7) abort(); count++; }
void finish(void) { if(pending) abort(); }
"""
        + emitted
        + "\nint main(void) {run(); return count!=3;}\n"
    )
    (tmp_path / "test.c").write_text(c)
    subprocess.run(["cc", str(tmp_path / "test.c"), "-o", str(tmp_path / "test")], check=True)
    subprocess.run([str(tmp_path / "test")], check=True)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "wrong_token", "stale", "timed"])
def test_missing_or_unconsumed_completion_decisions_refuse(mutation):
    kernel = source()
    plan = make_completion_plan(kernel)
    if mutation == "missing":
        plan.body.block.erase_op(plan.body.block.last_op)
    elif mutation == "duplicate":
        plan.body.block.add_op(plan.body.block.last_op.clone())
    elif mutation == "stale":
        kernel = replace(kernel, name="different")
    elif mutation == "timed":
        plan.body.block.add_op(
            schedule.DrainOp(properties={"cycle": IntegerAttr(0, i64), "clock": StringAttr("cycles")})
        )
    wait = (lambda c: Call("join", (), awaits=("wrong",))) if mutation == "wrong_token" else waiter
    with pytest.raises(ValueError):
        apply_completion_plan(kernel, plan, wait=wait)


def test_unknown_duration_is_not_admitted_as_fixed_time():
    call = source().body[0].body[0]
    with pytest.raises(ValueError, match="plain straight-line"):
        schedule_fixed_block(
            Kernel("bad", (), (call,)),
            project=lambda c: None,
            capacities={},
            issue_gap=lambda c: 1,
            model_identity="unknown",
            clock="core",
        )
