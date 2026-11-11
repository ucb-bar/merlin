"""Typed timing decisions consume actual operands and physical port demand."""

from dataclasses import replace

import pytest
from xdsl.dialects.builtin import IntegerAttr, ModuleOp, StringAttr, i64
from xdsl.ir import Block, Region

from merlin.compile.scheduling.memory_effects import EncodedAccess
from merlin.compile.scheduling.timed_schedule import (
    ResourceUse, TimedAccess, TimedEffects, verify_timed_block, verify_timing_plan,
)
from merlin.sched.ir import Call, Const, Kernel
from merlin.xdsl_dialects import _common, schedule


def read(address):
    return Call("read", (("address", Const(address)),))


def write(address):
    return Call("write", (("address", Const(address)),))


def project(call):
    if call.instr not in ("read", "write") or tuple(k for k, _ in call.args) != ("address",):
        raise ValueError("unknown instruction")
    address = dict(call.args)["address"].value
    writes = call.instr == "write"
    return TimedEffects(call.instr, 4,
                        (TimedAccess(EncodedAccess("registers", address, 1, writes), 0, 4),),
                        (ResourceUse(("write:" if writes else "read:") + str(address % 2), 0, 4),))


CAPACITIES = {"read:0": 1, "read:1": 1, "write:0": 1, "write:1": 1}


def test_issue_pair_projection_checks_spacing_overlap_and_unknown_rules():
    from merlin.compile.scheduling.timed_schedule import make_timing_plan, verify_issue_timing_plan

    kernel = Kernel("pair", (), (read(0), write(0)))
    def check(starts, distance=3, overlaps=()):
        decision = make_timing_plan(kernel, starts, model_identity="pair-model", clock="cycles", overlaps=overlaps)
        return verify_issue_timing_plan(kernel, decision, model_identity="pair-model", clock="cycles",
                                        completion=lambda call: 5, distance=lambda a, b: distance)
    assert check((0, 3), overlaps=((0, 1, "require"),)) == ((0, 3), (5, 8))
    with pytest.raises(ValueError, match="distance violated"):
        check((0, 2))
    with pytest.raises(ValueError, match="overlap"):
        check((0, 5), overlaps=((0, 1, "require"),))
    with pytest.raises(ValueError, match="unknown"):
        check((0, 5), distance=None)


def plan(kernel, starts, *, complete=None, overlap=None):
    ops, handles = [], []
    for i in range(len(kernel.body)):
        bind = schedule.BindOp(result_types=[schedule.HandleType()], properties={"target": StringAttr(f"call:{i}")})
        handles.append(bind.handle)
        ops.append(bind)
    for i, cycle in enumerate(starts):
        ops.append(schedule.TimeAnnotationOp(operands=[handles[i]], properties={
            "event": StringAttr("issue"), "clock": StringAttr("test-clock"), "cycle": IntegerAttr(cycle, i64),
        }))
    if complete is not None:
        ops.append(schedule.TimeAnnotationOp(operands=[handles[0]], properties={
            "event": StringAttr("complete"), "clock": StringAttr("test-clock"), "cycle": IntegerAttr(complete, i64),
        }))
    if overlap is not None:
        ops.append(schedule.OverlapOp(operands=handles[:2], properties={"mode": StringAttr(overlap)}))
    return schedule.PlanOp(properties={"sym_name": StringAttr("timed")}, attributes={
        "kernel_sha256": StringAttr(kernel.digest()), "timing_model": StringAttr("test-model"),
    }, regions=[Region(Block(ops))])


def check(kernel, decision):
    return verify_timing_plan(kernel, decision, model_identity="test-model", clock="test-clock",
                              project=project, capacities=CAPACITIES)


def test_independent_operations_overlap_and_roundtrip():
    kernel = Kernel("k", (), (read(0), read(1)))
    decision = plan(kernel, (0, 1), complete=4, overlap="require")
    assert check(kernel, decision) == ((0, 1), (4, 5))
    module = _common.roundtrip(ModuleOp([decision]), schedule.get_dialect())
    assert check(kernel, next(iter(module.body.block.ops))) == ((0, 1), (4, 5))


@pytest.mark.parametrize("calls", [(write(0), read(0)), (read(0), write(0)), (write(0), write(0))])
def test_raw_war_waw_require_actual_access_completion(calls):
    kernel = Kernel("k", (), calls)
    assert check(kernel, plan(kernel, (0, 4), overlap="forbid")) == ((0, 4), (4, 8))
    with pytest.raises(ValueError, match="memory dependence"):
        check(kernel, plan(kernel, (0, 3)))
    with pytest.raises(ValueError, match="memory dependence"):
        check(kernel, plan(kernel, (4, 0)))


def test_two_readers_still_contend_for_the_physical_port():
    kernel = Kernel("k", (), (read(0), read(2)))
    with pytest.raises(ValueError, match="resource read:0"):
        check(kernel, plan(kernel, (0, 1)))
    check(kernel, plan(kernel, (0, 4)))


def test_stale_or_unrealized_decisions_refuse():
    kernel = Kernel("k", (), (read(0), read(1)))
    with pytest.raises(ValueError, match="completion"):
        check(kernel, plan(kernel, (0, 1), complete=3))
    with pytest.raises(ValueError, match="not realized"):
        check(kernel, plan(kernel, (0, 5), overlap="require"))
    with pytest.raises(ValueError, match="not realized"):
        check(kernel, plan(kernel, (0, 1), overlap="forbid"))
    with pytest.raises(ValueError, match="current kernel"):
        check(replace(kernel, body=(read(2), read(3))), plan(kernel, (0, 1)))
    decision = plan(kernel, (0, 1))
    next(o for o in decision.body.block.ops if type(o) is schedule.TimeAnnotationOp).properties["clock"] = StringAttr("other")
    with pytest.raises(ValueError, match="clock"):
        check(kernel, decision)


def test_unknown_effects_resources_and_async_calls_refuse():
    kernel = Kernel("k", (), (read(0),))
    with pytest.raises(ValueError, match="unknown timing"):
        verify_timed_block(kernel, (0,), project=lambda _: None, capacities=CAPACITIES)
    with pytest.raises(ValueError, match="unknown capacity"):
        verify_timed_block(kernel, (0,), project=project, capacities={})
    with pytest.raises(ValueError, match="plain straight-line"):
        verify_timed_block(replace(kernel, body=(replace(read(0), produces="dma"),)), (0,),
                           project=project, capacities=CAPACITIES)


def test_repeated_decisions_and_missing_calls_refuse():
    kernel = Kernel("k", (), (read(0), read(1)))
    with pytest.raises(ValueError, match="every actual instruction"):
        check(kernel, plan(kernel, (0,)))
    decision = plan(kernel, (0, 1))
    annotation = next(o for o in decision.body.block.ops if type(o) is schedule.TimeAnnotationOp)
    decision.body.block.add_op(schedule.TimeAnnotationOp(operands=[annotation.target], properties=dict(annotation.properties)))
    with pytest.raises(ValueError, match="repeated timing"):
        check(kernel, decision)


def test_repeated_overlap_and_unconsumed_metadata_refuse():
    kernel = Kernel("k", (), (read(0), read(1)))
    decision = plan(kernel, (0, 1), overlap="require")
    overlap = next(o for o in decision.body.block.ops if type(o) is schedule.OverlapOp)
    decision.body.block.add_op(schedule.OverlapOp(operands=[overlap.second, overlap.first],
                                                properties={"mode": StringAttr("forbid")}))
    with pytest.raises(ValueError, match="repeated overlap"):
        check(kernel, decision)
    decision = plan(kernel, (0, 1))
    next(iter(decision.body.block.ops)).attributes["unconsumed"] = StringAttr("decision")
    with pytest.raises(ValueError, match="unknown timing decision"):
        check(kernel, decision)


def test_resource_capacity_counts_each_use_and_releases_at_interval_end():
    kernel = Kernel("k", (), (read(0), read(2)))
    capacities = {**CAPACITIES, "read:0": 2}
    assert verify_timed_block(kernel, (0, 0), project=project, capacities=capacities) == (4, 4)
    kernel = replace(kernel, body=(read(0), read(2), read(4)))
    with pytest.raises(ValueError, match="over capacity"):
        verify_timed_block(kernel, (0, 0, 3), project=project, capacities=capacities)
    assert verify_timed_block(kernel, (0, 0, 4), project=project, capacities=capacities) == (4, 4, 8)


@pytest.mark.parametrize("event,cycle,clock", [("unknown", 0, "clock"), ("issue", -1, "clock"), ("issue", 0, "")])
def test_invalid_annotation_contract_refuses(event, cycle, clock):
    from xdsl.utils.exceptions import VerifyException

    handle = Block(arg_types=[schedule.HandleType()]).args[0]
    annotation = schedule.TimeAnnotationOp(operands=[handle], properties={
        "event": StringAttr(event), "cycle": IntegerAttr(cycle, i64), "clock": StringAttr(clock),
    })
    with pytest.raises(VerifyException):
        annotation.verify()
