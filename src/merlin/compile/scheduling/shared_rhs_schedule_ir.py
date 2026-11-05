"""Compose existing executable dense decisions into a source-bound RHS group.

No new scheduling language: each member uses the ordinary dense consumer and
the final group_dispatch binds precisely those payloads into one retained epoch.
The consumer checks every operation; unsupported decisions are never ignored.
"""

from xdsl.dialects.builtin import ModuleOp, StringAttr
from xdsl.ir import Block, Region

from merlin.xdsl_dialects.schedule import (
    BindOp,
    DispatchGranularity,
    DispatchGranularityAttr,
    GroupDispatchOp,
    HandleType,
    PlanOp,
)

from .schedule_ir import (
    _binding,
    dense_schedule_text,
    make_dense_schedule_plan,
    parse_dense_schedule_plan,
    read_dense_schedule_plan,
)


def _group(
    problems, resources, storages, adapter, *, source_identity, context_identity, dtypes, static_budget, output_readouts
):
    common = BindOp.create(
        properties={"target": StringAttr(f"shared_rhs:{source_identity}:{context_identity}")},
        result_types=[HandleType()],
    )
    members = [
        BindOp.create(
            properties={
                "target": StringAttr(
                    _binding(
                        p,
                        resources,
                        s,
                        adapter,
                        source_identity=source_identity,
                        dtypes=dtypes,
                        static_budget=static_budget,
                        output_readout=r,
                    )
                )
            },
            result_types=[HandleType()],
        )
        for p, s, r in zip(problems, storages, output_readouts, strict=True)
    ]
    dispatch = GroupDispatchOp.create(
        operands=[common.handle, *(m.handle for m in members)],
        properties={"granularity": DispatchGranularityAttr(DispatchGranularity.FUSED_REGION)},
    )
    return PlanOp.create(
        properties={"sym_name": StringAttr("shared_rhs")}, regions=[Region(Block([common, *members, dispatch]))]
    )


def make_shared_rhs_schedule_plan(
    problems,
    resources,
    storages,
    adapter,
    *,
    source_identity,
    context_identity,
    dtypes,
    static_budget,
    block_tiles,
    output_readouts=None,
    asynchronous=False,
):
    """Propose ordinary member block/epoch decisions and one checked grouping."""
    readouts = (None,) * len(problems) if output_readouts is None else output_readouts
    plans = []
    for i, (p, s, tiles, readout) in enumerate(zip(problems, storages, block_tiles, readouts, strict=True)):
        member = make_dense_schedule_plan(
            p,
            resources,
            s,
            adapter,
            source_identity=source_identity,
            dtypes=dtypes,
            static_budget=static_budget,
            block_tiles=tiles,
            name=f"member{i}",
            output_readout=readout,
            pipeline_prefetch=False if asynchronous else None,
            pipeline_kind="retained_full_k" if asynchronous else None,
        )
        plans.append(member.body.block.first_op.clone())
    plans.append(
        _group(
            problems,
            resources,
            storages,
            adapter,
            source_identity=source_identity,
            context_identity=context_identity,
            dtypes=dtypes,
            static_budget=static_budget,
            output_readouts=readouts,
        )
    )
    module = ModuleOp(plans)
    module.verify()
    return module


def read_shared_rhs_schedule_plan(
    plan,
    problems,
    resources,
    storages,
    adapter,
    *,
    source_identity,
    context_identity,
    dtypes,
    static_budget,
    output_readouts=None,
    asynchronous=False,
):
    """Consume all member decisions; reject changed grouping, context or epochs."""
    if isinstance(plan, str):
        plan = parse_dense_schedule_plan(plan)
    if type(plan) is not ModuleOp:
        raise ValueError("shared RHS schedule requires a builtin.module")
    plan = plan.clone()
    plan.verify()
    ops = tuple(plan.body.block.ops)
    if plan.attributes or plan.properties or len(ops) != len(problems) + 1:
        raise ValueError("shared RHS schedule must retain every member and its group dispatch")
    readouts = (None,) * len(problems) if output_readouts is None else output_readouts
    expected = _group(
        problems,
        resources,
        storages,
        adapter,
        source_identity=source_identity,
        context_identity=context_identity,
        dtypes=dtypes,
        static_budget=static_budget,
        output_readouts=readouts,
    )
    if dense_schedule_text(ModuleOp([ops[-1].clone()])) != dense_schedule_text(ModuleOp([expected])):
        raise ValueError("shared RHS source/member/group binding changed")
    decisions = tuple(
        read_dense_schedule_plan(
            ModuleOp([op.clone()]),
            p,
            resources,
            s,
            adapter,
            source_identity=source_identity,
            dtypes=dtypes,
            static_budget=static_budget,
            output_readout=r,
        )
        for op, p, s, r in zip(ops[:-1], problems, storages, readouts, strict=True)
    )
    for d in decisions:
        if (
            d.outer_order != ("m", "n")
            or d.loop_transforms
            or d.pipeline_kind != ("retained_full_k" if asynchronous else None)
            or d.pipeline_prefetch != (False if asynchronous else None)
            or d.pipeline_issue_order is not None
            or d.pipeline_window_offsets is not None
        ):
            raise ValueError("shared RHS consumer cannot ignore unsupported member scheduling requests")
    return decisions
