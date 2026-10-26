"""Executable dense schedule decisions, conditional on the original source contract.

This consumes plan/bind/block_contraction and executed loop_transform requests. Other schedule
operations need their own verified consumers; accepting them as annotations
would silently ignore requested transformations. No target is discovered here.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, replace
from io import StringIO

from xdsl.context import Context
from xdsl.dialects.builtin import ArrayAttr, BoolAttr, Builtin, DenseArrayBase, IntegerAttr, ModuleOp, StringAttr, i64
from xdsl.ir import Block, Region
from xdsl.parser import Parser
from xdsl.printer import Printer

from merlin.transforms.dense_ir import DenseScheduleOp, dense_schedule_op
from merlin.transforms.dense_planning import dense_blocks
from merlin.xdsl_dialects.schedule import BindOp, HandleType, LoopTransformOp, PanelPipelineOp, PlanOp, get_dialect


def dense_schedule_text(plan):
    stream = StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(plan)
    return stream.getvalue()


def parse_dense_schedule_plan(text):
    """Parse ordinary editable xDSL text with only builtin and schedule loaded."""
    context = Context()
    context.load_dialect(Builtin)
    context.load_dialect(get_dialect())
    return Parser(context, text).parse_module()


def _binding(
    problem,
    resources,
    storage,
    adapter,
    *,
    source_identity,
    dtypes,
    static_budget,
    output_readout=None,
    bias_source=None,
):
    from .compact_dense import _digest

    if (type(source_identity) is not str or len(source_identity) != 64
            or any(c not in "0123456789abcdef" for c in source_identity)):
        raise ValueError("dense schedule needs the original source SHA256 identity")
    context = (
        asdict(problem),
        asdict(resources),
        asdict(storage),
        dtypes,
        static_budget,
        adapter.configuration_identity,
        adapter.numerical_policy_identity,
        adapter.synchronous_completion,
        adapter.exact_accumulator_continuation,
        adapter.index_bits,
        adapter.qualification_identity,
        None if output_readout is None else asdict(output_readout),
        None if bias_source is None else asdict(bias_source),
    )
    return f"dense:{source_identity}:{_digest(context)}"


def _layout(storage, bases):
    """Narrow assigned windows; never invent a backing, width or allocation."""
    from .compact_dense import DenseLoopStorage

    views = (storage.lhs, storage.rhs, storage.accumulator)
    if any(view is None for view in views):
        raise ValueError("dense schedule needs an original private accumulator assignment")
    if len(bases) != 3 or any(type(base) is not int for base in bases):
        raise ValueError("dense schedule needs three integer row bases")
    narrowed = []
    for view, base in zip(views, bases):
        if not view.base <= base < view.base + view.capacity:
            raise ValueError("dense schedule row base escapes original assigned window")
        narrowed.append(replace(view, base=base, capacity=view.base + view.capacity - base))
    return DenseLoopStorage(*narrowed)


def _block(problem, resources, storage, tiles):
    if (
        len(tiles) != 3
        or any(type(v) is not int or v <= 0 for v in tiles)
        or any(
            v > bound
            for v, bound in zip(tiles, (resources.max_m_blocks, resources.max_k_blocks, resources.max_n_blocks))
        )
    ):
        raise ValueError("dense schedule block exceeds original supported domain")
    m, k, n = tiles
    bounded = replace(resources, max_m_blocks=m, max_k_blocks=k, max_n_blocks=n)
    blocks = dense_blocks(problem, bounded, k_limit=lambda m, n: storage.k_limit(problem, resources, m, n))
    for block in blocks:
        if (block.m, block.k, block.n) == tiles and storage.admits(problem, block, resources):
            return block
    raise ValueError("dense schedule block does not fit original geometry/storage/tail capabilities")


@dataclass(frozen=True)
class DenseScheduleDecision:
    block: object
    storage: object
    outer_order: tuple[str, str]
    static_commands: int
    plan_text: str
    plan_sha256: str
    loop_transforms: tuple = ()
    pipeline_prefetch: bool | None = None
    pipeline_issue_order: str | None = None
    pipeline_window_offsets: tuple | None = None
    pipeline_kind: str | None = None


def make_dense_schedule_plan(
    problem,
    resources,
    storage,
    adapter,
    *,
    source_identity,
    dtypes,
    static_budget,
    block_tiles,
    outer_order=("m", "n"),
    row_bases=None,
    name="dense",
    output_readout=None,
    bias_source=None,
    loop_transforms=(),
    pipeline_prefetch=None,
    pipeline_issue_order=None,
    pipeline_window_offsets=None,
    pipeline_kind=None,
):
    """Build editable decision IR; this is a proposal, not a realization proof.

    ``source_identity`` comes from the unchanged original source. This binding
    detects mismatched contexts; it does not prove a supplied source or effect
    declaration. The normal source reader/selected contracts retain that duty.
    Static command count is measured from the actual adapter-emitted program.
    """
    from .compact_dense import _static_calls, emit_compact_dense

    binding = _binding(
        problem,
        resources,
        storage,
        adapter,
        source_identity=source_identity,
        dtypes=dtypes,
        static_budget=static_budget,
        output_readout=output_readout,
        bias_source=bias_source,
    )
    if storage.accumulator is None:
        raise ValueError("dense schedule needs an original private accumulator assignment")
    if row_bases is None:
        row_bases = tuple(v.base for v in (storage.lhs, storage.rhs, storage.accumulator))
    chosen_storage = _layout(storage, row_bases)
    block = _block(problem, resources, chosen_storage, block_tiles)
    kernel = emit_compact_dense(
        problem,
        block,
        chosen_storage,
        adapter,
        name=name,
        dtypes=dtypes,
        outer_order=outer_order,
        output_readout=output_readout,
        bias_source=bias_source,
    )
    if pipeline_prefetch is None and (
        pipeline_issue_order is not None or pipeline_window_offsets is not None or pipeline_kind is not None
    ):
        raise ValueError("panel issue order or windows require an explicit executable panel pipeline decision")
    if pipeline_kind not in (None, "two_window", "retained_full_k", "retained_lhs_segmented_k"):
        raise ValueError("unsupported executable panel pipeline kind")
    retained = pipeline_kind == "retained_full_k"
    segmented = pipeline_kind == "retained_lhs_segmented_k"
    if retained and (
        pipeline_prefetch is not False or pipeline_issue_order is not None or pipeline_window_offsets is not None
    ):
        raise ValueError("retained full-K stages do not accept prefetch or two-window decisions")
    if retained and loop_transforms:
        from .loop_schedule import propose_loop_transforms

        kernel = propose_loop_transforms(kernel, loop_transforms, index_bits=adapter.index_bits)
    if pipeline_prefetch is not None:
        if type(pipeline_prefetch) is not bool:
            raise ValueError("panel pipeline needs an explicit Boolean issue policy")
        from .dense_panel_pipeline import build_dense_panel_pipeline, complete_dense_retained_reduction

        if segmented:
            from .dense_segmented_pipeline import build_dense_segmented_pipeline

            kernel = build_dense_segmented_pipeline(
                problem, block, resources, chosen_storage, adapter,
                name=name, dtypes=dtypes, prefetch=pipeline_prefetch, outer_order=outer_order,
                output_readout=output_readout, bias_source=bias_source, window_offsets=pipeline_window_offsets,
                issue_order="compute_then_prefetch" if pipeline_issue_order is None else pipeline_issue_order,
            )
        else:
            kernel = (
                complete_dense_retained_reduction(kernel, problem, block, resources, chosen_storage, adapter)
                if retained
                else build_dense_panel_pipeline(
                    problem,
                    block,
                    resources,
                    chosen_storage,
                    adapter,
                    name=name,
                    dtypes=dtypes,
                    prefetch=pipeline_prefetch,
                    outer_order=outer_order,
                    output_readout=output_readout,
                    bias_source=bias_source,
                    issue_order="compute_then_prefetch" if pipeline_issue_order is None else pipeline_issue_order,
                    window_offsets=pipeline_window_offsets,
                )
            )
    if loop_transforms and not retained:
        from .loop_schedule import propose_loop_transforms

        kernel = propose_loop_transforms(kernel, loop_transforms, index_bits=adapter.index_bits)
    sites = sum(1 for _ in _static_calls(kernel.body))
    op = dense_schedule_op(problem, block, resources, sites, static_budget)
    op.attributes["outer_order"] = ArrayAttr(StringAttr(v) for v in outer_order)
    op.attributes["row_bases"] = DenseArrayBase.from_list(i64, row_bases)
    bind = BindOp.create(properties={"target": StringAttr(binding)}, result_types=[HandleType()])
    decisions = []
    if pipeline_prefetch is not None:
        pipeline_properties = {
            "prefetch": BoolAttr.from_bool(pipeline_prefetch),
        }
        if pipeline_kind is not None:
            pipeline_properties["kind"] = StringAttr(pipeline_kind)
        if not retained:
            pipeline_properties["issue_order"] = StringAttr(
                "compute_then_prefetch" if pipeline_issue_order is None else pipeline_issue_order
            )
        if pipeline_window_offsets is not None:
            pipeline_properties["window_offsets"] = DenseArrayBase.from_list(
                i64, tuple(offset for slot in pipeline_window_offsets for offset in slot)
            )
        decisions.append(
            PanelPipelineOp.create(
                operands=[bind.handle],
                properties=pipeline_properties,
            )
        )
    for request in loop_transforms:
        properties = {"action": StringAttr(request.action), "path": ArrayAttr(StringAttr(v) for v in request.path)}
        if request.factor is not None:
            properties["factor"] = IntegerAttr(request.factor, i64)
        if request.cut is not None:
            properties["cut"] = IntegerAttr(request.cut, i64)
        if request.names:
            properties["names"] = ArrayAttr(StringAttr(v) for v in request.names)
        if request.index is not None:
            properties["index"] = IntegerAttr(request.index, i64)
        if request.cuts:
            properties["cuts"] = DenseArrayBase.from_list(i64, request.cuts)
        decisions.append(LoopTransformOp.create(operands=[bind.handle], properties=properties))
    if retained:
        # Hoists act on the raw counted program before completion formation.
        decisions = [*decisions[1:], decisions[0]]
    plan = PlanOp.create(properties={"sym_name": StringAttr(name)}, regions=[Region(Block([bind, op, *decisions]))])
    module = ModuleOp([plan])
    module.verify()
    return module


def read_dense_schedule_plan(
    plan,
    problem,
    resources,
    storage,
    adapter,
    *,
    source_identity,
    dtypes,
    static_budget,
    output_readout=None,
    bias_source=None,
):
    """Read chosen knobs while re-deriving capacities/counts from original facts."""
    if type(plan) is not ModuleOp:
        raise ValueError("dense schedule consumer requires a typed builtin.module")
    # Consume a snapshot: later edits cannot alter the selected executable IR.
    plan = plan.clone()
    plan.verify()
    if (
        plan.attributes
        or plan.properties
        or len(plan.regions) != 1
        or len(plan.body.blocks) != 1
        or plan.body.block.args
    ):
        raise ValueError("dense schedule requires an ordinary unannotated module")
    members = list(plan.body.block.ops)
    if len(members) != 1 or type(members[0]) is not PlanOp:
        raise ValueError("dense schedule requires exactly one schedule.plan")
    owner = members[0]
    if (
        owner.attributes
        or set(owner.properties) != {"sym_name"}
        or owner.operands
        or owner.results
        or len(owner.body.blocks) != 1
        or owner.body.block.args
    ):
        raise ValueError("dense schedule plan has unsupported policies, arguments or attributes")
    ops = list(owner.body.block.ops)
    if (
        len(ops) < 2
        or type(ops[0]) is not BindOp
        or type(ops[1]) is not DenseScheduleOp
        or any(type(op) not in (LoopTransformOp, PanelPipelineOp) for op in ops[2:])
    ):
        raise ValueError("dense schedule consumer supports only bind then block_contraction and checked loop requests")
    bind, op = ops[:2]
    requests = ops[2:]
    pipeline_prefetch = None
    pipeline_issue_order = None
    pipeline_window_offsets = None
    pipeline_kind = None
    pipelines = [request for request in requests if type(request) is PanelPipelineOp]
    if len(pipelines) > 1:
        raise ValueError("one executable completion policy is required")
    if pipelines:
        selected_pipeline = pipelines[0]
        pipeline_kind = selected_pipeline.kind.data if selected_pipeline.kind is not None else "two_window"
        if selected_pipeline is not (requests[-1] if pipeline_kind == "retained_full_k" else requests[0]):
            raise ValueError("retained completion must follow loop transforms; two-window completion must precede them")
        if (
            selected_pipeline.target is not bind.handle
            or selected_pipeline.attributes
            or set(selected_pipeline.properties) - {"prefetch", "issue_order", "window_offsets", "kind"}
            or "prefetch" not in selected_pipeline.properties
        ):
            raise ValueError("panel pipeline has unknown effects or a different original source binding")
        pipeline_prefetch = bool(selected_pipeline.prefetch.value.data)
        pipeline_issue_order = (
            None
            if pipeline_kind == "retained_full_k"
            else (
                selected_pipeline.issue_order.data
                if selected_pipeline.issue_order is not None
                else "compute_then_prefetch"
            )
        )
        if pipeline_kind == "retained_lhs_segmented_k":
            from .dense_segmented_pipeline import SEGMENTED_ISSUE_ORDERS as supported_issue_orders
        else:
            from .dense_panel_pipeline import ISSUE_ORDERS as supported_issue_orders

        if pipeline_issue_order is not None and pipeline_issue_order not in supported_issue_orders:
            raise ValueError("unsupported executable panel issue order")
        if selected_pipeline.window_offsets is not None:
            offsets = tuple(selected_pipeline.window_offsets.get_values())
            pipeline_window_offsets = (offsets[:2], offsets[2:])
        requests = [request for request in requests if request is not selected_pipeline]
    if any(type(request) is not LoopTransformOp for request in requests):
        raise ValueError("one panel pipeline must precede the executed loop transforms")
    if requests:
        from .loop_schedule import LoopTransform

    transforms = []
    for request in requests:
        if (
            request.target is not bind.handle
            or request.attributes
            or request.results
            or request.regions
            or set(request.properties) - {"action", "path", "factor", "cut", "names", "index", "cuts"}
        ):
            raise ValueError("dense loop transform has unknown effects or a different bound source")
        transforms.append(
            LoopTransform(
                request.action.data,
                tuple(v.data for v in request.path),
                None if request.factor is None else request.factor.value.data,
                None if request.cut is None else request.cut.value.data,
                () if request.names is None else tuple(v.data for v in request.names),
                None if request.index is None else request.index.value.data,
                () if request.cuts is None else tuple(request.cuts.get_values()),
            )
        )
    expected = _binding(
        problem,
        resources,
        storage,
        adapter,
        source_identity=source_identity,
        dtypes=dtypes,
        static_budget=static_budget,
        output_readout=output_readout,
        bias_source=bias_source,
    )
    if (
        bind.target.data != expected
        or bind.attributes
        or set(bind.properties) != {"target"}
        or bind.operands
        or bind.regions
    ):
        raise ValueError("dense schedule binding differs from original source/effects/resources")
    required = {
        "problem",
        "primitive_tile",
        "block_tiles",
        "element_bytes",
        "storage_bytes",
        "storage_capacity",
        "static_commands",
        "command_budget",
        "outer_order",
        "row_bases",
    }
    if set(op.attributes) != required or op.properties or op.operands or op.results or op.regions:
        raise ValueError("dense schedule block has missing or unsupported executable decisions")
    chosen_storage = _layout(storage, tuple(op.row_bases.get_values()))
    block = _block(problem, resources, chosen_storage, tuple(op.block_tiles.get_values()))
    canonical = dense_schedule_op(problem, block, resources, op.static_commands.value.data, static_budget)
    if any(op.attributes[key] != value for key, value in canonical.attributes.items()):
        raise ValueError("dense schedule dimensions, capacity or command budget changed")
    text = dense_schedule_text(plan)
    return DenseScheduleDecision(
        block,
        chosen_storage,
        tuple(v.data for v in op.outer_order),
        op.static_commands.value.data,
        text,
        hashlib.sha256(text.encode()).hexdigest(),
        tuple(transforms),
        pipeline_prefetch,
        pipeline_issue_order,
        pipeline_window_offsets,
        pipeline_kind,
    )


def check_dense_schedule_commands(decision, actual_commands, static_budget):
    if actual_commands != decision.static_commands or actual_commands > static_budget:
        raise ValueError("dense schedule command count differs from actual emitted program")
