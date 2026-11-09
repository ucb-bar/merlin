"""Source-checked, two-window dense panel issue and explicit completion epochs.

The ordinary dense verifier checks the actual serial projection of every source
product. A separate epoch checker proves that only disjoint future DMA writes
cross current compute reads, that accumulator writes in an epoch are disjoint,
and that every epoch drains. The target must independently qualify those issue,
configuration and full-drain semantics; this module supplies no target answers."""

from __future__ import annotations

from dataclasses import asdict as asdict
from dataclasses import dataclass as dataclass
from dataclasses import replace as replace
from itertools import product as product
from math import prod as prod

from merlin.compiler.schedule.blocks import Compute as Compute
from merlin.compiler.schedule.blocks import InitializeAccumulator as InitializeAccumulator
from merlin.compiler.schedule.blocks import Load as Load
from merlin.compiler.schedule.blocks import Preload as Preload
from merlin.compiler.schedule.blocks import Store as Store
from merlin.compiler.schedule.compact_dense import DenseBiasSource as DenseBiasSource
from merlin.compiler.schedule.compact_dense import DenseLoopStorage as DenseLoopStorage
from merlin.compiler.schedule.compact_dense import _digest as _digest
from merlin.compiler.schedule.compact_dense import emit_compact_dense as emit_compact_dense
from merlin.compiler.schedule.dense_contracts import DenseIntegerReadout as DenseIntegerReadout
from merlin.compiler.schedule.dense_contracts import DenseTiledIntegerReadout as DenseTiledIntegerReadout
from merlin.compiler.schedule.dense_contracts import TiledIntegerStore as TiledIntegerStore
from merlin.compiler.schedule.dense_panel_contracts import DensePanelOverlapAdmission as DensePanelOverlapAdmission
from merlin.compiler.schedule.dense_panel_contracts import DenseReadoutComputeAdmission as DenseReadoutComputeAdmission
from merlin.compiler.schedule.dense_panel_contracts import (
    DenseReadoutConfigurationSnapshot as DenseReadoutConfigurationSnapshot,
)
from merlin.compiler.schedule.dense_panel_contracts import DenseReadyReadoutAdmission as DenseReadyReadoutAdmission
from merlin.compiler.schedule.dense_panel_ir import DRAIN as DRAIN
from merlin.compiler.schedule.dense_panel_ir import _calls as _calls
from merlin.compiler.schedule.dense_panel_ir import _loops as _loops
from merlin.compiler.schedule.dense_panel_ir import _serial_projection as _serial_projection
from merlin.compiler.schedule.dense_retained import DenseRetainedChoice as DenseRetainedChoice
from merlin.compiler.schedule.dense_retained import DenseRetainedReductionAdmission as DenseRetainedReductionAdmission
from merlin.compiler.schedule.dense_retained import _dense_retained_compute_nests as _dense_retained_compute_nests
from merlin.compiler.schedule.dense_retained import (
    complete_dense_retained_reduction as complete_dense_retained_reduction,
)
from merlin.compiler.schedule.dense_retained import dense_m_panel_capacity as dense_m_panel_capacity
from merlin.compiler.schedule.dense_retained import dense_retained_choices as dense_retained_choices
from merlin.compiler.schedule.dense_retained import dense_retained_compute_nests as dense_retained_compute_nests
from merlin.compiler.schedule.dense_retained import dense_retained_compute_order as dense_retained_compute_order
from merlin.compiler.schedule.dense_retained import dense_retained_compute_unroll as dense_retained_compute_unroll
from merlin.compiler.schedule.dense_retained import dense_retained_operand_hoists as dense_retained_operand_hoists
from merlin.compiler.schedule.dense_retained import verify_dense_retained_reduction as verify_dense_retained_reduction
from merlin.compiler.schedule.library_sources import admission_point as admission_point
from merlin.sched.ir import Call as Call
from merlin.sched.ir import Const as Const
from merlin.sched.ir import Kernel as Kernel
from merlin.sched.ir import Loop as Loop
from merlin.sched.ir import Ptr as Ptr
from merlin.sched.ir import Scaled as Scaled
from merlin.sched.ir import Select as Select
from merlin.sched.ir import Sum as Sum
from merlin.sched.ir import Var as Var
from merlin.sched.ir import check_structure as check_structure
from merlin.sched.ir.expr import add as add
from merlin.sched.ir.expr import mul as mul
from merlin.sched.ir.expr import select as select
from merlin.sched.ir.integer_domain import check_integer_domain as check_integer_domain

from .dense_panel_build import (
    ISSUE_ORDERS as ISSUE_ORDERS,
)
from .dense_panel_build import (
    _remap as _remap,
)
from .dense_panel_build import (
    _shift as _shift,
)
from .dense_panel_build import (
    _windows as _windows,
)
from .dense_panel_build import (
    build_dense_panel_pipeline as build_dense_panel_pipeline,
)
from .dense_panel_build import (
    dense_panel_rhs_stripes as dense_panel_rhs_stripes,
)
from .dense_panel_build import (
    select_dense_panel_window_offsets as select_dense_panel_window_offsets,
)
from .dense_panel_projection import (
    _collapse_compute_radices as _collapse_compute_radices,
)
from .dense_panel_projection import (
    _fission_noninterfering_epoch_loads as _fission_noninterfering_epoch_loads,
)
from .dense_panel_verify import (
    _verify_dense_epochs as _verify_dense_epochs,
)
from .dense_panel_verify import (
    verify_dense_panel_pipeline as verify_dense_panel_pipeline,
)


@dataclass(frozen=True)
class DensePanelPipelineResult:
    """Checked actual program; execution qualification stays with its owner."""

    kernel: Kernel
    realization: dict
    source_identity: str
    admission: DensePanelOverlapAdmission
    checked_kernel_sha256: str
    checked_admission_sha256: str
    checked_library_sources: tuple
    source_module: object = None
    selected_contracts: object = None
    output_readout: object = None
    bias_source: object = None
    readout_qualification_identity: str | None = None
    checked_boundary_sha256: str | None = None

    def revalidate(self):
        from merlin.compiler.schedule.library_sources import shared_library_sources
        from merlin.transforms.certificates import ir_digest

        if (
            self.output_readout is not None
            and type(self.output_readout) not in (DenseIntegerReadout, DenseTiledIntegerReadout)
            or self.bias_source is not None
            and type(self.bias_source) is not DenseBiasSource
        ):
            raise ValueError("panel original readout or bias boundary changed")
        if (
            self.kernel.digest() != self.checked_kernel_sha256
            or _digest(asdict(self.admission)) != self.checked_admission_sha256
            or tuple(sorted(shared_library_sources(version=26).items())) != self.checked_library_sources
            or self.source_module is not None
            and ir_digest(self.source_module) != self.source_identity
            or _digest(
                (
                    None if self.output_readout is None else asdict(self.output_readout),
                    None if self.bias_source is None else asdict(self.bias_source),
                    self.readout_qualification_identity,
                )
            )
            != self.checked_boundary_sha256
        ):
            raise ValueError("panel source, checked emitted program, execution premises or library changed")
        if self.selected_contracts is not None:
            self.selected_contracts.revalidate()

    def emit_llvm(self, builder, lower_call):
        """Emit the checked actual commands, including every explicit drain.

        The callback must realize the selected ordered issue and full completion
        semantics. This does not turn a callback into physical qualification.
        """
        from merlin.compiler.schedule.llvm_program import FnBuilderIndexAdapter, emit_index_program

        self.revalidate()
        receipt = emit_index_program(self.kernel, FnBuilderIndexAdapter(builder), lower_call)
        self.revalidate()
        return receipt

    def emit_index_program(self, control, lower_effect, *, complete_epoch):
        """Lower checked effects and explicit full-completion boundaries separately.

        ``lower_effect`` issues the actual instruction represented by each call;
        it must preserve the selected operand/configuration mapping. Only the
        schedule's explicit drains invoke ``complete_epoch``. That callback must
        emit a real full device completion, never a delay or an issue ticket.
        Both callbacks and target execution premises remain caller obligations.
        This entry neither chooses a schedule nor removes completion boundaries.
        """
        from merlin.compiler.schedule.llvm_program import emit_index_effect_program

        self.revalidate()
        receipt = emit_index_effect_program(
            self.kernel, control, lower_effect, complete_epoch=complete_epoch, completion_instruction=DRAIN
        )
        self.revalidate()
        return receipt

    def materialize_scf(self):
        from merlin.compiler.schedule.scf_commands import materialize_scf_commands

        self.revalidate()
        result = materialize_scf_commands(self.kernel)
        self.revalidate()
        return result


@admission_point(
    "merlin.compiler.schedule.dense_panel_pipeline.schedule_dense_panel_pipeline",
    fallback_route="caller-selected original route; this consumer emits nothing on refusal",
)
def schedule_dense_panel_pipeline(
    problem,
    resources,
    storage,
    adapter,
    admission,
    *,
    name,
    dtypes,
    schedule_plan,
    source_identity,
    external_buffers_disjoint,
    static_budget=128,
    timeout_ms=5000,
    output_readout=None,
    bias_source=None,
    readout_qualification_identity=None,
):
    """Consume typed block/layout/pipeline/loop decisions; check actual effects.

    Source identity/context binding confers no numerical or execution capability.
    The original source-aware caller and independently qualified admission retain
    those obligations. Every explicit decision executes or refuses; there is no
    alternate block/serialized fallback on verification failure.
    """
    from merlin.compiler.schedule.library_sources import shared_library_sources
    from merlin.compiler.schedule.loop_schedule import propose_loop_transforms
    from merlin.compiler.schedule.schedule_ir import check_dense_schedule_commands, read_dense_schedule_plan

    if external_buffers_disjoint is not True:
        raise ValueError("panel overlap requires original runtime external-buffer disjointness")
    if output_readout is not None:
        if (
            type(output_readout) not in (DenseIntegerReadout, DenseTiledIntegerReadout)
            or type(readout_qualification_identity) is not str
            or not readout_qualification_identity.strip()
        ):
            raise ValueError("panel readout requires original independent execution qualification")
    elif readout_qualification_identity is not None:
        raise ValueError("readout qualification requires an original typed output readout")
    decision = read_dense_schedule_plan(
        schedule_plan,
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
    if decision.pipeline_prefetch is None:
        raise ValueError("panel pipeline service requires an explicit schedule.panel_pipeline decision")
    if decision.pipeline_kind == "retained_lhs_segmented_k":
        raise ValueError("segmented decisions require the separate combined-capability scheduling service")
    sources = tuple(sorted(shared_library_sources(version=26).items()))
    retained = decision.pipeline_kind == "retained_full_k"
    kernel = (
        emit_compact_dense(
            problem,
            decision.block,
            decision.storage,
            adapter,
            name=name,
            dtypes=dtypes,
            outer_order=decision.outer_order,
            output_readout=output_readout,
            bias_source=bias_source,
        )
        if retained
        else build_dense_panel_pipeline(
            problem,
            decision.block,
            resources,
            decision.storage,
            adapter,
            name=name,
            dtypes=dtypes,
            prefetch=decision.pipeline_prefetch,
            outer_order=decision.outer_order,
            issue_order=decision.pipeline_issue_order,
            window_offsets=decision.pipeline_window_offsets,
            output_readout=output_readout,
            bias_source=bias_source,
        )
    )
    if decision.loop_transforms:
        kernel = propose_loop_transforms(kernel, decision.loop_transforms, index_bits=adapter.index_bits)
    if retained:
        kernel = complete_dense_retained_reduction(
            kernel, problem, decision.block, resources, decision.storage, adapter
        )
    check_dense_schedule_commands(decision, sum(1 for _ in _calls(kernel.body)), static_budget)
    verifier = verify_dense_retained_reduction if retained else verify_dense_panel_pipeline
    report = verifier(
        kernel,
        problem,
        decision.block,
        resources,
        decision.storage,
        adapter,
        admission,
        static_budget=static_budget,
        timeout_ms=timeout_ms,
        output_readout=output_readout,
        bias_source=bias_source,
    )
    report["schedule_ir"] = {
        "sha256": decision.plan_sha256,
        "text": decision.plan_text,
        "source_identity": source_identity,
        "pipeline_issue_order": decision.pipeline_issue_order,
        "pipeline_kind": decision.pipeline_kind,
        "prefetch": decision.pipeline_prefetch,
        "pipeline_window_offsets": decision.pipeline_window_offsets,
        "loop_transforms": [asdict(request) for request in decision.loop_transforms],
        "decisions_consumed": ("block_tiles", "outer_order", "row_bases", "panel_pipeline")
        + (("pipeline_window_offsets",) if decision.pipeline_window_offsets is not None else ())
        + (("loop_transforms",) if decision.loop_transforms else ()),
    }
    result = DensePanelPipelineResult(
        kernel,
        report,
        source_identity,
        admission,
        kernel.digest(),
        _digest(asdict(admission)),
        sources,
        output_readout=output_readout,
        bias_source=bias_source,
        readout_qualification_identity=readout_qualification_identity,
        checked_boundary_sha256=_digest(
            (
                None if output_readout is None else asdict(output_readout),
                None if bias_source is None else asdict(bias_source),
                readout_qualification_identity,
            )
        ),
    )
    result.revalidate()
    return result


def schedule_source_dense_panel(
    module,
    operation,
    *,
    selected_contracts,
    storage,
    adapter,
    admission,
    name,
    schedule_plan,
    resources=None,
    reservations=(),
    external_buffers_disjoint,
    static_budget=128,
    timeout_ms=5000,
    output_operation=None,
    output_operations=(),
    readout_qualification_identity=None,
):
    """Original typed source entry; no model/capsule identity dispatch."""
    from xdsl.dialects.builtin import ModuleOp

    from merlin.compiler.schedule.dense_source import (
        read_source_dense,
        read_source_dense_readout,
        read_source_dense_readout_chain,
    )
    from merlin.transforms.certificates import ir_digest

    if type(module) is not ModuleOp or not any(op is operation for op in module.walk()):
        raise ValueError("panel source must be an original typed operation inside its current module")
    module.verify()
    selected_contracts.revalidate()
    identity = ir_digest(module)
    problem = read_source_dense(operation, selected_contracts=selected_contracts)
    if type(output_operations) is not tuple or output_operations and output_operation is not None:
        raise ValueError("select one original readout or an explicit ordered readout chain")
    outputs = output_operations or (() if output_operation is None else (output_operation,))
    if any(not any(op is member for op in module.walk()) for member in outputs):
        raise ValueError("panel readout must belong to the current original source module")
    readout = None
    if output_operations:
        readout = read_source_dense_readout_chain(
            operation,
            output_operations,
            selected_contracts=selected_contracts,
            qualification_identity=readout_qualification_identity,
        )
    elif output_operation is not None:
        readout = read_source_dense_readout(
            operation,
            output_operation,
            selected_contracts=selected_contracts,
            qualification_identity=readout_qualification_identity,
        )
    elif readout_qualification_identity is not None:
        raise ValueError("panel readout qualification requires an original output operation")
    resources, bound = selected_contracts.dense.bind(
        problem,
        storage,
        adapter,
        dtypes=selected_contracts.dense.dtypes,
        hardware=selected_contracts.hardware_identity,
        policy=selected_contracts.numerical_policy,
        resources=resources,
        reservations=reservations,
        output_readout=readout,
    )
    result = schedule_dense_panel_pipeline(
        problem,
        resources,
        storage,
        bound,
        admission,
        name=name,
        dtypes=selected_contracts.dense.dtypes,
        schedule_plan=schedule_plan,
        source_identity=identity,
        external_buffers_disjoint=external_buffers_disjoint,
        static_budget=static_budget,
        timeout_ms=timeout_ms,
        output_readout=readout,
        readout_qualification_identity=readout_qualification_identity,
    )
    result = replace(result, source_module=module, selected_contracts=selected_contracts)
    result.revalidate()
    return result


@dataclass(frozen=True)
class DenseMPanelAttempt:
    """A checked current schedule or an explicit construction refusal per shape."""

    result: DensePanelPipelineResult | None
    feedback: dict


@admission_point(
    "merlin.compiler.schedule.dense_panel_pipeline.schedule_dense_m_panels",
    fallback_route="caller-selected original route; this consumer emits nothing on refusal",
)
def schedule_dense_m_panels(
    problem,
    resources,
    storage,
    adapter,
    admission,
    *,
    n_tiles,
    name,
    dtypes,
    source_identity,
    external_buffers_disjoint,
    m_tiles=None,
    static_budget=128,
    timeout_ms=5000,
    output_readout=None,
    bias_source=None,
    readout_qualification_identity=None,
):
    """Construct one counted invocation with RHS staging outside the M panels.

    An omitted M choice proposes the largest legal panel for the selected N
    width. It is a capacity choice, not a measured winner. The real scheduling
    dialect carries blocking, order and invariant hoists into the existing
    complete source/storage/epoch checker. It never concatenates thin kernels.
    A declined choice returns feedback before any alternative executes; errors
    in a concrete source or target verification propagate normally.
    """
    from merlin.compiler.schedule.compact_dense import emit_compact_dense
    from merlin.compiler.schedule.schedule_ir import make_dense_schedule_plan

    capacity = dense_m_panel_capacity(problem, resources, storage, n_tiles=n_tiles)
    feedback = dict(
        shape=dict(m=problem.m, k=problem.k, n=problem.n),
        capacity=capacity,
        mechanism=capacity["mechanism"],
        status="not_applied",
        emitted="not_observed",
        numerical_status="not_run",
        transfer_status="not_established",
    )
    if m_tiles is not None and (type(m_tiles) is not int or m_tiles < 1):
        raise ValueError("explicit M panel must be a positive original tile extent")
    blockers = list(capacity["limiting_conditions"]) if not capacity["maximum_m_tiles"] else []
    if admission is None:
        blockers.append("selected_read_only_operand_and_ordered_reduction_law")
    elif type(admission) is not DenseRetainedReductionAdmission:
        raise ValueError("M panels require the original typed retained-reduction admission")
    if external_buffers_disjoint is not True:
        blockers.append("original_external_buffer_disjointness")
    if blockers:
        return DenseMPanelAttempt(None, dict(feedback, extension_blockers=blockers))
    issue_limit = min(
        admission.max_issued_compute_pairs // (problem.tiles[1] * n_tiles),
        admission.max_issued_loads // problem.tiles[1] - n_tiles,
    )
    feedback["selected_issue_m_tile_limit"] = max(0, issue_limit)
    if issue_limit < 1:
        return DenseMPanelAttempt(None, dict(feedback, extension_blockers=["selected_issue_limit"]))
    bounded = replace(resources, max_m_blocks=min(capacity["maximum_m_tiles"], issue_limit))
    choices = [
        choice
        for choice in dense_retained_choices(problem, bounded, storage, roles=("weight",), n_choices=(n_tiles,))
        if choice.block.n == n_tiles
        and choice.block.m <= issue_limit
        and (m_tiles is None or choice.block.m == m_tiles)
    ]
    if not choices:
        return DenseMPanelAttempt(
            None,
            dict(
                feedback,
                extension_blockers=["requested_m_panel_exceeds_capacity_original_tail_domain_or_selected_issue_limit"],
            ),
        )
    choice = max(choices, key=lambda item: item.block.m)
    block = choice.block
    raw = emit_compact_dense(
        problem,
        block,
        storage,
        adapter,
        name=name,
        dtypes=dtypes,
        outer_order=choice.outer_order,
        output_readout=output_readout,
        bias_source=bias_source,
    )
    hoists = dense_retained_operand_hoists(raw, adapter, role="weight")
    plan = make_dense_schedule_plan(
        problem,
        resources,
        storage,
        adapter,
        source_identity=source_identity,
        dtypes=dtypes,
        static_budget=static_budget,
        block_tiles=(block.m, block.k, block.n),
        outer_order=choice.outer_order,
        loop_transforms=hoists,
        pipeline_kind="retained_full_k",
        pipeline_prefetch=False,
        output_readout=output_readout,
        bias_source=bias_source,
    )
    result = schedule_dense_panel_pipeline(
        problem,
        resources,
        storage,
        adapter,
        admission,
        name=name,
        dtypes=dtypes,
        schedule_plan=plan,
        source_identity=source_identity,
        external_buffers_disjoint=True,
        static_budget=static_budget,
        timeout_ms=timeout_ms,
        output_readout=output_readout,
        bias_source=bias_source,
        readout_qualification_identity=readout_qualification_identity,
    )
    actual = result.realization["source"]
    if actual["rhs_loads"] != choice.rhs_loads or actual["lhs_loads"] != choice.lhs_loads:
        raise ValueError("M-panel RHS reuse did not survive actual source and operand checking")
    feedback.update(
        status="checked_schedule_constructed",
        selected_m_tiles=block.m,
        selected_m_rows=block.m * problem.tile_m,
        counted_m_panels=(problem.tiles[0] + block.m - 1) // block.m,
        rhs_loads=actual["rhs_loads"],
        lhs_loads=actual["lhs_loads"],
        schedule_sha256=result.realization["schedule_ir"]["sha256"],
        checked_kernel_sha256=result.checked_kernel_sha256,
        extension_blockers=(
            capacity["limiting_conditions"]
            if block.m == capacity["maximum_m_tiles"]
            else ["selected_issue_limit"]
            if block.m == issue_limit
            else ["original_tail_emitter_domain"]
        ),
        tail_scope="full groups share RHS; separate partial M families retain their checked reloads",
    )
    return DenseMPanelAttempt(result, feedback)
