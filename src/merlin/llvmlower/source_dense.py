"""Discover source contractions for the existing resource-bound dense planner.

Hardware and software contracts constrain the existing planner; a backend only
supplies assigned physical storage and qualified instruction/ABI implementations.
Unsupported computations keep their source route. This interface checks source
and schedule IR, not arbitrary emitted C or machine-code semantics.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from .borrowed_buffer_calls import BorrowedBufferEffects, validate_effects

if TYPE_CHECKING:
    from merlin.compile.scheduling.compact_dense import DenseLoopAdapter, DenseLoopStorage
    from merlin.compile.scheduling.dense_source import DenseSourceResult
    from merlin.compile.scheduling.program_limits import AddressWindow
    from merlin.compile.scheduling.selected_contracts import SelectedSchedulingContracts
    from merlin.perf.offload_profitability import OffloadCostDecision
    from merlin.runtime.host_provider import HostProviderBuild, HostProviderContext
    from merlin.transforms.dense_planning import DenseResources
    from merlin.verify.host_packing_storage import PackingAllocationEffects

    from .fresh_tensor_writer import FreshTensorWriterContract
    from .logical_borrowed_abi import LogicalBorrowedABI
    from .offload_plan import OffloadPlanSelection
    from .source_dense_group import SourceDenseGrouping


@dataclass(frozen=True)
class SourceDenseLowering:
    """Fixed selected resources and implementation bindings for one invocation.

    Storage is an allocation output, not permission to occupy an entire memory.
    Optional resources restrict the existing search; selected contracts clamp
    them to physical limits. Runtime ownership and disjointness are independent
    premises, never inferred from distinct tensor SSA values.
    """

    selected_contracts: SelectedSchedulingContracts
    storage: DenseLoopStorage
    adapter: DenseLoopAdapter
    execution_domain: str
    allocation_alignment: int
    static_budget: int
    prepare: Callable[[tuple[SourceDenseBinding, ...], Path], None]
    build: Callable[[tuple[SourceDenseBinding, ...], HostProviderContext], HostProviderBuild]
    private_ownership: bool = False
    external_buffers_disjoint: bool = False
    resources: DenseResources | None = None
    reservations: tuple[AddressWindow, ...] = ()
    outer_order: tuple[str, str] = ("m", "n")
    max_source_elements: int = 64 << 20
    max_islands: int = 64
    verification_timeout_ms: int = 2000
    max_repairs: int = 1
    borrowed_effects: BorrowedBufferEffects | None = None
    logical_abi: LogicalBorrowedABI | None = None
    borrow_contiguous: bool = True
    fuse_output_readout: bool = True
    readout_qualification_identity: str | None = None
    max_convolution_packing_bytes: int | None = None
    convolution_route: str = "auto"
    max_convolution_candidates: int = 4096
    max_convolution_added_operations: int = 4096
    #: Original heap semantics for final packing-storage checks; an ID alone is insufficient.
    packing_allocation_effects: PackingAllocationEffects | None = None
    #: Original whole-program choice; actual source/resource admission still runs.
    offload_plan: OffloadPlanSelection | None = None
    #: Optional original complete-cost comparison; unknown costs retain the original route.
    offload_comparison: OffloadCostDecision | None = None
    #: Original qualified grouping implementation, absent for the singleton route.
    lhs_grouping: SourceDenseGrouping | None = None

    def __post_init__(self):
        from merlin.compile.scheduling.compact_dense import DenseLoopAdapter, DenseLoopStorage, DenseMemoryView
        from merlin.compile.scheduling.program_limits import AddressWindow
        from merlin.compile.scheduling.selected_contracts import SelectedSchedulingContracts
        from merlin.transforms.dense_planning import DenseResources

        from .logical_borrowed_abi import LogicalBorrowedABI

        validate_effects(self.borrowed_effects)
        if self.lhs_grouping is not None:
            from .source_dense_group import SourceDenseGrouping

            if type(self.lhs_grouping) is not SourceDenseGrouping or self.borrowed_effects is None:
                raise ValueError("dense grouping requires an original typed selection and borrowed ABI effects")
            self.lhs_grouping.__post_init__()
        if self.packing_allocation_effects is not None:
            from merlin.verify.host_packing_storage import PackingAllocationEffects

            if type(self.packing_allocation_effects) is not PackingAllocationEffects:
                raise ValueError("packing storage requires original typed allocation effects")
            self.packing_allocation_effects.__post_init__()
        if (
            type(self.convolution_route) is not str
            or self.convolution_route not in ("auto", "full", "panel")
            or self.convolution_route == "panel"
            and self.max_convolution_packing_bytes is None
            or self.max_convolution_packing_bytes is not None
            and (
                type(self.max_convolution_packing_bytes) is not int
                or not 0 < self.max_convolution_packing_bytes < 1 << 63
            )
            or any(
                type(value) is not int or not 0 < value <= 1 << 20
                for value in (
                    self.max_convolution_candidates,
                    self.max_convolution_added_operations,
                )
            )
        ):
            raise ValueError("convolution routes require explicit bounded packing payload and search/growth controls")
        if type(self.borrow_contiguous) is not bool:
            raise ValueError("logical buffer borrowing requires an explicit Boolean control")
        if type(self.fuse_output_readout) is not bool or (
            self.readout_qualification_identity is not None
            and (
                type(self.readout_qualification_identity) is not str
                or not self.readout_qualification_identity.strip()
                or len(self.readout_qualification_identity) > 4096
            )
        ):
            raise ValueError("dense readout requires a Boolean selection and independent qualification identity")
        if self.logical_abi is not None:
            if (
                type(self.logical_abi) is not LogicalBorrowedABI
                or type(self.selected_contracts) is not SelectedSchedulingContracts
                or self.selected_contracts.dense is None
                or self.logical_abi.declaration != self.selected_contracts.dense.logical_pointer_abi
                or self.logical_abi.qualified_effects != self.borrowed_effects
            ):
                raise ValueError("logical pointer ABI must match original software and qualified borrowed effects")
            self.logical_abi.__post_init__()
        alignment = self.allocation_alignment
        if (
            type(self.selected_contracts) is not SelectedSchedulingContracts
            or type(self.storage) is not DenseLoopStorage
            or type(self.adapter) is not DenseLoopAdapter
            or self.execution_domain not in ("host", "device")
            or type(alignment) is not int
            or not 0 < alignment < 1 << 63
            or alignment & (alignment - 1)
            or type(self.static_budget) is not int
            or self.static_budget < 0
            or any(not callable(f) for f in (self.prepare, self.build))
            or any(type(v) is not bool for v in (self.private_ownership, self.external_buffers_disjoint))
            or self.resources is not None
            and type(self.resources) is not DenseResources
            or type(self.reservations) is not tuple
            or any(type(r) is not AddressWindow for r in self.reservations)
            or type(self.outer_order) is not tuple
            or self.outer_order not in (("m", "n"), ("n", "m"))
            or any(
                type(v) is not int or v <= 0
                for v in (
                    self.max_source_elements,
                    self.max_islands,
                    self.verification_timeout_ms,
                )
            )
            or type(self.max_repairs) is not int
            or self.max_repairs not in (0, 1)
        ):
            raise ValueError("source dense lowering requires fixed typed capabilities and bounded controls")
        if any(
            type(v) is not DenseMemoryView
            for v in (
                self.storage.lhs,
                self.storage.rhs,
                *((self.storage.accumulator,) if self.storage.accumulator is not None else ()),
            )
        ):
            raise ValueError("source dense storage requires explicit physical memory views")
        self.selected_contracts.revalidate()
        validate_offload_selection(self)

    def with_complete_offload_cost(
        self,
        *,
        original,
        candidate,
        original_costs,
        candidate_costs,
        scope,
        regime,
        verify_original,
        max_regions=4096,
        max_evidence_bytes=4 << 20,
    ):
        """Bind original complete-cost evidence to this exact source plan.

        The original verifier still owns actual artifact, runtime, timer,
        input/state and qualification joins. Route identities are derived here
        so an observation cannot accidentally name a different candidate.
        """
        from merlin.perf.offload_profitability import compare_complete_offload_cost

        validate_offload_selection(self)
        if self.offload_plan is None:
            raise ValueError("complete offload costs require the original current source plan")
        plan = self.offload_plan
        comparison = compare_complete_offload_cost(
            original=original,
            candidate=candidate,
            original_costs=original_costs,
            candidate_costs=candidate_costs,
            scope=scope,
            regime=regime,
            stamp=plan.stamp,
            original_selection_sha256=ordinary_dense_route_identity(plan.stamp),
            candidate_selection_sha256=plan.plan.digest,
            verify_original=verify_original,
            max_regions=max_regions,
            max_evidence_bytes=max_evidence_bytes,
        )
        return replace(self, offload_comparison=comparison)


def offload_context(selection):
    """Current whole-source resources, effects and implementation identities.

    This scope includes the logical ABI and packing controls as well as dense
    scheduling. Callable identities cannot establish implementation equivalence;
    the original qualifier and actual provider/product checks remain mandatory.
    """
    from merlin.common.jsonio import canonical_sha256
    from merlin.compile.scheduling.library_sources import shared_library_sources
    from merlin.transforms.pipeline_state import PipelineContext

    if type(selection) is not SourceDenseLowering:
        raise ValueError("offload context requires the original typed dense selection")
    contracts, adapter = selection.selected_contracts, selection.adapter
    contracts.revalidate()

    def data(value):
        if is_dataclass(value):
            return {field.name: data(getattr(value, field.name)) for field in fields(value)}
        if isinstance(value, (tuple, list)):
            return [data(item) for item in value]
        return value

    controls = {}
    for field in fields(selection):
        if field.name in ("prepare", "build", "selected_contracts", "adapter", "offload_plan", "offload_comparison"):
            continue
        if field.name == "lhs_grouping" and selection.lhs_grouping is not None:
            from .source_dense_group import grouping_record

            controls[field.name] = grouping_record(selection)
            continue
        controls[field.name] = data(getattr(selection, field.name))
    execution = {
        "contracts": contracts.record(),
        "adapter": {
            field.name: getattr(adapter, field.name)
            for field in fields(adapter)
            if field.name not in ("encode", "project")
        },
    }
    from merlin.common.digest import sha256_file

    return PipelineContext(
        hardware=contracts.hardware_identity,
        numerical_policy=contracts.numerical_policy,
        resources=canonical_sha256(controls),
        execution=canonical_sha256(execution),
        library=canonical_sha256(
            {
                "shared": shared_library_sources(),
                "selection": {path.name: sha256_file(path) for path in offload_implementation_paths()},
            }
        ),
    )


def offload_implementation_paths():
    """Explicit optional source-binding/cost checker owners; no access grant."""
    import importlib

    modules = (
        "merlin.llvmlower.source_dense",
        "merlin.llvmlower.source_model",
        "merlin.llvmlower.offload_plan",
        "merlin.llvmlower.borrowed_buffer_calls",
        "merlin.llvmlower.logical_borrowed_abi",
        "merlin.llvmlower.fresh_tensor_writer",
        "merlin.llvmlower.prepared_operand_effects",
        "merlin.llvmlower.tensor_preparation_identity",
        "merlin.xdsl_dialects._common",
        "merlin.xdsl_dialects.lowering.dispatch_program",
        "merlin.xdsl_dialects.lowering.global_plan",
        "merlin.xdsl_dialects.lowering.global_plan_emission",
        "merlin.xdsl_dialects.lowering.outline",
        "merlin.perf.offload_profitability",
        "merlin.perf.component_cost",
        "merlin.perf.term",
    )
    return tuple(Path(importlib.import_module(name).__file__).resolve() for name in modules)


def ordinary_dense_route_identity(stamp):
    """Identity of the unchanged ordinary dense route under this source/context."""
    from merlin.common.jsonio import canonical_sha256

    return canonical_sha256({"schema": "merlin.original_dense_route.v1", "stamp": asdict(stamp)})


def validate_offload_selection(selection):
    """Reopen original source/context and cost ownership without granting legality."""
    plan, comparison = selection.offload_plan, selection.offload_comparison
    if plan is None:
        if comparison is not None:
            raise ValueError("offload comparison requires its original whole-source plan")
        return
    from .offload_plan import OffloadPlanSelection

    if type(plan) is not OffloadPlanSelection:
        raise ValueError("dense placement requires an original typed whole-program plan")
    plan.revalidate(context=offload_context(selection))
    if {alt.placement for alt in plan.device_alternatives} - {selection.selected_contracts.target} or any(
        alt.implementation != selection.adapter.qualification_identity or alt.required_capabilities or alt.metadata
        for alt in plan.device_alternatives
    ):
        raise ValueError("dense placement differs from the original endpoint or qualified implementation contract")
    if comparison is not None:
        from merlin.perf.offload_profitability import OffloadCostDecision

        if type(comparison) is not OffloadCostDecision:
            raise ValueError("dense profitability requires an original typed complete-cost comparison")
        record = comparison.record()
        if selection.offload_plan is not plan or selection.offload_comparison is not comparison:
            raise ValueError("dense profitability selection changed during original verification; replan")
        # The original verifier is a callback. Its cost query is checked by the
        # comparator, but outer source/resources must also remain current.
        plan.revalidate(context=offload_context(selection))
        if (
            record["stamp"] != asdict(plan.stamp)
            or record["original_selection_sha256"] != ordinary_dense_route_identity(plan.stamp)
            or record["candidate_selection_sha256"] != plan.plan.digest
        ):
            raise ValueError("dense profitability evidence differs from the original source or exact route selections")
        return record


@dataclass(frozen=True)
class SourceDenseBinding:
    compilation: DenseSourceResult
    writer: FreshTensorWriterContract
    logical_kernel_symbol: str | None = None
    logical_abi_source: Path | None = None
    logical_observer_symbol: str | None = None


def discover_dense(module, selection, validate):
    """Derive contractions and schedules from current IR; never ask a backend to select regions."""
    from merlin.compile.scheduling.dense_contracts import DenseSchedulingUnavailable
    from merlin.compile.scheduling.dense_source import (
        DenseSourceUnavailable,
        _problem,
        read_source_dense_readout,
        schedule_source_dense,
    )

    from .source_model import SourceRegion

    contracts, adapter = selection.selected_contracts, selection.adapter
    if contracts.dense is None:
        return (), ("selected software specification has no dense scheduling capability",)
    if not adapter.qualification_identity:
        return (), ("dense instructions lack independent execution qualification",)
    if not adapter.synchronous_completion or not adapter.exact_accumulator_continuation:
        return (), ("dense instructions lack synchronous completion or exact accumulator continuation",)
    if selection.storage.accumulator is None or any(s.capacity_bytes is None for s in contracts.dense.stores):
        return (), ("dense instructions lack assigned addressable storage or known physical capacity",)
    if not selection.private_ownership or not selection.external_buffers_disjoint:
        return (), ("dense runtime ownership and external disjointness are unavailable",)
    groups, group_declined = (), ()
    if selection.lhs_grouping is not None:
        from .source_dense_group import discover_dense_groups

        groups, group_declined = discover_dense_groups(module, selection, validate)
    regions, declined = list(groups), []
    claimed = frozenset(operation for group in groups for operation in group.operations)
    for operation in module.walk():
        if operation.name != "linalg.generic" or operation in claimed:
            continue
        try:
            # The transaction validates selected bytes before and after this
            # family and across all callbacks. Probe pure admission cheaply so
            # unrelated model operations do not start sessions or rescan files.
            problem = _problem(operation, contracts, selection.max_source_elements)
        except DenseSourceUnavailable:
            continue
        if len(regions) + len(declined) >= selection.max_islands:
            raise ValueError("source dense discovery exceeds the selected island budget")
        output_operation, readout = None, None
        if selection.fuse_output_readout and selection.readout_qualification_identity:
            uses = tuple(operation.results[0].uses)
            if len(uses) == 1:
                candidate = uses[0].operation
                try:
                    readout = read_source_dense_readout(
                        operation,
                        candidate,
                        selected_contracts=contracts,
                        qualification_identity=selection.readout_qualification_identity,
                        max_source_elements=selection.max_source_elements,
                    )
                    output_operation = candidate
                except DenseSourceUnavailable:
                    pass
            if selection.logical_abi is not None and (
                selection.logical_abi.readout_format_qualification_identity != selection.readout_qualification_identity
            ):
                output_operation, readout = None, None
        if selection.logical_abi is not None:
            from .logical_borrowed_abi import dense_buffers, scratch_bytes

            try:
                scratch_bytes(
                    dense_buffers(
                        problem,
                        contracts.dense.dtypes,
                        output_dtype=readout.output_dtype if readout is not None else None,
                    ),
                    selection.logical_abi,
                )
            except ValueError as error:
                declined.append(str(error))
                continue
        validate()
        options = dict(
            selected_contracts=contracts,
            storage=selection.storage,
            adapter=adapter,
            name="source_dense_" + str(len(regions)),
            static_budget=selection.static_budget,
            external_buffers_disjoint=selection.external_buffers_disjoint,
            resources=selection.resources,
            reservations=selection.reservations,
            outer_order=selection.outer_order,
            max_source_elements=selection.max_source_elements,
            verification_timeout_ms=selection.verification_timeout_ms,
            max_repairs=selection.max_repairs,
        )
        result = None
        prior_declines = len(declined)
        for candidate in (output_operation, None) if output_operation is not None else (None,):
            # A selected output path may fail allocation or code admission.
            # Retry the original dense result once; its CPU consumer survives.
            if candidate is None and selection.logical_abi is not None:
                try:
                    scratch_bytes(dense_buffers(problem, contracts.dense.dtypes), selection.logical_abi)
                except ValueError as error:
                    declined.append(str(error))
                    break
            try:
                result = schedule_source_dense(
                    module,
                    operation,
                    **options,
                    output_operation=candidate,
                    readout_qualification_identity=(
                        selection.readout_qualification_identity if candidate is not None else None
                    ),
                )
            except DenseSchedulingUnavailable as error:
                validate()
                if candidate is not None:
                    continue
                declined.append(str(error))
                break
            validate()
            if result is not None:
                break
        validate()
        if result is None:
            if len(declined) == prior_declines:
                declined.append("no legal dense blocking fits the selected physical windows and code budget")
            continue
        regions.append(SourceRegion(result, result.operations, tuple(operation.inputs), (problem.m, problem.n)))
    return tuple(regions), (*group_declined, *declined)
