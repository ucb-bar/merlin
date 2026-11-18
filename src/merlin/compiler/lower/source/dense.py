"""Discover source contractions for the existing resource-bound dense planner.

Hardware and software contracts constrain the existing planner; a backend only
supplies assigned physical storage and qualified instruction/ABI implementations.
Unsupported computations keep their source route. This interface checks source
and schedule IR, not arbitrary emitted C or machine-code semantics.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from xdsl.dialects import arith, builtin, func, tensor

from merlin.compiler.ir import mlir_query as mq
from merlin.compiler.lower.host.borrowed_buffer_calls import BorrowedBufferEffects, validate_effects
from merlin.compiler.transforms.dense.ir import recognize_signed_integer_matmul

# ---- source_dense ------------------------------------------------------------------------------

if TYPE_CHECKING:
    from merlin.analysis.optimization.evaluation.offload_profitability import OffloadCostDecision
    from merlin.compiler.lower.device.logical_borrowed_abi import LogicalBorrowedABI
    from merlin.compiler.lower.device.plan import OffloadPlanSelection
    from merlin.compiler.lower.host.writers import FreshTensorWriterContract
    from merlin.compiler.lower.source.dense_group import SourceDenseGrouping
    from merlin.compiler.schedule.dense.compact import DenseLoopAdapter, DenseLoopStorage
    from merlin.compiler.schedule.dense.source import DenseSourceResult
    from merlin.compiler.schedule.pipeline.program_limits import AddressWindow
    from merlin.compiler.schedule.pipeline.selected_contracts import SelectedSchedulingContracts
    from merlin.compiler.transforms.dense.planning import DenseResources
    from merlin.runtime.backends.baremetal.host_provider import HostProviderBuild, HostProviderContext
    from merlin.verification.host.packing import PackingAllocationEffects


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
        from merlin.compiler.lower.device.logical_borrowed_abi import LogicalBorrowedABI
        from merlin.compiler.schedule.dense.compact import DenseLoopAdapter, DenseLoopStorage, DenseMemoryView
        from merlin.compiler.schedule.pipeline.program_limits import AddressWindow
        from merlin.compiler.schedule.pipeline.selected_contracts import SelectedSchedulingContracts
        from merlin.compiler.transforms.dense.planning import DenseResources

        validate_effects(self.borrowed_effects)
        if self.lhs_grouping is not None:
            from merlin.compiler.lower.source.dense_group import SourceDenseGrouping

            if type(self.lhs_grouping) is not SourceDenseGrouping or self.borrowed_effects is None:
                raise ValueError("dense grouping requires an original typed selection and borrowed ABI effects")
            self.lhs_grouping.__post_init__()
        if self.packing_allocation_effects is not None:
            from merlin.verification.host.packing import PackingAllocationEffects

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
        from merlin.analysis.optimization.evaluation.offload_profitability import (
            compare_complete_offload_cost,
        )

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
    from merlin.common.digest import canonical_sha256
    from merlin.compiler.schedule.library_sources import shared_library_sources
    from merlin.compiler.transforms.pipeline_state import PipelineContext

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
            from merlin.compiler.lower.source.dense_group import grouping_record

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
        "merlin.compiler.lower.source.dense",
        "merlin.compiler.lower.source.model",
        "merlin.compiler.lower.device.plan",
        "merlin.compiler.lower.host.borrowed_buffer_calls",
        "merlin.compiler.lower.device.logical_borrowed_abi",
        "merlin.compiler.lower.host.writers",
        "merlin.compiler.lower.source.prepared",
        "merlin.compiler.lower.source.prepared",
        "merlin.compiler.dialects.xdsl_context",
        "merlin.compiler.partition.dispatch_program",
        "merlin.compiler.partition.global_plan",
        "merlin.compiler.partition.global_plan_emission",
        "merlin.compiler.partition.outline",
        "merlin.analysis.optimization.evaluation.offload_profitability",
        "merlin.analysis.optimization.evaluation.component_cost",
        "merlin.analysis.perf.contract.terms",
    )
    return tuple(Path(importlib.import_module(name).__file__).resolve() for name in modules)


def ordinary_dense_route_identity(stamp):
    """Identity of the unchanged ordinary dense route under this source/context."""
    from merlin.common.digest import canonical_sha256

    return canonical_sha256({"schema": "merlin.original_dense_route.v1", "stamp": asdict(stamp)})


def validate_offload_selection(selection):
    """Reopen original source/context and cost ownership without granting legality."""
    plan, comparison = selection.offload_plan, selection.offload_comparison
    if plan is None:
        if comparison is not None:
            raise ValueError("offload comparison requires its original whole-source plan")
        return
    from merlin.compiler.lower.device.plan import OffloadPlanSelection

    if type(plan) is not OffloadPlanSelection:
        raise ValueError("dense placement requires an original typed whole-program plan")
    plan.revalidate(context=offload_context(selection))
    if {alt.placement for alt in plan.device_alternatives} - {selection.selected_contracts.target} or any(
        alt.implementation != selection.adapter.qualification_identity or alt.required_capabilities or alt.metadata
        for alt in plan.device_alternatives
    ):
        raise ValueError("dense placement differs from the original endpoint or qualified implementation contract")
    if comparison is not None:
        from merlin.analysis.optimization.evaluation.offload_profitability import OffloadCostDecision

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
    from merlin.compiler.lower.source.model import SourceRegion
    from merlin.compiler.schedule.dense.contracts import DenseSchedulingUnavailable
    from merlin.compiler.schedule.dense.source import (
        DenseSourceUnavailable,
        _problem,
        read_source_dense_readout,
        schedule_source_dense,
    )

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
        from merlin.compiler.lower.source.dense_group import discover_dense_groups

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
            from merlin.compiler.lower.device.logical_borrowed_abi import dense_buffers, scratch_bytes

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


# ---- source_fma_batch --------------------------------------------------------------------------
# Explicit scheduling permission for independent source FMA chains.
#
# The selected implementation keeps each lane's multiply, addend and single
# rounding. It may schedule independent lanes together under the supplied source
# effects contract. CPU instructions and register constraints belong to explicitly
# selected host implementations, shared in Merlin independently of accelerators.


@dataclass(frozen=True)
class SourceFmaBatchContract:
    lanes: int
    fused_single_rounding: bool
    operand_order_preserved: bool
    finite_operands_and_results: bool
    private_disjoint_storage: bool
    stable_rounding: bool
    gradual_underflow: bool
    nontrapping: bool
    exception_flags_unobserved: bool
    errno_unobserved: bool

    def validate(self) -> None:
        if type(self.lanes) is not int or self.lanes != 8:
            raise ValueError("explicit eight independent source FMA lanes required")
        if any(type(value) is not bool or not value for key, value in self.__dict__.items() if key != "lanes"):
            raise ValueError("complete source arithmetic and effects contract required")


def emit_source_fma_batch_permission(contract: SourceFmaBatchContract) -> str:
    """Select an implementation for separately proved finite private chains."""
    if not isinstance(contract, SourceFmaBatchContract):
        raise ValueError("typed independent source FMA batch contract required")
    contract.validate()
    return (
        "#define MERLIN_ENABLE_SOURCE_FMA_BATCH_8 1\n"
        "#if !defined(MERLIN_SOURCE_F32_FMA_EIGHT)\n"
        '#error "explicit independent source FMA batch provider required"\n'
        "#endif\n"
    )


# ---- direct_source_contraction -----------------------------------------------------------------
# Exact single signed-dot source binding without a compiler global plan.
#
# This establishes typed source/positional pointer names only. Actual execution,
# complete products, host writes and readout still require independent checks.


def bind_direct_source_contraction(source: Path, command_buffer: dict, proof: dict) -> dict:
    source = Path(source)
    if source.is_symlink() or not source.is_file():
        raise ValueError("direct source binding needs its exact ordinary source")
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != proof.get("interface_sha256"):
        raise ValueError("direct source binding differs from its current typed proof")
    module = mq.parse(raw.decode())
    module.verify()
    functions = list(mq.walk(module, "func.func"))
    if len(functions) != 1 or len(functions[0].regions[0].blocks) != 1 or list(module.body.block.ops) != functions:
        raise ValueError("direct source binding requires one closed source function")
    block = functions[0].regions[0].block
    ops = list(block.ops)
    if (
        len(block.args) != 2
        or len(ops) != 4
        or not isinstance(ops[0], arith.ConstantOp)
        or not isinstance(ops[1], tensor.SplatOp)
        or not isinstance(ops[3], func.ReturnOp)
    ):
        raise ValueError("direct source binding supports only zero constant, splat, one signed dot and return")
    constant, splat, dot, returned = ops
    if set(constant.properties) != {"value"} or splat.properties or returned.properties:
        raise ValueError("direct source binding has unknown source properties")
    clean = dot.clone()
    for nested in clean.walk():
        for name in tuple(nested.attributes):
            if name.startswith("prov."):
                del nested.attributes[name]
    recognize_signed_integer_matmul(clean, ("i8", "i8", "i32"))
    if (
        not isinstance(constant.value, builtin.IntegerAttr)
        or constant.value.value.data != 0
        or constant.value.type != builtin.i32
        or tuple(splat.operands) != (constant.results[0],)
        or tuple(dot.inputs) != tuple(block.args)
        or tuple(dot.outputs) != (splat.results[0],)
        or tuple(returned.operands) != (dot.results[0],)
    ):
        raise ValueError("direct source binding changes signed operands, zero initialization or complete return")
    for op in (constant, splat, returned):
        if any(not name.startswith("prov.") for name in op.attributes):
            raise ValueError("direct source binding has unknown source effects")
    record = next((row for row in proof.get("operations", []) if row.get("ordinal") == 2), {})
    shapes = [tuple(value.type.get_shape()) for value in (*block.args, dot.results[0])]
    if any(len(shape) != 2 for shape in shapes):
        raise ValueError("direct source binding requires rank-two dense tensors")
    (m, k), (rhs_k, n), out = shapes
    if (
        min(m, k, n) <= 0
        or rhs_k != k
        or out != (m, n)
        or record.get("kind") != "multiply_accumulate"
        or record.get("initial_interval") != [0, 0]
        or record.get("reduction_extent") != k
        or record.get("result_bits") != 32
    ):
        raise ValueError("direct source proof does not establish the complete signed zero-initialized reduction")
    declaration = command_buffer.get("compiler_pointer_abi")
    tensors = command_buffer.get("tensors")
    if (
        type(declaration) is not dict
        or set(declaration) != {"version", "arguments"}
        or type(declaration["version"]) is not int
        or declaration["version"] != 1
        or type(declaration["arguments"]) is not list
        or len(declaration["arguments"]) != 3
        or any(type(name) is not str or not name for name in declaration["arguments"])
        or len(set(declaration["arguments"])) != 3
        or type(tensors) is not dict
        or set(tensors) != set(declaration["arguments"])
    ):
        raise ValueError("direct source binding has no complete original three-pointer ABI")
    for name, shape, dtype, role in zip(
        declaration["arguments"], shapes, ("i8", "i8", "i32"), ("input", "input", "output"), strict=True
    ):
        row = tensors[name]
        if (
            type(row) is not dict
            or row.get("shape") != list(shape)
            or row.get("dtype") != dtype
            or row.get("role") != role
            or set(row) - {"shape", "dtype", "role", "layout"}
            or row.get("layout", "dense_row_major") not in {"row_major", "dense_row_major"}
        ):
            raise ValueError("direct source logical pointer storage differs from typed source")
    lhs, rhs, output = declaration["arguments"]
    return {"ordinal": 2, "lhs": lhs, "rhs": rhs, "output": output, "m": m, "k": k, "n": n}
