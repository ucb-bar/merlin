"""Discover source contractions for the existing resource-bound dense planner.

Hardware and software contracts constrain the existing planner; a backend only
supplies assigned physical storage and qualified instruction/ABI implementations.
Unsupported computations keep their source route. This interface checks source
and schedule IR, not arbitrary emitted C or machine-code semantics.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .borrowed_buffer_calls import BorrowedBufferEffects, validate_effects

if TYPE_CHECKING:
    from merlin.compile.scheduling.compact_dense import DenseLoopAdapter, DenseLoopStorage
    from merlin.compile.scheduling.dense_source import DenseSourceResult
    from merlin.compile.scheduling.program_limits import AddressWindow
    from merlin.compile.scheduling.selected_contracts import SelectedSchedulingContracts
    from merlin.runtime.host_provider import HostProviderBuild, HostProviderContext
    from merlin.transforms.dense_planning import DenseResources

    from .fresh_tensor_writer import FreshTensorWriterContract


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

    def __post_init__(self):
        from merlin.compile.scheduling.compact_dense import DenseLoopAdapter, DenseLoopStorage, DenseMemoryView
        from merlin.compile.scheduling.program_limits import AddressWindow
        from merlin.compile.scheduling.selected_contracts import SelectedSchedulingContracts
        from merlin.transforms.dense_planning import DenseResources

        validate_effects(self.borrowed_effects)
        alignment = self.allocation_alignment
        if (
            type(self.selected_contracts) is not SelectedSchedulingContracts
            or type(self.storage) is not DenseLoopStorage or type(self.adapter) is not DenseLoopAdapter
            or self.execution_domain not in ("host", "device")
            or type(alignment) is not int or not 0 < alignment < 1 << 63 or alignment & (alignment - 1)
            or type(self.static_budget) is not int or self.static_budget < 0
            or any(not callable(f) for f in (self.prepare, self.build))
            or any(type(v) is not bool for v in (self.private_ownership, self.external_buffers_disjoint))
            or self.resources is not None and type(self.resources) is not DenseResources
            or type(self.reservations) is not tuple or any(type(r) is not AddressWindow for r in self.reservations)
            or type(self.outer_order) is not tuple or self.outer_order not in (("m", "n"), ("n", "m"))
            or any(type(v) is not int or v <= 0 for v in (
                self.max_source_elements, self.max_islands, self.verification_timeout_ms,
            ))
            or type(self.max_repairs) is not int or self.max_repairs not in (0, 1)
        ):
            raise ValueError("source dense lowering requires fixed typed capabilities and bounded controls")
        if any(type(v) is not DenseMemoryView for v in (
            self.storage.lhs, self.storage.rhs,
            *((self.storage.accumulator,) if self.storage.accumulator is not None else ()),
        )):
            raise ValueError("source dense storage requires explicit physical memory views")
        self.selected_contracts.revalidate()


@dataclass(frozen=True)
class SourceDenseBinding:
    compilation: DenseSourceResult
    writer: FreshTensorWriterContract


def discover_dense(module, selection, validate):
    """Derive contractions and schedules from current IR; never ask a backend to select regions."""
    from merlin.compile.scheduling.dense_contracts import DenseSchedulingUnavailable
    from merlin.compile.scheduling.dense_source import DenseSourceUnavailable, _problem, schedule_source_dense

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
    regions, declined = [], []
    for operation in module.walk():
        if operation.name != "linalg.generic":
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
        validate()
        try:
            result = schedule_source_dense(
                module, operation, selected_contracts=contracts, storage=selection.storage, adapter=adapter,
                name="source_dense_" + str(len(regions)), static_budget=selection.static_budget,
                external_buffers_disjoint=selection.external_buffers_disjoint, resources=selection.resources,
                reservations=selection.reservations, outer_order=selection.outer_order,
                max_source_elements=selection.max_source_elements,
                verification_timeout_ms=selection.verification_timeout_ms, max_repairs=selection.max_repairs,
            )
        except DenseSchedulingUnavailable as error:
            validate()
            declined.append(str(error))
            continue
        validate()
        if result is None:
            declined.append("no legal dense blocking fits the selected physical windows and code budget")
            continue
        regions.append(SourceRegion(result, (operation,), tuple(operation.inputs), (problem.m, problem.n)))
    return tuple(regions), tuple(declined)
