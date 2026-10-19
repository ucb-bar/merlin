"""Ordinary model source selection from declared integer scalar semantics.

The software contract supplies eligible arithmetic; target bindings map immutable
typed computations to original instructions and physical buffers. The existing
pointwise compiler independently checks both that original sequence and actual
final decoded bytes. Unknown source/mappings retain the original source route.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from merlin.compile.scheduling.schedule_pipeline import ScheduleBudgets

from .borrowed_buffer_calls import BorrowedBufferEffects, validate_effects

if TYPE_CHECKING:
    from merlin.compile.scheduling.pointwise_source import PointwiseBindings, PointwiseCompilation, PointwiseSource
    from merlin.compile.scheduling.program_limits import ProgramLimits
    from merlin.compile.scheduling.schedule_pipeline import ScheduleEncoding
    from merlin.compile.scheduling.selected_contracts import SelectedSchedulingContracts
    from merlin.compile.scheduling.temporary_storage import StorageAdapter, TemporaryObject
    from merlin.runtime.host_provider import HostProviderBuild, HostProviderContext
    from merlin.sched.ir import Kernel

    from .fresh_tensor_writer import FreshTensorWriterContract


@dataclass(frozen=True)
class PointwiseProgram:
    """Target instruction selection/mapping under the fixed selected contracts."""

    kernel: Kernel
    objects: tuple[TemporaryObject, ...]
    bindings: PointwiseBindings

    def __post_init__(self):
        from merlin.compile.scheduling.pointwise_source import PointwiseBindings
        from merlin.compile.scheduling.temporary_storage import TemporaryObject
        from merlin.sched.ir import Kernel

        if (
            type(self.kernel) is not Kernel or type(self.bindings) is not PointwiseBindings
            or type(self.objects) is not tuple or any(type(o) is not TemporaryObject for o in self.objects)
        ):
            raise ValueError("pointwise program requires a typed kernel, immutable temporaries and runtime bindings")


@dataclass(frozen=True)
class SourcePointwiseLowering:
    selected_contracts: SelectedSchedulingContracts
    adapter: StorageAdapter
    encoding: ScheduleEncoding
    limits: ProgramLimits
    execution_domain: str
    allocation_alignment: int
    lower: Callable[[PointwiseSource], PointwiseProgram | None]
    prepare: Callable[[tuple[SourcePointwiseBinding, ...], Path], None]
    build: Callable[[tuple[SourcePointwiseBinding, ...], HostProviderContext], HostProviderBuild]
    private_ownership: bool = False
    external_buffers_disjoint: bool = False
    scalar_qualification_identity: str | None = None
    fusion_emitters: tuple[tuple[str, Callable], ...] = ()
    configuration_qualification_identity: str | None = None
    spill_emitters: tuple[tuple[str, Callable], ...] = ()
    spill_qualification_identity: str | None = None
    budgets: ScheduleBudgets = field(default_factory=ScheduleBudgets)
    max_source_elements: int = 64 << 20
    max_islands: int = 64
    borrowed_effects: BorrowedBufferEffects | None = None

    def __post_init__(self):
        from merlin.compile.scheduling.program_limits import ProgramLimits
        from merlin.compile.scheduling.schedule_pipeline import ScheduleEncoding
        from merlin.compile.scheduling.selected_contracts import SelectedSchedulingContracts
        from merlin.compile.scheduling.temporary_storage import StorageAdapter

        validate_effects(self.borrowed_effects)
        alignment = self.allocation_alignment
        if (
            type(self.selected_contracts) is not SelectedSchedulingContracts
            or type(self.adapter) is not StorageAdapter or type(self.encoding) is not ScheduleEncoding
            or type(self.limits) is not ProgramLimits or type(self.budgets) is not ScheduleBudgets
            or self.execution_domain not in ("host", "device")
            or type(alignment) is not int or not 0 < alignment < 1 << 63 or alignment & (alignment - 1)
            or any(not callable(f) for f in (self.lower, self.prepare, self.build))
            or any(type(v) is not bool for v in (self.private_ownership, self.external_buffers_disjoint))
            or any(type(v) is not int or v <= 0 for v in (self.max_source_elements, self.max_islands))
            or any(v is not None and (type(v) is not str or not v.strip()) for v in (
                self.scalar_qualification_identity, self.configuration_qualification_identity,
                self.spill_qualification_identity,
            ))
        ):
            raise ValueError("source pointwise lowering requires fixed typed capabilities and bounded controls")
        for emitters in (self.fusion_emitters, self.spill_emitters):
            if (
                type(emitters) is not tuple
                or any(type(p) is not tuple or len(p) != 2 or type(p[0]) is not str or not p[0]
                       or not callable(p[1]) for p in emitters)
                or len({p[0] for p in emitters}) != len(emitters)
            ):
                raise ValueError("pointwise emitters require immutable distinct declared selector/callback pairs")
        self.selected_contracts.revalidate()


@dataclass(frozen=True)
class SourcePointwiseBinding:
    compilation: PointwiseCompilation
    writer: FreshTensorWriterContract


def discover_pointwise(module, selection, validate):
    """Find semantics from software, while instruction mapping remains target-owned."""
    from merlin.compile.scheduling.fusion_semantics import UnsupportedElementwise, scalar_graph
    from merlin.compile.scheduling.pointwise_source import compile_source_pointwise, read_pointwise_source
    from merlin.compile.scheduling.temporary_storage import StorageOptimizationUnavailable

    from .source_model import SourceRegion

    contracts, budgets = selection.selected_contracts, selection.budgets
    if contracts.storage is None or contracts.fusion is None:
        return (), ("selected software specification has no pointwise storage/scalar capability",)
    if selection.scalar_qualification_identity is None:
        return (), ("pointwise arithmetic lacks independent execution qualification",)
    if not selection.private_ownership or not selection.external_buffers_disjoint:
        return (), ("pointwise runtime ownership and external bindings are unavailable",)
    supported = []
    for declaration in json.loads(contracts.fusion.declarations_json)["elementwise"].values():
        if declaration["execution"] != "contiguous_read_then_write":
            continue
        try:
            supported.append(scalar_graph(
                declaration["scalar_ir"].encode(), budgets.max_scalar_operations, budgets.max_scalar_ir_bytes,
            ))
        except UnsupportedElementwise:
            continue
    regions, declined = [], []
    for operation in module.walk():
        if operation.name != "linalg.generic":
            continue
        try:
            source = read_pointwise_source(
                operation, max_elements=selection.max_source_elements,
                max_operations=budgets.max_scalar_operations, max_ir_bytes=budgets.max_scalar_ir_bytes,
            )
        except UnsupportedElementwise:
            continue
        if source.graph not in supported:
            continue
        if len(regions) + len(declined) >= selection.max_islands:
            raise ValueError("source pointwise discovery exceeds the selected island budget")
        validate()
        program = selection.lower(source)
        validate()
        if program is None:
            declined.append("qualified target binding has no mapping for the current pointwise computation")
            continue
        if type(program) is not PointwiseProgram:
            raise ValueError("pointwise target mapping must return a typed PointwiseProgram or None")
        if not program.bindings.external_buffers_disjoint:
            declined.append("pointwise target mapping lacks its external buffer ownership premise")
            continue
        try:
            compiled = compile_source_pointwise(
                module, operation, program.kernel, program.objects, selection.adapter, selection.encoding,
                selection.limits, selected_contracts=contracts, bindings=program.bindings, budgets=budgets,
                private_ownership=selection.private_ownership,
                fusion_emitters=dict(selection.fusion_emitters),
                fusion_qualification_identity=selection.scalar_qualification_identity,
                configuration_qualification_identity=selection.configuration_qualification_identity,
                spill_emitters=dict(selection.spill_emitters) if selection.spill_emitters else None,
                spill_qualification_identity=selection.spill_qualification_identity,
                max_source_elements=selection.max_source_elements,
            )
        except (UnsupportedElementwise, StorageOptimizationUnavailable) as error:
            declined.append(str(error))
            continue
        regions.append(SourceRegion(compiled, (operation,), tuple(operation.inputs), source.shape))
    return tuple(regions), tuple(declined)
