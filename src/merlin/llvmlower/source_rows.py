"""Select checked source rows during ordinary model preparation.

The shared compiler discovers and rewrites supported source islands. Explicit
backend bindings implement their primitives and ranked borrowed ABI, including
the declared input, workspace and completion premises. This transport neither
qualifies those bindings nor interprets opaque backend output.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from merlin.compile.scheduling.row_softmax import RowSoftmaxAdapter, RowSoftmaxSourceCompilation
    from merlin.compile.scheduling.selected_contracts import SelectedSchedulingContracts
    from merlin.runtime.host_provider import HostProviderBuild, HostProviderContext

    from .fresh_tensor_writer import FreshTensorWriterContract


@dataclass(frozen=True)
class SourceRowLowering:
    """Selected capabilities and implementation bindings, never source selection.

    ``prepare`` receives all checked bindings and an invocation-private directory;
    ``build`` receives those bindings and the existing HostProviderContext. The
    latter returns a HostProviderBuild subject to the ordinary object/link gates.
    Finite inputs must be established at each island's actual input, for example
    by a qualified runtime guard. Finite model inputs alone are insufficient.
    """

    selected_contracts: SelectedSchedulingContracts
    adapter: RowSoftmaxAdapter
    numerical_policy: str
    execution_domain: str
    allocation_alignment: int
    prepare: Callable[[tuple[SourceRowBinding, ...], Path], None]
    build: Callable[[tuple[SourceRowBinding, ...], HostProviderContext], HostProviderBuild]
    assume_finite_inputs: bool = False
    private_ownership: bool = False
    disjoint_arguments: bool = False
    chunk_elements: int | None = None
    max_source_elements: int = 64 << 20
    max_islands: int = 64
    timeout_ms: int = 2000

    def __post_init__(self):
        from merlin.compile.scheduling.row_softmax import RowSoftmaxAdapter
        from merlin.compile.scheduling.selected_contracts import SelectedSchedulingContracts

        alignment = self.allocation_alignment
        if (
            type(self.selected_contracts) is not SelectedSchedulingContracts
            or type(self.adapter) is not RowSoftmaxAdapter
            or not isinstance(self.numerical_policy, str)
            or not self.numerical_policy
            or self.execution_domain not in ("host", "device")
            or type(alignment) is not int
            or not 0 < alignment < 1 << 63
            or alignment & (alignment - 1)
            or not callable(self.prepare)
            or not callable(self.build)
            or any(type(value) is not bool for value in (
                self.assume_finite_inputs, self.private_ownership, self.disjoint_arguments,
            ))
            or any(type(value) is not int or value <= 0 for value in (
                self.max_source_elements, self.max_islands, self.timeout_ms,
            ))
            or self.chunk_elements is not None and (
                type(self.chunk_elements) is not int or self.chunk_elements <= 0
            )
        ):
            raise ValueError("source row lowering requires typed capabilities, explicit premises and bounded controls")
        self.selected_contracts.revalidate()


@dataclass(frozen=True)
class SourceRowBinding:
    """One checked source computation and its fresh-result borrowed ABI."""

    compilation: RowSoftmaxSourceCompilation
    writer: FreshTensorWriterContract


def discover_rows(module, selection, validate):
    """Compile supported canonical regions on the immutable original module."""
    from merlin.compile.scheduling.row_softmax import RowSoftmaxUnavailable, compile_source_rows, read_source_rows
    from merlin.compile.scheduling.temporary_storage import StorageOptimizationUnavailable

    from .source_model import SourceRegion

    if selection.selected_contracts.row_reduction is None:
        return (), ("selected software specification has no complete-row capability",)
    regions, declined = [], []
    for operation in module.walk():
        if operation.name != "linalg.generic":
            continue
        try:
            source = read_source_rows(
                operation, numerical_policy=selection.numerical_policy,
                assume_finite_inputs=selection.assume_finite_inputs,
                max_source_elements=selection.max_source_elements,
            )
        except RowSoftmaxUnavailable:
            continue
        if len(regions) + len(declined) >= selection.max_islands:
            raise ValueError("source row discovery exceeds the selected island budget")
        validate()
        try:
            compiled = compile_source_rows(
                module, operation, selected_contracts=selection.selected_contracts,
                adapter=selection.adapter, numerical_policy=selection.numerical_policy,
                assume_finite_inputs=selection.assume_finite_inputs,
                private_ownership=selection.private_ownership, disjoint_arguments=selection.disjoint_arguments,
                chunk_elements=selection.chunk_elements, timeout_ms=selection.timeout_ms,
                max_source_elements=selection.max_source_elements,
            )
        except (RowSoftmaxUnavailable, StorageOptimizationUnavailable) as error:
            declined.append(str(error))
            continue
        regions.append(SourceRegion(compiled, source.operations, (source.source,), source.shape))
    return tuple(regions), tuple(declined)


def bind_source_row_lowering(selection, transform, builder, *, device=None, device_admission=None):
    """Compatibility entry for one selected row family."""
    from .source_model import bind_source_lowerings

    return bind_source_lowerings((), selection, transform, builder, device=device, device_admission=device_admission)


def source_row_provider_builder(builder):
    from .source_model import source_provider_builder

    return source_provider_builder(builder)


def source_row_device_work(builder):
    from .source_model import source_device_work

    return source_device_work(builder)
