"""Ordinary model source selection from declared integer scalar semantics.

The software contract supplies eligible arithmetic; target bindings map immutable
typed computations to original instructions and physical buffers. The existing
pointwise compiler independently checks both that original sequence and actual
final decoded bytes. Unknown source/mappings retain the original source route.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import FunctionType, MethodType
from typing import TYPE_CHECKING

from merlin.common.digest import sha256_file
from merlin.common.paths import module_source_path
from merlin.compiler.lower.host.borrowed_buffer_calls import BorrowedBufferEffects, validate_effects
from merlin.compiler.schedule.pipeline.contracts import ScheduleBudgets
from merlin.compiler.transforms.certificates import ir_digest

# ---- source_pointwise --------------------------------------------------------------------------

if TYPE_CHECKING:
    from merlin.compiler.kernel_ir.ir import Kernel
    from merlin.compiler.lower.host.writers import FreshTensorWriterContract
    from merlin.compiler.schedule.elementwise.pointwise_source import (
        PointwiseBindings,
        PointwiseCompilation,
        PointwiseSource,
    )
    from merlin.compiler.schedule.pipeline.contracts import ScheduleEncoding
    from merlin.compiler.schedule.pipeline.program_limits import ProgramLimits
    from merlin.compiler.schedule.pipeline.selected_contracts import SelectedSchedulingContracts
    from merlin.compiler.schedule.storage.temporaries import StorageAdapter, TemporaryObject
    from merlin.runtime.backends.baremetal.host_provider import HostProviderBuild, HostProviderContext


@dataclass(frozen=True)
class PointwiseProgram:
    """Target instruction selection/mapping under the fixed selected contracts."""

    kernel: Kernel
    objects: tuple[TemporaryObject, ...]
    bindings: PointwiseBindings

    def __post_init__(self):
        from merlin.compiler.kernel_ir.ir import Kernel
        from merlin.compiler.schedule.elementwise.pointwise_source import PointwiseBindings
        from merlin.compiler.schedule.storage.temporaries import TemporaryObject

        if (
            type(self.kernel) is not Kernel
            or type(self.bindings) is not PointwiseBindings
            or type(self.objects) is not tuple
            or any(type(o) is not TemporaryObject for o in self.objects)
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
    prefetch_qualification_identity: str | None = None
    #: Original physical fanout composition; absent retains singleton discovery.
    multi_output: SourceMultiOutputLowering | None = None

    def __post_init__(self):
        from merlin.compiler.schedule.pipeline.contracts import ScheduleEncoding
        from merlin.compiler.schedule.pipeline.program_limits import ProgramLimits
        from merlin.compiler.schedule.pipeline.selected_contracts import SelectedSchedulingContracts
        from merlin.compiler.schedule.storage.temporaries import StorageAdapter

        validate_effects(self.borrowed_effects)
        if self.multi_output is not None:
            if type(self.multi_output) is not SourceMultiOutputLowering or self.borrowed_effects is None:
                raise ValueError("multi-output lowering requires original typed selection and borrowed effects")
            self.multi_output.__post_init__()
        alignment = self.allocation_alignment
        if (
            type(self.selected_contracts) is not SelectedSchedulingContracts
            or type(self.adapter) is not StorageAdapter
            or type(self.encoding) is not ScheduleEncoding
            or type(self.limits) is not ProgramLimits
            or type(self.budgets) is not ScheduleBudgets
            or self.execution_domain not in ("host", "device")
            or type(alignment) is not int
            or not 0 < alignment < 1 << 63
            or alignment & (alignment - 1)
            or any(not callable(f) for f in (self.lower, self.prepare, self.build))
            or any(type(v) is not bool for v in (self.private_ownership, self.external_buffers_disjoint))
            or any(type(v) is not int or v <= 0 for v in (self.max_source_elements, self.max_islands))
            or any(
                v is not None and (type(v) is not str or not v.strip())
                for v in (
                    self.scalar_qualification_identity,
                    self.configuration_qualification_identity,
                    self.spill_qualification_identity,
                    self.prefetch_qualification_identity,
                )
            )
        ):
            raise ValueError("source pointwise lowering requires fixed typed capabilities and bounded controls")
        for emitters in (self.fusion_emitters, self.spill_emitters):
            if (
                type(emitters) is not tuple
                or any(
                    type(p) is not tuple or len(p) != 2 or type(p[0]) is not str or not p[0] or not callable(p[1])
                    for p in emitters
                )
                or len({p[0] for p in emitters}) != len(emitters)
            ):
                raise ValueError("pointwise emitters require immutable distinct declared selector/callback pairs")
        self.selected_contracts.revalidate()


@dataclass(frozen=True)
class SourcePointwiseBinding:
    compilation: PointwiseCompilation
    writer: FreshTensorWriterContract


def discover_pointwise(module, selection, validate, *, excluded_operations=frozenset()):
    """Find semantics from software, while instruction mapping remains target-owned."""
    from merlin.compiler.lower.source.model import SourceRegion
    from merlin.compiler.schedule.elementwise.fusion import UnsupportedElementwise, scalar_graph
    from merlin.compiler.schedule.elementwise.pointwise_source import (
        compile_source_pointwise,
        read_pointwise_source,
    )
    from merlin.compiler.schedule.storage.temporaries import StorageOptimizationUnavailable

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
            supported.append(
                scalar_graph(
                    declaration["scalar_ir"].encode(),
                    budgets.max_scalar_operations,
                    budgets.max_scalar_ir_bytes,
                )
            )
        except UnsupportedElementwise:
            continue
    groups, group_declined = (), ()
    if selection.multi_output is not None:
        groups, group_declined = discover_multioutput(
            module, selection, validate, excluded_operations=excluded_operations
        )
    regions, declined = list(groups), list(group_declined)
    claimed = frozenset(operation for group in groups for operation in group.operations)
    for operation in module.walk():
        if operation in excluded_operations or operation in claimed or operation.name != "linalg.generic":
            continue
        try:
            source = read_pointwise_source(
                operation,
                max_elements=selection.max_source_elements,
                max_operations=budgets.max_scalar_operations,
                max_ir_bytes=budgets.max_scalar_ir_bytes,
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
                module,
                operation,
                program.kernel,
                program.objects,
                selection.adapter,
                selection.encoding,
                selection.limits,
                selected_contracts=contracts,
                bindings=program.bindings,
                budgets=budgets,
                private_ownership=selection.private_ownership,
                fusion_emitters=dict(selection.fusion_emitters),
                fusion_qualification_identity=selection.scalar_qualification_identity,
                configuration_qualification_identity=selection.configuration_qualification_identity,
                spill_emitters=dict(selection.spill_emitters) if selection.spill_emitters else None,
                spill_qualification_identity=selection.spill_qualification_identity,
                prefetch_qualification_identity=selection.prefetch_qualification_identity,
                max_source_elements=selection.max_source_elements,
            )
        except (UnsupportedElementwise, StorageOptimizationUnavailable) as error:
            declined.append(str(error))
            continue
        regions.append(SourceRegion(compiled, (operation,), tuple(operation.inputs), source.shape))
    return tuple(regions), tuple(declined)


# ---- source_rows -------------------------------------------------------------------------------
# Select checked source rows during ordinary model preparation.
#
# The shared compiler discovers and rewrites supported source islands. Explicit
# backend bindings implement their primitives and ranked borrowed ABI, including
# the declared input, workspace and completion premises. This transport neither
# qualifies those bindings nor interprets opaque backend output.

if TYPE_CHECKING:
    from merlin.compiler.lower.host.writers import FreshTensorWriterContract
    from merlin.compiler.schedule.elementwise.row_softmax import RowSoftmaxSourceCompilation
    from merlin.compiler.schedule.elementwise.row_softmax_checks import RowSoftmaxAdapter
    from merlin.compiler.schedule.pipeline.selected_contracts import SelectedSchedulingContracts
    from merlin.runtime.backends.baremetal.host_provider import HostProviderBuild, HostProviderContext


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
    borrowed_effects: BorrowedBufferEffects | None = None

    def __post_init__(self):
        validate_effects(self.borrowed_effects)
        from merlin.compiler.schedule.elementwise.row_softmax_checks import RowSoftmaxAdapter
        from merlin.compiler.schedule.pipeline.selected_contracts import SelectedSchedulingContracts

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
            or any(
                type(value) is not bool
                for value in (
                    self.assume_finite_inputs,
                    self.private_ownership,
                    self.disjoint_arguments,
                )
            )
            or any(
                type(value) is not int or value <= 0
                for value in (
                    self.max_source_elements,
                    self.max_islands,
                    self.timeout_ms,
                )
            )
            or self.chunk_elements is not None
            and (type(self.chunk_elements) is not int or self.chunk_elements <= 0)
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
    from merlin.compiler.lower.source.model import SourceRegion
    from merlin.compiler.schedule.elementwise.row_softmax import compile_source_rows, read_source_rows
    from merlin.compiler.schedule.elementwise.row_softmax_checks import RowSoftmaxUnavailable
    from merlin.compiler.schedule.storage.temporaries import StorageOptimizationUnavailable

    if selection.selected_contracts.row_reduction is None:
        return (), ("selected software specification has no complete-row capability",)
    regions, declined = [], []
    for operation in module.walk():
        if operation.name != "linalg.generic":
            continue
        try:
            source = read_source_rows(
                operation,
                numerical_policy=selection.numerical_policy,
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
                module,
                operation,
                selected_contracts=selection.selected_contracts,
                adapter=selection.adapter,
                numerical_policy=selection.numerical_policy,
                assume_finite_inputs=selection.assume_finite_inputs,
                private_ownership=selection.private_ownership,
                disjoint_arguments=selection.disjoint_arguments,
                chunk_elements=selection.chunk_elements,
                timeout_ms=selection.timeout_ms,
                max_source_elements=selection.max_source_elements,
            )
        except (RowSoftmaxUnavailable, StorageOptimizationUnavailable) as error:
            declined.append(str(error))
            continue
        regions.append(SourceRegion(compiled, source.operations, (source.source,), source.shape))
    return tuple(regions), tuple(declined)


def bind_source_row_lowering(selection, transform, builder, *, device=None, device_admission=None):
    """Compatibility entry for one selected row family."""
    from merlin.compiler.lower.source.model import bind_source_lowerings

    return bind_source_lowerings((), selection, transform, builder, device=device, device_admission=device_admission)


def source_row_provider_builder(builder):
    from merlin.compiler.lower.source.model import source_provider_builder

    return source_provider_builder(builder)


def source_row_device_work(builder):
    from merlin.compiler.lower.source.model import source_device_work

    return source_device_work(builder)


# ---- source_multioutput ------------------------------------------------------------------------
# Optional physical fanout composition through the ordinary source bridge.
#
# The original integration supplies instruction mapping and qualified composition
# effects. Shared source discovery and final-byte checks own correspondence. An
# unavailable composition retains the individual pointwise source route.

if TYPE_CHECKING:
    from merlin.compiler.kernel_ir.ir import Kernel
    from merlin.compiler.schedule.elementwise.multioutput_checks import MultiOutputCompositionLaw
    from merlin.compiler.schedule.storage.temporaries import TemporaryObject
    from merlin.compiler.transforms.source_rewrites.multioutput import MultiOutputBindings, MultiOutputSource


@dataclass(frozen=True)
class MultiOutputProgram:
    """Original instruction mapping, exact private objects and ordered outputs."""

    kernel: Kernel
    objects: tuple[TemporaryObject, ...]
    bindings: MultiOutputBindings

    def __post_init__(self):
        from merlin.compiler.kernel_ir.ir import Kernel
        from merlin.compiler.schedule.storage.temporaries import TemporaryObject
        from merlin.compiler.transforms.source_rewrites.multioutput import MultiOutputBindings

        if (
            type(self.kernel) is not Kernel
            or type(self.objects) is not tuple
            or any(type(obj) is not TemporaryObject for obj in self.objects)
            or type(self.bindings) is not MultiOutputBindings
        ):
            raise ValueError("multi-output mapping requires an actual kernel, typed objects and ordered bindings")
        self.bindings.__post_init__()


@dataclass(frozen=True)
class SourceMultiOutputLowering:
    """Original composition permission; the callback cannot choose source spans."""

    lower: Callable[[MultiOutputSource], MultiOutputProgram | None]
    composition_law: MultiOutputCompositionLaw

    def __post_init__(self):
        from merlin.compiler.schedule.elementwise.multioutput_checks import MultiOutputCompositionLaw

        if not callable(self.lower) or type(self.composition_law) is not MultiOutputCompositionLaw:
            raise ValueError("multi-output selection requires its original typed composition law and mapping")
        self.composition_law.__post_init__()


def _callbacks(selection):
    return (
        selection.multi_output.lower,
        selection.lower,
        selection.prepare,
        selection.build,
        selection.adapter.project,
        selection.adapter.rewrite,
        selection.encoding.encode,
        selection.encoding.decode_calls,
        selection.encoding.program_decoder.decode,
        selection.selected_contracts.revalidate,
        selection.selected_contracts.record,
        selection.selected_contracts.verify_image,
    )


def callback_snapshot(selection):
    """Freeze original Python implementations before the first owner callback."""
    return tuple(_callback(callback) for callback in _callbacks(selection))


def _callback(callback):
    target = callback.__func__ if type(callback) is MethodType else callback
    if type(target) is not FunctionType:
        raise ValueError("multi-output callbacks require original Python functions or bound methods")
    source = inspect.getsourcefile(target)
    if source is None or Path(source).is_symlink() or not Path(source).is_file():
        raise ValueError("multi-output callback source is missing or linked")
    path = Path(source).resolve()
    return getattr(callback, "__self__", None), target, target.__code__, path, sha256_file(path)


def _check_callback(callback, snapshot):
    owner, target, code, path, digest = snapshot
    if (
        getattr(callback, "__self__", None) is not owner
        or getattr(callback, "__func__", callback) is not target
        or target.__code__ is not code
        or path.is_symlink()
        or not path.is_file()
        or sha256_file(path) != digest
    ):
        raise ValueError("original multi-output implementation or callback changed; replan")


def check_callbacks(selection, snapshots):
    for callback, snapshot in zip(_callbacks(selection), snapshots, strict=True):
        _check_callback(callback, snapshot)


def _functions():
    from merlin.compiler.lower.source.model import _region_results, _SourceModelBuild
    from merlin.compiler.schedule.elementwise import multioutput_checks as multioutput_fusion_check
    from merlin.compiler.schedule.elementwise.multioutput_checks import MultiOutputCompositionLaw
    from merlin.compiler.transforms.source_rewrites import multioutput as multioutput_fusion
    from merlin.compiler.transforms.source_rewrites.multioutput import (
        MultiOutputCompilation,
        compile_source_multioutput,
        read_source_multioutput,
    )

    return (
        _callbacks,
        _callback,
        _check_callback,
        callback_snapshot,
        check_callbacks,
        _functions,
        function_snapshot,
        check_functions,
        implementation_paths,
        discover_multioutput,
        SourceMultiOutputLowering.__post_init__,
        MultiOutputProgram.__post_init__,
        SourcePointwiseLowering.__post_init__,
        discover_pointwise,
        _region_results,
        _SourceModelBuild.revalidate,
        _SourceModelBuild._check_group_callbacks,
        read_source_multioutput,
        compile_source_multioutput,
        MultiOutputCompositionLaw.__post_init__,
        MultiOutputCompositionLaw.flags,
        MultiOutputCompositionLaw.admits,
        MultiOutputCompilation.revalidate,
        multioutput_fusion._functions,
        multioutput_fusion_check._checker_functions,
        *multioutput_fusion._functions(),
        *multioutput_fusion_check._checker_functions(),
    )


def function_snapshot():
    reader = _functions
    return reader, reader.__code__, tuple((function, function.__code__) for function in reader())


def check_functions(snapshot):
    reader, reader_code, members = snapshot
    if _functions is not reader or reader.__code__ is not reader_code:
        raise ValueError("original multi-output implementation reader changed; replan")
    if any(
        function is not original or original.__code__ is not code
        for function, (original, code) in zip(reader(), members, strict=True)
    ):
        raise ValueError("original multi-output source checker or implementation code changed; replan")


def implementation_paths():
    from merlin.compiler.transforms.source_rewrites.multioutput import implementation_paths as core_paths

    return tuple(
        dict.fromkeys(
            (
                Path(__file__).resolve(),
                *(
                    module_source_path(
                        {
                            "source_model.py": "merlin.compiler.lower.source.model",
                            "source_pointwise.py": "merlin.compiler.lower.source.families",
                        }[name]
                    )
                    for name in ("source_pointwise.py", "source_model.py")
                ),
                *core_paths(),
            )
        )
    )


def discover_multioutput(module, selection, validate, *, excluded_operations=frozenset()):
    """Admit adjacent pure fanout only; all unsupported candidates keep singles."""
    from merlin.compiler.lower.source.model import SourceRegion, _region_results, _values
    from merlin.compiler.schedule.elementwise.fusion import UnsupportedElementwise
    from merlin.compiler.schedule.storage.temporaries import StorageOptimizationUnavailable
    from merlin.compiler.transforms.source_rewrites.multioutput import (
        compile_source_multioutput,
        read_source_multioutput,
    )

    if selection.multi_output is None:
        return (), ()
    before, selected = ir_digest(module), _values(selection)
    original_operations = tuple(module.walk())
    original_ssa = tuple(
        (operation.parent, tuple(operation.operands), tuple(operation.results)) for operation in original_operations
    )
    snapshots = callback_snapshot(selection)
    functions = function_snapshot()
    function_checker, function_checker_code = check_functions, check_functions.__code__
    validator = _callback(validate)
    pins = tuple((path, sha256_file(path)) for path in implementation_paths())

    def unchanged():
        if check_functions is not function_checker or function_checker.__code__ is not function_checker_code:
            raise ValueError("original multi-output implementation checker changed; replan")
        function_checker(functions)
        if ir_digest(module) != before or _values(selection) != selected:
            raise ValueError("original multi-output source or selection changed; replan")
        if (
            tuple(module.walk()) != original_operations
            or tuple(
                (operation.parent, tuple(operation.operands), tuple(operation.results))
                for operation in original_operations
            )
            != original_ssa
        ):
            raise ValueError("original multi-output SSA membership changed; replan")
        check_callbacks(selection, snapshots)
        _check_callback(validate, validator)
        for path, digest in pins:
            if path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
                raise ValueError("original multi-output implementation changed; replan")

    def fresh():
        unchanged()
        validate()
        unchanged()

    fresh()
    try:
        selection.multi_output.composition_law.admits(selection.adapter)
    except UnsupportedElementwise as error:
        fresh()
        return (), (str(error),)
    fresh()

    regions, declined, claimed = [], [], set()
    attempts = 0
    budgets = selection.budgets
    for producer in module.walk():
        if producer.name != "linalg.generic" or producer in claimed or producer in excluded_operations:
            continue
        first = producer.next_op
        second = None if first is None else first.next_op
        operations = (producer, first, second)
        if any(
            operation is None
            or operation.name != "linalg.generic"
            or operation in claimed
            or operation in excluded_operations
            for operation in operations
        ):
            continue
        try:
            source = read_source_multioutput(
                module,
                operations,
                max_source_elements=selection.max_source_elements,
                max_scalar_operations=budgets.max_scalar_operations,
                max_scalar_ir_bytes=budgets.max_scalar_ir_bytes,
            )
        except UnsupportedElementwise:
            continue
        if attempts >= selection.max_islands:
            declined.append("multi-output candidate budget exhausted; remaining source retains singles")
            break
        attempts += 1

        def source_state():
            return (
                source.shape,
                _values(source.producer),
                _values(source.epilogues),
                tuple((id(value), value) for value in source.inputs),
                tuple((id(value), value) for value in source.outputs),
                source.input_indices,
            )

        original_computation = source_state()
        fresh()
        try:
            program = selection.multi_output.lower(source)
        finally:
            fresh()
            if source_state() != original_computation:
                raise ValueError("multi-output mapping changed its original checked computation; replan")
        if program is None:
            declined.append("original multi-output implementation has no mapping; source retains singles")
            continue
        if type(program) is not MultiOutputProgram:
            raise ValueError("multi-output mapping must return a typed MultiOutputProgram or None")
        program.__post_init__()
        if not program.bindings.external_buffers_disjoint:
            declined.append("multi-output runtime disjointness is unavailable; source retains singles")
            continue
        try:
            compiled = compile_source_multioutput(
                module,
                operations,
                program.kernel,
                program.objects,
                selection.adapter,
                selection.encoding,
                selection.limits,
                selected_contracts=selection.selected_contracts,
                bindings=program.bindings,
                composition_law=selection.multi_output.composition_law,
                budgets=budgets,
                private_ownership=selection.private_ownership,
                configuration_qualification_identity=selection.configuration_qualification_identity,
                max_source_elements=selection.max_source_elements,
            )
        except (UnsupportedElementwise, StorageOptimizationUnavailable) as error:
            fresh()
            declined.append(str(error))
            continue
        fresh()
        if source_state() != original_computation:
            raise ValueError("multi-output compilation changed its original checked computation; replan")
        if compiled is None:
            declined.append("multi-output composition cannot satisfy the current resources; source retains singles")
            continue
        region = SourceRegion(compiled, operations, source.inputs, source.shape, outputs=source.outputs)
        _region_results(region)
        compiled.revalidate()
        fresh()
        regions.append(region)
        claimed.update(operations)
    return tuple(regions), tuple(declined)
