"""Original-qualified connected pointwise composition with two live outputs.

One existing instruction kernel, allocator and emission pipeline are reused.
This original-host entry grants neither device capabilities nor author access.
Missing composition support retains the caller's original individual route.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import prod
from pathlib import Path
from weakref import WeakKeyDictionary, ref

from merlin.common.digest import sha256_file
from merlin.sched.ir import Kernel
from merlin.transforms.certificates import ir_digest
from merlin.transforms.pipeline_state import PassSpec, PipelineContext, PipelineSession

from .compact_dense_group import _callback, _same_callback
from .fusion_semantics import UnsupportedElementwise, term
from .library_sources import shared_library_sources
from .multioutput_fusion_check import (
    MultiOutputBindings,
    MultiOutputCompositionLaw,
    _arguments,
    verify_multioutput_fusion,
)
from .pointwise_source import PointwiseSource, read_pointwise_source
from .schedule_pipeline import ScheduleBudgets, ScheduleEncoding, _rewrite_snapshot, compile_schedule
from .selected_contracts import SelectedSchedulingContracts
from .temporary_storage import StorageAdapter, TemporaryObject

_ISSUED = WeakKeyDictionary()


@dataclass(frozen=True)
class MultiOutputSource:
    shape: tuple[int, ...]
    producer: PointwiseSource
    epilogues: tuple[PointwiseSource, PointwiseSource]
    inputs: tuple
    outputs: tuple
    input_indices: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]


def read_source_multioutput(
    module, operations, *, max_source_elements=64 << 20, max_scalar_operations=128, max_scalar_ir_bytes=65536
):
    """Read an exact adjacent fan-out, preserving every external observer.

    P must have exactly its two epilogue users and no publication or escape.
    Consumer results remain the original ordered outputs. No source is cloned
    or mutated and no instruction/physical execution permission is inferred.
    """
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Operation

    if type(module) is not ModuleOp or type(operations) is not tuple or len(operations) != 3:
        raise ValueError("multi-output source requires one original module and exactly three operations")
    if type(max_source_elements) is not int or max_source_elements <= 0:
        raise ValueError("multi-output source needs a positive logical element budget")
    if len(set(operations)) != 3 or any(operation not in frozenset(module.walk()) for operation in operations):
        raise ValueError("multi-output members must be distinct current original operations")
    block = operations[0].parent
    if block is None or any(operation.parent is not block for operation in operations):
        raise UnsupportedElementwise("multi-output source requires one adjacent same-block composition")
    current = tuple(block.ops)
    index = current.index(operations[0])
    if current[index : index + 3] != operations:
        raise UnsupportedElementwise("multi-output composition crosses an operation or observer")
    stages = tuple(
        read_pointwise_source(
            operation,
            max_elements=max_source_elements,
            max_operations=max_scalar_operations,
            max_ir_bytes=max_scalar_ir_bytes,
            allow_constant_captures=True,
        )
        for operation in operations
    )
    if any(stage.shape != stages[0].shape for stage in stages):
        raise UnsupportedElementwise("multi-output stages change their common logical domain")
    producer = operations[0].results[0]
    if any(sum(value is producer for value in operation.inputs) != 1 for operation in operations[1:]):
        raise UnsupportedElementwise("each epilogue must consume the exact common producer once")
    if any(use.operation not in operations[1:] for use in producer.uses):
        raise UnsupportedElementwise("observed or escaping producer retains the original individual route")
    inputs, indices = [], []
    for operation in operations:
        row = []
        for value in operation.inputs:
            if value is producer:
                row.append(-1)
                continue
            if value.owner in operations:
                raise UnsupportedElementwise("multi-output epilogues must be independent apart from their common P")
            owner, anchor = value.owner, operations[0]
            while anchor is not None:
                if owner is anchor.parent or (
                    isinstance(owner, Operation) and owner.parent is anchor.parent and owner.is_before_in_block(anchor)
                ):
                    break
                anchor = anchor.parent_op()
            if anchor is None:
                raise UnsupportedElementwise("multi-output input does not dominate the whole composition")
            if value not in inputs:
                inputs.append(value)
            row.append(inputs.index(value))
        indices.append(tuple(row))
    outputs = tuple(operation.results[0] for operation in operations[1:])
    if (len(inputs) + 3) * prod(stages[0].shape) > max_source_elements:
        raise UnsupportedElementwise("multi-output complete source exceeds the original element budget")
    for value in outputs:
        for use in value.uses:
            observer = use.operation
            while observer is not None and observer.parent is not block:
                observer = observer.parent_op()
            if observer is None or current.index(observer) <= index + 2:
                raise UnsupportedElementwise("multi-output replacement cannot preserve an earlier observer")
    return MultiOutputSource(stages[0].shape, stages[0], stages[1:], tuple(inputs), outputs, tuple(indices))


def implementation_paths() -> tuple[Path, ...]:
    """Direct owners plus the already reviewed shared scheduling source roster."""
    import importlib

    from merlin.common import dense_float_ir
    from merlin.transforms import certificates

    from . import compact_dense_group, multioutput_fusion_check
    from .library_sources import library_modules

    return tuple(
        dict.fromkeys(
            (
                Path(__file__).resolve(),
                Path(multioutput_fusion_check.__file__).resolve(),
                Path(compact_dense_group.__file__).resolve(),
                Path(dense_float_ir.__file__).resolve(),
                *(Path(importlib.import_module(name).__file__).resolve() for name in library_modules()),
                *(
                    Path(__import__(obj.__module__, fromlist=["x"]).__file__).resolve()
                    for obj in (certificates.DenseFloatParser, certificates.DenseFloatPrinter)
                ),
            )
        )
    )


def _source_snapshot(module, operations):
    return (
        ir_digest(module),
        tuple(module.walk()),
        tuple(
            (operation, operation.parent, tuple(operation.operands), tuple(operation.results))
            for operation in operations
        ),
    )


def _callbacks(adapter, encoding, selected):
    return (
        adapter.project,
        adapter.rewrite,
        encoding.encode,
        encoding.decode_calls,
        encoding.program_decoder.decode,
        selected.revalidate,
        selected.record,
        selected.verify_image,
    )


def _functions():
    from . import multioutput_fusion_check as checker
    from .fusion_contracts import SelectedFusionContract
    from .storage_contracts import TemporaryStorageContract

    return (
        read_source_multioutput,
        _source_snapshot,
        _rewrite_snapshot,
        _arguments,
        _callback,
        _same_callback,
        read_pointwise_source,
        verify_multioutput_fusion,
        compile_schedule,
        SelectedSchedulingContracts.configuration_adapter,
        TemporaryStorageContract.bind,
        SelectedFusionContract.bind,
        implementation_paths,
        _callbacks,
        _functions,
        compile_source_multioutput,
        MultiOutputCompilation.revalidate,
        MultiOutputCompilation.images,
        MultiOutputCompilation.operation.fget,
        MultiOutputCompilation.operation_ordinal.fget,
        MultiOutputCompilation.implementation_paths,
        checker.verify_multioutput_fusion,
        checker._state,
        checker._current,
        checker._source_state,
        checker._arguments,
        MultiOutputCompositionLaw.__post_init__,
        MultiOutputCompositionLaw.flags,
        MultiOutputCompositionLaw.admits,
        MultiOutputBindings.__post_init__,
    )


@dataclass(frozen=True, eq=False)
class MultiOutputCompilation:
    source: object
    operations: tuple
    operation_ordinals: tuple[int, ...]
    inputs: tuple
    outputs: tuple
    compiled: object
    source_passes: tuple
    conformance: dict
    source_pins: tuple
    _guard: object

    @property
    def operation(self):
        return self.operations[0]

    @property
    def operation_ordinal(self):
        return self.operation_ordinals[0]

    def revalidate(self):
        issued = _ISSUED.get(self)
        if issued is None or self._guard is not issued[0]() or self._guard.__code__ is not issued[1]:
            raise ValueError("multi-output original issuance or validation guard changed; replan")
        self._guard(self)

    def images(self):
        self.revalidate()
        return tuple(self.compiled.program.segments)

    def implementation_paths(self):
        self.revalidate()
        return implementation_paths()


def compile_source_multioutput(
    module,
    operations,
    kernel,
    objects,
    adapter,
    encoding,
    limits,
    *,
    selected_contracts,
    bindings,
    composition_law,
    budgets=ScheduleBudgets(),
    banks=None,
    reservations=(),
    private_ownership=False,
    configuration_qualification_identity=None,
    max_source_elements=64 << 20,
    pass_validation=None,
):
    """Check one composed program before allocation and after actual byte decode.

    The original live caller supplies every semantic/effect implementation.
    The existing scheduler owns allocation, repair, limits and final emission;
    this entry cannot manufacture a missing physical composition capability.
    """
    if (
        type(selected_contracts) is not SelectedSchedulingContracts
        or selected_contracts.storage is None
        or selected_contracts.fusion is None
        or type(adapter) is not StorageAdapter
        or type(encoding) is not ScheduleEncoding
        or type(budgets) is not ScheduleBudgets
        or type(bindings) is not MultiOutputBindings
        or type(composition_law) is not MultiOutputCompositionLaw
    ):
        raise ValueError("multi-output compilation requires original typed scalar/storage contracts and bindings")
    composition_law.admits(adapter)
    # Bound admission and snapshots precede the first original-owner callback.
    source = read_source_multioutput(
        module,
        operations,
        max_source_elements=max_source_elements,
        max_scalar_operations=budgets.max_scalar_operations,
        max_scalar_ir_bytes=budgets.max_scalar_ir_bytes,
    )
    module.verify()
    if type(kernel) is not Kernel or kernel.args != _arguments(source):
        raise ValueError("multi-output program does not bind the exact original tensor arguments")
    if type(objects) is not tuple or len(objects) > budgets.max_objects or len(kernel.body) > budgets.max_calls:
        raise ValueError("multi-output program exceeds original bounded work")
    callbacks = tuple(_callback(function) for function in _callbacks(adapter, encoding, selected_contracts))
    same_callback, same_code = _same_callback, _same_callback.__code__
    function_reader, function_code = _functions, _functions.__code__
    functions = tuple(_callback(function) for function in function_reader())
    paths = tuple(dict.fromkeys((*implementation_paths(), *(row[3] for row in callbacks))))
    pins = tuple((path, sha256_file(path)) for path in paths)
    original_source, original_kernel = _source_snapshot(module, operations), kernel.digest()
    original_parameters = _rewrite_snapshot(
        (
            selected_contracts,
            encoding,
            limits,
            budgets,
            banks,
            reservations,
            private_ownership,
            configuration_qualification_identity,
            max_source_elements,
        )
    )
    raw_adapter_state = (
        adapter.configuration_identity,
        adapter.numerical_policy_identity,
        adapter.qualification_identity,
    )
    law_state, bindings_state = asdict(composition_law), asdict(bindings)
    objects_state = tuple(asdict(obj) for obj in objects)
    library = shared_library_sources()

    def static():
        if (
            _same_callback is not same_callback
            or same_callback.__code__ is not same_code
            or _functions is not function_reader
            or function_reader.__code__ is not function_code
            or any(
                not same_callback(function, frozen)
                for function, frozen in zip(function_reader(), functions, strict=True)
            )
        ):
            raise ValueError("multi-output original implementation function or code changed; replan")
        if (
            _source_snapshot(module, operations) != original_source
            or _rewrite_snapshot(
                (
                    selected_contracts,
                    encoding,
                    limits,
                    budgets,
                    banks,
                    reservations,
                    private_ownership,
                    configuration_qualification_identity,
                    max_source_elements,
                )
            )
            != original_parameters
            or kernel.digest() != original_kernel
            or asdict(composition_law) != law_state
            or asdict(bindings) != bindings_state
            or tuple(asdict(obj) for obj in objects) != objects_state
            or (adapter.configuration_identity, adapter.numerical_policy_identity, adapter.qualification_identity)
            != raw_adapter_state
            or shared_library_sources() != library
            or any(
                not same_callback(function, frozen)
                for function, frozen in zip(_callbacks(adapter, encoding, selected_contracts), callbacks, strict=True)
            )
            or any(
                not same_callback(function, frozen) for function, frozen in zip(_functions(), functions, strict=True)
            )
            or any(path.is_symlink() or not path.is_file() or sha256_file(path) != digest for path, digest in pins)
        ):
            raise ValueError("multi-output original source, implementation or selection changed; replan")

    def owner():
        static()
        selected_contracts.revalidate()
        static()
        value = selected_contracts.record()
        static()
        return term(value)

    selected_identity = owner()

    def fresh():
        if owner() != selected_identity:
            raise ValueError("multi-output selected RTL/software context changed; replan")

    configuration = (
        selected_contracts.configuration_adapter(configuration_qualification_identity)
        if configuration_qualification_identity is not None
        else None
    )
    fresh()
    physical_banks, fixed, external, effective = selected_contracts.storage.bind(
        selected_contracts.isa,
        adapter,
        banks=banks,
        reservations=reservations,
        external_windows=None,
        private_ownership=private_ownership,
        configuration_adapter=configuration,
    )
    fresh()
    effective, _, scalar_binding = selected_contracts.fusion.bind(
        effective,
        emitters={},
        qualification_identity=composition_law.qualification_identity,
        max_operations=budgets.max_scalar_operations,
        max_ir_bytes=budgets.max_scalar_ir_bytes,
    )
    fresh()

    def verify(actual, actual_objects, *, virtual=False):
        fresh()
        value = verify_multioutput_fusion(
            source,
            actual,
            actual_objects,
            physical_banks,
            effective,
            bindings,
            composition_law,
            external_windows=external,
            private_ownership=private_ownership,
            budgets=budgets,
            configuration_adapter=configuration,
            role_adapter=adapter,
            virtual_private_addresses=virtual,
        )
        fresh()
        return value

    context = PipelineContext(
        hardware=selected_contracts.hardware_identity,
        numerical_policy=selected_contracts.numerical_policy,
        resources=term((asdict(bindings), [asdict(bank) for bank in physical_banks], asdict(budgets))),
        execution=term((law_state, scalar_binding, selected_identity)),
        library=term(library),
    )
    session = PipelineSession(module, context=context, ir_level="linalg", max_repairs=0)
    specs = (
        PassSpec("multioutput-source", 0, produces=("source_conformance",), mutates_ir=False, ir_level="linalg"),
        PassSpec(
            "multioutput-lowering",
            1,
            requires=("source_conformance",),
            produces=("compiled",),
            mutates_ir=False,
            ir_level="linalg",
        ),
        PassSpec(
            "multioutput-final",
            2,
            requires=("compiled",),
            produces=("conformance",),
            mutates_ir=False,
            ir_level="linalg",
        ),
    )
    session.run(specs[0], lambda: {"source_conformance": verify(kernel, objects, virtual=True)})

    def lower():
        fresh()
        compiled = compile_schedule(
            kernel,
            objects,
            physical_banks,
            adapter,
            encoding,
            limits,
            budgets=budgets,
            reservations=fixed,
            external_windows=external,
            private_ownership=private_ownership,
            selected_contracts=selected_contracts,
            configuration_qualification_identity=configuration_qualification_identity,
            fusion_qualification_identity=composition_law.qualification_identity,
            pass_validation=pass_validation,
        )
        fresh()
        return {"compiled": compiled}

    session.run(specs[1], lower)
    compiled = session.analysis("compiled")

    def final():
        calls = []
        for image in compiled.program.segments:
            fresh()
            decoded = encoding.decode_calls(image)
            fresh()
            selected_contracts.verify_image(image, decoded)
            fresh()
            calls.extend(decoded)
        actual = Kernel(kernel.name, kernel.args, tuple(calls), kernel.attrs)
        storage = compiled.report["storage"]
        placed = tuple(
            replace(TemporaryObject(**row), source_base=storage["addresses"][row["name"]]) for row in storage["objects"]
        )
        return {"conformance": verify(actual, placed)}

    session.run(specs[2], final)
    ordinals = tuple(tuple(module.walk()).index(operation) for operation in operations)
    passes, conformance = tuple(session.records), session.analysis("conformance")
    compiled_state = (compiled.kernel.digest(), tuple(compiled.program.segments), term(compiled.report))
    final_state = term(conformance)
    passes_state = term([asdict(record) for record in passes])
    program_report = term(compiled.program.report)
    fresh()

    def guard(candidate):
        fresh()
        if (
            candidate.source is not module
            or candidate.operations != operations
            or candidate.operation_ordinals != ordinals
            or candidate.inputs != source.inputs
            or candidate.outputs != source.outputs
            or candidate.compiled is not compiled
            or candidate.source_passes != passes
            or term([asdict(record) for record in candidate.source_passes]) != passes_state
            or term(compiled.program.report) != program_report
            or candidate.source_pins != pins
            or term(candidate.conformance) != final_state
            or (compiled.kernel.digest(), tuple(compiled.program.segments), term(compiled.report)) != compiled_state
            or compiled.kernel.args != _arguments(source)
        ):
            raise ValueError("multi-output issued source-to-program binding changed; replan")

    result = MultiOutputCompilation(
        module, operations, ordinals, source.inputs, source.outputs, compiled, passes, conformance, pins, guard
    )
    _ISSUED[result] = ref(guard), guard.__code__
    result.revalidate()
    return result
