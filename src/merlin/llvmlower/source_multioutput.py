"""Optional physical fanout composition through the ordinary source bridge.

The original integration supplies instruction mapping and qualified composition
effects. Shared source discovery and final-byte checks own correspondence. An
unavailable composition retains the individual pointwise source route.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import FunctionType, MethodType
from typing import TYPE_CHECKING

from merlin.common.digest import sha256_file
from merlin.transforms.certificates import ir_digest

if TYPE_CHECKING:
    from merlin.compile.scheduling.multioutput_fusion import MultiOutputBindings, MultiOutputSource
    from merlin.compile.scheduling.multioutput_fusion_check import MultiOutputCompositionLaw
    from merlin.compile.scheduling.temporary_storage import TemporaryObject
    from merlin.sched.ir import Kernel


@dataclass(frozen=True)
class MultiOutputProgram:
    """Original instruction mapping, exact private objects and ordered outputs."""

    kernel: Kernel
    objects: tuple[TemporaryObject, ...]
    bindings: MultiOutputBindings

    def __post_init__(self):
        from merlin.compile.scheduling.multioutput_fusion import MultiOutputBindings
        from merlin.compile.scheduling.temporary_storage import TemporaryObject
        from merlin.sched.ir import Kernel

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
        from merlin.compile.scheduling.multioutput_fusion_check import MultiOutputCompositionLaw

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
    from merlin.compile.scheduling import multioutput_fusion, multioutput_fusion_check
    from merlin.compile.scheduling.multioutput_fusion import (
        MultiOutputCompilation,
        compile_source_multioutput,
        read_source_multioutput,
    )
    from merlin.compile.scheduling.multioutput_fusion_check import MultiOutputCompositionLaw

    from .source_model import _region_results, _SourceModelBuild
    from .source_pointwise import SourcePointwiseLowering, discover_pointwise

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
    from merlin.compile.scheduling.multioutput_fusion import implementation_paths as core_paths

    return tuple(
        dict.fromkeys(
            (
                Path(__file__).resolve(),
                *(Path(__file__).with_name(name) for name in ("source_pointwise.py", "source_model.py")),
                *core_paths(),
            )
        )
    )


def discover_multioutput(module, selection, validate, *, excluded_operations=frozenset()):
    """Admit adjacent pure fanout only; all unsupported candidates keep singles."""
    from merlin.compile.scheduling.fusion_semantics import UnsupportedElementwise
    from merlin.compile.scheduling.multioutput_fusion import compile_source_multioutput, read_source_multioutput
    from merlin.compile.scheduling.temporary_storage import StorageOptimizationUnavailable

    from .source_model import SourceRegion, _region_results, _values

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
