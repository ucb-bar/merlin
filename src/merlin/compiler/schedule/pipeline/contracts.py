"""Schedule compilation selections, budgets and shared emission checks."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path
from types import FunctionType, MethodType

from merlin.compiler.kernel_ir.ir import Call, Kernel, check_structure
from merlin.compiler.schedule.library_sources import shared_library_sources as shared_library_sources
from merlin.compiler.schedule.pipeline.program_limits import (
    FinalProgram,
    ProgramDecoder,
    _decode,
)
from merlin.compiler.schedule.storage.temporaries import (
    StorageOptimizationUnavailable,
    _project,
)


@dataclass(frozen=True)
class ScheduleEncoding:
    """Selected target implementation and actual-byte semantic decoder.

    decode_calls must reconstruct every instruction's actual operands from
    bytes alone, without consulting the optimizer's retained request stream.
    This family maps one decoded instruction to one schedule call. Qualification
    establishes that all state and numerical semantics are represented.
    """

    encode: Callable[[Kernel], bytes]
    decode_calls: Callable[[bytes], tuple[Call, ...]]
    program_decoder: ProgramDecoder
    numerical_policy_identity: str
    qualification_identity: str

    def __post_init__(self):
        if (
            not callable(self.encode)
            or not callable(self.decode_calls)
            or type(self.program_decoder) is not ProgramDecoder
            or not isinstance(self.numerical_policy_identity, str)
            or not self.numerical_policy_identity
            or not isinstance(self.qualification_identity, str)
            or not self.qualification_identity
        ):
            raise ValueError("schedule emission requires explicit typed encoding and qualified actual-byte decoding")


@dataclass(frozen=True)
class ScheduleBudgets:
    max_calls: int = 4096
    max_objects: int = 256
    max_spills: int = 4
    max_repairs: int = 4
    solver_timeout_ms: int = 5000
    eliminate_copies: bool = True
    max_fusions: int = 8
    max_scalar_operations: int = 128
    max_scalar_ir_bytes: int = 65536
    eliminate_configuration: bool = True
    optimize_storage: bool = True
    prefetch_distance: int = 1
    max_prefetch_moves: int = 8

    def __post_init__(self):
        if (
            any(
                type(value) is not int or value <= 0
                for value in (
                    self.max_calls,
                    self.max_objects,
                    self.solver_timeout_ms,
                    self.max_scalar_operations,
                    self.max_scalar_ir_bytes,
                )
            )
            or any(
                type(value) is not int or value < 0 for value in (self.max_spills, self.max_repairs, self.max_fusions)
            )
            or type(self.eliminate_copies) is not bool
            or type(self.eliminate_configuration) is not bool
            or type(self.optimize_storage) is not bool
            or any(
                type(value) is not int or not 0 <= value <= 1 << 20
                for value in (self.prefetch_distance, self.max_prefetch_moves)
            )
        ):
            raise ValueError("schedule compilation requires explicit finite work, repair and solver budgets")


@dataclass(frozen=True)
class PostStorageRewrite:
    """One original-host deletion transport, not qualification or a manager.

    Only the original host selects the live optimizer and separate verifier.
    Neither construction nor records grant permissions; author request data
    cannot choose these callbacks, source owners or verification premises.
    """

    name: str
    propose: Callable
    verify: Callable
    revalidate: Callable
    record: Callable
    sources: tuple[Path, ...]

    def __post_init__(self):
        if (
            type(self.name) is not str
            or not self.name
            or len(self.name) > 64
            or any(not (c.isascii() and (c.isalnum() or c == "_")) for c in self.name)
            or self.name
            in {
                "semantic_admission",
                "configuration",
                "fusion",
                "prefetch",
                "storage",
                "realization",
                "completion",
                "emission",
                "final_program",
                "encoded",
                "final_conformance",
                "configuration_candidate",
                "fusion_candidate",
                "prefetch_candidate",
                "source_effects",
            }
            or not all(
                type(f) in (FunctionType, MethodType)
                and getattr(getattr(f, "__func__", f), "__code__", None) is not None
                for f in (self.propose, self.verify, self.revalidate, self.record)
            )
            or type(self.sources) is not tuple
            or not 0 < len(self.sources) <= 256
            or any(not isinstance(path, Path) for path in self.sources)
        ):
            raise ValueError("post-storage rewrite requires a bounded original name, sources and separate verifier")


@dataclass(frozen=True)
class CopyPipelineMotion:
    """One original-owner qualified copy-motion slot, separate from deletion.

    The base owns placement, stage order, exact membership and final completion.
    A reviewed companion supplies the actual algorithm and separate law checker.
    Records create neither qualification nor candidate import permissions.
    """

    propose: Callable
    verify: Callable
    revalidate: Callable
    record: Callable
    sources: tuple[Path, ...]

    def __post_init__(self):
        if (
            not all(
                type(f) in (FunctionType, MethodType) for f in (self.propose, self.verify, self.revalidate, self.record)
            )
            or type(self.sources) is not tuple
            or not 0 < len(self.sources) <= 256
            or any(not isinstance(path, Path) for path in self.sources)
        ):
            raise ValueError("copy motion requires original callbacks and bounded implementation sources")


@dataclass(frozen=True)
class ScheduleCompilation:
    kernel: Kernel
    program: FinalProgram
    report: dict


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _rewrite_snapshot(value):
    """Read original typed data without invoking its record/provider methods."""
    if value is None or type(value) in (str, int, float, bool, bytes):
        return (type(value), value)
    if isinstance(value, Path):
        return (Path, str(value))
    if isinstance(value, (tuple, list, set, frozenset)):
        items = tuple(_rewrite_snapshot(item) for item in value)
        return (type(value), frozenset(items) if isinstance(value, (set, frozenset)) else items)
    if type(value) is dict:
        return (dict, tuple((key, _rewrite_snapshot(item)) for key, item in value.items()))
    if is_dataclass(value) and not isinstance(value, type):
        return (type(value), tuple((f.name, _rewrite_snapshot(getattr(value, f.name))) for f in fields(value)))
    if callable(value):
        return ("callback", value, getattr(getattr(value, "__func__", value), "__code__", None))
    raise ValueError("post-storage input requires supported immutable typed declarations")


def _verify_kernel(kernel):
    if type(kernel) is not Kernel or check_structure(kernel):
        raise ValueError("schedule pipeline requires structurally valid typed kernel IR")


def _program_completions(effects, *, checked_copy_motion=None):
    """Project exact waits onto the finalizer's qualified global-drain model.

    The storage checker remains responsible for token validity and hazards.
    A final decoder's drain cannot stand in for a wait for only some pending
    operations: that would permit an unsafe cut and storage reuse after launch.
    """
    from merlin.compiler.schedule.pipeline.contracts import CopyPipelineMotion

    pending, completions = set(), []
    for effect in effects:
        if effect.completion == "asynchronous":
            pending.add(effect.token)
            completions.append("asynchronous")
        elif effect.completion == "wait":
            if checked_copy_motion is not None and type(checked_copy_motion) is not CopyPipelineMotion:
                raise ValueError("partial waits require the original checked copy motion")
            if set(effect.waits) != pending and checked_copy_motion is None:
                raise StorageOptimizationUnavailable(
                    "encoded schedule requires a wait for all pending tokens; "
                    "partial waits need a qualified final model"
                )
            pending.difference_update(effect.waits)
            completions.append("synchronous" if pending else "drain")
        else:
            completions.append(effect.completion)
    if pending:
        raise ValueError("encoded schedule returns before actual completion")
    return tuple(completions)


def _emission_effects(
    images,
    kernel,
    adapter,
    encoding,
    limits,
    max_calls,
    configuration_adapter=None,
    selected_contracts=None,
    checked_copy_motion=None,
):
    """Reconstruct complete effects from actual output, independently of requests."""
    from merlin.compiler.schedule.pipeline.contracts import _program_completions

    actual = []
    actual_configuration = []
    for image in images:
        instructions, _targets = _decode(image, limits, encoding.program_decoder)
        calls = encoding.decode_calls(image)
        if type(calls) is not tuple or any(type(call) is not Call for call in calls):
            raise ValueError("semantic decoder must reconstruct every actual instruction as typed calls")
        if selected_contracts is not None:
            selected_contracts.verify_image(image, calls)
        effects = _project(replace(kernel, body=calls), adapter, max_calls)
        if len(effects) != len(instructions):
            raise ValueError("actual semantic decoding and final instruction decoding disagree on coverage")
        for effect, instruction, completion in zip(
            effects, instructions, _program_completions(effects, checked_copy_motion=checked_copy_motion), strict=True
        ):
            if (
                instruction.completion != completion
                or instruction.branches
                or not instruction.fallthrough
                or effect.accesses != instruction.accesses
            ):
                raise ValueError(
                    "actual emitted effects require matching straight-line access and completion semantics"
                )
        actual.extend(effects)
        if configuration_adapter is not None:
            actual_configuration.extend(configuration_adapter.project(call) for call in calls)
        if len(actual) > max_calls:
            raise ValueError("actual emitted program exceeds its selected call budget")
    expected = _project(kernel, adapter, max_calls)
    if tuple(actual) != expected:
        raise ValueError("actual emitted numerical, storage or completion effects differ from the verified schedule")
    if configuration_adapter is not None and tuple(actual_configuration) != tuple(
        configuration_adapter.project(call) for call in kernel.body
    ):
        raise ValueError("actual emitted configuration effects differ from the verified schedule")
    return tuple(actual)
