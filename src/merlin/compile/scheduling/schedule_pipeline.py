"""Execute checked schedule passes through final independently decoded bytes.

This entry supports finite straight-line byte-addressed programs. Target
encoding and semantic decoding remain selected, independently qualified
capabilities. Unknown instructions, hidden effects and counted loops retain
their separate lowering routes; opaque output is never interpreted here.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from merlin.sched.ir import Call, Kernel
from merlin.sched.ir.kernel import check_structure
from merlin.semantic_compiler.allocation_types import StorageBank
from merlin.transforms.pipeline import PassSpec, PipelineContext, PipelineSession, validate_pass_order

from .fusion import plan_fusion, validate_fusion_capabilities
from .program_limits import (
    AddressWindow,
    Continuation,
    EncodedAccess,
    FinalProgram,
    ProgramDecoder,
    ProgramLimits,
    _decode,
    finalize_program,
)
from .spill_conformance import validate_spill_classes
from .temporary_storage import (
    StorageAdapter,
    StorageOptimizationUnavailable,
    TemporaryObject,
    _check_reservations,
    _completion,
    _problem,
    _project,
    allocate_temporary_storage,
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
        ):
            raise ValueError("schedule compilation requires explicit finite work, repair and solver budgets")


@dataclass(frozen=True)
class ScheduleCompilation:
    kernel: Kernel
    program: FinalProgram
    report: dict


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def shared_library_sources():
    """Current shared implementation identities, equal for every treatment."""
    names = (
        "merlin.transforms.pipeline",
        "merlin.compile.scheduling.schedule_pipeline",
        "merlin.compile.scheduling.temporary_storage",
        "merlin.compile.scheduling.copy_elimination",
        "merlin.compile.scheduling.device_spills",
        "merlin.compile.scheduling.spill_conformance",
        "merlin.compile.scheduling.program_limits",
        "merlin.compile.scheduling.fusion",
        "merlin.compile.scheduling.fusion_semantics",
        "merlin.compile.scheduling.row_softmax",
        "merlin.compile.scheduling.row_softmax_verification",
        "merlin.compile.scheduling.compact_dense",
        "merlin.compile.scheduling.compact_verification",
        "merlin.transforms.dense_planning",
        "merlin.llvmlower.cpu_softmax",
        "merlin.transforms.pure_scalar",
        "merlin.transforms.certificates",
        "merlin.verify.epilogue_applicability",
        "merlin.semantic_compiler.allocate",
        "merlin.semantic_compiler.allocation_types",
        "merlin.semantic_compiler.rules",
        "merlin.sched.ir.kernel",
        "merlin.sched.ir.expr",
        "merlin.sched.codegen",
    )
    return {
        name: hashlib.sha256(Path(importlib.import_module(name).__file__).read_bytes()).hexdigest() for name in names
    }


def _verify_kernel(kernel):
    if type(kernel) is not Kernel or check_structure(kernel):
        raise ValueError("schedule pipeline requires structurally valid typed kernel IR")


def _emission_effects(images, kernel, adapter, encoding, limits, max_calls):
    """Reconstruct complete effects from actual output, independently of requests."""
    actual = []
    for image in images:
        instructions, _targets = _decode(image, limits, encoding.program_decoder)
        calls = encoding.decode_calls(image)
        if type(calls) is not tuple or any(type(call) is not Call for call in calls):
            raise ValueError("semantic decoder must reconstruct every actual instruction as typed calls")
        effects = _project(replace(kernel, body=calls), adapter, max_calls)
        if len(effects) != len(instructions):
            raise ValueError("actual semantic decoding and final instruction decoding disagree on coverage")
        for effect, instruction in zip(effects, instructions, strict=True):
            if (
                effect.completion != "synchronous"
                or instruction.completion != "synchronous"
                or instruction.branches
                or effect.accesses != instruction.accesses
            ):
                raise ValueError(
                    "actual emitted effects require matching synchronous straight-line instruction semantics"
                )
        actual.extend(effects)
        if len(actual) > max_calls:
            raise ValueError("actual emitted program exceeds its selected call budget")
    expected = _project(kernel, adapter, max_calls)
    if tuple(actual) != expected:
        raise ValueError("actual emitted numerical, storage or completion effects differ from the verified schedule")
    return tuple(actual)


def compile_schedule(
    kernel,
    objects,
    banks,
    adapter,
    encoding,
    limits,
    *,
    budgets=ScheduleBudgets(),
    reservations=(),
    external_windows=(),
    private_ownership=False,
    spill_classes=(),
    continuation=None,
    pass_order=(),
    fusion_capabilities=(),
) -> ScheduleCompilation:
    """Allocate, check, encode and finalize through the existing pass session.

    Missing capabilities or infeasible allocation diagnose for the caller's
    existing legal route. Never emit a partially verified candidate. Target
    execution qualification is separate from shared transformation correctness.
    """
    if (
        type(adapter) is not StorageAdapter
        or type(encoding) is not ScheduleEncoding
        or type(limits) is not ProgramLimits
        or type(budgets) is not ScheduleBudgets
        or encoding.program_decoder.configuration_identity != adapter.configuration_identity
        or encoding.numerical_policy_identity != adapter.numerical_policy_identity
    ):
        raise ValueError("schedule compilation requires matching selected typed contracts and budgets")
    specs = (
        PassSpec("semantic_admission", 0, produces=("source_effects",), mutates_ir=False, ir_level="schedule"),
        PassSpec(
            "storage",
            3,
            requires=("source_effects",),
            produces=("storage", "realization"),
            invalidates=("*",),
            ir_level="schedule",
        ),
        PassSpec(
            "completion",
            4,
            requires=("storage", "realization"),
            produces=("completion",),
            mutates_ir=False,
            ir_level="schedule",
        ),
        PassSpec(
            "emission",
            5,
            requires=("completion", "realization"),
            produces=("encoded",),
            mutates_ir=False,
            ir_level="schedule",
        ),
        PassSpec(
            "final_program",
            6,
            requires=("encoded", "completion"),
            produces=("final_program", "final_conformance"),
            mutates_ir=False,
            ir_level="schedule",
        ),
    )
    validate_fusion_capabilities(fusion_capabilities, adapter)
    if fusion_capabilities:
        fusion_spec = PassSpec(
            "fusion",
            2,
            requires=("source_effects",),
            produces=("fusion_candidate",),
            mutates_ir=False,
            ir_level="schedule",
        )
        specs = (specs[0], fusion_spec, replace(specs[1], requires=("source_effects", "fusion_candidate")), *specs[2:])
    if pass_order:
        names = {spec.name: spec for spec in specs}
        if type(pass_order) is not tuple or len(pass_order) != len(specs) or set(pass_order) != set(names):
            raise ValueError("explicit schedule order must contain every selected pass exactly once")
        specs = tuple(names[name] for name in pass_order)
    validate_pass_order(specs)
    _verify_kernel(kernel)
    if (
        type(objects) is not tuple
        or any(type(obj) is not TemporaryObject for obj in objects)
        or type(banks) is not tuple
        or any(type(bank) is not StorageBank for bank in banks)
        or type(external_windows) is not tuple
        or any(type(window) is not AddressWindow for window in external_windows)
        or continuation is not None
        and type(continuation) is not Continuation
    ):
        raise ValueError("schedule compilation requires immutable typed storage and continuation declarations")
    _check_reservations(reservations)
    validate_spill_classes(spill_classes, banks)
    sources = shared_library_sources()
    resources = {
        "objects": [asdict(obj) for obj in objects],
        "banks": [bank.record() for bank in banks],
        "reservations": [asdict(row) for row in reservations],
        "external_windows": [asdict(w) for w in external_windows],
        "private_ownership": private_ownership,
        "limits": asdict(limits),
        "budgets": asdict(budgets),
        "fusion_capabilities": [capability.record() for capability in fusion_capabilities],
        "spill_classes": [
            {
                name: getattr(capability, name)
                for name in (
                    "register_bank",
                    "spill_bank",
                    "width_bytes",
                    "alignment",
                    "qualification_identity",
                )
            }
            for capability in spill_classes
        ],
    }
    context = PipelineContext(
        hardware=adapter.configuration_identity,
        numerical_policy=adapter.numerical_policy_identity,
        resources=_hash(resources),
        execution=_hash(
            {
                "storage": adapter.qualification_identity,
                "encoding": encoding.qualification_identity,
                "decoder": encoding.program_decoder.qualification_identity,
                "continuation": asdict(continuation) if continuation is not None else None,
            }
        ),
        library=_hash(sources),
    )
    session = PipelineSession(
        kernel,
        context=context,
        ir_level="schedule",
        max_repairs=budgets.max_repairs,
        fingerprint=lambda value: value.digest(),
        verify_ir=_verify_kernel,
    )
    model = dict(
        reservations=reservations,
        external_windows=external_windows,
        private_ownership=private_ownership,
        max_calls=budgets.max_calls,
        max_objects=budgets.max_objects,
    )

    def admission():
        _bank_map, effects, *_ = _problem(
            kernel, objects, banks, adapter, external_windows, private_ownership, budgets.max_calls, budgets.max_objects
        )
        if any(effect.completion != "synchronous" for effect in effects):
            raise StorageOptimizationUnavailable("encoded schedule pipeline requires qualified synchronous completion")
        return {"source_effects": effects}

    def fusion():
        return {
            "fusion_candidate": plan_fusion(
                session.module,
                objects,
                banks,
                adapter,
                fusion_capabilities,
                external_windows=external_windows,
                private_ownership=private_ownership,
                max_calls=budgets.max_calls,
                max_objects=budgets.max_objects,
                max_fusions=budgets.max_fusions,
                max_scalar_operations=budgets.max_scalar_operations,
                max_scalar_ir_bytes=budgets.max_scalar_ir_bytes,
            )
        }

    def storage():
        def allocate(source, source_objects):
            return allocate_temporary_storage(
                source,
                source_objects,
                banks,
                adapter,
                timeout_ms=budgets.solver_timeout_ms,
                spill_classes=spill_classes,
                max_spills=budgets.max_spills,
                eliminate_copies=budgets.eliminate_copies,
                on_repair=lambda reason: session.repair(stage=3, reason=reason),
                **model,
            )

        candidate = session.analysis("fusion_candidate") if fusion_capabilities else None
        result = allocate(candidate.kernel, candidate.objects) if candidate is not None else None
        fusion_status = "realized"
        if candidate is not None and result is None:
            session.repair(stage=3, reason="fusion allocation failed; restore original composition and replan")
            fusion_status = "allocation_fallback"
        if result is None:
            result = allocate(session.module, objects)
        if result is None:
            raise StorageOptimizationUnavailable(
                "no legal temporary allocation within the selected schedule repair family"
            )
        session.module = result.kernel
        report = dict(result.report)
        if candidate is not None:
            report["fusion"] = {"status": fusion_status, "checked_steps": list(candidate.checks)}
        return {"storage": report, "realization": report}

    def completion():
        effects = _project(session.module, adapter, budgets.max_calls)
        return {"completion": _completion(effects, {bank.name: bank for bank in banks})}

    def emission():
        image = encoding.encode(session.module)
        _emission_effects((image,), session.module, adapter, encoding, limits, budgets.max_calls)
        return {"encoded": image}

    def final():
        # All external windows are published conservatively, including input
        # aliases and carried state. A caller cannot omit an output to permit
        # an otherwise illegal segmentation boundary.
        outputs = tuple(EncodedAccess(window.space, window.start, window.size) for window in external_windows)
        program = finalize_program(
            session.analysis("encoded"), limits, encoding.program_decoder, continuation=continuation, outputs=outputs
        )
        effects = _emission_effects(program.segments, session.module, adapter, encoding, limits, budgets.max_calls)
        return {
            "final_program": program,
            "final_conformance": {
                "schedule_sha256": session.module.digest(),
                "image_sha256": hashlib.sha256(b"".join(program.segments)).hexdigest(),
                "actual_commands": len(effects),
                "requested_read_bytes": sum(a.size for effect in effects for a in effect.accesses if not a.writes),
                "requested_write_bytes": sum(a.size for effect in effects for a in effect.accesses if a.writes),
                "encoding_qualification_identity": encoding.qualification_identity,
                "machine_effects": "checked_under_selected_decoders",
                "execution_qualification": "not_established",
                "timing": "UNKNOWN",
            },
        }

    actions = {
        "semantic_admission": admission,
        "fusion": fusion,
        "storage": storage,
        "completion": completion,
        "emission": emission,
        "final_program": final,
    }
    for spec in specs:
        session.run(spec, actions[spec.name])
    return ScheduleCompilation(
        session.module,
        session.analysis("final_program"),
        {
            "schema": "merlin.schedule_compilation.v1",
            "source_sha256": kernel.digest(),
            "schedule_sha256": session.module.digest(),
            "context": asdict(context),
            "library_sources_sha256": sources,
            "resources": resources,
            "passes": [asdict(record) for record in session.records],
            "repair_rounds": session.repair_round,
            "storage": session.analysis("storage"),
            "final_conformance": session.analysis("final_conformance"),
        },
    )
