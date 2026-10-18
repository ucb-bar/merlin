"""Execute checked schedule passes through final independently decoded bytes.

This entry supports finite straight-line byte-addressed programs. Target
encoding and semantic decoding remain selected, independently qualified
capabilities. Unknown instructions, hidden effects and counted loops retain
their separate lowering routes; opaque output is never interpreted here.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace

from merlin.sched.ir import Call, Kernel
from merlin.sched.ir.kernel import check_structure
from merlin.semantic_compiler.allocation_types import StorageBank
from merlin.transforms.pipeline_state import PassSpec, PipelineContext, PipelineSession, validate_pass_order

from .configuration import (
    ConfigurationAdapter,
    ConfigurationEffects,
    eliminate_redundant_configuration,
    verify_configuration_state,
)
from .fusion import plan_fusion, validate_fusion_capabilities
from .library_sources import shared_library_sources as shared_library_sources
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
from .selected_contracts import SelectedSchedulingContracts
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
    eliminate_configuration: bool = True
    optimize_storage: bool = True

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
        ):
            raise ValueError("schedule compilation requires explicit finite work, repair and solver budgets")


@dataclass(frozen=True)
class ScheduleCompilation:
    kernel: Kernel
    program: FinalProgram
    report: dict


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()




def _verify_kernel(kernel):
    if type(kernel) is not Kernel or check_structure(kernel):
        raise ValueError("schedule pipeline requires structurally valid typed kernel IR")


def _program_completions(effects):
    """Project exact waits onto the finalizer's qualified global-drain model.

    The storage checker remains responsible for token validity and hazards.
    A final decoder's drain cannot stand in for a wait for only some pending
    operations: that would permit an unsafe cut and storage reuse after launch.
    """
    pending, completions = set(), []
    for effect in effects:
        if effect.completion == "asynchronous":
            pending.add(effect.token)
            completions.append("asynchronous")
        elif effect.completion == "wait":
            if set(effect.waits) != pending:
                raise StorageOptimizationUnavailable(
                    "encoded schedule requires a wait for all pending tokens; "
                    "partial waits need a qualified final model"
                )
            pending.clear()
            completions.append("drain")
        else:
            completions.append(effect.completion)
    if pending:
        raise ValueError("encoded schedule returns before actual completion")
    return tuple(completions)


def _emission_effects(
    images, kernel, adapter, encoding, limits, max_calls, configuration_adapter=None, selected_contracts=None
):
    """Reconstruct complete effects from actual output, independently of requests."""
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
        for effect, instruction, completion in zip(effects, instructions, _program_completions(effects), strict=True):
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
    external_windows=None,
    private_ownership=False,
    spill_classes=(),
    continuation=None,
    pass_order=(),
    fusion_capabilities=(),
    configuration_adapter=None,
    selected_contracts=None,
    configuration_qualification_identity=None,
    fusion_emitters=None,
    fusion_qualification_identity=None,
    spill_emitters=None,
    spill_qualification_identity=None,
    continuation_qualification_identity=None,
) -> ScheduleCompilation:
    """Allocate, check, encode and finalize through the existing pass session.

    Missing capabilities or infeasible allocation diagnose for the caller's
    existing legal route. Never emit a partially verified candidate. Target
    execution qualification is separate from shared transformation correctness.
    """
    if type(budgets) is not ScheduleBudgets:
        raise ValueError("schedule compilation requires typed work and repair budgets")
    binding = None
    if selected_contracts is not None:
        if type(selected_contracts) is not SelectedSchedulingContracts:
            raise ValueError("schedule inputs require explicitly selected RTL/software contracts")
        selected_contracts.revalidate()
        if (
            type(adapter) is not StorageAdapter
            or adapter.configuration_identity != selected_contracts.hardware_identity
            or adapter.numerical_policy_identity != selected_contracts.numerical_policy
            or configuration_adapter is not None
        ):
            raise ValueError("schedule adapters disagree with the selected RTL/software contracts")
        if configuration_qualification_identity is not None:
            configuration_adapter = selected_contracts.configuration_adapter(configuration_qualification_identity)
        scopes = ["instruction_encoding", "configuration_fields"]
        if selected_contracts.storage is not None:
            banks, reservations, external_windows, adapter = selected_contracts.storage.bind(
                selected_contracts.isa, adapter, banks=banks, reservations=reservations,
                external_windows=external_windows, private_ownership=private_ownership,
            )
            scopes.extend(("temporary_memory_windows", "physical_aliases", "storage_effects_and_completion"))
        fusion_binding = None
        if selected_contracts.fusion is not None:
            if fusion_capabilities:
                raise ValueError("selected fusion declarations cannot be overridden by arbitrary capabilities")
            adapter, fusion_capabilities, fusion_binding = selected_contracts.fusion.bind(
                adapter, emitters=fusion_emitters, qualification_identity=fusion_qualification_identity,
                max_operations=budgets.max_scalar_operations, max_ir_bytes=budgets.max_scalar_ir_bytes,
            )
            scopes.extend(("scalar_computation_and_layouts", "physical_fusion_routes", "fusion_resource_bounds"))
        elif fusion_emitters is not None or fusion_qualification_identity is not None:
            raise ValueError("fusion emitters and qualification require a selected fusion contract")
        spill_binding = None
        if spill_classes:
            raise ValueError("selected device spill declarations cannot be overridden by arbitrary capabilities")
        if selected_contracts.spills is not None:
            spill_classes, spill_binding = selected_contracts.spills.bind(
                adapter, banks, emitters=spill_emitters, qualification_identity=spill_qualification_identity,
            )
            scopes.extend(("device_register_resources", "exact_spill_reload_routes"))
        elif spill_emitters is not None or spill_qualification_identity is not None:
            raise ValueError("device spill emitters and qualification require a selected spill contract")
        program_binding = None
        if selected_contracts.program is not None:
            if type(encoding) is not ScheduleEncoding:
                raise ValueError("selected final program requires typed schedule encoding")
            limits, decoder, continuation, program_binding = selected_contracts.program.bind(
                selected_contracts, limits, encoding.program_decoder, continuation,
                continuation_qualification_identity,
            )
            encoding = replace(encoding, program_decoder=decoder)
            scopes.extend(("final_program_limits", "physical_address_windows", "branch_encodings",
                           "launch_state_continuation"))
        elif continuation_qualification_identity is not None:
            raise ValueError("continuation qualification requires a selected program contract")
        binding = selected_contracts.record(scopes=tuple(scopes))
        if fusion_binding is not None:
            binding["fusion"] = fusion_binding
        if spill_binding is not None:
            binding["spills"] = spill_binding
        if program_binding is not None:
            binding["program"] = program_binding
        binding["configuration_pass"] = (
            "selected_qualified_declaration" if configuration_adapter is not None
            else "unavailable_missing_qualification"
        )
    elif configuration_qualification_identity is not None:
        raise ValueError("configuration qualification requires selected RTL/software contracts")
    elif fusion_emitters is not None or fusion_qualification_identity is not None:
        raise ValueError("fusion emitters and qualification require selected RTL/software contracts")
    elif spill_emitters is not None or spill_qualification_identity is not None:
        raise ValueError("device spill emitters and qualification require selected RTL/software contracts")
    elif continuation_qualification_identity is not None:
        raise ValueError("continuation qualification requires selected RTL/software program contracts")
    if external_windows is None:
        external_windows = ()
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
    if configuration_adapter is not None:
        if (
            type(configuration_adapter) is not ConfigurationAdapter
            or configuration_adapter.configuration_identity != adapter.configuration_identity
        ):
            raise ValueError("configuration decoder must match the selected hardware contract")
        configuration_spec = PassSpec(
            "configuration",
            1,
            requires=("source_effects",),
            produces=("configuration_candidate",),
            mutates_ir=False,
            ir_level="schedule",
        )
        specs = (
            specs[0],
            configuration_spec,
            *(
                replace(spec, requires=(*spec.requires, "configuration_candidate"))
                if spec.name in {"fusion", "storage"}
                else spec
                for spec in specs[1:]
            ),
        )
    if pass_order:
        names = {spec.name: spec for spec in specs}
        if type(pass_order) is not tuple or len(pass_order) != len(specs) or set(pass_order) != set(names):
            raise ValueError("explicit schedule order must contain every selected pass exactly once")
        specs = tuple(names[name] for name in pass_order)
    validate_pass_order(specs)
    _verify_kernel(kernel)
    source_sha256 = kernel.digest()
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
        "configuration_adapter": configuration_adapter.record() if configuration_adapter is not None else None,
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
                "selected_contracts": binding,
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
        _program_completions(effects)
        if configuration_adapter is not None:
            verify_configuration_state(kernel, configuration_adapter, max_calls=budgets.max_calls)
        return {"source_effects": effects}

    def fusion():
        return {
            "fusion_candidate": plan_fusion(
                configured_kernel(),
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
        configured = configured_kernel()
        configuration_report = (
            session.analysis("configuration_candidate").report if configuration_adapter is not None else None
        )

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
                optimize=budgets.optimize_storage,
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
            result = allocate(configured, objects)
        if result is None:
            unavailable = binding.get("spills", {}).get("unavailable_classes", {}) if binding else {}
            raise StorageOptimizationUnavailable(
                "no legal temporary allocation within the selected schedule repair family"
                + ("; unavailable device spills: " + json.dumps(unavailable, sort_keys=True) if unavailable else "")
            )
        session.module = result.kernel
        report = dict(result.report)
        if candidate is not None:
            report["fusion"] = {"status": fusion_status, "checked_steps": list(candidate.checks)}
        if configuration_report is not None:
            report["configuration"] = configuration_report
        return {"storage": report, "realization": report}

    def configured_kernel():
        return (
            session.analysis("configuration_candidate").kernel if configuration_adapter is not None else session.module
        )

    def configuration():
        # Pure setup must also be memory-free and synchronous according to the
        # independent storage decoder used through final actual-byte checks.
        for call, effect in zip(session.module.body, session.analysis("source_effects"), strict=True):
            config = configuration_adapter.project(call)
            if (
                config is not None
                and config.pure_write
                and (
                    effect.accesses
                    or effect.completion != "synchronous"
                    or effect.exact_copy
                    or effect.elementwise is not None
                )
            ):
                raise ValueError("pure configuration claim disagrees with decoded execution or memory effects")
        return {
            "configuration_candidate": eliminate_redundant_configuration(
                session.module,
                configuration_adapter,
                enabled=budgets.eliminate_configuration,
                max_calls=budgets.max_calls,
            )
        }

    def completion():
        effects = _project(session.module, adapter, budgets.max_calls)
        result = {"memory": _completion(effects, {bank.name: bank for bank in banks})}
        if configuration_adapter is not None:
            result["configuration"] = verify_configuration_state(
                session.module, configuration_adapter, max_calls=budgets.max_calls
            )
        return {"completion": result}

    def emission():
        image = encoding.encode(session.module)
        _emission_effects(
            (image,), session.module, adapter, encoding, limits, budgets.max_calls,
            configuration_adapter, selected_contracts,
        )
        return {"encoded": image}

    def final():
        # All external windows are published conservatively, including input
        # aliases and carried state. A caller cannot omit an output to permit
        # an otherwise illegal segmentation boundary.
        outputs = tuple(EncodedAccess(window.space, window.start, window.size) for window in external_windows)
        program = finalize_program(
            session.analysis("encoded"), limits, encoding.program_decoder, continuation=continuation, outputs=outputs,
            selected_contracts=(selected_contracts if selected_contracts is not None
                                and selected_contracts.program is not None else None),
        )
        effects = _emission_effects(
            program.segments, session.module, adapter, encoding, limits, budgets.max_calls,
            configuration_adapter, selected_contracts,
        )
        if len(program.segments) > 1 and configuration_adapter is not None:
            state_effects = tuple(configuration_adapter.project(call) for call in session.module.body)
            if any(type(effect) is not ConfigurationEffects or not effect.ordered for effect in state_effects):
                raise StorageOptimizationUnavailable(
                    "segmentation cannot preserve unknown configuration effects or ordering"
                )
            fields = {
                key for effect in state_effects for key in (*dict(effect.reads), *dict(effect.writes), *effect.clobbers)
            }
            if not fields <= set(continuation.preserved_configuration):
                raise StorageOptimizationUnavailable("segmentation lacks explicit configuration-register persistence")
        return {
            "final_program": program,
            "final_conformance": {
                "schedule_sha256": session.module.digest(),
                "image_sha256": hashlib.sha256(b"".join(program.segments)).hexdigest(),
                "actual_commands": len(effects),
                "requested_read_bytes": sum(a.size for effect in effects for a in effect.accesses if not a.writes),
                "requested_write_bytes": sum(a.size for effect in effects for a in effect.accesses if a.writes),
                "completion_model": ("explicit_global_drains" if any(
                    effect.completion == "asynchronous" for effect in effects) else "declared_synchronous"),
                "encoding_qualification_identity": encoding.qualification_identity,
                "machine_effects": "checked_under_selected_decoders",
                "execution_qualification": "not_established",
                "timing": "UNKNOWN",
            },
        }

    actions = {
        "semantic_admission": admission,
        "configuration": configuration,
        "fusion": fusion,
        "storage": storage,
        "completion": completion,
        "emission": emission,
        "final_program": final,
    }
    for spec in specs:
        if kernel.digest() != source_sha256:
            raise ValueError("source schedule changed during compilation; recompile")
        if selected_contracts is not None:
            selected_contracts.revalidate()
        if sources != shared_library_sources():
            raise ValueError("shared scheduling library changed during compilation; recompile")
        session.run(spec, actions[spec.name])
        if kernel.digest() != source_sha256:
            raise ValueError("source schedule changed during compilation; recompile")
        if sources != shared_library_sources():
            raise ValueError("shared scheduling library changed during compilation; recompile")
        if selected_contracts is not None:
            selected_contracts.revalidate()
    return ScheduleCompilation(
        session.module,
        session.analysis("final_program"),
        {
            "schema": "merlin.schedule_compilation.v1",
            "source_sha256": source_sha256,
            "schedule_sha256": session.module.digest(),
            "context": asdict(context),
            "library_sources_sha256": sources,
            "resources": resources,
            "selected_contracts": binding,
            "passes": [asdict(record) for record in session.records],
            "repair_rounds": session.repair_round,
            "storage": session.analysis("storage"),
            "final_conformance": session.analysis("final_conformance"),
        },
    )
