"""Execute checked schedule passes through final independently decoded bytes.

This entry supports finite straight-line byte-addressed programs. Target
encoding and semantic decoding remain selected, independently qualified
capabilities. Unknown instructions, hidden effects and counted loops retain
their separate lowering routes; opaque output is never interpreted here.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from pathlib import Path
from types import FunctionType, MethodType

from merlin.sched.ir import Call, Kernel
from merlin.sched.ir.kernel import check_structure
from merlin.semantic_compiler.allocation_types import StorageBank
from merlin.transforms.pipeline_state import PassSpec, PipelineContext, PipelineSession, validate_pass_order

from .configuration import (
    ConfigurationAdapter,
    ConfigurationEffects,
    _check_pure_configuration_storage,
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
from .selected_contracts import SelectedSchedulingContracts, _parse
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
    prefetch_private_copies,
    verify_temporary_storage,
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
    prefetch_qualification_identity=None,
    post_storage: PostStorageRewrite | None = None,
    copy_pipeline: CopyPipelineMotion | None = None,
    pass_validation=None,
) -> ScheduleCompilation:
    """Allocate, check, encode and finalize through the existing pass session.

    Missing capabilities or infeasible allocation diagnose for the caller's
    existing legal route. Never emit a partially verified candidate. Target
    execution qualification is separate from shared transformation correctness.
    """
    owner = sys.modules.get("merlin.verify.pass_validation")
    active_validation = owner.current() if owner is not None else None
    if pass_validation is not None and active_validation is not None and pass_validation is not active_validation:
        raise ValueError("schedule validation differs from active original compiler owner")
    if type(budgets) is not ScheduleBudgets:
        raise ValueError("schedule compilation requires typed work and repair budgets")
    copy_source = copy_objects = copy_positions = None
    copy_source_digest = copy_object_snapshot = None
    copy_checked = False
    copy_sources, copy_record = {}, None
    copy_effective_inputs = None
    copy_attempts = []
    if copy_pipeline is not None:
        if type(copy_pipeline) is not CopyPipelineMotion:
            raise ValueError("copy pipeline requires the original typed motion selection")
        copy_pipeline.__post_init__()
        motion = copy_pipeline
        copy_shared = shared_library_sources()
        copy_base_functions = (_program_completions, _emission_effects, _verify_kernel)
        copy_base_codes = tuple(f.__code__ for f in copy_base_functions)
        copy_callbacks = (motion.propose, motion.verify, motion.revalidate, motion.record)
        copy_codes = tuple(getattr(f, "__func__", f).__code__ for f in copy_callbacks)
        copy_paths = motion.sources
        if any(
            (path := inspect.getsourcefile(f)) is None or Path(path).resolve() not in {p.resolve() for p in copy_paths}
            for f in copy_callbacks
        ):
            raise ValueError("copy pipeline callbacks require their actual source owners")
        copy_sources = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in copy_paths}
        copy_inputs = (
            kernel,
            objects,
            banks,
            adapter,
            encoding,
            limits,
            budgets,
            reservations,
            external_windows,
            private_ownership,
            spill_classes,
            continuation,
            pass_order,
            fusion_capabilities,
            configuration_adapter,
            selected_contracts,
            configuration_qualification_identity,
            fusion_emitters,
            fusion_qualification_identity,
            spill_emitters,
            spill_qualification_identity,
            continuation_qualification_identity,
            prefetch_qualification_identity,
            post_storage,
        )
        copy_snapshot_reader, copy_snapshot_code = _rewrite_snapshot, _rewrite_snapshot.__code__
        copy_snapshot = copy_snapshot_reader(copy_inputs)

    def require_copy_inputs():
        if copy_pipeline is None:
            return
        if shared_library_sources() != copy_shared or any(
            f is not old or f.__code__ is not code
            for f, old, code in zip(
                (_program_completions, _emission_effects, _verify_kernel),
                copy_base_functions,
                copy_base_codes,
                strict=True,
            )
        ):
            raise ValueError("original copy pipeline shared proof implementation changed")
        if _rewrite_snapshot is not copy_snapshot_reader or copy_snapshot_reader.__code__ is not copy_snapshot_code:
            raise ValueError("original copy pipeline snapshot reader changed")
        if (
            copy_pipeline is not motion
            or motion.sources != copy_paths
            or copy_snapshot_reader(copy_inputs) != copy_snapshot
            or {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in copy_paths} != copy_sources
        ):
            raise ValueError("original copy pipeline source, inputs or resources changed")
        if (
            copy_effective_inputs is not None
            and copy_snapshot_reader((adapter, encoding, banks, reservations, external_windows, limits))
            != copy_effective_inputs
        ):
            raise ValueError("original copy pipeline bound adapter or decoder changed")
        current = (motion.propose, motion.verify, motion.revalidate, motion.record)
        if any(
            f is not old or getattr(f, "__func__", f).__code__ is not code
            for f, old, code in zip(current, copy_callbacks, copy_codes, strict=True)
        ):
            raise ValueError("original copy pipeline callback changed")
        for attempted, digest, slots, snapshot in copy_attempts:
            if attempted.digest() != digest or tuple(asdict(obj) for obj in slots) != snapshot:
                raise ValueError("copy pipeline allocated source or slots changed")
        if copy_source is not None and (
            copy_source.digest() != copy_source_digest
            or tuple(asdict(obj) for obj in copy_objects) != copy_object_snapshot
        ):
            raise ValueError("checked copy pipeline allocation changed")

    def revalidate_copy():
        require_copy_inputs()
        if copy_pipeline is not None:
            copy_callbacks[2]()
            require_copy_inputs()
            record = json.dumps(copy_callbacks[3](), sort_keys=True, allow_nan=False)
            require_copy_inputs()
            if copy_record is not None and record != copy_record:
                raise ValueError("original copy pipeline admission or policy changed")
            return record
        return None

    copy_record = revalidate_copy()
    if copy_record is not None and len(copy_record.encode()) > 1 << 20:
        raise ValueError("copy pipeline owner record exceeds its bounded byte budget")
    rewrite_source = rewrite_objects = rewrite_digest = rewrite_object_record = None
    rewrite_candidate = rewrite_candidate_digest = None
    rewrite_applied = False
    rewrite_sources = {}
    if post_storage is not None:
        if type(post_storage) is not PostStorageRewrite or type(adapter) is not StorageAdapter:
            raise ValueError("post-storage rewrite requires the original typed host selection and adapter")
        if (
            type(kernel) is not Kernel
            or type(kernel.body) is not tuple
            or len(kernel.body) > budgets.max_calls
            or type(objects) is not tuple
            or len(objects) > budgets.max_objects
            or any(type(obj) is not TemporaryObject for obj in objects)
            or banks is not None
            and (type(banks) is not tuple or len(banks) > budgets.max_calls)
            or any(
                value is not None and (type(value) is not tuple or len(value) > budgets.max_calls)
                for value in (reservations, external_windows, spill_classes, fusion_capabilities, pass_order)
            )
        ):
            raise ValueError("post-storage original source or declarations exceed typed work budgets")
        _verify_kernel(kernel)
        post_storage.__post_init__()
        original_slot = post_storage
        callbacks = (post_storage.propose, post_storage.verify, post_storage.revalidate, post_storage.record)
        callback_snapshot = tuple((f, getattr(getattr(f, "__func__", f), "__code__", None)) for f in callbacks)
        original_name, original_paths = post_storage.name, post_storage.sources
        pinned_paths = {path.resolve() for path in original_paths}
        if any(
            (owner := inspect.getsourcefile(f)) is None or Path(owner).resolve() not in pinned_paths for f in callbacks
        ):
            raise ValueError("post-storage callbacks require their actual Python source owners in pinned sources")
        rewrite_sources = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in original_paths}
        rewrite_shared = shared_library_sources()
        original_adapter = adapter
        original_adapter_state = (
            adapter.configuration_identity,
            adapter.numerical_policy_identity,
            adapter.qualification_identity,
        )
        original_adapter_callbacks = (adapter.project, adapter.rewrite)
        original_adapter_codes = tuple(
            getattr(getattr(f, "__func__", f), "__code__", None) for f in original_adapter_callbacks
        )
        snapshot_reader, snapshot_code = _rewrite_snapshot, _rewrite_snapshot.__code__
        # Retain raw references before selected contract binding can legitimately
        # replace adapters/windows. No owner runs before this value snapshot.
        original_inputs = (
            kernel,
            objects,
            banks,
            encoding,
            limits,
            budgets,
            reservations,
            external_windows,
            private_ownership,
            spill_classes,
            continuation,
            pass_order,
            fusion_capabilities,
            configuration_adapter,
            selected_contracts,
            configuration_qualification_identity,
            fusion_emitters,
            fusion_qualification_identity,
            spill_emitters,
            spill_qualification_identity,
            continuation_qualification_identity,
            prefetch_qualification_identity,
        )
        original_input_snapshot = snapshot_reader(original_inputs)

    def require_rewrite_inputs():
        if post_storage is None:
            return
        if _rewrite_snapshot is not snapshot_reader or snapshot_reader.__code__ is not snapshot_code:
            raise ValueError("original post-storage snapshot reader changed")
        if snapshot_reader(original_inputs) != original_input_snapshot:
            raise ValueError("original post-storage source, resources or controls changed")
        if (
            post_storage is not original_slot
            or post_storage.name != original_name
            or post_storage.sources != original_paths
            or {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in original_paths} != rewrite_sources
            or shared_library_sources() != rewrite_shared
        ):
            raise ValueError("original post-storage selection, sources or pass specification changed")
        current = (post_storage.propose, post_storage.verify, post_storage.revalidate, post_storage.record)
        if any(
            f is not original or getattr(getattr(f, "__func__", f), "__code__", None) is not code
            for f, (original, code) in zip(current, callback_snapshot, strict=True)
        ):
            raise ValueError("original post-storage callback changed")
        for selected, identities, functions, codes in (
            (original_adapter, original_adapter_state, original_adapter_callbacks, original_adapter_codes),
            (adapter, effective_adapter_state, effective_adapter_callbacks, effective_adapter_codes),
        ):
            if (
                selected.configuration_identity,
                selected.numerical_policy_identity,
                selected.qualification_identity,
            ) != identities or any(
                f is not original or getattr(getattr(f, "__func__", f), "__code__", None) is not code
                for f, original, code in zip((selected.project, selected.rewrite), functions, codes, strict=True)
            ):
                raise ValueError("original post-storage adapter identities or callbacks changed")
        if rewrite_source is not None and (
            rewrite_source.digest() != rewrite_digest
            or tuple(asdict(obj) for obj in rewrite_objects) != rewrite_object_record
        ):
            raise ValueError("original post-storage source or lifetimes changed")
        if rewrite_candidate is not None and rewrite_candidate.digest() != rewrite_candidate_digest:
            raise ValueError("verified post-storage candidate changed")

    def revalidate_rewrite():
        require_rewrite_inputs()
        if post_storage is not None:
            callbacks[2]()
            require_rewrite_inputs()
            current_record = json.dumps(callbacks[3](), sort_keys=True, allow_nan=False)
            require_rewrite_inputs()
            if current_record != rewrite_record:
                raise ValueError("original post-storage record changed")

    effective_adapter_state = original_adapter_state if post_storage is not None else ()
    effective_adapter_callbacks = original_adapter_callbacks if post_storage is not None else ()
    effective_adapter_codes = original_adapter_codes if post_storage is not None else ()
    if post_storage is not None:
        require_rewrite_inputs()
        rewrite_record = json.dumps(callbacks[3](), sort_keys=True, allow_nan=False)
        require_rewrite_inputs()
        if len(rewrite_record.encode()) > 1 << 20:
            raise ValueError("post-storage original record exceeds its byte budget")
    revalidate_rewrite()
    binding, prefetch_admission = None, None
    original_prefetch = None
    if selected_contracts is not None:
        if type(selected_contracts) is not SelectedSchedulingContracts:
            raise ValueError("schedule inputs require explicitly selected RTL/software contracts")
        selected_contracts.revalidate()
        if prefetch_qualification_identity is not None:
            # Parsed objects are not authority: compare to the original frozen
            # files before binding and retain immutable bytes across callbacks.
            original_storage = _parse(
                selected_contracts.facts_bytes, selected_contracts.software_bytes, selected_contracts.target
            )[5]
            original_prefetch = original_storage.prefetch if original_storage is not None else None
            current = selected_contracts.storage.prefetch if selected_contracts.storage is not None else None
            if current != original_prefetch:
                raise ValueError("selected prefetch contract differs from its original software/RTL declaration")
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
                selected_contracts.isa,
                adapter,
                banks=banks,
                reservations=reservations,
                external_windows=external_windows,
                private_ownership=private_ownership,
                configuration_adapter=configuration_adapter,
            )
            scopes.extend(("temporary_memory_windows", "physical_aliases", "storage_effects_and_completion"))
            prefetch_admission = selected_contracts.storage.bind_prefetch(prefetch_qualification_identity)
            if prefetch_admission is not None and prefetch_admission.available:
                scopes.append("qualified_async_copy_overlap")
        elif prefetch_qualification_identity is not None:
            raise ValueError("prefetch qualification requires selected storage declarations")
        fusion_binding = None
        if selected_contracts.fusion is not None:
            if fusion_capabilities:
                raise ValueError("selected fusion declarations cannot be overridden by arbitrary capabilities")
            adapter, fusion_capabilities, fusion_binding = selected_contracts.fusion.bind(
                adapter,
                emitters=fusion_emitters,
                qualification_identity=fusion_qualification_identity,
                max_operations=budgets.max_scalar_operations,
                max_ir_bytes=budgets.max_scalar_ir_bytes,
            )
            scopes.extend(("scalar_computation_and_layouts", "physical_fusion_routes", "fusion_resource_bounds"))
        elif fusion_emitters is not None or fusion_qualification_identity is not None:
            raise ValueError("fusion emitters and qualification require a selected fusion contract")
        spill_binding = None
        if spill_classes:
            raise ValueError("selected device spill declarations cannot be overridden by arbitrary capabilities")
        if selected_contracts.spills is not None:
            spill_classes, spill_binding = selected_contracts.spills.bind(
                adapter,
                banks,
                emitters=spill_emitters,
                qualification_identity=spill_qualification_identity,
            )
            scopes.extend(("device_register_resources", "exact_spill_reload_routes"))
        elif spill_emitters is not None or spill_qualification_identity is not None:
            raise ValueError("device spill emitters and qualification require a selected spill contract")
        program_binding = None
        if selected_contracts.program is not None:
            if type(encoding) is not ScheduleEncoding:
                raise ValueError("selected final program requires typed schedule encoding")
            limits, decoder, continuation, program_binding = selected_contracts.program.bind(
                selected_contracts,
                limits,
                encoding.program_decoder,
                continuation,
                continuation_qualification_identity,
            )
            encoding = replace(encoding, program_decoder=decoder)
            scopes.extend(
                ("final_program_limits", "physical_address_windows", "branch_encodings", "launch_state_continuation")
            )
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
            "selected_qualified_declaration"
            if configuration_adapter is not None
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
    elif prefetch_qualification_identity is not None:
        raise ValueError("prefetch qualification requires selected RTL/software storage contracts")
    effective_adapter_state = (
        (adapter.configuration_identity, adapter.numerical_policy_identity, adapter.qualification_identity)
        if post_storage is not None
        else ()
    )
    effective_adapter_callbacks = (adapter.project, adapter.rewrite) if post_storage is not None else ()
    effective_adapter_codes = tuple(
        getattr(getattr(f, "__func__", f), "__code__", None) for f in effective_adapter_callbacks
    )
    revalidate_rewrite()
    if external_windows is None:
        external_windows = ()
    if copy_pipeline is not None:
        copy_effective_inputs = copy_snapshot_reader((adapter, encoding, banks, reservations, external_windows, limits))
        revalidate_copy()
    prefetch_snapshot = _hash(asdict(prefetch_admission) if prefetch_admission is not None else None)
    original_prefetch_snapshot = _hash(asdict(original_prefetch) if original_prefetch is not None else None)

    def revalidate_prefetch():
        if _hash(asdict(prefetch_admission) if prefetch_admission is not None else None) != prefetch_snapshot:
            raise ValueError("prefetch overlap qualification or resources changed during compilation")
        if selected_contracts is not None and prefetch_qualification_identity is not None:
            current = selected_contracts.storage.prefetch if selected_contracts.storage is not None else None
            if _hash(asdict(current) if current is not None else None) != original_prefetch_snapshot:
                raise ValueError("selected prefetch overlap declaration changed during compilation")

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
    # Missing original capability/qualification preserves the legacy roster.
    if prefetch_admission is not None and prefetch_admission.available and budgets.prefetch_distance:
        requires = ("source_effects",)
        if configuration_adapter is not None:
            requires += ("configuration_candidate",)
        if fusion_capabilities:
            requires += ("fusion_candidate",)
        prefetch_spec = PassSpec(
            "prefetch", 2, requires=requires, produces=("prefetch_candidate",), mutates_ir=False, ir_level="schedule"
        )
        insertion = next(i for i, spec in enumerate(specs) if spec.name == "storage")
        specs = (
            *specs[:insertion],
            prefetch_spec,
            replace(specs[insertion], requires=(*specs[insertion].requires, "prefetch_candidate")),
            *specs[insertion + 1 :],
        )
    if post_storage is not None:
        insertion = next(i for i, spec in enumerate(specs) if spec.name == "completion")
        # Fixed host ordering, prerequisites and invalidation; callers supply no
        # arbitrary PassSpec or action map.
        spec = PassSpec(
            post_storage.name,
            4,
            requires=("storage", "realization"),
            produces=("storage", "realization", post_storage.name),
            invalidates=("*",),
            ir_level="schedule",
        )
        specs = (
            *specs[:insertion],
            spec,
            replace(specs[insertion], requires=(*specs[insertion].requires, post_storage.name)),
            *specs[insertion + 1 :],
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

    def current_resources():
        # Typed descriptors may be shared with provider callbacks. Freezing a
        # report once must not let later object changes inherit its analyses.
        return {
            **(
                {
                    "post_storage_rewrite": {
                        "record": json.loads(rewrite_record),
                        "name": original_name,
                        "implementation_sources": rewrite_sources,
                    }
                }
                if post_storage is not None
                else {}
            ),
            "objects": [asdict(obj) for obj in objects],
            "banks": [bank.record() for bank in banks],
            "reservations": [asdict(row) for row in reservations],
            "external_windows": [asdict(w) for w in external_windows],
            "private_ownership": private_ownership,
            "limits": asdict(limits),
            "budgets": asdict(budgets),
            "fusion_capabilities": [capability.record() for capability in fusion_capabilities],
            "configuration_adapter": configuration_adapter.record() if configuration_adapter is not None else None,
            "prefetch_admission": asdict(prefetch_admission) if prefetch_admission is not None else None,
            **(
                {"copy_pipeline": json.loads(copy_record), "copy_pipeline_sources": copy_sources}
                if copy_pipeline is not None
                else {}
            ),
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

    resources = current_resources()

    def revalidate_resources():
        if current_resources() != resources:
            raise ValueError("schedule resources or budgets changed during compilation; replan")

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
                "prefetch_qualification": prefetch_qualification_identity,
                **({"post_storage_rewrite": json.loads(rewrite_record)} if post_storage is not None else {}),
            }
        ),
        library=_hash(
            {
                "shared": sources,
                "post_storage": rewrite_sources,
                **({"copy_pipeline": copy_sources} if copy_pipeline is not None else {}),
            }
        )
        if post_storage is not None or copy_pipeline is not None
        else _hash(sources),
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
        nonlocal copy_source, copy_objects, copy_positions, copy_checked, copy_source_digest, copy_object_snapshot
        configured = configured_kernel()
        configuration_report = (
            session.analysis("configuration_candidate").report if configuration_adapter is not None else None
        )

        allocated_source = allocated_objects = None

        def allocate(source, source_objects):
            nonlocal allocated_source, allocated_objects
            result = allocate_temporary_storage(
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
            if result is not None:
                allocated_source, allocated_objects = source, source_objects
            return result

        candidate = session.analysis("fusion_candidate") if fusion_capabilities else None
        source = candidate.kernel if candidate is not None else configured
        source_objects = candidate.objects if candidate is not None else objects
        prefetch_report = {"status": "unchanged", "reason": "disabled_or_missing_qualified_overlap", "moves": []}
        prefetch_selected = any(spec.name == "prefetch" for spec in specs)
        if prefetch_selected:
            reordered, prefetch_report = session.analysis("prefetch_candidate")
            if prefetch_report["source_sha256"] != source.digest():
                raise ValueError("prefetch candidate belongs to a different source composition")
            result = allocate(reordered, source_objects)
            if result is None and reordered != source:
                session.repair(stage=3, reason="prefetch allocation failed; restore original issue order and replan")
                prefetch_report = {**prefetch_report, "status": "allocation_fallback"}
                result = allocate(source, source_objects)
        else:
            result = allocate(source, source_objects) if candidate is not None else None
        fusion_status = "realized"
        if candidate is not None and result is None:
            session.repair(stage=3, reason="fusion allocation failed; restore original composition and replan")
            fusion_status = "allocation_fallback"
        if result is None:
            result = allocate(configured, objects)
            if candidate is not None and prefetch_selected:
                prefetch_report = {**prefetch_report, "status": "composition_fallback"}
        if result is None:
            unavailable = binding.get("spills", {}).get("unavailable_classes", {}) if binding else {}
            raise StorageOptimizationUnavailable(
                "no legal temporary allocation within the selected schedule repair family"
                + ("; unavailable device spills: " + json.dumps(unavailable, sort_keys=True) if unavailable else "")
            )
        session.module = result.kernel
        report = dict(result.report)
        if copy_pipeline is not None:
            copy_report = {"status": "unchanged", "reason": "unsupported_post_storage_composition"}
            if post_storage is None:
                bank_map = {bank.name: bank for bank in banks}

                def attempt(allocated):
                    placed = tuple(
                        replace(TemporaryObject(**row), source_base=allocated.report["addresses"][row["name"]])
                        for row in allocated.report["objects"]
                    )
                    if any(
                        bank_map[a.bank].backing == bank_map[b.bank].backing
                        and a.source_base < b.source_base + b.size
                        and b.source_base < a.source_base + a.size
                        for i, a in enumerate(placed)
                        for b in placed[:i]
                    ):
                        return None
                    allocated_digest = allocated.kernel.digest()
                    slot_snapshot = tuple(asdict(obj) for obj in placed)
                    copy_attempts.append((allocated.kernel, allocated_digest, placed, slot_snapshot))

                    def require_allocated():
                        require_copy_inputs()
                        if (
                            allocated.kernel.digest() != allocated_digest
                            or tuple(asdict(obj) for obj in placed) != slot_snapshot
                        ):
                            raise ValueError("copy pipeline allocated source or slots changed")

                    require_allocated()
                    proposal = copy_callbacks[0](
                        allocated.kernel, placed, banks, adapter, external_windows, private_ownership
                    )
                    require_allocated()
                    if proposal is None:
                        return None
                    actual, positions = proposal
                    _verify_kernel(actual)
                    if (
                        type(positions) is not tuple
                        or any(type(i) is not int for i in positions)
                        or sorted(positions) != list(range(len(allocated.kernel.body)))
                        or replace(actual, body=allocated.kernel.body) != allocated.kernel
                        or actual.body != tuple(allocated.kernel.body[i] for i in positions)
                    ):
                        raise ValueError("copy pipeline changed exact original call membership or boundary")
                    actual_digest = actual.digest()
                    proof = copy_callbacks[1](
                        allocated.kernel, actual, positions, placed, banks, adapter, external_windows, private_ownership
                    )
                    require_allocated()
                    checked_storage = verify_temporary_storage(actual, actual, placed, banks, adapter, **model)
                    require_allocated()
                    if actual.digest() != actual_digest:
                        raise ValueError("copy pipeline realized candidate changed during verification")
                    return actual, positions, placed, {**proof, "storage": checked_storage}

                selected_copy = attempt(result)
                # One existing repair alternative: original slots must themselves
                # pass the ordinary allocation/checker, never invent new buffers.
                if (
                    (selected_copy is None or selected_copy[0] == result.kernel)
                    and budgets.optimize_storage
                    and session.repair_round < budgets.max_repairs
                ):
                    try:
                        fixed = allocate_temporary_storage(
                            allocated_source,
                            allocated_objects,
                            banks,
                            adapter,
                            timeout_ms=budgets.solver_timeout_ms,
                            spill_classes=(),
                            max_spills=0,
                            eliminate_copies=False,
                            optimize=False,
                            **model,
                        )
                    except (StorageOptimizationUnavailable, ValueError):
                        revalidate_copy()
                        fixed = None
                    if fixed is not None:
                        alternative = attempt(fixed)
                        if alternative is not None and alternative[0] != fixed.kernel:
                            session.repair(stage=3, reason="copy pipeline preserves checked original private slots")
                            result, selected_copy = fixed, alternative
                            report = {**dict(fixed.report), "before_copy_slot_repair": report}
                if selected_copy is None:
                    copy_report = {"status": "unchanged", "reason": "unsupported_fixed_slots_or_copy_effects"}
                else:
                    actual, copy_positions, copy_objects, copy_report = selected_copy
                    copy_source, copy_source_digest = result.kernel, result.kernel.digest()
                    copy_object_snapshot = tuple(asdict(obj) for obj in copy_objects)
                    copy_checked = True
                    session.module = actual
                    checked_storage = copy_report["storage"]
                    if checked_storage["addresses"] != result.report["addresses"]:
                        raise ValueError("copy pipeline changed its checked physical placement")
                    report = {**report, **checked_storage, "before_copy_pipeline": result.report}
            report["copy_pipeline"] = copy_report
        if candidate is not None:
            report["fusion"] = {"status": fusion_status, "checked_steps": list(candidate.checks)}
        if configuration_report is not None:
            report["configuration"] = configuration_report
        report["prefetch"] = prefetch_report
        return {"storage": report, "realization": report}

    def prefetch():
        candidate = session.analysis("fusion_candidate") if fusion_capabilities else None
        source = candidate.kernel if candidate is not None else configured_kernel()
        source_objects = candidate.objects if candidate is not None else objects
        reordered, report = prefetch_private_copies(
            source,
            source_objects,
            banks,
            adapter,
            prefetch_admission,
            distance=budgets.prefetch_distance,
            max_moves=budgets.max_prefetch_moves,
            external_windows=external_windows,
            private_ownership=private_ownership,
            max_calls=budgets.max_calls,
            max_objects=budgets.max_objects,
        )
        return {"prefetch_candidate": (reordered, {"source_sha256": source.digest(), **report})}

    def configured_kernel():
        return (
            session.analysis("configuration_candidate").kernel if configuration_adapter is not None else session.module
        )

    def configuration():
        validation = pass_validation
        if validation is None:
            owner = sys.modules.get("merlin.verify.pass_validation")
            validation = owner.current() if owner is not None else None
        checking = validation is not None and "configuration_write_elision" in validation.passes
        original = session.module
        original_text = original.text() if checking else None
        _check_pure_configuration_storage(
            session.module.body, session.analysis("source_effects"), configuration_adapter
        )
        result = eliminate_redundant_configuration(
            session.module,
            configuration_adapter,
            enabled=budgets.eliminate_configuration,
            max_calls=budgets.max_calls,
        )
        if checking:
            if original.text() != original_text:
                raise ValueError("configuration producer changed the original Kernel before proof")
            from merlin.verify.compiler_configuration import validated_kernel_result

            result = validated_kernel_result(
                selected_contracts,
                configuration_qualification_identity,
                original,
                result,
                validation,
                max_calls=budgets.max_calls,
                enabled=budgets.eliminate_configuration,
            )
            if original.text() != original_text:
                raise ValueError("configuration original Kernel changed during proof")
        return {"configuration_candidate": result}

    def rewrite_storage():
        nonlocal rewrite_source, rewrite_objects, rewrite_digest, rewrite_object_record
        nonlocal rewrite_candidate, rewrite_candidate_digest, rewrite_applied
        previous = session.analysis("storage")
        placed = tuple(
            replace(TemporaryObject(**row), source_base=previous["addresses"][row["name"]])
            for row in previous["objects"]
        )
        bank_map = {bank.name: bank for bank in banks}
        overlaps = any(
            bank_map[a.bank].backing == bank_map[b.bank].backing
            and a.source_base < b.source_base + b.size
            and b.source_base < a.source_base + a.size
            for i, a in enumerate(placed)
            for b in placed[:i]
        )
        proof = {"status": "unchanged", "reason": "unsupported_overlapping_physical_leases", "deleted_calls": 0}
        result = None
        if not overlaps:
            # Retain the checked allocation BEFORE the optimizer can see it.
            rewrite_source, rewrite_objects = session.module, placed
            rewrite_digest, rewrite_object_record = rewrite_source.digest(), tuple(asdict(obj) for obj in placed)
            require_rewrite_inputs()
            result = callbacks[0](rewrite_source, placed, banks, adapter, external_windows, private_ownership)
            require_rewrite_inputs()
            if result is None:
                proof = {"status": "unchanged", "reason": "disabled_or_unavailable_deletion_proof", "deleted_calls": 0}
        report = dict(previous)
        if result is not None:
            _verify_kernel(result)
            if replace(result, body=session.module.body) != session.module or len(result.body) >= len(
                session.module.body
            ):
                raise ValueError("post-storage selection must only remove original instructions")
            remaining = iter(rewrite_source.body)
            if any(not any(call == original for original in remaining) for call in result.body):
                raise ValueError("post-storage selection must retain unchanged original instructions in order")
            rewrite_candidate, rewrite_candidate_digest = result, result.digest()
            proof = {
                "status": "applied",
                **callbacks[1](rewrite_source, result, placed, banks, adapter, external_windows, private_ownership),
            }
            require_rewrite_inputs()
            rewrite_applied = True
            session.module = result
            effects = _project(session.module, adapter, budgets.max_calls)
            require_rewrite_inputs()
            lifetimes = {}
            for index, effect in enumerate(effects):
                for access in effect.accesses:
                    if access.space not in bank_map:
                        continue
                    matches = [
                        obj
                        for obj in placed
                        if bank_map[obj.bank].backing == bank_map[access.space].backing
                        and obj.source_base <= access.start
                        and access.start + access.size <= obj.source_base + obj.size
                    ]
                    if len(matches) != 1:
                        raise ValueError("post-storage rewrite lost its original checked physical allocation")
                    lifetimes.setdefault(matches[0].name, [index, index])[1] = index
            report = {
                **report,
                "before_post_storage": previous,
                "realized_sha256": session.module.digest(),
                "lifetime_calls": lifetimes,
                "static_commands": len(effects),
                "requested_read_bytes": sum(a.size for e in effects for a in e.accesses if not a.writes),
                "requested_write_bytes": sum(a.size for e in effects for a in e.accesses if a.writes),
                "allocation_after_post_storage": "original checked assignment retained; only lifetimes shortened",
            }
        report["post_storage_rewrite"] = {"name": post_storage.name, "proof": proof}
        return {"storage": report, "realization": report, post_storage.name: proof}

    def completion():
        effects = _project(session.module, adapter, budgets.max_calls)
        _program_completions(effects, checked_copy_motion=copy_pipeline if copy_checked else None)
        result = {"memory": _completion(effects, {bank.name: bank for bank in banks})}
        if configuration_adapter is not None:
            result["configuration"] = verify_configuration_state(
                session.module, configuration_adapter, max_calls=budgets.max_calls
            )
        return {"completion": result}

    def emission():
        image = encoding.encode(session.module)
        _emission_effects(
            (image,),
            session.module,
            adapter,
            encoding,
            limits,
            budgets.max_calls,
            configuration_adapter,
            selected_contracts,
            checked_copy_motion=copy_pipeline if copy_checked else None,
        )
        return {"encoded": image}

    def final():
        # All external windows are published conservatively, including input
        # aliases and carried state. A caller cannot omit an output to permit
        # an otherwise illegal segmentation boundary.
        outputs = tuple(EncodedAccess(window.space, window.start, window.size) for window in external_windows)
        program = finalize_program(
            session.analysis("encoded"),
            limits,
            encoding.program_decoder,
            continuation=continuation,
            outputs=outputs,
            selected_contracts=(
                selected_contracts
                if selected_contracts is not None and selected_contracts.program is not None
                else None
            ),
        )
        effects = _emission_effects(
            program.segments,
            session.module,
            adapter,
            encoding,
            limits,
            budgets.max_calls,
            configuration_adapter,
            selected_contracts,
            checked_copy_motion=copy_pipeline if copy_checked else None,
        )
        actual_rewrite = None
        if rewrite_applied:
            require_rewrite_inputs()
            actual_calls = tuple(call for image in program.segments for call in encoding.decode_calls(image))
            require_rewrite_inputs()
            actual_rewrite = callbacks[1](
                rewrite_source,
                replace(session.module, body=actual_calls),
                rewrite_objects,
                banks,
                adapter,
                external_windows,
                private_ownership,
            )
            require_rewrite_inputs()
        actual_copy = None
        if copy_checked:
            require_copy_inputs()
            actual_calls = tuple(call for image in program.segments for call in encoding.decode_calls(image))
            require_copy_inputs()
            actual_copy = copy_callbacks[1](
                copy_source,
                replace(session.module, body=actual_calls),
                copy_positions,
                copy_objects,
                banks,
                adapter,
                external_windows,
                private_ownership,
            )
            require_copy_inputs()
        prefetch_report = session.analysis("storage")["prefetch"]
        observed_prefetch = []
        if prefetch_report["status"] == "applied":
            for move in prefetch_report["moves"]:
                positions = [(i, effect) for i, effect in enumerate(effects) if effect.token == move["token"]]
                if len(positions) != 1:
                    raise ValueError("actual final bytes lost the selected prefetch launch")
                index, effect = positions[0]
                observed_prefetch.append(
                    {
                        "actual_instruction_index": index,
                        "token": effect.token,
                        "accesses": [asdict(access) for access in effect.accesses],
                    }
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
                "completion_model": (
                    "explicit_global_drains"
                    if any(effect.completion == "asynchronous" for effect in effects)
                    else "declared_synchronous"
                ),
                "encoding_qualification_identity": encoding.qualification_identity,
                "machine_effects": "checked_under_selected_decoders",
                "execution_qualification": "not_established",
                "timing": "UNKNOWN",
                "prefetch": {"status": prefetch_report["status"], "actual_launches": observed_prefetch},
                **({"copy_pipeline": actual_copy} if copy_pipeline is not None else {}),
                **(
                    {"post_storage_rewrite": {"name": post_storage.name, "proof": actual_rewrite}}
                    if post_storage is not None
                    else {}
                ),
            },
        }

    actions = {
        "semantic_admission": admission,
        "configuration": configuration,
        "fusion": fusion,
        "prefetch": prefetch,
        "storage": storage,
        "completion": completion,
        "emission": emission,
        "final_program": final,
    }
    if post_storage is not None:
        actions[post_storage.name] = rewrite_storage
    for spec in specs:
        revalidate_copy()
        revalidate_rewrite()
        revalidate_resources()
        revalidate_prefetch()
        if kernel.digest() != source_sha256:
            raise ValueError("source schedule changed during compilation; recompile")
        if selected_contracts is not None:
            selected_contracts.revalidate()
        if sources != shared_library_sources():
            raise ValueError("shared scheduling library changed during compilation; recompile")
        session.run(spec, actions[spec.name])
        revalidate_copy()
        revalidate_rewrite()
        revalidate_resources()
        revalidate_prefetch()
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
            **({"conditional_sources_sha256": rewrite_sources} if post_storage is not None else {}),
            "resources": resources,
            "selected_contracts": binding,
            "passes": [asdict(record) for record in session.records],
            "repair_rounds": session.repair_round,
            "storage": session.analysis("storage"),
            "final_conformance": session.analysis("final_conformance"),
        },
    )
