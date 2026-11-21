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
from dataclasses import asdict, replace
from pathlib import Path

from merlin.compiler.kernel_ir.ir import Kernel
from merlin.compiler.schedule.elementwise.fusion import validate_fusion_capabilities
from merlin.compiler.schedule.library_sources import shared_library_sources as shared_library_sources
from merlin.compiler.schedule.loops.configuration import (
    ConfigurationAdapter,
    verify_configuration_state,
)
from merlin.compiler.schedule.pipeline import actions as implementations
from merlin.compiler.schedule.pipeline.contracts import (
    CopyPipelineMotion,
    PostStorageRewrite,
    ScheduleBudgets,
    ScheduleCompilation,
    ScheduleEncoding,
    _emission_effects,
    _hash,
    _program_completions,
    _rewrite_snapshot,
    _verify_kernel,
)
from merlin.compiler.schedule.pipeline.program_limits import (
    AddressWindow,
    Continuation,
    ProgramLimits,
)
from merlin.compiler.schedule.pipeline.selected_contracts import SelectedSchedulingContracts, _parse
from merlin.compiler.schedule.storage.spills import validate_spill_classes
from merlin.compiler.schedule.storage.temporaries import (
    StorageAdapter,
    TemporaryObject,
    _check_reservations,
    _completion,
    _problem,
    _project,
)
from merlin.compiler.semantic.allocation_types import StorageBank
from merlin.compiler.transforms.pipeline_state import PassSpec, PipelineContext, PipelineSession, validate_pass_order

# ---- schedule_pipeline -------------------------------------------------------------------------


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
) -> ScheduleCompilation:
    """Allocate, check, encode and finalize through the existing pass session.

    Missing capabilities or infeasible allocation diagnose for the caller's
    existing legal route. Never emit a partially verified candidate. Target
    execution qualification is separate from shared transformation correctness.
    """
    if type(budgets) is not ScheduleBudgets:
        raise ValueError("schedule compilation requires typed work and repair budgets")
    copy_source = copy_objects = copy_positions = None
    copy_callbacks = None
    callbacks = ()
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
        return implementations.fusion(
            adapter=adapter,
            banks=banks,
            budgets=budgets,
            configured_kernel=configured_kernel,
            external_windows=external_windows,
            fusion_capabilities=fusion_capabilities,
            objects=objects,
            private_ownership=private_ownership,
        )

    def storage():
        nonlocal copy_source, copy_objects, copy_positions, copy_checked, copy_source_digest, copy_object_snapshot
        copy_source, copy_objects, copy_positions, copy_checked, copy_source_digest, copy_object_snapshot, result = (
            implementations.storage(
                adapter=adapter,
                banks=banks,
                binding=binding,
                budgets=budgets,
                configuration_adapter=configuration_adapter,
                configured_kernel=configured_kernel,
                copy_attempts=copy_attempts,
                copy_callbacks=copy_callbacks,
                copy_checked=copy_checked,
                copy_object_snapshot=copy_object_snapshot,
                copy_objects=copy_objects,
                copy_pipeline=copy_pipeline,
                copy_positions=copy_positions,
                copy_source=copy_source,
                copy_source_digest=copy_source_digest,
                external_windows=external_windows,
                fusion_capabilities=fusion_capabilities,
                model=model,
                objects=objects,
                post_storage=post_storage,
                private_ownership=private_ownership,
                require_copy_inputs=require_copy_inputs,
                revalidate_copy=revalidate_copy,
                session=session,
                specs=specs,
                spill_classes=spill_classes,
            )
        )
        return result

    def prefetch():
        return implementations.prefetch(
            adapter=adapter,
            banks=banks,
            budgets=budgets,
            configured_kernel=configured_kernel,
            external_windows=external_windows,
            fusion_capabilities=fusion_capabilities,
            objects=objects,
            prefetch_admission=prefetch_admission,
            private_ownership=private_ownership,
            session=session,
        )

    def configured_kernel():
        return (
            session.analysis("configuration_candidate").kernel if configuration_adapter is not None else session.module
        )

    def configuration():
        return implementations.configuration(
            budgets=budgets,
            configuration_adapter=configuration_adapter,
            session=session,
        )

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
        return implementations.final(
            adapter=adapter,
            banks=banks,
            budgets=budgets,
            callbacks=callbacks,
            configuration_adapter=configuration_adapter,
            continuation=continuation,
            copy_callbacks=copy_callbacks,
            copy_checked=copy_checked,
            copy_objects=copy_objects,
            copy_pipeline=copy_pipeline,
            copy_positions=copy_positions,
            copy_source=copy_source,
            encoding=encoding,
            external_windows=external_windows,
            limits=limits,
            post_storage=post_storage,
            private_ownership=private_ownership,
            require_copy_inputs=require_copy_inputs,
            require_rewrite_inputs=require_rewrite_inputs,
            rewrite_applied=rewrite_applied,
            rewrite_objects=rewrite_objects,
            rewrite_source=rewrite_source,
            selected_contracts=selected_contracts,
            session=session,
        )

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
