"""Storage, fusion, prefetch, configuration and final emission pass actions.

The pipeline retains selection and source-admission lifetimes. Each action receives
its actual dependencies explicitly and returns any updated closure state.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace

from merlin.compiler.schedule.elementwise.fusion import plan_fusion
from merlin.compiler.schedule.library_sources import shared_library_sources as shared_library_sources
from merlin.compiler.schedule.loops.configuration import (
    ConfigurationEffects,
    eliminate_redundant_configuration,
)
from merlin.compiler.schedule.pipeline.contracts import _emission_effects, _verify_kernel
from merlin.compiler.schedule.pipeline.program_limits import (
    EncodedAccess,
    finalize_program,
)
from merlin.compiler.schedule.storage.temporaries import (
    StorageOptimizationUnavailable,
    TemporaryObject,
    allocate_temporary_storage,
    prefetch_private_copies,
    verify_temporary_storage,
)


def fusion(
    *,
    adapter,
    banks,
    budgets,
    configured_kernel,
    external_windows,
    fusion_capabilities,
    objects,
    private_ownership,
):
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


def storage(
    *,
    adapter,
    banks,
    binding,
    budgets,
    configuration_adapter,
    configured_kernel,
    copy_attempts,
    copy_callbacks,
    copy_checked,
    copy_object_snapshot,
    copy_objects,
    copy_pipeline,
    copy_positions,
    copy_source,
    copy_source_digest,
    external_windows,
    fusion_capabilities,
    model,
    objects,
    post_storage,
    private_ownership,
    require_copy_inputs,
    revalidate_copy,
    session,
    specs,
    spill_classes,
):
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
    return (
        copy_source,
        copy_objects,
        copy_positions,
        copy_checked,
        copy_source_digest,
        copy_object_snapshot,
        {"storage": report, "realization": report},
    )


def prefetch(
    *,
    adapter,
    banks,
    budgets,
    configured_kernel,
    external_windows,
    fusion_capabilities,
    objects,
    prefetch_admission,
    private_ownership,
    session,
):
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


def configuration(
    *,
    budgets,
    configuration_adapter,
    session,
):
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


def final(
    *,
    adapter,
    banks,
    budgets,
    callbacks,
    configuration_adapter,
    continuation,
    copy_callbacks,
    copy_checked,
    copy_objects,
    copy_pipeline,
    copy_positions,
    copy_source,
    encoding,
    external_windows,
    limits,
    post_storage,
    private_ownership,
    require_copy_inputs,
    require_rewrite_inputs,
    rewrite_applied,
    rewrite_objects,
    rewrite_source,
    selected_contracts,
    session,
):
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
            selected_contracts if selected_contracts is not None and selected_contracts.program is not None else None
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
