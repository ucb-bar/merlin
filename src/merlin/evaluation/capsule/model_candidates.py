"""Capsule evaluation: model candidates."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from merlin.evaluation.capsule.bundle import _model_runtime_bundle as _model_runtime_bundle
from merlin.evaluation.capsule.diagnostics import _candidate_exception_diagnostic as _candidate_exception_diagnostic
from merlin.evaluation.capsule.identity import CONTRACT_VERSION
from merlin.targetgen.capsule_common import make_run_paths, run_entrypoints
from merlin.targetgen.package_runtime import Package, build_package, integrity_scan, load_package


def _grade_candidate_model_capsule_inline(capsule, *, target, timeout, package_dir, context, pkg=None):
    """Keep explicit public-program treatment separate from owned-host model grading."""
    from merlin.targetgen.owned_host_model_selection import owned_model_required
    from merlin.targetgen.owned_host_slice_execution import selected_source_slice

    if selected_source_slice(capsule):
        if context.get("public_program_execution") is not None or "checked_source_diagnostic" in context:
            raise ValueError("source slices require Merlin host ownership")
        return _grade_owned_model_capsule_inline(
            capsule, target=target, timeout=timeout, package_dir=package_dir, context=context, pkg=pkg
        )
    if "checked_source_diagnostic" in context:
        from merlin.targetgen.checked_source_model_diagnostic_execution import grade_from_context

        return grade_from_context(
            sys.modules[__name__],
            capsule,
            target=target,
            timeout=timeout,
            package_dir=package_dir,
            context=context,
            pkg=pkg,
        )
    selected = (
        _grade_public_program_capsule_inline
        if context.get("public_program_execution") is not None
        else (
            _grade_owned_model_capsule_inline
            if owned_model_required(capsule, target=target)
            else _grade_public_program_capsule_inline
        )
    )
    return selected(capsule, target=target, timeout=timeout, package_dir=package_dir, context=context, pkg=pkg)


def _grade_owned_model_capsule_inline(
    capsule: dict,
    *,
    target: str,
    timeout: int,
    package_dir: str | Path,
    context: dict,
    pkg: Package | None = None,
) -> dict:
    """Grade only the submitted whole-program model under the shared model budget.

    The parent or suite retains a frozen capsule copy until this function has
    enforced the candidate-only source, native and mandatory-tier checks.
    """
    from merlin.targetgen.capsule_grade import enforce_model_execution_check
    from merlin.targetgen.native_model_execution import _digest, independent_frozen_source_eligibility

    attrs = (capsule.get("operation") or {}).get("attributes") or {}
    result: dict = {
        "capsule": capsule["name"],
        "kind": "model",
        "label": capsule.get("label"),
        "operation": {
            "op": "model",
            "model": attrs.get("model"),
            "dtype": attrs.get("compile_dtype", "fp32"),
            "run": "submitted_whole_program",
            "target": target,
        },
        "contract_version": CONTRACT_VERSION,
        "status": "incomplete",
        "legacy_model_diagnostic": {
            "status": "not_run",
            "scope": "runner-owned host-dispatch graph; not candidate whole-program evidence",
        },
    }
    try:
        from merlin.targetgen.owned_host_slice_execution import require_slice_placement, selected_source_slice

        if selected_source_slice(capsule):
            require_slice_placement(capsule)
        result["candidate_source_eligibility"] = independent_frozen_source_eligibility(capsule, target=target)
    except Exception as exc:  # noqa: BLE001 -- no source denominator means no submitted-program run
        result["candidate_source_eligibility_failure"] = {"type": type(exc).__name__, "detail": str(exc)[:2000]}
        return enforce_model_execution_check(result, capsule, target=target)

    readback_policy = None
    stage = "prepare_model_package"
    try:
        # The child receives a JSON context, not a Python capability. Reconstruct
        # only the exact operator-selected policy record; absent means legacy.
        readback_record = context.get("readback_policy")
        if readback_record is not None:
            from merlin.targetgen.contract.readback_policy import ReadbackPolicy

            readback_policy = ReadbackPolicy.from_record(readback_record)
        else:
            readback_policy = None
        selected_pkg = pkg
        if selected_pkg is None:
            selected_pkg = load_package(package_dir, contract=context["contract"])
            integrity_scan(selected_pkg)
            if not context["package_prebuilt"]:
                build_package(selected_pkg)
        paths = make_run_paths(
            context["runs_root"],
            context["run_id"],
            suite=context["suite"],
            target=target,
            dtype=context["dtype"],
            benchmark=capsule["name"],
        )
        # Host lowering is a trusted Merlin service. The candidate is invoked
        # only for independently source-bound device groups by that service.
        from merlin.targetgen.owned_host_model_execution import execute_owned_host_model

        result["owned_host_source"] = {
            "capsule_declaration": _digest(Path(capsule["__dir__"]) / "capsule.yaml"),
            "source_interface": _digest(
                Path(capsule["__dir__"]) / capsule.get("interface_mlir", "capsule.interface.mlir")
            ),
        }

        stage = "prepare_runtime_bundle"
        with _model_runtime_bundle(capsule, timeout=timeout, target=target) as (bundle, provenance, verify):
            result["candidate_capture"] = provenance
            stage = "owned_host_execution"
            result["owned_host_native_execution"] = execute_owned_host_model(
                capsule=capsule,
                candidate_package=package_dir,
                target=target,
                bundle=bundle,
                provenance=provenance,
                verify=verify,
                output=paths.run_path / "owned_host_native",
                selection_record=context.get("owned_host_build"),
                simulator=os.environ.get("MERLIN_MODEL_NATIVE_SIMULATOR") or "none",
                timeout=timeout,
                source_certificate=result["candidate_source_eligibility"],
                readback_policy=readback_policy,
                reachable_oracle_tiers=context.get("reachable_oracle_tiers"),
            )
            stage = "verify_runtime_bundle"
            verify()
    except Exception as exc:  # noqa: BLE001 -- candidate exceptions cannot borrow a legacy pass
        result["candidate_emission_failure"] = _candidate_exception_diagnostic(exc, stage=stage)
        # A post-execution source/bundle verification failure invalidates even
        # a numerically green native receipt. Keep its durable run artifacts
        # for diagnosis, but never feed that receipt to the pass gate.
        result.pop("candidate_native_execution", None)
        result.pop("owned_host_native_execution", None)
    return enforce_model_execution_check(
        result,
        capsule,
        target=target,
        **({"readback_policy": readback_policy} if readback_policy is not None else {}),
    )


def _grade_public_program_capsule_inline(
    capsule: dict,
    *,
    target: str,
    timeout: int,
    package_dir: str | Path,
    context: dict,
    pkg: Package | None = None,
    diagnostic_selection=None,
) -> dict:
    """Grade only the submitted whole-program model under the shared model budget.

    The parent or suite retains a frozen capsule copy until this function has
    enforced the candidate-only source, native and mandatory-tier checks.
    """
    from merlin.targetgen.capsule_grade import enforce_model_execution_check
    from merlin.targetgen.checked_source_model_diagnostic_execution import (
        execute_native_model as execute_candidate_model,
    )
    from merlin.targetgen.native_model_execution import _digest, independent_frozen_source_eligibility

    diagnostic_deadline = time.monotonic() + timeout
    diagnostic_native_result = diagnostic_work_root = None
    attrs = (capsule.get("operation") or {}).get("attributes") or {}
    result: dict = {
        "capsule": capsule["name"],
        "kind": "model",
        "label": capsule.get("label"),
        "operation": {
            "op": "model",
            "model": attrs.get("model"),
            "dtype": attrs.get("compile_dtype", "fp32"),
            "run": "submitted_whole_program",
            "target": target,
        },
        "contract_version": CONTRACT_VERSION,
        "status": "incomplete",
        "legacy_model_diagnostic": {
            "status": "not_run",
            "scope": "runner-owned host-dispatch graph; not candidate whole-program evidence",
        },
    }
    try:
        result["candidate_source_eligibility"] = independent_frozen_source_eligibility(capsule, target=target)
    except Exception as exc:  # noqa: BLE001 -- no source denominator means no submitted-program run
        result["candidate_source_eligibility_failure"] = {"type": type(exc).__name__, "detail": str(exc)[:2000]}
        if diagnostic_selection is not None:
            from merlin.targetgen.checked_source_model_diagnostic_execution import finish_unavailable

            return finish_unavailable(
                diagnostic_selection, result, capsule, target=target, deadline=diagnostic_deadline
            )
        return enforce_model_execution_check(result, capsule, target=target)

    readback_policy = None
    stage = "prepare_model_package"
    try:
        execution_kwargs = {}
        selection = context.get("public_program_execution")
        if diagnostic_selection is not None:
            from merlin.targetgen.checked_source_model_diagnostic_execution import execution_kwargs as checked_execution

            if selection is not None:
                raise ValueError("checked source diagnostics cannot select public-program grading")
            execution_kwargs = checked_execution(diagnostic_selection, capsule, target=target)
        if selection is not None:
            from merlin.targetgen.public_program_timing import verify_selection

            execution_kwargs = verify_selection(capsule, selection)
            result["public_program_execution"] = selection
        # The child receives a JSON context, not a Python capability. Reconstruct
        # only the exact operator-selected policy record; absent means legacy.
        readback_record = context.get("readback_policy")
        if readback_record is not None:
            from merlin.targetgen.contract.readback_policy import ReadbackPolicy

            readback_policy = ReadbackPolicy.from_record(readback_record)
        else:
            readback_policy = None
        selected_pkg = pkg
        if selected_pkg is None:
            selected_pkg = load_package(package_dir, contract=context["contract"])
            integrity_scan(selected_pkg)
            if not context["package_prebuilt"]:
                build_package(selected_pkg)
        paths = make_run_paths(
            context["runs_root"],
            context["run_id"],
            suite=context["suite"],
            target=target,
            dtype=context["dtype"],
            benchmark=capsule["name"],
        )
        if diagnostic_selection is not None:
            diagnostic_native_result = paths.run_path / "candidate_native/result.json"
            diagnostic_work_root = paths.run_path / "checked_source_realization"
        stage = "candidate_entrypoints"
        _, candidate_cb, candidate_llvm = run_entrypoints(
            selected_pkg,
            package_dir,
            capsule,
            paths,
            contract=context["contract"],
            timeout=timeout,
            fourth_output_name=context["fourth_output_name"],
        )
        source_interface = Path(capsule["__dir__"]) / capsule.get("interface_mlir", "capsule.interface.mlir")
        if source_interface.is_symlink():
            raise ValueError("frozen model interface cannot be a symlink")
        result["candidate_emission"] = {
            "capsule_declaration": _digest((Path(capsule["__dir__"]) / "capsule.yaml").resolve(strict=True)),
            "source_interface": _digest(source_interface.resolve(strict=True)),
            "command_buffer": _digest((paths.generated / "command_buffer.json").resolve(strict=True)),
            "lowered_mlir": _digest((paths.generated / context["fourth_output_name"]).resolve(strict=True)),
        }
        if "__model_source_transport" in capsule:
            result["candidate_source_transport"] = capsule["__model_source_transport"]
        from merlin.perf.work_volume import command_buffer_evidence

        result["work_volume"], result["command_buffer_artifact"] = command_buffer_evidence(
            candidate_cb,
            compiler_provenance="actual submitted whole-program command buffer; physical pricing remains independent",
            store=str(paths.run_path),
        )
        bundle_kwargs = (
            {"artifact_root": paths.run_path / ".private_runtime_bundle"}
            if (selection is not None or diagnostic_selection is not None)
            else {}
        )
        stage = "prepare_runtime_bundle"
        with _model_runtime_bundle(capsule, timeout=timeout, **bundle_kwargs) as (bundle, provenance, verify):
            result["candidate_capture"] = provenance
            policy_kwargs = {"readback_policy": readback_policy} if readback_policy is not None else {}
            stage = "candidate_model_execution"
            result["candidate_native_execution"] = execute_candidate_model(
                command_buffer=candidate_cb,
                lowered_mlir_text=candidate_llvm,
                capsule_dir=capsule["__dir__"],
                capture_bundle=bundle,
                target=target,
                out_dir=paths.run_path / "candidate_native",
                timeout=timeout,
                **(
                    execution_kwargs
                    or {
                        "simulator": os.environ.get("MERLIN_MODEL_NATIVE_SIMULATOR") or None,
                        "rtl_facts": os.environ.get("MERLIN_MODEL_NATIVE_RTL_FACTS") or None,
                        "board_config": os.environ.get("MERLIN_MODEL_NATIVE_BOARD_CONFIG") or None,
                    }
                ),
                **policy_kwargs,
            )
            stage = "verify_runtime_bundle"
            verify()
    except Exception as exc:  # noqa: BLE001 -- candidate exceptions cannot borrow a legacy pass
        result["candidate_emission_failure"] = _candidate_exception_diagnostic(exc, stage=stage)
        # A post-execution source/bundle verification failure invalidates even
        # a numerically green native receipt. Keep its durable run artifacts
        # for diagnosis, but never feed that receipt to the pass gate.
        result.pop("candidate_native_execution", None)
    if diagnostic_selection is not None:
        from merlin.targetgen.checked_source_model_diagnostic_execution import finish

        return finish(
            diagnostic_selection,
            result,
            capsule,
            target=target,
            native_result=diagnostic_native_result,
            work_root=diagnostic_work_root,
            remaining_s=max(0, diagnostic_deadline - time.monotonic()),
        )
    return enforce_model_execution_check(
        result,
        capsule,
        target=target,
        **({"readback_policy": readback_policy} if readback_policy is not None else {}),
    )
