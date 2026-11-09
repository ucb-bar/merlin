"""Capsule evaluation: publication."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from merlin.evaluation.capsule.cpu import _grade_cpu_step
from merlin.evaluation.capsule.identity import submission_identity
from merlin.evaluation.capsule.lanes_evidence import (
    whole_program_completion_lane_report as whole_program_completion_lane_report,
)
from merlin.evaluation.capsule.result import _finalize_capsule_result
from merlin.evaluation.capsule.source import _finish_compiler_source


def publish_operation(
    capsule,
    package_dir,
    paths,
    cfg,
    contract,
    eff_target,
    name,
    run_id,
    required,
    no_oracle,
    _defer_result_publication,
    evidence,
):
    from merlin.perf.work_volume import command_buffer_evidence as _cb_evidence

    _work_volume, _cb_artifact = _grade_cpu_step(  # raw buffer stored beside the result, referenced
        _cb_evidence,
        evidence.command_buffer,
        compiler_provenance="submission command-buffer contract entrypoint",
        store=paths.run_path,
    )
    _lane_extra: dict[str, Any] = {}
    if (
        evidence.whole_program
        and evidence.command_buffer is not None
        and ((capsule.get("lanes") or {}).get("require") or (capsule.get("lanes") or {}).get("forbid"))
    ):
        try:
            import hashlib as _hashlib

            from merlin.benchharness import hash_tree as _hash_tree
            from merlin.perf.compiler_plan_evidence import verify_compiler_global_plan

            _source_name = capsule.get("linalg_mlir") or capsule.get("interface_mlir")
            _source_path = Path(str(capsule.get("__dir__") or "")) / str(_source_name or "")
            _candidate_digest = _grade_cpu_step(_hash_tree, Path(package_dir)).get("sha256")
            if not isinstance(_candidate_digest, str):
                raise ValueError("whole-program package has no content identity")
            _plan_proof = _grade_cpu_step(
                verify_compiler_global_plan,
                source_text=_source_path.read_text(encoding="utf-8"),
                lowered_text=evidence.lowered_text,
                command_buffer=evidence.command_buffer,
                candidate_sha256=_candidate_digest,
                # The module parsed from these exact bytes above, in this invocation, unmutated --
                # the hook's own precondition. Verification still runs; only the reparse is saved.
                parsed_lowered_module=evidence.parsed_lowered_module,
                command_buffer_sha256=_hashlib.sha256(
                    json.dumps(evidence.command_buffer, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
                ).hexdigest(),
            )
            _lane_extra["global_plan_verification"] = _plan_proof
            _completed_lane_report = whole_program_completion_lane_report(
                capsule,
                evidence.command_buffer,
                global_plan_proof=_plan_proof,
                numeric=evidence.numeric,
                tiers=evidence.tiers,
                decoded_trace=evidence.decoded_trace,
            )
            if _completed_lane_report is not None:
                _lane_extra["lane_report"] = _completed_lane_report
        except Exception as _lane_exc:  # noqa: BLE001 -- missing proof stays unmeasured, never passes
            _lane_extra["global_plan_verification"] = {
                "status": "UNKNOWN",
                "reason": (f"whole-program lane proof unavailable: {type(_lane_exc).__name__}: {_lane_exc}"),
            }
    # THE MOVEMENT AXIS, counted over the SAME buffer. An operational intensity is work over traffic,
    # and the two halves must come from one emitted program or the ratio describes neither -- so this
    # rides beside the work total under the artifact digest they share. Emitted for every graded
    # capsule for the same reason the work total is: an absent key reads as "no movement axis applies",
    # a zero reads as "this program moved no data", and on a roofline that means infinite intensity.
    from merlin.perf.movement_volume import movement_evidence as _mv_evidence

    _movement_volume = _grade_cpu_step(
        _mv_evidence, evidence.command_buffer, compiler_provenance="submission command-buffer contract entrypoint"
    )
    return _finalize_capsule_result(
        result_filename=".operator_child_result.json" if _defer_result_publication else "capsule_result.json",
        submission=submission_identity(package_dir, run_id=run_id),
        name=name,
        capsule=capsule,
        status=evidence.status,
        failure=evidence.failure,
        tiers=evidence.tiers,
        trace_check_res=evidence.trace_check,
        numeric=evidence.numeric,
        required=required,
        no_oracle=no_oracle,
        eff_target=eff_target,
        paths=paths,
        run_id=run_id,
        cfg=cfg,
        contract=contract,
        executability=evidence.executability,
        epilogue_applicability=evidence.epilogue_applicability,
        declined=evidence.declined,
        decoded_trace=evidence.decoded_trace,
        execution_tiers=evidence.execution_tiers,
        extra={
            "compiler_source": _grade_cpu_step(_finish_compiler_source, evidence.compiler_source, paths.generated),
            "work_volume": _work_volume,
            "command_buffer_artifact": _cb_artifact,
            "movement_volume": _movement_volume,
            **_lane_extra,
            **evidence.owned_source_compilation,
            **({"order_check": evidence.order_check} if evidence.order_check is not None else {}),
            **({"completion_order": evidence.completion_order} if evidence.completion_order is not None else {}),
        },
    )
