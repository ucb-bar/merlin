"""Capsule evaluation: publication."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from merlin.development.targetgen.corpus.capsule_io import _cat
from merlin.development.targetgen.grading import trace_check as TCK
from merlin.development.targetgen.grading.elf_lanes import ACCELERATOR_LANE as _ACCELERATOR_LANE
from merlin.development.targetgen.packaging.loading import CertFailure
from merlin.evaluation.capsule.evidence.lanes import accelerator_lane_violated
from merlin.evaluation.capsule.evidence.lanes import (
    whole_program_completion_lane_report as whole_program_completion_lane_report,
)
from merlin.evaluation.capsule.evidence.results import _finalize_capsule_result
from merlin.evaluation.capsule.evidence.source_custody import _finish_compiler_source, submission_identity
from merlin.evaluation.capsule.operations.numerics import _grade_cpu_step
from merlin.target.contract.interfaces import abi as schemas
from merlin.target.isa.rocc import decode as RD

# ---- publication -------------------------------------------------------------------------------


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
    from merlin.analysis.programs.work_volume import command_buffer_evidence as _cb_evidence

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

            from merlin.analysis.programs.plan_evidence import verify_compiler_global_plan
            from merlin.common.digest import hash_tree as _hash_tree

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
    from merlin.analysis.programs.movement_volume import movement_evidence as _mv_evidence

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


# ---- instructions ------------------------------------------------------------------------------
# The experiment's prohibited instruction roles, applied to a graded capsule's program.
#
# A run may declare instruction roles its candidate programs must not use (``prohibited_instruction_roles``,
# a statement in the closed vocabulary of :mod:`merlin.analysis.observation.instruction_roles`). Which
# instructions carry such a role is the target's derived instruction facts, never a list written here
# (:mod:`merlin.analysis.whole_model.grading.isa_prohibition`). This module holds the capsule grader's side of that rule, kept out
# of :mod:`.operation` so the runner only calls it.


def graded_prohibited_roles(capsule: dict) -> tuple[str, ...]:
    """The prohibited instruction roles a graded capsule's program is held to.

    The run's declared policy, from the environment the Phase 1 adapter hands every grader
    (:data:`merlin.analysis.whole_model.grading.gate.ROLES_ENV`), together with the roles the sealed corpus stamped
    into the capsule's own candidate arm (``performance.arms.candidate.instruction_policy``). Either
    source alone is enough to make the scan required: a capsule graded outside the adapter keeps the
    rule its corpus was sealed under."""
    from merlin.analysis.whole_model.grading.gate import ROLES_ENV

    roles = {r.strip() for r in (os.environ.get(ROLES_ENV) or "").split(",") if r.strip()}
    arm = (((capsule.get("performance") or {}).get("arms") or {}).get("candidate") or {}).get("instruction_policy")
    if isinstance(arm, dict):
        roles |= {str(r) for r in arm.get("prohibited_instruction_roles") or () if str(r)}
    return tuple(sorted(roles))


def prohibited_instruction_report(capsule: dict, generated, *, target: str) -> dict | None:
    """The whole-ELF prohibited-instruction scan of every executable this grade linked, or ``None`` when
    the capsule's program is held to no role (:func:`graded_prohibited_roles`).

    Every ELF :func:`run_executables` finds is walked in full (:func:`merlin.analysis.whole_model.grading.isa_prohibition.scan_elf`
    reads every executable section, so an instruction reached only through a function pointer, or never
    executed, is found exactly like one on the hot path). ``clean`` is ``True`` only when every ELF was
    measured and none carries a prohibited instruction, ``False`` when any carries one, and ``None``
    (unmeasured) otherwise -- including a grade that linked no ELF at all, or a target whose facts give
    the roles no instruction. Unmeasured is never clean."""
    roles = graded_prohibited_roles(capsule)
    if not roles:
        return None
    from merlin.analysis.whole_model.grading.isa_prohibition import SCHEMA, scan_elf
    from merlin.evaluation.capsule.evidence.certificates import run_executables

    elfs = []
    for path in run_executables(generated):
        try:
            with path.open("rb") as handle:
                if handle.read(4) == b"\x7fELF":
                    elfs.append(path)
        except OSError:
            continue
    base = {"schema": SCHEMA, "scope": "whole_elf", "roles": list(roles)}
    if not elfs:
        return {
            **base,
            "status": "unmeasured",
            "clean": None,
            "summary": {},
            "scans": [],
            "detail": "the grade linked no ELF to scan, so the prohibited-instruction rule was not checked",
        }
    scans = [scan_elf(path, target=target, roles=roles) for path in elfs]
    summary: dict[str, int] = {}
    for scan in scans:
        for name, count in (scan.get("summary") or {}).items():
            summary[name] = summary.get(name, 0) + int(count)
    measured = all(scan.get("status") == "measured" for scan in scans)
    return {
        **base,
        "status": "measured" if measured else "unmeasured",
        "clean": False if summary else (True if measured else None),
        "summary": dict(sorted(summary.items())),
        "scans": scans,
        **(
            {"detail": "; ".join(str(s.get("detail")) for s in scans if s.get("status") != "measured")}
            if not measured
            else {}
        ),
    }


def apply_prohibited_instruction_rule(status: str, failure: dict | None, report: dict | None):
    """``(status, failure)`` after the instruction rule: a prohibited instruction anywhere in the linked
    program FAILS the capsule (it is a fact about the submission's bytes), and a passing capsule whose
    program could not be scanned is ``incomplete`` -- never a pass on an unchecked program."""
    if report is None:
        return status, failure
    if report.get("clean") is False:
        hits = ", ".join(f"{name} x{count}" for name, count in (report.get("summary") or {}).items())
        prohibited = {k: v for scan in report.get("scans") or () for k, v in (scan.get("prohibited") or {}).items()}
        if failure and failure.get("plane") == "instruction_policy":
            # The executed stream already failed the same rule at a named command: keep that reason first.
            return "fail", {**failure, "whole_elf": {"summary": report.get("summary"), "prohibited": prohibited}}
        return "fail", {
            "plane": "instruction_policy",
            "category": "PROHIBITED_INSTRUCTION",
            "detail": (
                f"the linked program carries instruction(s) the experiment prohibits (roles "
                f"{report.get('roles')}): {hits}. The whole ELF is scanned, so an instruction in code that "
                f"is never called still counts; emit the work without the prohibited instructions"
            ),
            "prohibited_instructions": prohibited,
            **({"previous_failure": failure} if failure else {}),
        }
    if report.get("clean") is not True and status == "pass":
        return "incomplete", failure or {
            "plane": "instruction_policy",
            "category": "PROHIBITION_NOT_MEASURED",
            "detail": (
                f"the capsule's program is held to prohibited roles {report.get('roles')}, and the "
                f"whole-ELF scan could not settle it: {report.get('detail') or 'no measured verdict'}"
            ),
        }
    return status, failure

    #: The tier a refusal of the decoded command stream is reported at: the loop tier, where the trace gate
    #: runs before any oracle.


STRUCTURAL_TIER = "L2"


def executed_stream_roles(capsule: dict) -> tuple[str, ...]:
    """The roles a capsule's EXECUTED command stream is held to.

    :func:`graded_prohibited_roles` (the run's environment and the capsule's sealed arm) together with
    the selected joint campaign's policy (``prohibited_instruction_roles``), which every grader of the
    campaign reads from the run's stored state rather than from its environment."""
    from merlin.development.backend.campaign import control as joint_control

    return tuple(sorted(set(graded_prohibited_roles(capsule)) | set(joint_control.prohibited_instruction_roles())))


def enforce_executed_stream(capsule: dict, trace: dict, *, target: str, tiers: dict) -> dict | None:
    """Hold the decoded command stream to the prohibited roles; ``None`` when the capsule has none.

    A command with a prohibited role that the stream executes fails the case at :data:`STRUCTURAL_TIER`
    (raised as a :class:`CertFailure` naming the role, the command and its command index). Otherwise
    the stream report is returned for the trace-check result; an UNKNOWN one is settled by
    :func:`apply_executed_stream_rule` and never passes."""
    roles = executed_stream_roles(capsule)
    if not roles:
        return None
    from merlin.analysis.whole_model.grading.isa_prohibition import stream_reason, stream_report
    from merlin.development.targetgen.corpus.capsule_io import _cat
    from merlin.development.targetgen.packaging.loading import CertFailure
    from merlin.evaluation.capsule.evidence.tiers import TierResult

    report = stream_report(trace, target=target, roles=roles)
    if report["clean"] is not False:
        return report
    reason = stream_reason(report)
    tiers[STRUCTURAL_TIER] = TierResult(STRUCTURAL_TIER, "fail", mandatory=True, reason=reason)
    error = CertFailure(
        "instruction_policy",
        _cat("FORBIDDEN_PATTERN"),
        f"{reason}. Build the schedule from the target's other commands",
    )
    error.facts = {"tier": STRUCTURAL_TIER, "instruction_policy": report}
    raise error


def apply_executed_stream_rule(status: str, failure: dict | None, trace_check_res: dict | None):
    """``(status, failure)``: a passing capsule whose decoded command stream the rule judged UNKNOWN (the
    role mapping is not derivable, or a command's role cannot be told) is ``incomplete`` -- reported,
    never read as compliance. A grade that decoded no accelerator stream (a host-only leaf, a target
    without a command decoder) carries no stream report and is left to its own verdict."""
    report = (trace_check_res or {}).get("instruction_policy")
    if status != "pass" or not isinstance(report, dict) or report.get("clean") is True:
        return status, failure
    return "incomplete", {
        "plane": "instruction_policy",
        "category": "PROHIBITION_NOT_MEASURED",
        "tier": STRUCTURAL_TIER,
        "detail": (
            f"prohibited instruction roles {report.get('roles')}: UNKNOWN for the executed command stream "
            f"({report.get('detail') or 'no measured verdict'})"
        ),
    }


# ---- trace -------------------------------------------------------------------------------------
# Capsule evaluation: trace.


def check_trace(capsule, paths, cfg, contract, eff_target, required, no_oracle, evidence):
    evidence.parsed_lowered_module = None

    # --- trace gate (optional; only for targets whose ISA has a decoder+checker plugin) --------
    # A command-ISA target (systolic, trace_gate="rocc_insn") decodes the emitted .insn stream to a
    # RoCC trace and checks instruction coverage. A SIMT/other target (trace_gate=None) has no analog
    # and skips it — the L0/L1 math floor + the oracle tiers carry the verdict either way.
    if cfg.trace_gate == "rocc_insn":
        # PARSED ONCE, then shared. This is the first of up to three full xDSL parses of the same
        # bytes in one grade; on a 29 MB artifact each costs about 44 s plus 5 s to verify, so the
        # plan evidence below is handed THIS module instead of parsing the text again. The decoder
        # keeps its own text path for an artifact that is not a parseable module.
        evidence.parsed_lowered_module = _grade_cpu_step(RD._parse_module, evidence.lowered_text)
        trace = (
            _grade_cpu_step(
                RD.decode_module,
                evidence.parsed_lowered_module,
                target=eff_target,
                source=str(paths.generated / cfg.fourth_output_name),
            )
            if evidence.parsed_lowered_module is not None
            else _grade_cpu_step(
                RD.decode_text,
                evidence.lowered_text,
                source=str(paths.generated / cfg.fourth_output_name),
                target=eff_target,
            )
        )
        evidence.decoded_trace = trace  # hand to the advisory localizer on an oracle divergence
        # Validating OUR OWN decoder's output must never crash the capsule before the oracle runs: a
        # schema hiccup (e.g. the manifest maps a funct to a class the enum lacks) is our tooling's
        # limitation, not the backend's defect — record it and continue, not a pre-oracle RUNNER_CRASH.
        try:
            schemas.validate(trace, "instruction_trace", contract=contract)
        except schemas.ContractViolation as _sv:
            import sys as _sys

            _sys.stderr.write(f"WARNING: instruction_trace self-validation failed (advisory): {_sv}\n")
        (paths.generated / "instruction_trace.json").write_text(json.dumps(trace, indent=2), encoding="utf-8")
        # Advisory silicon-liveness screen (HW-agnostic): would this emitted stream stall/fault on real
        # silicon? It uses only the target's OWN derived facts (scratchpad/accumulator capacity, address
        # map, ISA legality) + this decoded trace, never a target literal, and — like trace_check — is
        # advisory: it writes liveness_report.json for the author and never gates or crashes the capsule.
        try:
            from merlin.target.isa.machine_model import dram_window_for
            from merlin.verification.liveness import Program as _LProg
            from merlin.verification.liveness import assess as _lassess

            _lname = capsule.get("id") or capsule.get("name") or "kernel"
            # THE DRAM WINDOW IS A DERIVABLE FACT AND WAS NEVER HANDED OVER. The screen's address-map
            # check bounds a movement transaction against [base, base+size); nothing supplied `size`,
            # so EVERY report — including targets whose green card ships `start ~ end` — recorded
            # `dram-window-unknown` and the upper bound went unchecked. The window comes from the
            # target's OWN memory map (never a board default: fabricating a size would manufacture
            # false unmapped-address faults), and stays UNKNOWN with its reason named when the target
            # ships no map.
            _dwin = dram_window_for(eff_target)  # (base, size_bytes | None, provenance_why)
            _lrep = _lassess(
                _LProg(
                    name=_lname,
                    trace=trace,
                    address_model="pointer_args",
                    dram_bytes=_dwin[1],
                    dram_window_why=_dwin[2],
                ),
                eff_target,
            )
            (paths.generated / "liveness_report.json").write_text(
                json.dumps(_lrep.to_dict(), indent=2), encoding="utf-8"
            )
        except Exception as _le:  # noqa: BLE001 — advisory: our screener's limit is not the backend's defect
            import sys as _sys

            _sys.stderr.write(f"WARNING: liveness screen failed (advisory): {_le}\n")
            # Structural observations supplement numerical execution. Applicable trace
            # requirements gate qualification; numerical passes alone are not certificates.
            # The RoCC/inline-asm bare-metal harness passes each operand buffer as a POINTER argument, so
            # its address model is pointer_args: DRAM addresses must derive from the kernel's args, not be
            # baked. (A fixed-preload oracle — e.g. the program oracle — would pass "fixed_preload"; the
            # model is the harness's, derivable per oracle, not a target literal.)
        evidence.trace_check = _grade_cpu_step(
            TCK.check,
            trace,
            capsule.get("expected", {}),
            cb=evidence.command_buffer,
            address_model="pointer_args",
            # A design with a hardware loop performs several declared
            # classes with one instruction; its contract says which.
            subsumes=TCK.subsumption_for(eff_target),
            # Whose derived facts a declared residency mode reads.
            target=eff_target,
            lowered_module=evidence.parsed_lowered_module,
        )
        # Prohibited instruction roles on the executed stream: a hit fails the case at L2 here.
        _stream_policy = enforce_executed_stream(capsule, trace, target=eff_target, tiers=evidence.tiers)
        if _stream_policy is not None:
            evidence.trace_check["instruction_policy"] = _stream_policy
            # The MODE-FREE movement diagnostic, kept OUT of `violations` on purpose. It is a
            # whole-program performance observation, not a conformance one: `violations` feeds
            # `trace_all_pass`, which gates the run's formal verdict, and a program that moves one tile
            # per transfer is slow, not wrong. Reported under its own key so the author can see the
            # declared transfer width going unused without a performance note being able to fail a
            # correct capsule. Silent unless the target declares a payload bound.
        _movement = TCK.movement_findings(trace, TCK.movement_bound_for(eff_target))
        if _movement:
            evidence.trace_check["movement"] = _movement
            # Anti-cheese floor (must drive the accelerator) is the ONE trace-derived gate — but ONLY where
            # the RTL oracle will NOT run to prove it. When a required oracle tier executes the emitted
            # stream on real hardware, THAT is the spelling-independent anti-cheese (a host-computed fake
            # produces wrong output on RTL); the decoder-shaped floor only recognizes the `.insn r` form, so
            # gating on it here would false-fail a conformant `.word`-encoded kernel before the oracle.
        if (no_oracle or not required) and not TCK.drives_accelerator(trace):
            raise CertFailure(
                "trace_check",
                _cat("PROTOCOL_VIOLATION"),
                "no accelerator instructions emitted — the kernel did not drive the "
                "target's ISA (compute must happen on the device; correctness cannot "
                "come from the host).",
            )

            # A FORBIDDEN ACCELERATOR LANE, judged on the one thing this path can prove. `lane_report`
            # needs a routing plan and an execution record, which only the whole-model path owns, so an
            # op or model-slice capsule forbidding the accelerator had no verdict at all (see
            # LANE_CONTRACT_NOT_EVALUATED).
            #
            # The asymmetry is deliberate and is the whole reason this is sound here. A DECODED
            # accelerator instruction proves work went there: that is a violation, and it is a fail. Its
            # ABSENCE proves nothing -- the decoder recognizes the `.insn r` form, so a `.word`-encoded
            # kernel that does drive the device decodes as silent. Reading that silence as "the host
            # carried it" would hand a forbidding capsule a free pass, which is exactly the shape of
            # vacuous evidence this contract exists to refuse. So: violated, or unmeasured. Never
            # satisfied.
        if accelerator_lane_violated(capsule, trace):
            # PROTOCOL_VIOLATION is the enum member this is: the emitted stream broke a contract the
            # capsule declared. `_cat` would silently resolve an unknown name to RUNNER_CRASH, which
            # would report a compiler defect as a harness crash, so the plane carries the lane
            # vocabulary and the category stays one the enum actually has.
            raise CertFailure(
                "lanes",
                _cat("PROTOCOL_VIOLATION"),
                f"capsule forbids lane {_ACCELERATOR_LANE!r} and the emitted stream "
                f"decodes accelerator instructions; this target's capability manifest "
                f"does not admit this capsule's family, so the compiler must leave it "
                f"on the host lane.",
            )
