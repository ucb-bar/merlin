"""The experiment's prohibited instruction roles, applied to a graded capsule's program.

A run may declare instruction roles its candidate programs must not use (``prohibited_instruction_roles``,
a statement in the closed vocabulary of :mod:`merlin.compile.analysis.instruction_roles`). Which
instructions carry such a role is the target's derived instruction facts, never a list written here
(:mod:`merlin.perf.isa_prohibition`). This module holds the capsule grader's side of that rule, kept out
of :mod:`.capsule_runner` so the runner only calls it.
"""

from __future__ import annotations

import os


def graded_prohibited_roles(capsule: dict) -> tuple[str, ...]:
    """The prohibited instruction roles a graded capsule's program is held to.

    The run's declared policy, from the environment the Phase 1 adapter hands every grader
    (:data:`merlin.perf.whole_model_gate.ROLES_ENV`), together with the roles the sealed corpus stamped
    into the capsule's own candidate arm (``performance.arms.candidate.instruction_policy``). Either
    source alone is enough to make the scan required: a capsule graded outside the adapter keeps the
    rule its corpus was sealed under."""
    from merlin.perf.whole_model_gate import ROLES_ENV

    roles = {r.strip() for r in (os.environ.get(ROLES_ENV) or "").split(",") if r.strip()}
    arm = (((capsule.get("performance") or {}).get("arms") or {}).get("candidate") or {}).get("instruction_policy")
    if isinstance(arm, dict):
        roles |= {str(r) for r in arm.get("prohibited_instruction_roles") or () if str(r)}
    return tuple(sorted(roles))


def prohibited_instruction_report(capsule: dict, generated, *, target: str) -> dict | None:
    """The whole-ELF prohibited-instruction scan of every executable this grade linked, or ``None`` when
    the capsule's program is held to no role (:func:`graded_prohibited_roles`).

    Every ELF :func:`run_executables` finds is walked in full (:func:`merlin.perf.isa_prohibition.scan_elf`
    reads every executable section, so an instruction reached only through a function pointer, or never
    executed, is found exactly like one on the hot path). ``clean`` is ``True`` only when every ELF was
    measured and none carries a prohibited instruction, ``False`` when any carries one, and ``None``
    (unmeasured) otherwise -- including a grade that linked no ELF at all, or a target whose facts give
    the roles no instruction. Unmeasured is never clean."""
    roles = graded_prohibited_roles(capsule)
    if not roles:
        return None
    from merlin.perf.isa_prohibition import SCHEMA, scan_elf

    from .capsule_runner import run_executables

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
    from merlin_experiments.phase1 import joint_control

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
    from merlin.perf.isa_prohibition import stream_reason, stream_report
    from merlin.targetgen.capsule_common import _cat
    from merlin.targetgen.package_runtime import CertFailure

    from .capsule_tier_result import TierResult

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
