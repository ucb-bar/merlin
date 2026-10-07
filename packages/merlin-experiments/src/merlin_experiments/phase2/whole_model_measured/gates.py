"""The refusals a whole-model candidate meets BEFORE any machine time, and the infra-fault classifier.

In the order they apply to one request:

1. **The capsule screen** (:func:`pre_measure_check`): a declared check (an argv with ``{package}``,
   ``{out}`` and ``{capsules}`` placeholders) run against the SNAPSHOT at request time.  A required
   screen that fails ends the job without a build.
2. **The coverage gate** (:func:`coverage_gate`): a candidate whose package answers less of the work
   than the seed measures the library, not the compiler; refused right after its build.
3. **The instruction rule** (:func:`isa_gate`): the experiment's ``prohibited_instruction_roles``,
   checked over the WHOLE linked program -- every group, the library's groups and host code
   included -- before any run, FAILING CLOSED: a check that cannot run refuses, it never passes.
4. **The functional gate** (:func:`functional_gate`): a program the functional model grades wrong gets
   no board time.

An infrastructure fault (a harness import break, a capsule the harness could not resolve, a builder
traceback) is never a fact about the candidate's own bytes: :func:`is_infra_refusal` classifies one
structurally, :func:`infra_circuit_breaker` stops a run that sees a streak of them.
"""

from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from merlin.perf import whole_model_verdict as V

from . import jobs as J
from .identity import IdentityError, locked, package_digest, read_json, write_json_atomic

#: Substrings that name an INFRASTRUCTURE fault -- never a fact about the candidate's own bytes.
#: Structural fault-CLASS markers, not a target or model name: the class recurs across targets and
#: sessions even though the exact message differs every time.  Measured 2026-09-29: a whole night
#: (104 iterations across three run dirs) was refused this way with nothing surfacing it.
INFRA_REFUSAL_MARKERS = (
    "unknown capsule",
    "importerror",
    "modulenotfounderror",
    "keyerror",
    "traceback (most recent call last)",
    "tool_crash",
    "no sized symbol",
)

#: The build option the builder routes by and this gate checks the linked program for, so the rule
#: that shapes a build and the rule it is judged by cannot drift apart.
PROHIBITED_ROLES = "prohibited_roles"


def is_infra_refusal(text: Any) -> bool:
    """Whether ``text`` (a screen's output tail or error, or a build's refusal) names an
    infrastructure fault rather than anything about the candidate itself."""
    lowered = str(text or "").lower()
    return any(marker in lowered for marker in INFRA_REFUSAL_MARKERS)


def infra_circuit_breaker(history: Sequence[Mapping[str, Any]], *, limit: int = 3) -> str | None:
    """None while fewer than ``limit`` of the MOST RECENT candidates (oldest first, as
    :meth:`.service.MeasurementService.history` returns them) were each refused for an
    infrastructure reason in an unbroken streak; otherwise the reason to stop offering candidates.

    A row without a refusal (still pending, or measured) ends the streak: this counts only a streak of
    infra refusals ENDING the history, never one broken up by an intervening success."""
    if limit < 1:
        raise ValueError("the infra-refusal limit must be a positive integer")
    consecutive = 0
    latest = ""
    for row in reversed(list(history)):
        refusal = row.get("refusal")
        if not refusal or not is_infra_refusal(refusal):
            break
        consecutive += 1
        latest = latest or str(refusal)
        if consecutive >= limit:
            return (
                f"{consecutive} consecutive candidate(s) refused for an infrastructure reason, not "
                f"the package's own bytes (latest: {latest[:200]!r})"
            )
    return None


def screen_was_infra(check: Mapping[str, Any] | None) -> bool:
    check = check or {}
    return (
        is_infra_refusal(check.get("output_tail"))
        or is_infra_refusal(check.get("error"))
        or is_infra_refusal((check.get("summary") or {}).get("error"))
    )


# --------------------------------------------------------------- 1. the capsule screen
def _argv(spec: Mapping[str, Any], package: Path, out: Path, capsules: str | None = None) -> list[str]:
    names = capsules if capsules is not None else str(spec.get("capsules") or "")
    return [
        str(token).replace("{package}", str(package)).replace("{out}", str(out)).replace("{capsules}", names)
        for token in spec["argv"]
    ]


def pre_measure_check(job: Mapping[str, Any], package: Path, job_dir: Path) -> dict[str, Any] | None:
    """Run the check the launch declared against the SNAPSHOT, before the build (or reuse a stored
    screen of the SAME argv whose failure was never infrastructure's)."""
    spec = job.get("pre_measure_check")
    if not isinstance(spec, Mapping) or not spec.get("argv"):
        return None
    out = Path(job_dir) / "pre_measure_check.json"
    screened = read_json(Path(job_dir) / "pre_measure_check_result.json")
    if screened is not None and screen_result_reusable(screened, spec, package, out):
        return screened
    return run_capsule_check(spec, package, out, cwd=Path(job_dir))


def screen_result_reusable(screened: Mapping[str, Any], spec: Mapping[str, Any], package: Path, out: Path) -> bool:
    """A stored screen is reused only when its OWN recorded ``argv`` equals what THIS spec would run
    now (a screen recorded under one snapshot's harness says nothing about another's), and its failure
    was never infrastructure's."""
    if screened.get("argv") != _argv(spec, package, out):
        return False
    return not screen_was_infra(screened)


def run_capsule_check(
    spec: Mapping[str, Any], package: Path, out: Path, *, cwd: Path, capsules: str | None = None
) -> dict[str, Any]:
    """Run the declared capsule check on ``package``; ``capsules`` overrides the spec's own list."""
    argv = _argv(spec, package, out, capsules)
    started = time.time()
    try:
        done = subprocess.run(
            argv,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=float(spec.get("timeout_seconds") or 900),
            env={**os.environ, **{str(k): str(v) for k, v in (spec.get("environment") or {}).items()}},
        )
        returncode, tail = done.returncode, (done.stdout or "")[-4000:] + (done.stderr or "")[-2000:]
    except subprocess.TimeoutExpired:
        returncode, tail = None, "the check exceeded its timeout"
    except OSError as exc:
        returncode, tail = None, f"the check could not start: {type(exc).__name__}: {exc}"
    report = read_json(out)
    passed = capsule_report_passed(report, returncode)
    summary = None
    if isinstance(report, dict):
        rows = report.get("per_capsule") or report.get("results") or []
        failing = [
            row
            for row in rows
            if isinstance(row, Mapping) and not (row.get("pass") is True or row.get("status") == "pass")
        ]
        # THE REPORT'S `all_pass` IS ITS CERTIFICATION FLAG (every mandatory tier cleared), not whether
        # this check passed: a subset screen on a functional model certifies nothing by design, so it
        # reads false beside a `passed: true` that is correct.  It is carried as what it means.
        summary = {
            "certified_all": report.get("all_pass"),
            "scope": report.get("scope"),
            "n_passed": report.get("n_passed"),
            "n_capsules": report.get("n_capsules"),
            "n_certified": report.get("n_certified"),
            "n_screened_only": report.get("n_screened_only"),
            "error": report.get("error"),
            "failing": [row.get("capsule") or row.get("name") for row in failing],
            "mismatches": {str(row.get("capsule") or row.get("name")): _mismatch_summary(row) for row in failing},
        }
    return {
        "label": spec.get("label") or "pre-measure check",
        "required": bool(spec.get("required")),
        "argv": argv,
        "returncode": returncode,
        "passed": passed,
        "passed_basis": PASSED_BASIS,
        "summary": summary,
        "report": str(out) if out.is_file() else None,
        "output_tail": tail[-3000:],
        "wall_seconds": round(time.time() - started, 3),
        "provenance": check_provenance(spec, argv, package, out),
    }


#: What ``passed`` on a capsule-check record means, stated in the record itself.
PASSED_BASIS = (
    "every capsule the check graded passed at the tiers its simulator runs (capsule_report_passed); "
    "summary.certified_all is the report's own flag for having also cleared every mandatory tier"
)


def check_provenance(spec: Mapping[str, Any], argv: list[str], package: Path, out: Path) -> dict[str, Any]:
    """The program files the check ran (by content), the machine binaries the launch declares for it,
    the report it wrote, and the digest of the package it graded."""
    from merlin.common import provenance

    sources = [token for token in argv[1:] if token.endswith(".py") and Path(token).is_file()]
    artifacts = {str(k): str(v) for k, v in (spec.get("artifacts") or {}).items() if Path(str(v)).is_file()}
    if out.is_file():
        artifacts["report"] = str(out)
    try:
        package_sha256 = package_digest(package)
    except IdentityError:
        package_sha256 = None
    return provenance.record(
        sources=sources,
        artifacts=artifacts,
        extra={"check": spec.get("label"), "argv": list(argv), "package_sha256": package_sha256},
    )


def capsule_report_passed(report: Any, returncode: int | None) -> bool:
    """Whether a capsule check passed, read from its report.

    A SUBSET report (``scope: subset``) states ``all_pass: false`` by design -- it covers only what
    was asked for -- so it passes when it reports no error, graded at least one capsule, and every
    graded row passed.  A full-suite report keeps its own ``all_pass``.  No report at all falls back to
    the exit status."""
    if not isinstance(report, Mapping):
        return returncode == 0
    if report.get("error"):
        return False
    if report.get("scope") == "subset" or "all_pass" not in report:
        rows = [r for r in report.get("per_capsule") or report.get("results") or [] if isinstance(r, Mapping)]
        count = report.get("n_capsules")
        return (
            bool(rows)
            and (count is None or (int(count) > 0 and int(report.get("n_passed") or 0) == int(count) == len(rows)))
            and all(r.get("pass") is True for r in rows)
        )
    return bool(report.get("all_pass"))


def _mismatch_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    numeric = row.get("numeric") if isinstance(row.get("numeric"), Mapping) else {}
    return {
        "tiers": row.get("tiers"),
        "status": numeric.get("status"),
        "policy": numeric.get("policy"),
        "mismatch_count": numeric.get("mismatch_count"),
        "max_abs_diff": numeric.get("max_abs_diff"),
        "first_mismatch": numeric.get("first_mismatch"),
        "failure": row.get("failure") or row.get("error") or row.get("detail"),
    }


# --------------------------------------------------------------- 2. the coverage gate
def coverage_assessment(job: Mapping[str, Any], build: Mapping[str, Any]) -> dict[str, Any] | None:
    """What the coverage gate decided about ``build``, PASS OR FAIL, or None when no gate applies.

    Kept on the build record: a gate that leaves a trace only when it refuses cannot be told apart
    from one that never ran, and "did the candidate pass the coverage gate" was answerable only by
    recomputing it from the job and the build."""
    gate = job.get("coverage_gate")
    if not isinstance(gate, Mapping) or job.get("screen_exempt") or job.get("role") == J.ROLE_REFERENCE:
        return None
    price = {str(k): int(v) for k, v in (gate.get("price") or {}).items()}
    total = sum(price.values())
    if not total:
        return None
    answered = {
        str(r.get("group")) for r in build.get("groups") or () if isinstance(r, Mapping) and r.get("on") == "package"
    }
    share = sum(price.get(g, 0) for g in answered) / total
    floor_groups = set(map(str, gate.get("floor_groups") or ()))
    floor = (
        sum(price.get(g, 0) for g in floor_groups) / total if floor_groups else float(gate.get("floor_share") or 0.0)
    )
    declined = sorted(floor_groups - answered, key=V._order)
    return {
        "passed": share + 1.0 / total >= floor or not declined,
        "priced_share": round(share, 4),
        "floor_share": round(floor, 4),
        "declined_groups": declined,
        "floor_package_sha256": gate.get("floor_package_sha256"),
    }


def coverage_gate(
    job: Mapping[str, Any], build: Mapping[str, Any], check: Mapping[str, Any] | None
) -> dict[str, Any] | None:
    """Refuse, right after the build, a candidate whose package answers less of the work than the
    seed: its board cycles would be the library's, which the objective never counts."""
    gate = job.get("coverage_gate")
    if not isinstance(gate, Mapping) or job.get("screen_exempt") or job.get("role") == J.ROLE_REFERENCE:
        return None
    price = {str(k): int(v) for k, v in (gate.get("price") or {}).items()}
    total = sum(price.values())
    if not total:
        return None
    answered = {
        str(r.get("group")) for r in build.get("groups") or () if isinstance(r, Mapping) and r.get("on") == "package"
    }
    share = sum(price.get(g, 0) for g in answered) / total
    floor_groups = set(map(str, gate.get("floor_groups") or ()))
    # THE SAME PRICED BASIS FOR BOTH: the floor's share is recomputed from its groups with THIS price
    # table, and a difference below one cycle of the total is a tie, and a tie passes.
    floor = (
        sum(price.get(g, 0) for g in floor_groups) / total if floor_groups else float(gate.get("floor_share") or 0.0)
    )
    if share + 1.0 / total >= floor:
        return None
    declined = sorted(floor_groups - answered, key=V._order)
    if not declined:
        return None  # every group the floor's package answered, this one answers too
    regression = {
        "declined_groups": declined,
        "priced_share": round(share, 4),
        "floor_share": round(floor, 4),
        "floor_package_sha256": gate.get("floor_package_sha256"),
    }
    return J.refused(
        job,
        f"coverage_regression: {len(declined)} group(s) declined to the library "
        f"({', '.join('g' + g for g in declined[:20])}); package-authored work {100 * share:.1f}% < the seed's "
        f"{100 * floor:.1f}%; no board time",
        coverage_regression=regression,
        build=dict(build),
        pre_measure_check=check,
    )


# --------------------------------------------------------------- 3. the instruction rule
def _default_checker() -> Callable[..., Mapping[str, Any]]:
    """The whole-program scanner (``merlin.perf.isa_prohibition.check_build``).  Resolved lazily so a
    missing scanner is a REFUSAL at gate time, never an import error that skips the gate."""
    import importlib

    return importlib.import_module("merlin.perf.isa_prohibition").check_build


def isa_gate(
    job: Mapping[str, Any],
    build: dict[str, Any],
    check: Mapping[str, Any] | None,
    role: str,
    *,
    checker: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Refuse, before any run, a candidate whose LINKED PROGRAM emits a prohibited instruction.

    The whole program is read -- every group, including the ones the target's library answers, and the
    host code around them -- because a rule on the program is broken by any of it.  The reference is
    the one program exempt: it is what the rule is measured against, not a candidate under it.  A CLEAN
    build keeps the scan's per-group instruction census on ``build`` for diagnostic feedback."""
    roles = list((job.get("build_options") or {}).get(PROHIBITED_ROLES) or ())
    if not roles or job.get("role") == J.ROLE_REFERENCE:
        return None
    sealed = job.get("instruction_policy")
    unsealed = sealed_policy_problems(sealed, roles)
    if unsealed:
        # Refused BEFORE the scan: a scan judges the program against what the roles derive to now, and
        # only the sealed Phase 0 policy says what they must forbid. With none, a scan that finds
        # nothing cannot be told apart from a rule that forbids nothing.
        report = {"clean": None, "summary": {}, "roles": roles, "error": "; ".join(unsealed)}
        report.update(checked_build=role, scope="whole_elf")
        return J.refused(
            job,
            f"isa_prohibited: the run carries no enforceable sealed instruction policy ({report['error']}); "
            "no board time",
            isa_prohibited=report,
            build=dict(build),
            pre_measure_check=check,
        )
    try:
        scan = checker or _default_checker()
        report = dict(scan(build, target=str(job["target"]), roles=roles))
    except Exception as exc:  # noqa: BLE001 -- a check that cannot run refuses; it never passes
        report = {"clean": False, "summary": {}, "error": f"{type(exc).__name__}: {exc}"}
    report["checked_build"] = role
    report.setdefault("roles", roles)
    report["scope"] = "whole_elf"
    if report.get("census") is not None:
        build["isa_census"] = report["census"]
    weaker = scan_weaker_than_sealed(report, sealed, roles)
    if weaker and not report.get("error"):
        report["error"] = weaker
    status = report.get("status")
    if report.get("clean") is True and status != "measured" and not report.get("error"):
        # A clean verdict the scanner did not mark measured is not one this gate can record as such.
        report["error"] = f"the scan reported clean with status {status!r}, not 'measured'"
    if report.get("clean") is True and not report.get("error"):
        build["isa_prohibition"] = {
            "scope": "whole_elf",
            "status": status,
            "verdict": "clean",
            "roles": list(roles),
            "prohibited": dict(report.get("prohibited") or {}),
            "sealed_source": dict(sealed.get("sealed_source") or {}) if isinstance(sealed, Mapping) else None,
        }
        return None
    if report.get("summary"):
        lead = "isa_prohibited: " + ", ".join(sorted(report.get("summary") or {})[:12])
    else:
        why = report.get("error") or report.get("detail") or "no clean verdict"
        lead = f"isa_prohibited: the program could not be checked ({why})"
    return J.refused(job, f"{lead}; no board time", isa_prohibited=report, build=dict(build), pre_measure_check=check)


def sealed_policy_problems(policy: Mapping[str, Any] | None, roles: Sequence[str]) -> list[str]:
    """Why the sealed Phase 0 ``policy`` a job carries cannot hold a program to ``roles``; empty if it can."""
    from merlin_experiments.phase0.instruction_roles import enforcement_problems

    return enforcement_problems(policy, roles) if roles else []


def _selector(value: Any) -> str:
    """One selector's spelling, normalised to its integer value where it has one ("0x8" and "8" agree)."""
    try:
        return str(int(str(value), 0))
    except ValueError:
        return str(value)


def _selectors(policy: Mapping[str, Any], roles: Sequence[str]) -> set[str]:
    prohibited = policy.get("prohibited_instructions") if isinstance(policy, Mapping) else None
    rows = [row for role in roles for row in ((prohibited or {}).get(role) or ())]
    return {_selector(row.get("selector")) for row in rows if isinstance(row, Mapping)}


def scan_weaker_than_sealed(report: Mapping[str, Any], sealed: Mapping[str, Any] | None, roles: Sequence[str]) -> str:
    """Non-empty when the scan's prohibited set is EMPTY or omits an instruction Phase 0 sealed.

    A scan whose derived prohibited set is smaller than the sealed one is enforcing a weaker rule than
    the one the experiment was frozen under (a facts tree that lost a role binding, a contract the
    process read from the wrong place); the program it calls clean was not checked for what is
    prohibited."""
    scanned = {_selector(k) for k in (report.get("prohibited") or {})}
    if not scanned:
        return "the scan prohibited no instruction (the roles matched nothing in the target's facts)"
    missing = sorted(_selectors(sealed or {}, roles) - scanned, key=lambda sel: (len(sel), sel))
    if missing:
        return f"the scan does not prohibit sealed selector(s) {missing}; it enforces a weaker rule than Phase 0 sealed"
    return ""


# --------------------------------------------------------------- 4. the functional gate
def functional_gate(
    job: Mapping[str, Any],
    job_dir: Path,
    builds: Mapping[str, Mapping[str, Any]],
    local: Mapping[str, Any],
    local_device: Mapping[str, Any],
    check: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """A program the functional model grades WRONG gets no board time (the seed excepted, once, for
    attribution -- labelled so, and never a best)."""
    verdict = read_json(Path(job_dir) / J.LOCAL_VERDICT) or {}
    if verdict.get("status") != "incorrect":
        return None
    failed = list((verdict.get("correctness") or {}).get("groups_failed") or [])
    if job.get("screen_exempt"):
        with locked(Path(job_dir)):
            fresh = read_json(Path(job_dir) / "job.json") or dict(job)
            fresh.update(attribution_only=True, attribution_label=J.ATTRIBUTION_ONLY)
            write_json_atomic(Path(job_dir) / "job.json", fresh)
        return None
    return J.refused(
        job,
        f"functional_gate: the functional model grades {len(failed)} group(s) wrong "
        f"({', '.join('g' + g for g in failed[:20])}); a wrong program gets no board time",
        functional_gate_failed=True,
        verdict={
            "timing_status": V.TIMING_REFUSED,
            "correctness": verdict.get("correctness"),
            "groups": verdict.get("groups") or [],
        },
        build=builds.get("timing") or builds.get("local"),
        local_build=builds.get("local"),
        local_device=dict(local_device),
        local_run={k: v for k, v in (local.get("run") or {}).items() if k != "device"},
        pre_measure_check=check,
    )


__all__ = [
    "INFRA_REFUSAL_MARKERS",
    "PROHIBITED_ROLES",
    "capsule_report_passed",
    "check_provenance",
    "coverage_gate",
    "functional_gate",
    "infra_circuit_breaker",
    "is_infra_refusal",
    "isa_gate",
    "pre_measure_check",
    "run_capsule_check",
    "scan_weaker_than_sealed",
    "screen_result_reusable",
    "screen_was_infra",
    "sealed_policy_problems",
]
