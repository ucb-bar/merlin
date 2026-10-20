"""Project closed self-check results without exposing expected values or private paths."""

from __future__ import annotations

import json
from pathlib import Path

from . import qa as _qc


def _successful_expected_refusal(result: dict) -> bool:
    """A completed refusal obligation has no executable or numerical tier to measure."""
    record = result.get("expected_refusal")
    return (
        result.get("status") == "pass"
        and result.get("failure") is None
        and isinstance(record, dict)
        and record.get("status") == "met"
        and isinstance(result.get("declined"), dict)
        and bool(result["declined"])
    )


def _operator_structural_feedback(result, capsule, *, target):
    """Qualification requires original applicable structure, independently of numerical screens."""
    from merlin.targetgen.expected_refusal import validated_result
    from merlin.targetgen.owned_host_leaf_execution import qualified_host_result, selected_host_leaf
    from merlin.targetgen.trace_check import requirements_complete

    if not isinstance(capsule, dict) or capsule.get("name") != result.get("capsule"):
        return {"status": "unavailable", "component": "original_capsule_binding"}
    if validated_result(capsule, result):
        return {"status": "pass", "component": "original_expected_refusal"}
    if (result.get("failure") or {}).get("category") == "host_compute_in_accelerator_group":
        return {"status": "failed", "component": "host_accelerator_placement"}
    if selected_host_leaf(capsule):
        return {
            "status": "pass" if qualified_host_result(capsule, result, target=target) else "unavailable",
            "component": "provided_host_infrastructure",
        }
    trace = result.get("trace_check") or {}
    requirement = trace.get("trace_requirement_status")
    complete = requirements_complete(trace)
    return {
        "status": "pass"
        if complete
        else requirement or ("failed" if trace.get("status") == "fail" else "unavailable_record"),
        "component": "instruction_trace",
        "violations": [str(reason)[:240] for reason in (trace.get("violations") or [])[:8]],
        "unavailable_requirements": list(trace.get("unavailable_requirements") or [])[:16],
    }


def _operator_feedback_state(result, capsule, *, target, barrier_status):
    """Keep numerical screens separate from original applicable qualification."""
    from merlin.targetgen.expected_refusal import validated_result

    refusal = isinstance(capsule, dict) and validated_result(capsule, result)
    structural = _operator_structural_feedback(result, capsule, target=target)
    numeric_native_pass = result.get("status") == "pass" and barrier_status == "pass"
    certified = refusal or (numeric_native_pass and structural["status"] == "pass")
    failure = result.get("failure") or {}
    selection_blocked = failure.get("category") == "NOT_RUN_IS_NOT_PASS" and failure.get("tier_status") == "unavailable"
    ran = {
        tier: row.get("status")
        for tier, row in (result.get("tiers") or {}).items()
        if isinstance(row, dict) and row.get("status") not in (None, "skipped", "unavailable")
    }
    screened = not certified and (
        (
            selection_blocked
            and bool(ran)
            and all(status == "pass" for status in ran.values())
            and barrier_status in ("pass", None)
        )
        or numeric_native_pass
    )
    return certified, screened, refusal, structural


def _public_candidate_emission_failure(result: dict) -> dict | None:
    """Expose only bounded structural compiler errors, never receipt payloads."""
    failure = result.get("candidate_emission_failure")
    detail = failure.get("detail") if isinstance(failure, dict) else failure
    if not isinstance(detail, str) or not detail.strip():
        return None
    # Long GCC argv used the entire public detail allowance before the actual
    # linker error. Project only structural symbol names from its stderr; never
    # copy generated source, array values or other fields of the raw receipt.
    # The unchanged original result/compilation recipe retains full diagnostics.
    if isinstance(failure, dict) and failure.get("stage") == "owned_host_execution":
        stderr = detail[:2000].partition("\nSTDERR:")[2]
        symbols = []
        for line in stderr.splitlines():
            for marker, end in (("undefined reference to `", "'"),
                                ("undefined reference to '", "'"), ("undefined symbol: ", " ")):
                symbol = line.partition(marker)[2].partition(end)[0].strip()
                if (0 < len(symbol) <= 128 and symbol.isascii()
                        and (symbol[0].isalpha() or symbol[0] in "_.$")
                        and all(character.isalnum() or character in "_.$@" for character in symbol)
                        and symbol not in symbols):
                    symbols.append(symbol)
        if symbols:
            hint = "Linker undefined symbols: " + ", ".join(symbols[:4])
            detail = hint + ". Check the selected executable entry and per-group symbol binding.\n" + detail
    return {
        "plane": "candidate_compilation",
        "category": "CANDIDATE_EMISSION_FAILED",
        "detail": _qc._redact_detail(detail[:2000]),
    }


def _candidate_selfcheck_row(result: dict, closed: dict | None, *, name: str, barrier_tier: str) -> tuple[dict, bool]:
    """Project one submitted whole-model artifact from QA's closed candidate fields.

    The runner-owned graph's tiers, numeric output, trace, console and executable
    are different artifacts. An absent closed row never promotes raw result data.
    """
    projected = closed if isinstance(closed, dict) else {}
    summary = projected.get("candidate_native_verification")
    if not isinstance(summary, dict):
        summary = {"status": "unverified", "violations": ["candidate_verification_record_unrecognized"]}
    tiers = summary.get("tiers")
    tiers = tiers if isinstance(tiers, dict) else {}
    required = summary.get("required_tiers")
    required = required if isinstance(required, list) else []
    required_pass = bool(required) and all(tiers.get(tier) == "pass" for tier in required)
    components = summary.get("components")
    components = components if isinstance(components, dict) else {}
    source_coverage = summary.get("source_coverage")
    source_verified = isinstance(source_coverage, dict) and source_coverage.get("status") == "verified"
    component_pass = all(
        components.get(key) == expected
        for key, expected in (
            ("emitted_host_compute", "clean"),
            ("source_placement", "clean"),
            ("completed_dispatch", "verified"),
            ("candidate_required_tiers", "pass"),
        )
    )
    bar_used = barrier_tier
    bar = tiers.get(barrier_tier)
    if bar is None:
        deeper = [
            tier for tier in required if isinstance(tier, str) and tier > barrier_tier and tiers.get(tier) == "pass"
        ]
        if deeper:
            bar_used = max(deeper)
            bar = tiers[bar_used]
    certified = (
        bool(projected)
        and result.get("status") == "pass"
        and (summary.get("status") == "pass" and required_pass and source_verified and component_pass and bar == "pass")
    )
    screened = (
        barrier_tier == "L2"
        and source_verified
        and summary.get("status") == "incomplete"
        and not summary.get("violations")
        and all(tiers.get(tier) == "pass" for tier in ("L0", "L1", "L2"))
        and tiers.get("L3") == "unverified"
        and set(required) == {"L0", "L1", "L2", "L3"}
        and all(
            components.get(key) == expected
            for key, expected in (
                ("emitted_host_compute", "clean"),
                ("source_placement", "clean"),
                ("completed_dispatch", "verified"),
                ("candidate_required_tiers", "incomplete"),
            )
        )
    )
    row = {
        "capsule": name,
        "pass": certified or screened,
        "screened_only": screened,
        "certified_complete": certified,
        "execution_digest": None,
        "barrier_tier": bar_used,
        "barrier_declared": barrier_tier,
        "barrier_status": bar,
        "candidate_native_verification": summary,
    }
    if not certified:
        numeric = {
            key: projected.get(field)
            for key, field in (("status", "numeric_status"), ("mismatch_count", "mismatch_count"))
            if projected.get(field) is not None
        }
        row.update(
            tiers=tiers,
            numeric=numeric,
            failure={
                "plane": projected.get("failure_plane"),
                "category": projected.get("failure_category"),
                "tier": projected.get("failure_tier"),
                "detail": projected.get("failure_detail"),
            },
        )
    if not certified:
        emission_failure = _public_candidate_emission_failure(result)
        if emission_failure is not None:
            row["failure"] = emission_failure
    return row, certified


def _log_telemetry(out: dict, capsules_arg: str) -> None:
    """Append ONE operator-side dev-trajectory line per self-check (set by the driver via $SELFCHECK_LOG).
    Records EVERY verdict — including the degenerate build_failed / no_results ones — so a blind or
    flat-lining loop is visible in the operator log, not only in the agent's transcript. Best-effort."""
    import os
    import time as _t

    log_path = os.environ.get("SELFCHECK_LOG")
    if not log_path:
        return
    try:
        from merlin_experiments.phase1.brokers.selfcheck_sources import LABEL_ENV, SUBMISSION

        t0 = float(os.environ.get("SELFCHECK_T0", "0") or 0)
        rows = out.get("per_capsule") or []
        line = {
            "wall_offset_s": round(_t.time() - t0, 1) if t0 else None,
            # Which tree: a variants/<name> line is an author-side experiment, not submission/ progress.
            "submission": os.environ.get(LABEL_ENV) or SUBMISSION,
            "sim": out.get("sim"),
            "barrier_tier": out.get("barrier_tier"),
            # The EFFECTIVE barrier each capsule was scored against, distinct from the declared one.
            # When they differ the declared tier never ran, and scoring against it silently fails
            # every capsule -- the atlas defect that hid behind 110 identical 0/11 lines. Recording
            # both makes that visible in the log instead of only in a per-row field nobody reads.
            "barrier_used": sorted({r.get("barrier_tier") for r in rows if r.get("barrier_tier")}) or None,
            "capsules": capsules_arg,
            "n_passed": out.get("n_passed"),
            "n_capsules": out.get("n_capsules"),
            "all_pass": out.get("all_pass"),
            "build_failed": out.get("build_failed", False),
            "no_results": out.get("no_results", False),
            "failing": [r["capsule"] for r in rows if not r.get("pass")],
        }
        with open(log_path, "a") as lf:
            lf.write(json.dumps(line) + "\n")
    except Exception:
        pass


def _gated_without_result_rows(
    score: dict, *, requested: set[str], models: set[str], represented: set[str]
) -> list[dict]:
    """Project only requested, unexecuted model gates from the grader's score.

    The grader records a gated model in its score but writes no
    ``capsule_result.json``. Its gate is a known non-verdict even when operation
    results exist beside it. Every score identity must belong to this request;
    malformed or contradictory score rows cannot be turned into feedback.
    """
    rows = score.get("per_capsule") if isinstance(score, dict) else None
    if not isinstance(rows, list) or not represented <= requested or not models <= requested:
        raise ValueError("self-check score identities are invalid")
    seen: set[str] = set()
    gated: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("self-check score row is invalid")
        name, status = row.get("capsule"), row.get("status")
        if (
            not isinstance(name, str)
            or not name
            or name not in requested
            or name in seen
            or not isinstance(status, str)
            or not status
        ):
            raise ValueError("self-check score row identity is invalid")
        seen.add(name)
        if status != "gated":
            continue
        reason = row.get("gate_reason")
        if (
            row.get("kind") != "model"
            or name not in models
            or name in represented
            or not isinstance(reason, str)
            or not reason
        ):
            raise ValueError("self-check gated row is inconsistent with capsule results")
        gated.append({"capsule": name, "pass": False, "status": "gated", "reason": reason})
    if not represented <= seen:
        raise ValueError("self-check capsule result is absent from the grader score")
    return gated


def _selfcheck_counts(
    *, result_rows: int, gated_rows: int, passed: int, certified: int, requested_size: int, suite_size: int, scope: str
) -> dict:
    """Keep performed results distinct from requested models deferred without execution."""
    reported = result_rows + gated_rows
    if (
        min(result_rows, gated_rows, passed, certified, requested_size, suite_size) < 0
        or reported > requested_size
        or passed > result_rows
        or certified > result_rows
        or (suite_size and requested_size > suite_size)
    ):
        raise ValueError("self-check counts disagree with the selected public capsules")
    complete_request = suite_size > 0 and requested_size > 0 and reported == requested_size
    all_pass = complete_request and gated_rows == 0 and certified == requested_size
    return {
        "n_capsules": reported,
        "n_result_rows": result_rows,
        "n_gated": gated_rows,
        "n_unchecked": max(0, suite_size - result_rows) if suite_size else None,
        "n_unknown": max(0, suite_size - reported) if suite_size else None,
        "certified_complete": bool(scope == "all" and suite_size == requested_size and all_pass),
        "all_pass": all_pass,
        "n_passed": passed,
        "n_certified": certified,
    }


def _offload_census(
    sub: Path,
    out_path: str,
    *,
    context: InvocationContext,
    capsules_root: Path,
    contract: Path | None = None,
    timeout: int = 300,
    workers: int = 0,
) -> int:
    """The agent-facing offload census (see ``--offload-census``).

    Leaks nothing: it reads only the command buffers the agent's own package emits for the public
    capsules -- no golden, no reference value, no simulator.
    """

    from merlin.targetgen import offload_census as OC

    try:
        report = OC.census_of_corpus(
            sub,
            target=context.target,
            capsule_roots=[capsules_root],
            contract=str(contract if contract is not None else context.repo / "merlin/contract"),
            timeout=timeout,
            workers=workers,
        )
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"error": f"offload census failed: {type(e).__name__}: {e}"}))
        return 2
    report["note"] = (
        "One row per public program, from the command buffer YOUR package emitted for it. "
        "`offloaded` put work on a unit; `host_only` placed everything on the host lane and said so; "
        "`declined` refused with a reason; `silent` emitted no command, declared no placement and "
        "declined nothing -- at the numeric tier that is an output of zeros, indistinguishable from "
        "wrong arithmetic, so say what you did instead. `placement_undeclared` lists programs that "
        "reached a unit without a `params.lane_placement` record: work nobody can attribute. A host "
        "placement is not a defect, an unexplained one is. `routed_macs` is the work that reached the "
        "accelerator; watch it across rounds -- a pass count can rise while it falls. "
        "`host_compute` audits the HOST code of each program: issuing commands and computing "
        "addresses is command-scale work and is fine; a host loop that loads, computes and stores "
        "every element of a program that claims to run on the unit is "
        "`host_compute_in_accelerator_group`, named with its dominant block. Such a program passes "
        "numerics and the trace check and spends its time on the host. If the unit cannot apply a "
        "stage, place that stage on the host and SAY so; that is reported, not penalised."
    )
    txt = json.dumps(report, indent=2)
    print(txt)
    if out_path:
        Path(out_path).write_text(txt)
    return 0 if report.get("all_accounted") and not report.get("host_compute_vetoed") else 1
