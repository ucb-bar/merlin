"""Project closed self-check results without exposing expected values or private paths."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from merlin.development.backend.feedback.grading import package_rejection
from merlin.development.backend.feedback.grading import qa as _qc
from merlin.development.backend.tools.result_details import (
    _CACHE,
    MAX_DETAIL_BYTES,
    SCHEMA,
    append_rows,
    hydrate,
    load_row,
)
from merlin.development.backend.tools.result_details import (
    compact_document as _compact_document,
)
from merlin.development.backend.tools.result_details import (
    compact_row as _compact_row,
)

# ---- selfcheck_results -------------------------------------------------------------------------

if TYPE_CHECKING:
    from merlin.development.backend.context import InvocationContext


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
    from merlin.development.targetgen.execution.host.leaf_execution import (
        qualified_host_result,
        selected_host_leaf,
    )
    from merlin.development.targetgen.grading.boundary import validated_result
    from merlin.development.targetgen.grading.trace_check import requirements_complete

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
    from merlin.development.targetgen.grading.boundary import validated_result

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
            for marker, end in (
                ("undefined reference to `", "'"),
                ("undefined reference to '", "'"),
                ("undefined symbol: ", " "),
            ):
                symbol = line.partition(marker)[2].partition(end)[0].strip()
                if (
                    0 < len(symbol) <= 128
                    and symbol.isascii()
                    and (symbol[0].isalpha() or symbol[0] in "_.$")
                    and all(character.isalnum() or character in "_.$@" for character in symbol)
                    and symbol not in symbols
                ):
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
    from merlin.development.backend.feedback.guidance.composition_brief import from_case

    composition = from_case(result)
    if composition is not None:
        row["source_composition"] = composition
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
        from merlin.development.backend.brokers.selfcheck_sources import LABEL_ENV, SUBMISSION

        origin = os.environ.get("SELFCHECK_T0")
        t0 = float(origin) if origin else None
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
            "build_failed": out.get("build_failed"),
            "no_results": out.get("no_results"),
            "failing": [r["capsule"] for r in rows if not r.get("pass")],
        }
        with open(log_path, "a") as lf:
            lf.write(json.dumps(line) + "\n")
    except Exception:
        logging.getLogger(__name__).warning(
            "Optional self-check telemetry unavailable; verdict unchanged", exc_info=True
        )


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

    from merlin.development.targetgen.conformance import offload_census as OC

    try:
        report = OC.census_of_corpus(
            sub,
            target=context.target,
            capsule_roots=[capsules_root],
            contract=str(contract if contract is not None else context.repo / "src/merlin/resources/contracts"),
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


# ---- selfcheck_progress ------------------------------------------------------------------------
# Incremental self-check progress: each case's correctness as soon as the grader decides it.
#
# ``run_capsule`` writes ``capsule_result.json`` when a case's whole ladder is done, and a large suite
# waits on its slowest case before it answers. Measured on one 132-case scalability self-check: the
# response came 72 minutes after the request while the last case's numerics were decided at minute 53;
# after the numeric comparison a large case still spent up to 16 minutes in the completion-order check
# and the result write, so a correct case reached the agent up to a quarter of an hour late and the
# sweep's answer waited for all of them.
#
# This view therefore publishes two kinds of row, both golden-free:
#
# * a FINISHED row, from a durable ``capsule_result.json``: the case's final status, its observed operator
#   tier statuses, the failing plane/category (labels the final response already carries) and the
#   numeric status;
# * a PROVISIONAL row (``status: "running"``), from ``generated/numeric_report.yaml``, written by the
#   runner the moment the numeric comparison is made: ``numerics: pass|fail`` with ``final: false``.
#   Later checks (completion order, RTL tiers, the host-compute audit) can still fail the case; they can
#   never turn a numeric fail into a pass.
#
# Results are parsed once per ``(size, mtime)``: the snapshot is re-taken every second inside the
# grading process, and re-reading every result each time held that process's interpreter lock for
# seconds per snapshot once large results existed. One directory walk finds both kinds of file, and a
# finished case's numeric report is not looked at again.
#
# The document is published as it is. Its rows are already small and golden-free; routing them through
# the response compactor stored a detail file per row in ``qa/cases`` and hashed every row on every
# publish (measured: 76 ms of interpreter time per publish of a 300-case suite, 69% of it there), and the
# compacted rows lost ``numerics``/``final``, so the client printed ``numerics None`` for a running case.

_TIERS = frozenset({"L0", "L1", "L2", "L3"})

_TIER_STATUSES = frozenset({"pass", "fail", "unverified", "unavailable", "incomplete", "skipped"})

_NUMERIC_STATUSES = frozenset({"pass", "fail"})

NOTE = (
    "Incremental self-check progress from completed capsule records. This is not a final "
    "verdict and contains no golden/reference values. Reported tiers are observed public "
    "operator screens, not full-case certification; raw model tiers are omitted. A row with "
    "status 'running' has its numerics decided ('numerics', final: false) while later checks "
    "still run; they can fail it, never turn a numeric fail into a pass."
)


def _signature(path: Path):
    try:
        stat = path.stat()
    except OSError:
        return None
    return stat.st_size, stat.st_mtime_ns


def _cached(cache: dict | None, path: Path, parse):
    """``parse(path)`` once per file version; ``None`` (not cached) while the file is unreadable."""
    signature = _signature(path)
    if signature is None:
        return None
    key = str(path)
    if cache is not None:
        hit = cache.get(key)
        if hit is not None and hit[0] == signature:
            return hit[1]
    value = parse(path)
    if cache is not None and value is not None:
        cache[key] = (signature, value)
    return value


def _finished_row(result_path: Path) -> dict | None:
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None  # writer still publishing; the next snapshot will see it
    if not isinstance(result, dict):
        return None
    row = {
        "capsule": str(result.get("capsule") or result_path.parent.name),
        "status": str(result.get("status") or "unknown"),
    }
    # Raw whole-model graph tiers do not qualify the submitted compiler's
    # artifact. Model progress keeps its legacy status-only projection.
    model = result.get("kind") == "model" or any(
        key in result for key in ("model_execution_check", "candidate_native_model_check", "candidate_native_execution")
    )
    recorded = result.get("tiers")
    if not model and isinstance(recorded, dict):
        tiers = {
            tier: record["status"]
            for tier, record in recorded.items()
            if tier in _TIERS
            and isinstance(record, dict)
            and isinstance(record.get("status"), str)
            and record["status"] in _TIER_STATUSES
        }
        if tiers:
            row["tiers"] = tiers
    numeric = result.get("numeric")
    if isinstance(numeric, dict) and isinstance(numeric.get("status"), str) and numeric["status"] in _NUMERIC_STATUSES:
        row["numerics"] = numeric["status"]
    failure = result.get("failure")
    if row["status"] != "pass" and isinstance(failure, dict):
        for key in ("plane", "category"):
            if isinstance(failure.get(key), str) and failure[key]:
                row[f"failure_{key}"] = failure[key]
    return row


def _numeric_status(report_path: Path) -> str | None:
    import yaml

    try:
        report = yaml.safe_load(report_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None  # written in place: a torn read is retried on the next snapshot
    status = report.get("status") if isinstance(report, dict) else None
    return status if isinstance(status, str) and status in _NUMERIC_STATUSES else None


def progress_snapshot(cb_root: Path, *, expected: int, started_ns: int, cache: dict | None = None) -> dict:
    """A deliberately small, golden-free view of durable and provisional per-capsule results."""
    rows: list[dict] = []
    provisional: list[dict] = []
    for case in _case_dirs(Path(cb_root)):
        row = _cached(cache, case / "capsule_result.json", _finished_row)
        if row is not None:
            rows.append(dict(row))
            continue
        status = _cached(cache, case / "generated" / "numeric_report.yaml", _numeric_status)
        if status is not None:
            provisional.append({"capsule": case.name, "status": "running", "numerics": status, "final": False})
    counts: dict[str, int] = {}
    for row in rows:
        status = row["status"]
        if status not in counts:
            counts[status] = 1
        else:
            counts[status] += 1
    numerics: dict[str, int] = {}
    for row in (*rows, *provisional):
        if row.get("numerics") in _NUMERIC_STATUSES:
            status = row["numerics"]
            if status not in numerics:
                numerics[status] = 1
            else:
                numerics[status] += 1
    return {
        "status": "running",
        "started_at_unix_ns": int(started_ns),
        "updated_at_unix_ns": time.time_ns(),
        "n_expected": int(expected),
        "n_finished": len(rows),
        "n_remaining": max(0, int(expected) - len(rows)),
        "n_numerics_decided": sum(numerics.values()),
        "counts": counts,
        "numerics_counts": numerics,
        "per_capsule": sorted([*rows, *provisional], key=lambda row: row["capsule"]),
        "note": NOTE,
    }


def _case_dirs(root: Path) -> list[Path]:
    """The case directories of one grade, in name order (one directory read)."""
    try:
        with os.scandir(root) as entries:
            return sorted((Path(entry.path) for entry in entries if entry.is_dir()), key=lambda path: path.name)
    except OSError:
        return []


def publish(path: Path, doc: dict) -> None:
    """Replace ``path`` with ``doc`` atomically; a reader never sees a partial document."""
    from merlin.development.backend.feedback.disclosure import project

    doc = project(doc)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        tmp.write_text(json.dumps(doc, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def progress_publisher(
    path: Path, cb_root: Path, *, expected: int, started_ns: int, stop: threading.Event, poll_s: float = 1.0
) -> None:
    """Continuously publish changing progress until the grade's final roll-up is available."""
    cache: dict = {}
    last_rows = None
    while not stop.is_set():
        doc = progress_snapshot(cb_root, expected=expected, started_ns=started_ns, cache=cache)
        if doc["per_capsule"] != last_rows:
            publish(path, doc)
            last_rows = doc["per_capsule"]
        stop.wait(poll_s)
    doc = progress_snapshot(cb_root, expected=expected, started_ns=started_ns, cache=cache)
    doc["status"] = "grading_complete"
    publish(path, doc)


def partial_results(progress_path: Path) -> dict:
    """The decided cases of a check that ended without a final verdict (timeout, abort, crash).

    ``{}`` when no progress was published. Otherwise ``{"partial": {...}}``: every finished row and every
    case whose numerics were decided, labelled as partial, so a timed-out suite still returns what it
    learned instead of nothing (measured: one suite timed out three times at 31 minutes, each time
    returning only an error while most of its cases had long been decided)."""
    try:
        progress = json.loads(Path(progress_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(progress, dict) or not isinstance(progress.get("per_capsule"), list):
        return {}
    keys = ("n_expected", "n_finished", "n_numerics_decided", "counts", "numerics_counts", "per_capsule")
    return {
        "partial": {
            **{key: progress.get(key) for key in keys},
            "note": "the check ended before its final verdict; these are the cases it had decided. "
            "Not a pass for the cases listed as running or absent.",
        }
    }


# ---- selfcheck_certificates --------------------------------------------------------------------
# Keep the certificate tier an author self-check already executed on its exact snapshot.
#
# When a self-check's own selection ran the cert tier, the broker skips promotion -- re-running those
# bytes buys the same simulation twice -- but nothing recorded the result either, so a passing gSIM L3
# run of a required case (whole models among them) never reached the physical-correctness ledger and
# the grade kept it ``unverified``. This records it, under the promotion ledger's own rules:
#
# * the certificate belongs to the source snapshot the broker verified the check graded
#   (``source_ws``), by its whole-submission digest and, when the row carries one, its execution
#   identity -- never to the live, possibly edited, workspace;
# * only rows that state a passed, certified cert-tier barrier, run on the engine a promotion would
#   use and under the instrument a promotion would record, are imported;
# * bytes that already hold a cert-tier record (outstanding, passed or failed) are left alone;
# * the result is written by ``record_cert`` against a record created for exactly those bytes, so
#   attribution is the recorder's, not this module's.
#
# Joint-campaign required cases only. Like promotion, it is an optimisation and never gates a run.


def _rows(verdict, cert_tier, roster):
    for row in verdict.get("per_capsule") or ():
        if (
            isinstance(row, dict)
            and row.get("capsule") in roster
            and row.get("pass") is True
            and row.get("certified_complete") is True
            and row.get("screened_only") is not True
            and row.get("barrier_tier") == cert_tier
            and row.get("barrier_status") == "pass"
        ):
            yield row


def import_executed(ws, verdict, *, loop_tier, cert_tier, source_ws, context, log=None) -> list[str]:
    """Record the cert-tier passes ``verdict`` already executed; returns ``record_cert``'s lines."""
    from merlin.development.backend.campaign import control as joint_control
    from merlin.development.backend.feedback.campaign import correctness as physical
    from merlin.development.backend.feedback.certification import promotion as P

    roster = physical.roster() if joint_control.selected() is not None else set()
    if (
        not roster
        or (loop_tier, cert_tier) != ("L2", "L3")
        or not isinstance(verdict, dict)
        or verdict.get("error")
        or not P.cert_already_executed(verdict, cert_tier)
    ):
        return []
    rows = list(_rows(verdict, cert_tier, roster))
    if not rows:
        return []
    engine = (verdict.get("tier_engines") or {}).get(cert_tier)
    sim = P.cert_sim(cert_tier, context=context)
    instrument = P.cert_instrument(cert_tier, context=context)
    if sim is None or engine != sim or instrument is None:
        if log is not None:
            print(
                f"[promote] self-check {cert_tier} not imported: ran on {engine!r}, a promotion would use "
                f"{sim!r}" + ("" if instrument else "; certificate instrument unavailable"),
                file=log,
                flush=True,
            )
        return []
    return P._ledger_transaction(_record)(
        ws, rows, cert_tier=cert_tier, instrument=instrument, source_ws=Path(source_ws), log=log
    )


def _record(ws, rows, *, cert_tier, instrument, source_ws, log):
    from merlin.development.backend.feedback.certification import promotion as P

    digest, comps, _ = P.submission_digests(source_ws)
    state = P._tier_state(ws)
    staged = []
    for row in rows:
        name = str(row["capsule"])
        execution = row.get("execution_digest") if P._valid_execution(row.get("execution_digest")) else None
        identity = P.current_identity(execution, digest)
        if identity in P._slots_ro(state, name, cert_tier):
            continue  # these bytes already hold a record; a running promotion resolves its own
        entry = {
            "status": "pending",
            "digest": digest,
            "components": dict(comps),
            "instrument": instrument,
            "origin": "author_selfcheck",
        }
        if execution is not None:
            entry["execution_digest"] = execution
        P._record(state, name, cert_tier, entry)
        staged.append((name, identity, execution))
    if not staged:
        return []
    P._save_tier_state(ws, state)
    recorded = []
    for name, identity, execution in staged:
        row = {
            "capsule": name,
            "pass": True,
            "barrier_tier": cert_tier,
            "barrier_status": "pass",
            **({"execution_digest": execution} if execution is not None else {}),
        }
        resolved = P.record_cert.__wrapped__(
            ws, {"per_capsule": [row]}, cert_tier, log, identity=identity, source_identity_verified=True
        )
        if not resolved:
            # Never leave a record outstanding for a job nobody will run.
            state = P._tier_state(ws)
            slots = P._slots(state, name, cert_tier)
            if (slots.get(identity) or {}).get("status") == "pending":
                del slots[identity]
                if P.record_identity(state[name].get(cert_tier)) == identity:
                    state[name][cert_tier] = next(reversed(slots.values()), None) if slots else None
                    if state[name][cert_tier] is None:
                        del state[name][cert_tier]
                P._save_tier_state(ws, state)
        recorded += resolved
    if log is not None and recorded:
        print(
            f"[promote] self-check {cert_tier} certificates kept for source {digest}: {recorded}", file=log, flush=True
        )
    return recorded


# ---- raw_selfcheck_evidence --------------------------------------------------------------------
# Retain raw ordinary selfchecks in an operator-selected host-only directory.

ROOT_ENV = "MERLIN_PHASE1_SELFCHECK_EVIDENCE_ROOT"

WORKSPACE_ENV = "MERLIN_PHASE1_SELFCHECK_VISIBLE_WORKSPACE"


def _ordinary_absolute(raw: str, label: str) -> Path:
    path = Path(raw)
    if not raw or not path.is_absolute() or path.resolve() != path:
        raise ValueError(f"{label} must be a canonical absolute directory without symlinks")
    if path.exists() and not path.is_dir():
        raise ValueError(f"{label} must be an ordinary directory")
    return path


def _selected_root(environment: Mapping[str, str]) -> Path | None:
    if ROOT_ENV not in environment and WORKSPACE_ENV not in environment:
        return None
    root = _ordinary_absolute(environment.get(ROOT_ENV, ""), "selfcheck evidence root")
    workspace = _ordinary_absolute(environment.get(WORKSPACE_ENV, ""), "visible workspace")
    if not workspace.is_dir():
        raise ValueError("visible workspace must exist before selecting selfcheck evidence")
    if root == workspace or root in workspace.parents or workspace in root.parents:
        raise ValueError("selfcheck evidence root must be disjoint from the visible workspace")
    return root


def broker_environment(root: Path | None, workspace: Path, *, environ=None) -> dict[str, str]:
    """Select from trusted broker configuration; requests cannot choose raw paths."""
    environment = dict(os.environ if environ is None else environ)
    for key in (ROOT_ENV, WORKSPACE_ENV):
        environment.pop(key, None)
    if root is not None:
        environment[ROOT_ENV] = str(root)
        environment[WORKSPACE_ENV] = str(workspace)
        _selected_root(environment)
    return environment


def operator_log_dir(*, environ=None) -> Path | None:
    """Operator-only directory for broker child logs, or ``None`` when no evidence root is selected.

    A child's raw stdout/stderr can name host paths; spooled into the agent's channel they were
    readable by the agent in full. With a selected evidence root they live beside the raw runs.
    """
    environment = os.environ if environ is None else environ
    try:
        root = _selected_root(environment)
    except ValueError:
        return None  # an unusable selection keeps the historical channel location rather than failing a check
    if root is None:
        return None
    directory = root / "broker-logs"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    return directory


def create_runs_root(*, environ=None) -> Path:
    """Keep failed and successful raw runs; legacy unselected calls use TMPDIR."""
    environment = os.environ if environ is None else environ
    root = _selected_root(environment)
    if root is not None:
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        _selected_root(environment)
    return Path(tempfile.mkdtemp(prefix="selfcheck_", dir=root))


# ---- selfcheck_notes ---------------------------------------------------------------------------
# The note a self-check shows when the author's package ran no test case, chosen by failure plane.
#
# Every package failure used to carry the same note about cmake builds. An integrity refusal (for
# example an editable module under a namespace the full-compiler selection does not contain) is not a
# build problem, and the build note sent the author looking at cmake instead of the rejected path.

BUILD_NOTE = (
    "Your package did NOT build/scan, so ZERO capsules ran — this is NOT a stubbed "
    "grader. Fix the build first (see 'detail' for the cmake/compiler stderr). NOTE: "
    "the grade builds an ISOLATED COPY of your package WITHOUT the build/ tree, so your "
    "manifest 'build' must support a CLEAN configure (provide a 'configure' step, e.g. "
    "cmake -S <src> -B <build>, not only 'cmake --build'); build commands run from the "
    "package root, so relative paths and $MLIR_DIR/$LLVM_DIR/$CM env vars are honored."
)

INTEGRITY_NOTE = (
    "Your package was REFUSED by its integrity scan before any build, so ZERO capsules ran — this is "
    "not a build or grader failure. 'detail' names what the scan refused"
)


def package_failure_note(failure) -> str:
    """The note for a package failure: integrity refusals explain the rule, build failures the build."""
    plane = str((failure or {}).get("plane") or "") if isinstance(failure, dict) else ""
    if plane != "integrity":
        return BUILD_NOTE
    hint = package_rejection.namespace_hint(failure)
    if not hint:
        return INTEGRITY_NOTE + "; remove or change it and re-run the check."
    allowed = hint.get("allowed_namespaces")
    return (
        INTEGRITY_NOTE
        + f": {hint['rejected_path']}. {hint['rule']}"
        + (f" Allowed: {', '.join(allowed)}." if allowed else "")
    )


# ---- result_details ----------------------------------------------------------------------------
# Host import of the standalone public result index/detail transport.


def compact_document(document, *, workspace, namespace="grade"):
    from merlin.development.backend.feedback.disclosure import project

    return _compact_document(project(document, workspace=workspace), workspace=workspace, namespace=namespace)


def compact_row(row, *, workspace):
    from merlin.development.backend.feedback.disclosure import project, restricted

    if restricted():
        row = project(load_row(row, workspace=workspace))
    return _compact_row(row, workspace=workspace)


__all__ = [
    "MAX_DETAIL_BYTES",
    "SCHEMA",
    "append_rows",
    "compact_document",
    "compact_row",
    "hydrate",
    "load_row",
]
