"""Read an experiment's own records into one summary: what the dashboard and ``watch`` show.

NOTHING HERE MEASURES, GRADES OR BUILDS.  Every number is a field an owner already wrote -- the
orchestrator's ``orchestration.json``; a phase-1 run's ``qa_history/verdict_*.json``, ``plateau.json``,
``oot_commits.jsonl``, ``freeze.json``, ``run_manifest.yaml`` and ``qa_loop_summary.yaml``; a phase-2
run's ``run.json``, ``resumed_seed.json``, ``iterations.jsonl`` and ``stage/`` records, and its
measurement store's ``job.json`` / ``result.json`` / ``plateau.json`` / ``board_outages.jsonl`` -- read
through the owners' own vocabulary (:mod:`..phase2.whole_model_measured.jobs`, ``gates``, ``feedback``).
The few derived figures are arithmetic over those fields (a running minimum, an age, a sum of
per-group rooflines), and each is named as such where it is shown.

A RECORD THAT IS ABSENT IS SAID TO BE ABSENT.  A field whose record does not exist is ``None`` and
renders as "not recorded"; the inventory says which file was read, missing, unreadable or of an
unexpected schema.  A missing record is never a crash and never a default.

LIVENESS IS STATED FROM RECORDS.  A run that recorded why it stopped is STOPPED; a run whose newest
measured candidate (or, before any, whose start) is older than the stall threshold is STALLED.  A
process check, where one is shown, is labelled as the state at generation time.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ---- records -----------------------------------------------------------------------------------


def phase1(
    run_dir: Path,
    inventory: Inventory,
    *,
    now: float,
    stall_hours: float,
    orchestration_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """A phase-1 run's grades, capsule tiers, plateau, freeze and liveness, from its own records."""
    from merlin.experiments.tracking.readers.records import (
        ENDED,
        FINISHED,
        FROZEN,
        LIVE,
        STALLED,
        UNFINALIZED,
        UNKNOWN,
        Inventory,
        _age_state,
        _capsule_row,
        _int,
        _orchestration_ended,
        _text,
        epoch,
        run_started,
        stamp,
        tier_key,
    )

    run_dir = Path(run_dir)
    history = run_dir / "qa_history"
    files = sorted(history.glob("verdict_*.json")) if history.is_dir() else []
    if not files:
        inventory.note("qa_history/verdict_*.json", history, "absent")
    grades = []
    for path in files:
        document = inventory.json(f"qa_history/{path.name}", path)
        if document is None:
            continue
        label, _, key = path.stem[len("verdict_") :].partition("_")
        capsules = [_capsule_row(row) for row in document.get("per_capsule") or () if isinstance(row, Mapping)]
        grades.append(
            {
                "name": path.stem,
                "label": label,
                "key": key,
                "at": epoch(document.get("graded_at")),
                "n_passed": _int(document.get("n_passed")),
                "n_capsules": _int(document.get("n_capsules")),
                "all_pass": document.get("all_pass"),
                "highest_tier": document.get("highest_tier"),
                "first_failure_planes": dict(document.get("first_failure_planes") or {}),
                "package_failure": document.get("package_failure"),
                "capsules": capsules,
            }
        )
    grades.sort(key=lambda g: (g["at"] is None, g["at"] or 0.0, g["name"]))
    tiers = sorted({t for g in grades for c in g["capsules"] for t in c["tiers"]}, key=tier_key)
    latest = grades[-1] if grades else None
    failing = (
        sorted(
            (c for c in latest["capsules"] if c["status"] != "pass"),
            key=lambda c: (c["failure_plane"] is None, str(c["failure_plane"]), c["capsule"]),
        )
        if latest
        else []
    )

    plateau_doc = inventory.json("plateau.json", run_dir / "plateau.json")
    plateau = None
    if plateau_doc is not None:
        never = list(plateau_doc.get("never_passed") or [])
        plateau = {
            "stuck": plateau_doc.get("stuck"),
            "sentence": plateau_doc.get("sentence") or plateau_doc.get("reason"),
            "stalled_grades": plateau_doc.get("stalled_grades"),
            "n_grades": plateau_doc.get("n_grades"),
            "best_passed": plateau_doc.get("best_passed"),
            "latest_passed": plateau_doc.get("latest_passed"),
            "n_capsules": plateau_doc.get("n_capsules"),
            "never_passed": len(never),
            "regressed": list(plateau_doc.get("regressed") or []),
        }

    commits = inventory.jsonl("oot_commits.jsonl", run_dir / "oot_commits.jsonl")
    oot_rows = None
    if commits is not None:
        oot_rows = [
            {
                "label": row.get("label"),
                "key": row.get("key"),
                "commit": (row.get("oot") or {}).get("commit"),
                "at": epoch((row.get("oot") or {}).get("committed_at")),
                "package": (row.get("oot") or {}).get("package_digest"),
                "n_passed": _int(row.get("n_passed")),
                "n_capsules": _int(row.get("n_capsules")),
                "error": _text(row.get("error")),
            }
            for row in commits
        ]

    freeze_doc = inventory.json("freeze.json", run_dir / "freeze.json")
    freeze = None
    if freeze_doc is not None:
        frozen_oot = freeze_doc.get("oot") or {}
        freeze = {
            "at": epoch(freeze_doc.get("frozen_at")),
            "submission_sha256": freeze_doc.get("submission_sha256"),
            "frozen_commit": frozen_oot.get("frozen_commit"),
            "oot_error": _text(frozen_oot.get("error")),
            "mutated_after_freeze": freeze_doc.get("workspace_mutable_after_freeze"),
        }

    manifest_doc = inventory.yaml("run_manifest.yaml", run_dir / "run_manifest.yaml")
    manifest = None
    if manifest_doc is not None:
        process = manifest_doc.get("process") or {}
        manifest = {
            "arm": manifest_doc.get("arm"),
            "model": manifest_doc.get("model"),
            "public": {k: (manifest_doc.get("public_dev") or {}).get(k) for k in ("passed", "highest_tier")},
            "hidden": {k: (manifest_doc.get("hidden") or {}).get(k) for k in ("passed", "highest_tier")},
            "formal_grade_complete": (manifest_doc.get("completion") or {}).get("formal_grade_complete"),
            "completion_failures": list((manifest_doc.get("completion") or {}).get("failures") or []),
            "wall_time_seconds": process.get("wall_time_seconds"),
            "tokens_total": process.get("tokens_total"),
            "estimated_cost_usd": process.get("estimated_cost_usd"),
        }

    summary_doc = inventory.yaml("qa_loop_summary.yaml", run_dir / "qa_loop_summary.yaml")
    summary = None
    if summary_doc is not None:
        summary = {
            "n_rounds": summary_doc.get("n_rounds"),
            "converged": summary_doc.get("converged"),
            "formal_complete": summary_doc.get("formal_complete"),
            "numeric_all_pass": summary_doc.get("numeric_all_pass"),
            "wall_seconds": summary_doc.get("wall_seconds"),
            "cost_capped": summary_doc.get("cost_capped"),
            "feedback_healthy": (summary_doc.get("feedback_health") or {}).get("healthy"),
        }

    last_grade = max((g["at"] for g in grades if g["at"] is not None), default=None)
    ended = _orchestration_ended(orchestration_record)
    if ended is not None:
        liveness = {"state": ENDED, "hours": None, "detail": f"orchestration state {ended}"}
    elif summary is not None or (manifest is not None and manifest["formal_grade_complete"] is True):
        liveness = {
            "state": FINISHED,
            "hours": None,
            "detail": (
                f"qa_loop_summary.yaml: formal_complete={summary['formal_complete']}"
                if summary is not None
                else "run_manifest.yaml: formal_grade_complete=True"
            ),
        }
    elif freeze is not None:
        liveness = {
            "state": FROZEN,
            "hours": None,
            "detail": f"frozen {stamp(freeze['at']) or 'at an unrecorded time'}; formal completion not recorded",
        }
    else:
        reference = last_grade if last_grade is not None else run_started(run_dir, inventory)
        state, hours, detail = _age_state(
            reference, now, stall_hours, "the last grade" if last_grade else "the run start"
        )
        liveness = {
            "state": UNFINALIZED,
            "hours": hours,
            "activity_state": "STALE" if state == STALLED else "FRESH" if state == LIVE else UNKNOWN,
            "detail": f"completion and current process liveness not recorded; {detail}",
        }
    liveness["last_grade"] = last_grade

    return {
        "grades": grades,
        "tiers": tiers,
        "latest": latest,
        "failing": failing,
        "plateau": plateau,
        "oot_commits": oot_rows,
        "freeze": freeze,
        "manifest": manifest,
        "qa_loop_summary": summary,
        "liveness": liveness,
    }


def _phase2_names() -> dict[str, Any]:
    from merlin.development.backend.optimization.whole_model_measured.measurement import board_batches as B
    from merlin.development.backend.optimization.whole_model_measured.measurement import jobs as J
    from merlin.development.backend.optimization.whole_model_measured.runs import ledger as L
    from merlin.development.backend.optimization.whole_model_measured.runs import runs as R
    from merlin.development.backend.optimization.whole_model_measured.runs import sessions as S
    from merlin.development.backend.optimization.whole_model_measured.runs import watchdog as W

    return {"B": B, "J": J, "L": L, "R": R, "S": S, "W": W}


def phase2_markers() -> tuple[str, ...]:
    names = _phase2_names()
    return ("run.json", "resumed_seed.json", names["R"].CONFIG_NAME, names["L"].ITERATIONS)


def failure_class(job: Mapping[str, Any], result: Mapping[str, Any] | None) -> tuple[str, str | None]:
    """``(class, reason)`` of one store job, from the fields its owners wrote.

    ``measured`` (a valid reading), ``pending`` (not finished), ``superseded`` or one of
    :data:`FAILURE_CLASSES`: ``correctness`` (a wrong program: ``MEASURED_INVALID``, the functional gate,
    a capsule screen that failed on the candidate's own bytes), ``infra`` (a fault the record names as
    infrastructure's, never a fact about the bytes), ``prohibited_instruction`` (the whole-ELF
    instruction rule refused it), ``declined`` (the coverage gate: groups declined to the library) and
    ``refused`` (no admissible reading, for any other recorded reason)."""
    from merlin.analysis.whole_model.grading import verdict as V
    from merlin.development.backend.optimization.whole_model_measured.measurement import jobs as J
    from merlin.development.backend.optimization.whole_model_measured.scoring import gates as G
    from merlin.experiments.tracking.readers.records import MEASURED, OPEN, SUPERSEDED, _text

    state = job.get("state")
    if state == J.SUPERSEDED:
        return SUPERSEDED, _text((result or {}).get("refusal") or job.get("notice"))
    if result is None:
        if state in J.TERMINAL:
            failure = str(job.get("failure") or "")
            cls = "infra" if failure.startswith("infra_") or G.is_infra_refusal(failure) else "refused"
            return cls, _text(failure) or "the job ended without a result.json"
        return OPEN, _text(job.get("notice"))
    status = result.get("timing_status")
    refusal = str(result.get("refusal") or (result.get("verdict") or {}).get("refusal") or "")
    if status == V.TIMING_MEASURED:
        return MEASURED, None
    if status == V.TIMING_MEASURED_INVALID:
        return "correctness", _text(
            (result.get("verdict") or {}).get("invalid_reason") or refusal
        ) or "MEASURED_INVALID"
    if result.get("isa_prohibited"):
        return "prohibited_instruction", _text(refusal)
    if result.get("coverage_regression"):
        return "declined", _text(refusal)
    if result.get("functional_gate_failed"):
        return "correctness", _text(refusal)
    if result.get("screen_failed"):
        infra = G.screen_was_infra(result.get("pre_measure_check"))
        return ("infra" if infra else "correctness"), _text(refusal)
    if any(str(k).startswith("infra_") and v is True for k, v in result.items()):
        return "infra", _text(refusal)
    if refusal.startswith("infra_") or G.is_infra_refusal(refusal):
        return "infra", _text(refusal)
    if status in (None, J.TIMING_PENDING) and state not in J.TERMINAL:
        return OPEN, _text(job.get("notice"))
    return "refused", _text(refusal) or f"timing_status {status}"
