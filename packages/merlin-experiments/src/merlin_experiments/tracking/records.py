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

SCHEMA = "merlin_experiment_tracking_summary_v1"
#: Hours without a measured candidate (or a grade) after which a run that recorded no stop is STALLED.
DEFAULT_STALL_HOURS = 6.0

LIVE, STALLED, STOPPED, FINISHED, ENDED, RELAUNCHED, UNKNOWN = (
    "LIVE",
    "STALLED",
    "STOPPED",
    "FINISHED",
    "ENDED",
    "RELAUNCHED",
    "UNKNOWN",
)

#: Failure classes of a phase-2 candidate, in the order the views list them.
MEASURED = "measured"
FAILURE_CLASSES = ("correctness", "infra", "refused", "prohibited_instruction", "declined")
OPEN, SUPERSEDED = "pending", "superseded"

#: Orchestration states that mean an attempt is still in flight.
_IN_FLIGHT = ("pending", "running")


# --------------------------------------------------------------------------- reading
class Inventory:
    """Every record a summary consulted, and what was found."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def note(self, name: str, path: Path, state: str, detail: str | None = None) -> None:
        self.rows.append({"record": name, "path": str(path), "state": state, "detail": detail})

    def _text(self, name: str, path: Path) -> str | None:
        path = Path(path)
        if path.is_symlink() and not path.exists():
            self.note(name, path, "unreadable", "dangling symlink")
            return None
        if not path.is_file():
            self.note(name, path, "absent")
            return None
        try:
            return path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            self.note(name, path, "unreadable", f"{type(exc).__name__}: {exc}")
            return None

    def json(self, name: str, path: Path, *, schema: str | None = None) -> dict[str, Any] | None:
        text = self._text(name, path)
        if text is None:
            return None
        try:
            document = json.loads(text)
        except ValueError as exc:
            self.note(name, path, "unreadable", f"invalid JSON: {exc}")
            return None
        return self._mapping(name, path, document, schema)

    def yaml(self, name: str, path: Path) -> dict[str, Any] | None:
        from merlin.common.yaml import safe_load_text

        text = self._text(name, path)
        if text is None:
            return None
        try:
            document = safe_load_text(text)
        except Exception as exc:  # noqa: BLE001 -- any parser error is an unreadable record
            self.note(name, path, "unreadable", f"invalid YAML: {type(exc).__name__}")
            return None
        return self._mapping(name, path, document, None)

    def jsonl(self, name: str, path: Path) -> list[dict[str, Any]] | None:
        text = self._text(name, path)
        if text is None:
            return None
        rows, bad = [], 0
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                bad += 1
                continue
            if isinstance(row, dict):
                rows.append(row)
            else:
                bad += 1
        self.note(name, path, "read", f"{len(rows)} rows" + (f", {bad} unreadable" if bad else ""))
        return rows

    def _mapping(self, name: str, path: Path, document: Any, schema: str | None) -> dict[str, Any] | None:
        if not isinstance(document, dict):
            self.note(name, path, "unreadable", "not a mapping")
            return None
        found = document.get("schema")
        if schema is not None and found != schema:
            self.note(name, path, "unexpected schema", f"{found!r}; this reader follows {schema!r}")
        else:
            self.note(name, path, "read")
        return document


def epoch(value: Any) -> float | None:
    """A recorded time as epoch seconds: a number, a ``YYYYmmddTHHMMSSZ`` stamp or ISO-8601; else None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    try:
        return datetime.strptime(text, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC).timestamp()
    except ValueError:
        pass
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.timestamp()


def stamp(value: float | None) -> str | None:
    """Epoch seconds as ``YYYY-mm-dd HH:MM UTC`` (None stays None)."""
    if value is None:
        return None
    return datetime.fromtimestamp(float(value), UTC).strftime("%Y-%m-%d %H:%M UTC")


def tier_key(tier: str) -> tuple[str, int, str]:
    """Order tier names by their letter prefix, then their number (``L2`` before ``L10``)."""
    prefix = tier.rstrip("0123456789")
    digits = tier[len(prefix) :]
    return (prefix, int(digits) if digits else -1, tier)


def _text(value: Any, limit: int = 300) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text[:limit] if text else None


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def process_present(pid: Any) -> bool | None:
    """Whether a process with ``pid`` exists on THIS host now (None for no pid). Generation-time state."""
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return None
    return Path(f"/proc/{pid}").is_dir()


def run_started(run_dir: Path, inventory: Inventory) -> float | None:
    """When the run began: its run record's timestamp, else the leading stamp of its sortable name."""
    record = inventory.json("run_record.json", Path(run_dir) / "run_record.json")
    for key in ("timestamp", "created_at"):
        found = epoch((record or {}).get(key))
        if found is not None:
            return found
    return epoch(Path(run_dir).name.partition("_")[0])


def _age_state(reference: float | None, now: float, stall_hours: float, what: str) -> tuple[str, float | None, str]:
    if reference is None:
        return UNKNOWN, None, f"no {what} and no run start recorded"
    hours = max(0.0, (now - reference) / 3600.0)
    state = STALLED if hours >= stall_hours else LIVE
    return state, round(hours, 2), f"{hours:.1f} h since {what} (stall threshold {stall_hours:g} h)"


# --------------------------------------------------------------------------- orchestration
def orchestration(run_dir: Path, inventory: Inventory) -> dict[str, Any] | None:
    """The orchestrator's record of a run: its frozen plan's binding, attempts and phase engines."""
    run_dir = Path(run_dir)
    plan_path = run_dir / "resolved-plan.json"
    plan = inventory.json("resolved-plan.json", plan_path)
    record = inventory.json("orchestration.json", run_dir / "orchestration.json")
    if plan is None and record is None:
        return None
    binding = "not recorded"
    if plan is not None and record is not None:
        digest = hashlib.sha256(plan_path.read_bytes()).hexdigest()
        binding = "bound" if digest == record.get("plan_sha256") else "frozen plan changed since the run froze it"
    attempts = []
    for entry in (record or {}).get("attempts") or ():
        if not isinstance(entry, Mapping):
            continue
        in_flight = entry.get("state") in _IN_FLIGHT
        attempts.append(
            {
                "phase": entry.get("phase"),
                "adapter": entry.get("adapter"),
                "state": entry.get("state"),
                "started": epoch(entry.get("started_at")),
                "ended": epoch(entry.get("ended_at")),
                "returncode": entry.get("returncode"),
                "pid": entry.get("pid"),
                "process_present": process_present(entry.get("pid")) if in_flight else None,
                "engine_output": entry.get("engine_output"),
            }
        )
    phases = {}
    for number, command in ((plan or {}).get("phases") or {}).items():
        if not isinstance(command, Mapping):
            continue
        latest = next((a for a in reversed(attempts) if str(a.get("phase")) == str(number)), {})
        phases[str(number)] = {
            "adapter": command.get("adapter"),
            "state": latest.get("state", "not_started"),
            "engine_output": latest.get("engine_output") or command.get("engine_output"),
        }
    return {
        "experiment": (plan or {}).get("experiment"),
        "target": (plan or {}).get("target"),
        "frozen_at": (plan or {}).get("frozen_at"),
        "state": (record or {}).get("state"),
        "plan_binding": binding,
        "attempts": attempts,
        "phases": phases,
    }


def _orchestration_ended(record: Mapping[str, Any] | None) -> str | None:
    state = (record or {}).get("state")
    return str(state) if state and state not in _IN_FLIGHT else None


# --------------------------------------------------------------------------- phase 1
PHASE1_MARKERS = ("qa_history", "oot_commits.jsonl", "freeze.json", "run_manifest.yaml", "qa_loop_summary.yaml")


def _capsule_row(row: Mapping[str, Any]) -> dict[str, Any]:
    tiers: dict[str, str | None] = {}
    for tier, value in (row.get("tiers") or {}).items():
        status = value.get("status") if isinstance(value, Mapping) else value
        tiers[str(tier)] = str(status) if status is not None else None
    passed = [tier for tier, status in tiers.items() if status == "pass"]
    return {
        "capsule": str(row.get("capsule") or row.get("name") or "?"),
        "label": row.get("label"),
        "status": row.get("status"),
        "tiers": tiers,
        "highest_pass": max(passed, key=tier_key) if passed else None,
        "failure_plane": row.get("failure_plane"),
        "failure_category": row.get("failure_category"),
        "failure_detail": _text(row.get("failure_detail") or row.get("detail"), 400),
    }


def phase1(
    run_dir: Path,
    inventory: Inventory,
    *,
    now: float,
    stall_hours: float,
    orchestration_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """A phase-1 run's grades, capsule tiers, plateau, freeze and liveness, from its own records."""
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
    if summary is not None:
        liveness = {
            "state": FINISHED,
            "hours": None,
            "detail": f"qa_loop_summary.yaml: formal_complete={summary['formal_complete']}",
        }
    elif freeze is not None:
        liveness = {
            "state": FINISHED,
            "hours": None,
            "detail": f"frozen {stamp(freeze['at']) or 'at an unrecorded time'}",
        }
    elif ended is not None:
        liveness = {"state": ENDED, "hours": None, "detail": f"orchestration state {ended}"}
    else:
        reference = last_grade if last_grade is not None else run_started(run_dir, inventory)
        state, hours, detail = _age_state(
            reference, now, stall_hours, "the last grade" if last_grade else "the run start"
        )
        liveness = {"state": state, "hours": hours, "detail": detail}
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


# --------------------------------------------------------------------------- phase 2
def _phase2_names() -> dict[str, Any]:
    from ..phase2.whole_model_measured import batch as B
    from ..phase2.whole_model_measured import jobs as J
    from ..phase2.whole_model_measured import ledger as L
    from ..phase2.whole_model_measured import runs as R
    from ..phase2.whole_model_measured import sessions as S
    from ..phase2.whole_model_measured import watchdog as W

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
    from merlin.perf import whole_model_verdict as V

    from ..phase2.whole_model_measured import gates as G
    from ..phase2.whole_model_measured import jobs as J

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


def _group_rows(result: Mapping[str, Any] | None) -> dict[str, Mapping[str, Any]]:
    from merlin.perf import whole_model_verdict as V

    return {
        str(row["group"]): row
        for row in V.group_table((result or {}).get("verdict") or {})
        if isinstance(row, Mapping) and row.get("group") is not None
    }


def _coverage(result: Mapping[str, Any], reference: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The package-authored share recorded by the candidate's own build routes (feedback's projection)."""
    from ..phase2.whole_model_measured import feedback as F

    if not ((result.get("build") or {}).get("groups")):
        return None
    authored = F.package_authored(result, reference)
    return {k: authored.get(k) for k in ("groups_answered", "groups_total", "priced_share")}


def _roofline(result: Mapping[str, Any], reference: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Per-group distance to the derived roofline the result recorded (``diagnostics.per_group``)."""
    from ..phase2.whole_model_measured import feedback as F

    if not ((result.get("diagnostics") or {}).get("per_group")):
        return []
    return F.roofline_gaps(result, _group_rows(result), _group_rows(reference))


class StoreCache:
    """Loaded measurement stores and reference results, so a target view reads each once."""

    def __init__(self) -> None:
        self.stores: dict[str, dict[str, Any]] = {}
        self.results: dict[str, dict[str, Any] | None] = {}

    def result(self, path: str | None, inventory: Inventory, name: str) -> dict[str, Any] | None:
        if not path:
            return None
        if path not in self.results:
            self.results[path] = inventory.json(name, Path(path))
        return self.results[path]


def _standing_result(job_dir: Path) -> dict[str, Any] | None:
    """The result that stands for a job across all its attempts, read by the measured mode's own reader
    (:func:`..phase2.whole_model_measured.attempts.effective_result`): a re-queued job whose current
    attempt was lost to the host still shows the earlier verdict about its bytes, marked
    ``from_attempt``, instead of the infrastructure outcome that displaced its ``result.json``."""
    from ..phase2.whole_model_measured import attempts as A

    try:
        result = A.effective_result(job_dir)
    except (OSError, ValueError):
        return None
    return dict(result) if isinstance(result, Mapping) else None


def load_store(root: Path, inventory: Inventory, cache: StoreCache) -> dict[str, Any]:
    """Every job of one measurement store, classified, plus the store's own plateau and board records."""
    from ..phase2.whole_model_measured.machines import INFRA_BOARD_UNAVAILABLE

    names = _phase2_names()
    B, J, S = names["B"], names["J"], names["S"]
    key = str(Path(root).resolve())
    if key in cache.stores:
        return cache.stores[key]
    root = Path(root)
    candidates: list[dict[str, Any]] = []
    roofline_best: dict[str, Any] | None = None
    unreadable = 0
    for job_path in sorted(root.glob("*/job.json")):
        try:
            job = json.loads(job_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            job = None
        if not isinstance(job, dict):
            unreadable += 1
            continue
        job_dir = job_path.parent
        result = _standing_result(job_dir)
        try:
            attribution = json.loads((job_dir / J.ATTRIBUTION_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            attribution = None
        reference_path = (job.get("reference") or {}).get("path") if isinstance(job.get("reference"), Mapping) else None
        reference = cache.result(reference_path, inventory, "job reference result.json")
        cls, reason = failure_class(job, result)
        verdict = (result or {}).get("verdict") or {}
        batch = (result or {}).get("batch") if isinstance((result or {}).get("batch"), Mapping) else None
        gaps = _roofline(result, reference) if result else []
        row = {
            "key": job_dir.name,
            "package": job.get("package_sha256"),
            "replicate": _int(job.get("replicate")) or 0,
            "role": job.get("role"),
            "label": _text(job.get("label"), 120),
            "state": job.get("state"),
            "solo": bool(job.get("solo")),
            "requested": epoch(job.get("requested_epoch")) or epoch(job.get("requested_at")),
            "finished": epoch((result or {}).get("finished_at")) if result else None,
            "from_attempt": ((result or {}).get("from_attempt") or {}).get("attempt"),
            "board_ready": epoch(job.get("board_ready_epoch")),
            "timing_status": (result or {}).get("timing_status") or job.get("timing_status"),
            "cycles": _int((result or {}).get("objective_cycles")),
            "whole_window": _int(verdict.get("whole_window_cycles")),
            "class": cls,
            "reason": reason,
            "attribution": J.attribution_state(attribution if isinstance(attribution, Mapping) else None),
            "attributable": J.attributable(attribution if isinstance(attribution, Mapping) else None),
            "batch": {
                "id": batch.get("batch"),
                "size": batch.get("size"),
                "control_ok": (batch.get("control") or {}).get("ok"),
            }
            if batch
            else None,
            "coverage": _coverage(result, reference) if result else None,
            "roofline": {
                "ours": sum(g["ours"] for g in gaps),
                "roofline": sum(g["roofline"] for g in gaps),
                "groups": len(gaps),
            }
            if gaps
            else None,
        }
        candidates.append(row)
        if (
            gaps
            and row["class"] == MEASURED
            and (roofline_best is None or (row["cycles"] or 0) < roofline_best["cycles"])
        ):
            roofline_best = {"key": row["key"], "cycles": row["cycles"] or 0, "gaps": gaps}
    inventory.note(
        "store job.json/result.json",
        root,
        "read" if candidates else "absent",
        f"{len(candidates)} jobs" + (f", {unreadable} unreadable job.json" if unreadable else ""),
    )
    candidates.sort(key=lambda c: (c["finished"] or c["requested"] or 0.0, c["key"]))

    plateau_doc = inventory.json("store plateau.json", root / S.PLATEAU_FILE, schema=S.PLATEAU_SCHEMA)
    plateau = None
    if plateau_doc is not None:
        trace = [row for row in plateau_doc.get("trace") or () if isinstance(row, Mapping)]
        plateau = {
            "rule": plateau_doc.get("rule"),
            "last_improvement": plateau_doc.get("last_improvement"),
            "sessions": [
                {
                    "at": epoch(row.get("epoch")) or epoch(row.get("at")),
                    "run": Path(str(row.get("run") or "")).parent.name if row.get("run") else None,
                    "session": row.get("session"),
                    "cycles": _int(row.get("cycles")),
                    "package": row.get("package_sha256"),
                    "reset": bool(row.get("reset")),
                    "abandoned": _text(row.get("abandoned"), 200),
                    "model": row.get("model"),
                }
                for row in trace
            ],
        }

    outage_doc = inventory.json("store board_outage.json", root / B.BOARD_OUTAGE)
    closed = inventory.jsonl("store board_outages.jsonl", root / B.BOARD_OUTAGES)

    def outage_row(doc: Mapping[str, Any]) -> dict[str, Any]:
        failures = [f for f in doc.get("failures") or () if isinstance(f, Mapping)]
        return {
            "opened": epoch(doc.get("opened_epoch")) or epoch(doc.get("opened_at")),
            "closed": epoch(doc.get("closed_at")),
            "failures": len(failures),
            "last_reason": _text((failures[-1] if failures else {}).get("reason"), 240),
            "retry_after": epoch(doc.get("retry_after_epoch")),
            "closed_by_batch": doc.get("closed_by_batch"),
        }

    streak = inventory.json("store solo_streak.json", root / B.SOLO_STREAK_FILE)
    runner = inventory.json("store batch_runner.json", root / "batch_runner.json")
    batches_dir = root / "batches"
    batch_dirs = sorted(p for p in batches_dir.iterdir() if p.is_dir()) if batches_dir.is_dir() else []
    recent_batches = []
    for batch_dir in batch_dirs[-12:]:
        try:
            record = json.loads((batch_dir / "batch.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            record = None
        record = record if isinstance(record, dict) else {}
        control = record.get("control") if isinstance(record.get("control"), Mapping) else {}
        recent_batches.append(
            {
                "id": batch_dir.name,
                "jobs": len(record.get("jobs") or ()),
                "control_ok": control.get("ok"),
                "control_ratio": control.get("ratio"),
                "board_unavailable": bool(record.get(INFRA_BOARD_UNAVAILABLE)),
                "failure": _text(record.get("failure"), 200),
            }
        )
    store = {
        "root": str(root),
        "candidates": candidates,
        "roofline_best": roofline_best,
        "plateau": plateau,
        "board": {
            "open_outage": outage_row(outage_doc) if outage_doc else None,
            "closed_outages": [outage_row(row) for row in closed] if closed is not None else None,
            "solo_streak": _int((streak or {}).get("count")) if streak is not None else None,
            "batch_runner": {
                "pid": (runner or {}).get("pid"),
                "started": epoch((runner or {}).get("started_at")),
                "process_present": process_present((runner or {}).get("pid")),
            }
            if runner is not None
            else None,
            "batches": len(batch_dirs),
            "recent_batches": recent_batches,
        },
    }
    cache.stores[key] = store
    return store


def _lineage_chain(seed_doc: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    chain, node, depth = [], seed_doc, 0
    while isinstance(node, Mapping) and depth < 64:
        if node.get("resumed_from_run"):
            chain.append(
                {
                    "run": str(node["resumed_from_run"]),
                    "seed_package": node.get("seed_package_sha256"),
                    "kind": node.get("lineage_kind"),
                }
            )
        node, depth = node.get("lineage"), depth + 1
    return chain


def _relaunched_as(run_dir: Path, run_schema: str) -> str | None:
    """A sibling run whose ``run.json`` says it was resumed from this run, if any."""
    here = Path(run_dir).resolve()
    parent = here.parent
    for sibling in sorted(parent.iterdir()) if parent.is_dir() else ():
        if sibling == here or not sibling.is_dir():
            continue
        try:
            record = json.loads((sibling / "run.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(record, dict) and record.get("schema") == run_schema and record.get("resumed_from_run"):
            if Path(str(record["resumed_from_run"])).resolve() == here:
                return str(sibling)
    return None


def _index_rows(target: str | None, run_dir: Path, inventory: Inventory) -> dict[str, list] | None:
    """The target's INDEX.yaml rows that cite this run (read from the written file; never regenerated)."""
    if not target:
        return None
    from merlin.common import paths
    from merlin.targetgen import target_index

    try:
        path = target_index.index_path(target)
    except (ValueError, OSError):
        return None
    document = inventory.yaml("INDEX.yaml", path)
    if document is None:
        return None
    wanted = {str(Path(run_dir).resolve())}
    try:
        wanted.add(Path(run_dir).resolve().relative_to(paths.out_dir().resolve()).as_posix())
    except ValueError:
        pass

    def cites(row: Any) -> bool:
        if not isinstance(row, Mapping):
            return False
        values = [row.get("run"), row.get("source_run"), *((row.get("lineage") or {}).values())]
        return any(isinstance(v, str) and v in wanted for v in values)

    sections = ("phase0_releases", "phase1_frozen", "phase2_best", "champions")
    return {name: [row for row in document.get(name) or () if cites(row)] for name in sections}


def phase2(
    run_dir: Path,
    inventory: Inventory,
    *,
    now: float,
    stall_hours: float,
    store: Path | None = None,
    cache: StoreCache | None = None,
    target: str | None = None,
) -> dict[str, Any]:
    """A phase-2 measured run: its candidates over time, failures, plateau, liveness, board and lineage."""
    names = _phase2_names()
    R, L, S, J = names["R"], names["L"], names["S"], names["J"]
    cache = cache or StoreCache()
    run_dir = Path(run_dir)
    run = inventory.json("run.json", run_dir / "run.json", schema=R.RUN_SCHEMA)
    seed = inventory.json("resumed_seed.json", run_dir / "resumed_seed.json", schema=R.RESUMED_SEED_SCHEMA)
    config = inventory.json(R.CONFIG_NAME, run_dir / R.CONFIG_NAME)
    ledger = inventory.jsonl(L.ITERATIONS, run_dir / L.ITERATIONS)
    stage = run_dir / "stage"
    sessions_doc = inventory.json("stage/sessions.json", stage / "sessions.json", schema=S.SEQUENCE_SCHEMA)
    breaker = inventory.json("stage/infra_circuit_breaker.json", stage / "infra_circuit_breaker.json")
    host = inventory.json("stage/host_resource_telemetry.json", stage / "host_resource_telemetry.json")
    rounds = []
    rounds_dir = stage / "rounds"
    for path in sorted(rounds_dir.glob("round_*.round.json")) if rounds_dir.is_dir() else ():
        document = inventory.json(f"stage/rounds/{path.name}", path)
        if document is not None:
            rounds.append(
                {
                    "round": document.get("round"),
                    "status": document.get("status"),
                    "why": _text(document.get("why"), 200),
                    "wall_seconds": document.get("wall_seconds"),
                    "requested": len(document.get("requested") or ()),
                    "final_package": document.get("final_package_sha256"),
                    "final_timing_status": document.get("final_timing_status"),
                }
            )

    store_roots = (seed or {}).get("store_roots") if isinstance((seed or {}).get("store_roots"), Mapping) else {}
    store_root = (
        Path(store) if store is not None else (Path(store_roots["screen"]) if store_roots.get("screen") else None)
    )
    store_source = "--store" if store is not None else ("resumed_seed.json store_roots.screen" if store_root else None)
    store_doc = None
    if store_root is not None:
        if store_root.is_dir():
            store_doc = load_store(store_root, inventory, cache)
        else:
            inventory.note("measurement store", store_root, "absent")
    else:
        inventory.note("measurement store", run_dir / "resumed_seed.json", "absent", "no store_roots.screen recorded")

    screen_section = (config or {}).get("screen") if isinstance((config or {}).get("screen"), Mapping) else {}
    bar_path = screen_section.get("reference")
    bar_doc = cache.result(str(bar_path), inventory, "screen reference result.json") if bar_path else None
    from merlin.perf import whole_model_verdict as V

    bar = None
    if bar_doc is not None:
        bar = {
            "cycles": _int((bar_doc.get("verdict") or {}).get("whole_window_cycles")),
            "timing_status": bar_doc.get("timing_status"),
            "label": _text(bar_doc.get("label"), 120),
            "admissible_as_bar": bar_doc.get("timing_status") == V.TIMING_MEASURED,
        }
    orientation = [
        {
            "label": _text(row.get("label"), 160),
            "cycles": _int(row.get("whole_model_cycles")),
            "note": _text(row.get("note"), 200),
        }
        for row in (config or {}).get("orientation") or ()
        if isinstance(row, Mapping) and _int(row.get("whole_model_cycles"))
    ]

    started = run_started(run_dir, inventory)
    candidates = [dict(c) for c in (store_doc or {}).get("candidates") or ()]
    for row in candidates:
        row["in_run"] = None if started is None or row["requested"] is None else row["requested"] >= started
    measured = [c for c in candidates if c["class"] == MEASURED and c["cycles"] is not None]
    eligible = [c for c in measured if c["replicate"] == 0 and c["attributable"] and c["role"] != J.ROLE_REFERENCE]
    best_line, lowest = [], None
    for row in sorted(eligible, key=lambda c: c["finished"] or c["requested"] or 0.0):
        if lowest is None or row["cycles"] < lowest["cycles"]:
            lowest = row
        best_line.append({"at": row["finished"] or row["requested"], "cycles": lowest["cycles"], "key": lowest["key"]})
    counts = {name: 0 for name in (MEASURED, *FAILURE_CLASSES, OPEN, SUPERSEDED)}
    counts_in_run = dict(counts)
    for row in candidates:
        counts[row["class"]] = counts.get(row["class"], 0) + 1
        if row["in_run"]:
            counts_in_run[row["class"]] = counts_in_run.get(row["class"], 0) + 1

    landed = [
        c["finished"]
        for c in candidates
        if c["class"] in (MEASURED, "correctness")
        and c["whole_window"] is not None
        and c["finished"] is not None
        and (started is None or c["finished"] >= started)
    ]
    last_measured = max(landed, default=None)
    last_measured_store = max(
        (
            c["finished"]
            for c in candidates
            if c["class"] in (MEASURED, "correctness") and c["whole_window"] is not None and c["finished"]
        ),
        default=None,
    )
    stopped = (sessions_doc or {}).get("stopped") if isinstance((sessions_doc or {}).get("stopped"), Mapping) else None
    relaunched = _relaunched_as(run_dir, R.RUN_SCHEMA)
    if stopped:
        liveness = {
            "state": STOPPED,
            "hours": None,
            "detail": f"{stopped.get('kind')}: {_text(stopped.get('reason'), 240)}",
        }
    elif relaunched:
        liveness = {"state": RELAUNCHED, "hours": None, "detail": f"resumed as {Path(relaunched).name}"}
    else:
        what = (
            "the last measured candidate"
            if last_measured is not None
            else "the run start (no candidate measured since)"
        )
        state, hours, detail = _age_state(
            last_measured if last_measured is not None else started, now, stall_hours, what
        )
        if store_doc is None:
            # No store: nothing was ever requested from it, or it lives where the run does not record.
            state = STALLED if state == STALLED else UNKNOWN
            detail = f"no measurement store found; {detail}"
        liveness = {"state": state, "hours": hours, "detail": detail}
    liveness.update(
        last_measured=last_measured,
        last_measured_store=last_measured_store,
        run_started=started,
        stall_hours=stall_hours,
    )

    board = dict((store_doc or {}).get("board") or {})
    if store_doc is not None:
        waiting = [c for c in candidates if c["state"] == J.BOARD]
        board.update(
            queue=len(waiting),
            oldest_waiting=min((c["board_ready"] for c in waiting if c["board_ready"]), default=None),
            running=sum(1 for c in candidates if c["state"] == J.RUNNING),
            pending=sum(1 for c in candidates if c["state"] in (J.PENDING, J.SCREENING)),
        )

    ledger_rows = ledger or []
    ledger_view = None
    if ledger is not None:
        ledger_view = {
            "candidates": sum(1 for r in ledger_rows if r.get("kind") == "candidate"),
            "measured": [
                {
                    "n": r.get("n"),
                    "tag": r.get("tag"),
                    "timing_status": r.get("timing_status"),
                    "cycles": _int(r.get("objective_cycles")),
                    "at": epoch(r.get("at")),
                }
                for r in ledger_rows
                if r.get("kind") == "measured"
            ],
            "best": [
                {"n": r.get("n"), "package": r.get("package_sha256"), "at": epoch(r.get("at"))}
                for r in ledger_rows
                if r.get("kind") == "best"
            ],
        }
    target = target or (run or {}).get("target")
    roofline_best = (store_doc or {}).get("roofline_best")
    by_kind: dict[str, dict[str, Any]] = {}
    for gap in (roofline_best or {}).get("gaps") or ():
        entry = by_kind.setdefault(
            str(gap.get("kind")), {"kind": str(gap.get("kind")), "groups": 0, "ours": 0, "roofline": 0}
        )
        entry["groups"] += 1
        entry["ours"] += int(gap["ours"])
        entry["roofline"] += int(gap["roofline"])
    return {
        "run": {
            "target": (run or {}).get("target"),
            "method": (run or {}).get("method"),
            "prohibited_roles": list((run or {}).get("prohibited_instruction_roles") or []) if run else None,
            "config_sha256": (run or {}).get("config_sha256"),
            "resumed_from": (run or {}).get("resumed_from_run") or (seed or {}).get("resumed_from_run"),
            "started": started,
        },
        "store": {
            "root": str(store_root) if store_root else None,
            "source": store_source,
            "loaded": store_doc is not None,
        },
        "config": {
            "plateau_hours": (config or {}).get("plateau_hours"),
            "plateau_min_sessions": (config or {}).get("plateau_min_sessions"),
            "stop_at_bar": (config or {}).get("stop_at_bar"),
        }
        if config is not None
        else None,
        "bar": bar,
        "orientation": orientation,
        "candidates": candidates,
        "best_line": best_line,
        "best": lowest,
        "counts": counts,
        "counts_in_run": counts_in_run if started is not None else None,
        "roofline": {
            "key": roofline_best["key"],
            "gaps": roofline_best["gaps"][:20],
            "by_kind": sorted(by_kind.values(), key=lambda e: -(e["ours"] - e["roofline"])),
        }
        if roofline_best
        else None,
        "plateau": (store_doc or {}).get("plateau"),
        "board": board if store_doc is not None else None,
        "sessions": {
            "driver": sessions_doc.get("driver"),
            "model": sessions_doc.get("model"),
            "rows": list(sessions_doc.get("sessions") or []),
            "stopped": stopped,
        }
        if sessions_doc is not None
        else None,
        "rounds": rounds or None,
        "circuit_breaker": {"at": breaker.get("at"), "reason": _text(breaker.get("reason"), 300)} if breaker else None,
        "host_resource": {"status": host.get("status")} if host else None,
        "ledger": ledger_view,
        "lineage": {
            "seed_package": (seed or {}).get("seed_package_sha256"),
            "lineage_kind": (seed or {}).get("lineage_kind"),
            "origin_kind": (seed or {}).get("origin_kind"),
            "frozen": (seed or {}).get("frozen"),
            "why": _text((seed or {}).get("why"), 600),
            "imported": bool((seed or {}).get("imported")),
            "chain": _lineage_chain(seed),
            "index_rows": _index_rows(target, run_dir, inventory),
        }
        if seed is not None
        else None,
        "liveness": liveness,
    }


# --------------------------------------------------------------------------- the documents
def _phases_present(run_dir: Path, plan_phases: Mapping[str, Any]) -> list[str]:
    phases = {
        str(n)
        for n, p in plan_phases.items()
        if Path(str(p.get("engine_output") or "")).resolve() == Path(run_dir).resolve()
    }
    if any((Path(run_dir) / name).exists() for name in PHASE1_MARKERS):
        phases.add("1")
    if any((Path(run_dir) / name).exists() for name in phase2_markers()):
        phases.add("2")
    return sorted(phases & {"1", "2"})


def run_summary(
    run_dir: Path,
    *,
    now: float | None = None,
    stall_hours: float = DEFAULT_STALL_HOURS,
    store: Path | None = None,
    cache: StoreCache | None = None,
    clock: Callable[[], float] | None = None,
) -> dict[str, Any]:
    """The whole summary of one run directory (an orchestration, a phase-1 run or a phase-2 run)."""
    import time

    run_dir = Path(run_dir).expanduser()
    if not run_dir.is_dir():
        from ..spec import SpecError

        raise SpecError(f"not a run directory: {run_dir}")
    now = (clock or time.time)() if now is None else now
    cache = cache or StoreCache()
    inventory = Inventory()
    record = orchestration(run_dir, inventory)
    phases = _phases_present(run_dir, (record or {}).get("phases") or {})
    target = (record or {}).get("target")
    run_json = run_dir / "run.json"
    if target is None and run_json.is_file():
        try:
            target = (json.loads(run_json.read_text(encoding="utf-8")) or {}).get("target")
        except (OSError, ValueError, AttributeError):
            target = None
    if target is None:
        record_doc = Inventory().json("run_record.json", run_dir / "run_record.json")
        target = (record_doc or {}).get("target")
    if target is None and "1" in phases:
        environment = inventory.yaml("environment.yaml", run_dir / "environment.yaml")
        scope = (environment or {}).get("task_scope")
        target = scope.get("target") if isinstance(scope, Mapping) else None
    phase_views: dict[str, Any] = {}
    if "1" in phases:
        phase_views["1"] = phase1(run_dir, inventory, now=now, stall_hours=stall_hours, orchestration_record=record)
    if "2" in phases:
        phase_views["2"] = phase2(
            run_dir, inventory, now=now, stall_hours=stall_hours, store=store, cache=cache, target=target
        )
    engines = {}
    for number, entry in ((record or {}).get("phases") or {}).items():
        output = entry.get("engine_output")
        if not output or number in phase_views or Path(output).resolve() == run_dir.resolve():
            continue
        if Path(output).is_dir():
            engines[number] = run_summary(Path(output), now=now, stall_hours=stall_hours, store=store, cache=cache)
    liveness = next(
        (phase_views[n]["liveness"] for n in ("2", "1") if n in phase_views),
        None,
    )
    if liveness is None and record is not None:
        ended = _orchestration_ended(record)
        liveness = {
            "state": ENDED if ended else LIVE if record.get("state") == "running" else UNKNOWN,
            "hours": None,
            "detail": f"orchestration state {record.get('state')}",
        }
    return {
        "schema": SCHEMA,
        "kind": "run",
        "run_dir": str(run_dir.resolve()),
        "run_id": run_dir.resolve().name,
        "target": target,
        "generated": now,
        "phases": phases,
        "orchestration": record,
        "phase1": phase_views.get("1"),
        "phase2": phase_views.get("2"),
        "engines": engines or None,
        "liveness": liveness or {"state": UNKNOWN, "hours": None, "detail": "no phase records found"},
        "inventory": inventory.rows,
    }


def target_summary(
    target: str,
    *,
    now: float | None = None,
    stall_hours: float = DEFAULT_STALL_HOURS,
    max_runs: int = 40,
) -> dict[str, Any]:
    """Every run of ``target`` across phases, the target's INDEX.yaml and its champion lineage."""
    import time

    from merlin.common import paths
    from merlin.targetgen import package_records, target_index

    from ..history import runs

    package_records.component(target)
    now = time.time() if now is None else now
    inventory = Inventory()
    index = inventory.yaml("INDEX.yaml", target_index.index_path(target))
    try:
        orchestrations = runs(target=target)
    except Exception as exc:  # noqa: BLE001 -- discovery problems are shown, never fatal
        orchestrations = {"runs": [], "problems": [{"run_dir": str(paths.runs_dir()), "error": str(exc)}]}
    cache = StoreCache()
    phase_runs: dict[str, list[dict[str, Any]]] = {}
    truncated: dict[str, int] = {}
    for phase in ("0", "1", "2"):
        root = paths.phase_runs_root(target, phase)
        directories = (
            sorted((p for p in root.iterdir() if p.is_dir() and not p.is_symlink()), reverse=True)
            if root.is_dir()
            else []
        )
        truncated[phase] = max(0, len(directories) - max_runs)
        rows = []
        for run_dir in directories[:max_runs]:
            try:
                summary = run_summary(run_dir, now=now, stall_hours=stall_hours, cache=cache)
            except Exception as exc:  # noqa: BLE001 -- one unreadable run never hides the others
                rows.append(
                    {"run_id": run_dir.name, "run_dir": str(run_dir), "problem": f"{type(exc).__name__}: {exc}"}
                )
                continue
            rows.append(_run_row(summary))
        phase_runs[phase] = rows
    return {
        "schema": SCHEMA,
        "kind": "target",
        "target": target,
        "generated": now,
        "index": index,
        "index_path": str(target_index.index_path(target)),
        "orchestrations": orchestrations,
        "phase_runs": phase_runs,
        "truncated": truncated,
        "inventory": inventory.rows,
    }


def _run_row(summary: Mapping[str, Any]) -> dict[str, Any]:
    """One line of a run for the target view."""
    p1, p2 = summary.get("phase1") or {}, summary.get("phase2") or {}
    latest = p1.get("latest") or {}
    best = p2.get("best") or {}
    return {
        "run_id": summary["run_id"],
        "run_dir": summary["run_dir"],
        "phases": summary["phases"],
        "state": (summary.get("liveness") or {}).get("state"),
        "detail": (summary.get("liveness") or {}).get("detail"),
        "orchestration_state": (summary.get("orchestration") or {}).get("state"),
        "grades": len(p1.get("grades") or ()) if p1 else None,
        "latest_passed": latest.get("n_passed"),
        "latest_capsules": latest.get("n_capsules"),
        "frozen_commit": (p1.get("freeze") or {}).get("frozen_commit") if p1 else None,
        "method": (p2.get("run") or {}).get("method") if p2 else None,
        "measured_in_run": (p2.get("counts_in_run") or {}).get(MEASURED) if p2 else None,
        "best_cycles": best.get("cycles"),
        "best_key": best.get("key"),
    }


__all__ = [
    "DEFAULT_STALL_HOURS",
    "ENDED",
    "FAILURE_CLASSES",
    "FINISHED",
    "Inventory",
    "LIVE",
    "MEASURED",
    "RELAUNCHED",
    "STALLED",
    "STOPPED",
    "StoreCache",
    "UNKNOWN",
    "epoch",
    "failure_class",
    "load_store",
    "orchestration",
    "phase1",
    "phase2",
    "run_summary",
    "stamp",
    "target_summary",
    "tier_key",
]
