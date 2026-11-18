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
import os
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ---- records -----------------------------------------------------------------------------------

SCHEMA = "merlin_experiment_tracking_summary_v1"


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


UNFINALIZED, FROZEN = "UNFINALIZED", "FROZEN"


MEASURED = "measured"


FAILURE_CLASSES = ("correctness", "infra", "refused", "prohibited_instruction", "declined")


OPEN, SUPERSEDED = "pending", "superseded"


_IN_FLIGHT = ("pending", "running")


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
        from merlin.common.record_formats import safe_load_text

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


def _group_rows(result: Mapping[str, Any] | None) -> dict[str, Mapping[str, Any]]:
    from merlin.analysis.whole_model.grading import verdict as V

    return {
        str(row["group"]): row
        for row in V.group_table((result or {}).get("verdict") or {})
        if isinstance(row, Mapping) and row.get("group") is not None
    }


def _coverage(result: Mapping[str, Any], reference: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The package-authored share recorded by the candidate's own build routes (feedback's projection)."""
    from merlin.development.backend.optimization.whole_model_measured.scoring import feedback as F

    if not ((result.get("build") or {}).get("groups")):
        return None
    authored = F.package_authored(result, reference)
    return {k: authored.get(k) for k in ("groups_answered", "groups_total", "priced_share")}


def _roofline(result: Mapping[str, Any], reference: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Per-group distance to the derived roofline the result recorded (``diagnostics.per_group``)."""
    from merlin.development.backend.optimization.whole_model_measured.scoring import feedback as F

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
    from merlin.development.backend.optimization.whole_model_measured.measurement import attempts as A

    try:
        result = A.effective_result(job_dir)
    except (OSError, ValueError):
        return None
    return dict(result) if isinstance(result, Mapping) else None


def load_store(root: Path, inventory: Inventory, cache: StoreCache) -> dict[str, Any]:
    """Every job of one measurement store, classified, plus the store's own plateau and board records."""
    from merlin.development.backend.optimization.whole_model_measured.measurement.machines import (
        INFRA_BOARD_UNAVAILABLE,
    )
    from merlin.experiments.tracking.readers.backend_generation_records import _phase2_names, failure_class

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
    from merlin.development.targetgen.packaging import target_index

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


def _phases_present(run_dir: Path, plan_phases: Mapping[str, Any]) -> list[str]:
    from merlin.experiments.tracking.readers.backend_generation_records import phase2_markers

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

    from merlin.experiments.tracking.readers.backend_generation_records import phase1
    from merlin.experiments.tracking.readers.optimization_records import phase2

    run_dir = Path(run_dir).expanduser()
    if not run_dir.is_dir():
        from merlin.experiments.definition import SpecError

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
    from merlin.common.paths import generation_run_roots
    from merlin.development.targetgen.packaging import records as package_records
    from merlin.development.targetgen.packaging import target_index
    from merlin.experiments.runs.catalogs import runs

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
        directories = sorted(
            (
                p
                for root in generation_run_roots(target, phase)
                if root.is_dir()
                for p in root.iterdir()
                if p.is_dir() and not p.is_symlink() and not (phase == "1" and p.name == "optimization")
            ),
            reverse=True,
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


# ---- tail --------------------------------------------------------------------------------------
# Read growing records cheaply and without touching them: offset tailing and stat signatures.
#
# A live view polls files a running phase is still writing.  It must cost the run nothing, so:
#
# * every file is opened read-only, never locked, renamed, truncated or written;
# * a JSONL file is read from the byte offset the previous poll stopped at, and only up to its last
#   newline -- a trailing line still being written is left for the next poll, never parsed half-way;
# * a file that shrank or was replaced (another inode) is read again from the start, and a file that
#   disappeared is simply absent this poll;
# * :func:`signature` is the stat of a set of paths, so a caller can skip work whose inputs did not change.

#: Bytes read per ``read`` call while catching up (bounds the transient buffer, not the file size).
CHUNK = 4 << 20


class JsonlTail:
    """One JSONL file read incrementally: :meth:`poll` returns only the rows completed since last time."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.offset = 0
        self.identity: tuple[int, int] | None = None
        self.rows_read = 0
        self.bad = 0
        self.bytes_read = 0
        self.resets = 0
        self.present = False

    def poll(self, consume: Callable[[dict[str, Any]], None], on_reset: Callable[[], None] | None = None) -> int:
        """Hand each row completed since the last poll to ``consume``; return how many.

        When the file was replaced or truncated, ``on_reset`` runs first and the whole file is read
        again, so a consumer rebuilds what it derived.  Rows are handed over one parsed chunk at a time
        (never the whole file at once), which bounds memory while catching up on a large file."""
        try:
            stat = os.stat(self.path)
        except OSError:
            self.present = False
            return 0
        self.present = True
        identity = (stat.st_dev, stat.st_ino)
        if self.identity is not None and (identity != self.identity or stat.st_size < self.offset):
            self.offset = 0
            self.resets += 1
            if on_reset is not None:
                on_reset()
        self.identity = identity
        if stat.st_size == self.offset:
            return 0
        count = 0
        try:
            with open(self.path, "rb") as handle:
                handle.seek(self.offset)
                pending = b""
                while True:
                    chunk = handle.read(CHUNK)
                    if not chunk:
                        break
                    self.bytes_read += len(chunk)
                    data = pending + chunk
                    cut = data.rfind(b"\n")
                    if cut < 0:
                        pending = data
                        continue
                    pending = data[cut + 1 :]
                    count += self._parse(data[: cut + 1], consume)
                    # Everything up to the last newline is consumed; a partial tail line waits.
                    self.offset = handle.tell() - len(pending)
        except OSError:
            return count
        return count

    def _parse(self, block: bytes, consume: Callable[[dict[str, Any]], None]) -> int:
        count = 0
        for line in block.split(b"\n"):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                self.bad += 1
                continue
            if isinstance(row, dict):
                consume(row)
                count += 1
                self.rows_read += 1
            else:
                self.bad += 1
        return count


class Tails:
    """The tails a view keeps between polls, by path (one-shot views use a fresh one)."""

    def __init__(self) -> None:
        self.files: dict[str, JsonlTail] = {}
        self.state: dict[str, Any] = {}

    def tail(self, path: Path) -> JsonlTail:
        key = str(Path(path))
        if key not in self.files:
            self.files[key] = JsonlTail(Path(path))
        return self.files[key]


def signature(paths: Iterable[Path]) -> tuple[tuple[str, int, int], ...]:
    """``(path, mtime_ns, size)`` for each existing path, sorted: equal signatures mean unchanged inputs."""
    out = []
    for path in paths:
        try:
            stat = os.stat(path)
        except OSError:
            continue
        out.append((str(path), stat.st_mtime_ns, stat.st_size))
    return tuple(sorted(out))


def tree_signature(root: Path, *, max_entries: int = 20000) -> tuple[tuple[str, int, int], ...]:
    """The signature of every file under ``root`` (bounded), for inputs read by a whole-tree reader."""
    paths: list[Path] = []
    for directory, _dirs, names in os.walk(root):
        for name in names:
            paths.append(Path(directory) / name)
            if len(paths) >= max_entries:
                return signature(paths)
    return signature(paths)


__all__ = [
    "DEFAULT_STALL_HOURS",
    "ENDED",
    "FAILURE_CLASSES",
    "FINISHED",
    "FROZEN",
    "Inventory",
    "JsonlTail",
    "LIVE",
    "MEASURED",
    "RELAUNCHED",
    "STALLED",
    "STOPPED",
    "StoreCache",
    "Tails",
    "UNFINALIZED",
    "UNKNOWN",
    "epoch",
    "failure_class",
    "load_store",
    "orchestration",
    "phase1",
    "phase2",
    "run_summary",
    "signature",
    "stamp",
    "target_summary",
    "tier_key",
    "tree_signature",
]
