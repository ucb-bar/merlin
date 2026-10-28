"""Incremental self-check progress: each case's correctness as soon as the grader decides it.

``run_capsule`` writes ``capsule_result.json`` when a case's whole ladder is done, and a large suite
waits on its slowest case before it answers. Measured on one 132-case scalability self-check: the
response came 72 minutes after the request while the last case's numerics were decided at minute 53;
after the numeric comparison a large case still spent up to 16 minutes in the completion-order check
and the result write, so a correct case reached the agent up to a quarter of an hour late and the
sweep's answer waited for all of them.

This view therefore publishes two kinds of row, both golden-free:

* a FINISHED row, from a durable ``capsule_result.json``: the case's final status, its observed operator
  tier statuses, the failing plane/category (labels the final response already carries) and the
  numeric status;
* a PROVISIONAL row (``status: "running"``), from ``generated/numeric_report.yaml``, written by the
  runner the moment the numeric comparison is made: ``numerics: pass|fail`` with ``final: false``.
  Later checks (completion order, RTL tiers, the host-compute audit) can still fail the case; they can
  never turn a numeric fail into a pass.

Results are parsed once per ``(size, mtime)``: the snapshot is re-taken every second inside the
grading process, and re-reading every result each time held that process's interpreter lock for
seconds per snapshot once large results existed. One directory walk finds both kinds of file, and a
finished case's numeric report is not looked at again.

The document is published as it is. Its rows are already small and golden-free; routing them through
the response compactor stored a detail file per row in ``qa/cases`` and hashed every row on every
publish (measured: 76 ms of interpreter time per publish of a 300-case suite, 69% of it there), and the
compacted rows lost ``numerics``/``final``, so the client printed ``numerics None`` for a running case.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

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
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    numerics: dict[str, int] = {}
    for row in (*rows, *provisional):
        if row.get("numerics") in _NUMERIC_STATUSES:
            numerics[row["numerics"]] = numerics.get(row["numerics"], 0) + 1
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
