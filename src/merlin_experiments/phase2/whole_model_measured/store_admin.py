"""Operator surgery on a measurement store: every edit states why, is logged, and deletes nothing.

    python -m merlin_experiments.phase2.whole_model_measured admin <store> <operation> ...

Operations (each takes ``--why``, which is required):

* ``mark-infra --key K... --kind NAME`` -- record that a result was refused for an INFRASTRUCTURE reason
  (a contaminated snapshot, a harness import break), not its bytes; the service then reopens it under
  a new harness instead of serving the old refusal.
* ``reopen --key K [--retract]`` -- set a finished job's attempt aside as ``attempts/<n>/`` (never
  deleted) and queue it again.  Its verdict keeps standing until the new attempt reaches one, unless
  ``--retract`` says the earlier verdict no longer stands (:mod:`.attempts`).
* ``requeue-solo --key K [--retract]`` -- like ``reopen``, and measured ALONE (no batch, no neighbour).
* ``reset-plateau [--at TS]`` / ``backfill-plateau --rows FILE`` / ``abandon-session --run R --session N``
  -- the store's plateau record (:mod:`.sessions`).
* ``correct-citation --key K --field a.b.c --value JSON`` -- correct one citation field of a job and its
  result (for example a stale builder citation), keeping the old value.  The fields a MEASUREMENT consists
  of (status, cycles, verdict, digests) are refused: a measurement is re-taken, never edited.  A result
  file is never rewritten: the correction is an annotation beside it (``annotations.json``), laid over
  the result whenever it is read.
* ``outage-retry-now`` -- the board was reported back: the next batch may try it now instead of at the
  outage's retry interval.  The outage stays OPEN, with the report recorded on it, until a batch actually
  runs its workload (which closes it); a report is not evidence the board works.

A running job is never touched, nothing is ever deleted and no ``result.json`` is ever rewritten: an
attempt is moved aside whole, an operator's note on a result is an annotation beside it, and every
operation appends one line to the store's ``admin_log.jsonl``.
"""

from __future__ import annotations

import argparse
import getpass
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import attempts as A
from . import batch as BATCH
from . import jobs as J
from . import retention as RET
from . import sessions as SES
from .identity import locked, now, read_json, write_json_atomic

ADMIN_LOG = "admin_log.jsonl"
#: What a measurement IS; a correction may never edit these.
PROTECTED_FIELDS = frozenset(
    {
        "timing_status",
        "objective_cycles",
        "verdict",
        "package_sha256",
        "program_sha256",
        "build",
        "device",
        "run",
        "state",
    }
)
INFRA_MARK = "infra_marked"


class AdminError(RuntimeError):
    """The operation is refused, and the reason is stated."""


def _require_why(why: str) -> str:
    if not str(why or "").strip():
        raise AdminError("every store edit states why")
    return str(why).strip()


def _log(store: Path, operation: str, why: str, **fields: Any) -> dict[str, Any]:
    entry = {"at": now(), "operator": getpass.getuser(), "operation": operation, "why": why, **fields}
    with (Path(store) / ADMIN_LOG).open("a", encoding="utf-8") as log:
        log.write(json.dumps(entry, sort_keys=True, default=str) + "\n")
    return entry


def _job_dir(store: Path, key: str) -> Path:
    matches = [p for p in Path(store).iterdir() if p.is_dir() and p.name.startswith(key) and (p / "job.json").is_file()]
    if len(matches) != 1:
        raise AdminError(f"{len(matches)} job(s) in {store} match {key!r}; name one exactly")
    return matches[0]


def _annotate(job_dir: Path, update: Mapping[str, Any]) -> dict[str, Any]:
    """Merge ``update`` into the current result's annotations (``annotations.json`` beside it)."""
    notes = read_json(job_dir / J.ANNOTATIONS_FILE) or {"schema": J.ANNOTATIONS_SCHEMA}
    for name, value in update.items():
        if isinstance(value, list):
            notes[name] = [*(notes.get(name) or ()), *value]
        else:
            notes[name] = value
    write_json_atomic(job_dir / J.ANNOTATIONS_FILE, notes)
    return notes


def mark_infra(store: Path, keys: Sequence[str], *, kind: str, why: str) -> list[str]:
    why = _require_why(why)
    if not kind or not kind.replace("_", "").isalnum():
        raise AdminError(f"an infra kind is one identifier, not {kind!r}")
    marked = []
    for key in keys:
        job_dir = _job_dir(store, key)
        document = read_json(job_dir / "result.json")
        if document is None:
            raise AdminError(f"{job_dir.name} has no result to mark")
        if document.get("timing_status") != "REFUSED":
            raise AdminError(f"{job_dir.name} is {document.get('timing_status')}: only a refusal can be infra-caused")
        _annotate(job_dir, {INFRA_MARK: {"kind": kind, "why": why, "at": now()}})
        marked.append(job_dir.name)
    _log(store, "mark-infra", why, kind=kind, keys=marked)
    return marked


def _requeue(store: Path, key: str, *, why: str, solo: bool, operation: str, retract: bool = False) -> dict[str, Any]:
    why = _require_why(why)
    job_dir = _job_dir(store, key)
    with locked(Path(store)):
        job = read_json(job_dir / "job.json") or {}
        if job.get("state") not in J.TERMINAL:
            raise AdminError(f"{job_dir.name} is {job.get('state')}: only a finished job is requeued")
        attempt = J.archive_attempt(job_dir, "paused_attempt_", why=f"{operation}: {why}", retracted=retract)
        RET.prune_archived_attempt(attempt, job.get("retain"))
        job.setdefault("admin_requeues", []).append(
            {"at": now(), "why": why, "moved_to": str(attempt), "solo": solo, "retracted": retract}
        )
        job.update(state=J.PENDING, notice=f"requeued by the operator: {why} (not a verdict)")
        if solo:
            job["solo"] = True
        for stale in (
            "worker_pid",
            "dispatched_at",
            "started_at",
            "batch",
            "failed_at",
            "failure",
            "finished_at",
            "timing_status",
        ):
            job.pop(stale, None)
        write_json_atomic(job_dir / "job.json", job)
    _log(store, operation, why, key=job_dir.name, moved_to=str(attempt), retracted=retract)
    return job


def reopen(store: Path, key: str, *, why: str, retract: bool = False) -> dict[str, Any]:
    return _requeue(store, key, why=why, solo=False, operation="reopen", retract=retract)


def requeue_solo(store: Path, key: str, *, why: str, retract: bool = False) -> dict[str, Any]:
    return _requeue(store, key, why=why, solo=True, operation="requeue-solo", retract=retract)


def correct_citation(store: Path, key: str, *, field: str, value: Any, why: str) -> dict[str, Any]:
    """Set ``field`` (dotted) on the job and its result, keeping the old value and the reason."""
    why = _require_why(why)
    path = [part for part in str(field).split(".") if part]
    if not path or path[0] in PROTECTED_FIELDS:
        raise AdminError(f"{field!r} is part of what was measured; a measurement is re-taken, never edited")
    job_dir = _job_dir(store, key)
    corrected = {}
    for name in ("job.json", "result.json"):
        document = read_json(job_dir / name)
        if document is None:
            continue
        if name == "result.json":
            # A RESULT IS NEVER REWRITTEN: the correction is an annotation beside it, laid over it on read.
            document = A.annotated(document, job_dir) or document
        node: Any = document
        for part in path[:-1]:
            node = node.setdefault(part, {}) if isinstance(node, dict) else None
            if not isinstance(node, dict):
                raise AdminError(f"{field!r} does not name a field of {name}")
        old = node.get(path[-1])
        entry = {"at": now(), "field": field, "was": old, "now": value, "why": why}
        if name == "result.json":
            _annotate(job_dir, {"citation_corrections": [entry]})
        else:
            node[path[-1]] = value
            document.setdefault("citation_corrections", []).append(entry)
            write_json_atomic(job_dir / name, document)
        corrected[name] = old
    if not corrected:
        raise AdminError(f"{job_dir.name} has no record to correct")
    _log(store, "correct-citation", why, key=job_dir.name, field=field, was=corrected, now=value)
    return corrected


def outage_retry_now(store: Path, *, why: str) -> dict[str, Any]:
    """Let the next batch try the board now; the outage stays open until a batch runs its workload."""
    why = _require_why(why)
    with locked(Path(store)):
        outage = BATCH.board_outage(Path(store))
        if outage is None:
            raise AdminError(f"{store} has no open board outage")
        outage["retry_after_epoch"] = time.time()
        outage.setdefault("reported_back", []).append({"at": now(), "why": why})
        write_json_atomic(Path(store) / BATCH.BOARD_OUTAGE, outage)
    _log(store, "outage-retry-now", why, opened_at=outage.get("opened_at"))
    return outage


def _parse_stamp(stamp: str | None) -> float | None:
    if stamp is None:
        return None
    import calendar

    return float(calendar.timegm(time.strptime(stamp, "%Y%m%dT%H%M%SZ")))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="whole_model_measured admin", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("store", type=Path)
    sub = parser.add_subparsers(dest="operation", required=True)
    op = sub.add_parser("mark-infra")
    op.add_argument("--key", action="append", required=True)
    op.add_argument("--kind", required=True)
    for name in ("reopen", "requeue-solo"):
        op = sub.add_parser(name)
        op.add_argument("--key", required=True)
        op.add_argument(
            "--retract", action="store_true", help="the earlier attempt's verdict no longer stands (say why)"
        )
    op = sub.add_parser("reset-plateau")
    op.add_argument("--at")
    sub.add_parser("backfill-plateau").add_argument("--rows", type=Path, required=True)
    op = sub.add_parser("abandon-session")
    op.add_argument("--run", required=True)
    op.add_argument("--session", type=int, required=True)
    op = sub.add_parser("correct-citation")
    op.add_argument("--key", required=True)
    op.add_argument("--field", required=True)
    op.add_argument("--value", required=True, help="the new value, as JSON")
    sub.add_parser("outage-retry-now")
    for child in sub.choices.values():
        child.add_argument("--why", required=True)
    args = parser.parse_args(argv)
    store = args.store
    if not store.is_dir():
        raise SystemExit(f"no store at {store}")
    result: Any
    if args.operation == "mark-infra":
        result = mark_infra(store, args.key, kind=args.kind, why=args.why)
    elif args.operation == "reopen":
        result = reopen(store, args.key, why=args.why, retract=args.retract)
    elif args.operation == "requeue-solo":
        result = requeue_solo(store, args.key, why=args.why, retract=args.retract)
    elif args.operation == "reset-plateau":
        result = SES.reset_plateau(store / SES.PLATEAU_FILE, reason=args.why, now=_parse_stamp(args.at))
        _log(store, "reset-plateau", args.why, at=args.at)
    elif args.operation == "backfill-plateau":
        rows = json.loads(args.rows.read_text(encoding="utf-8"))
        result = SES.backfill_plateau(store / SES.PLATEAU_FILE, rows, reason=args.why)
        _log(store, "backfill-plateau", args.why, rows=len(result))
    elif args.operation == "outage-retry-now":
        result = outage_retry_now(store, why=args.why)
    elif args.operation == "abandon-session":
        result = SES.abandon_session(store / SES.PLATEAU_FILE, run=args.run, session=args.session, reason=args.why)
        _log(store, "abandon-session", args.why, run=args.run, session=args.session)
    else:
        result = correct_citation(store, args.key, field=args.field, value=json.loads(args.value), why=args.why)
    print(
        json.dumps(result, indent=1, default=str)
        if not isinstance(result, Mapping)
        else json.dumps(dict(result), indent=1, default=str)
    )
    return 0


__all__ = [
    "ADMIN_LOG",
    "AdminError",
    "correct_citation",
    "main",
    "mark_infra",
    "outage_retry_now",
    "reopen",
    "requeue_solo",
]
