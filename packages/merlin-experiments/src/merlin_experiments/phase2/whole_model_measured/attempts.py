"""A job's measurements across ALL its attempts, and which one stands.

A job is keyed by the bytes it measures, but the same bytes can be measured more than once: a worker
lost to host pressure is re-queued, an operator re-queues a job solo, a builder change re-opens a
finished job, a batch whose board objects vanished is rebuilt.  Every such retry first moves the earlier
attempt aside, whole, as ``attempts/<n>/`` (:func:`.jobs.archive_attempt`), and every ``result.json`` is
written once and never replaced (:func:`.jobs.write_result`).  So the history is complete on disk; this
module is the one reader of it.

WHICH RESULT STANDS.  The newest attempt that reached a VERDICT about the bytes -- a measurement
(``MEASURED`` or ``MEASURED_INVALID``) or a refusal that is about the candidate.  An outcome that says
nothing about the bytes (a worker lost to the host, an infrastructure refusal, an operator's infra mark,
a supersede) never displaces an earlier verdict: on 2026-10 a crashed loop's resubmit left a champion's
store reading ``infra_worker_lost`` while its real board reading sat in an archive nothing read, and
the best quietly changed.  A verdict an operator RETRACTED (``reopen --retract``) no longer stands.
When the standing result is an earlier attempt's, the document says so (``from_attempt``) and carries
the current attempt's outcome beside it (``current_attempt``).

LEGACY STORES.  Attempts archived before this layout (``lost_attempt_0/``, ``paused_attempt_0/``, ...)
are read the same way, ordered by when their result finished.

The operator's notes on a result (``annotations.json``: an infra mark, a corrected citation) are read
here too and laid over the result as it is returned -- the file on disk is never edited.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from merlin.perf import whole_model_verdict as V

from . import jobs as J
from .identity import epoch_of, read_json

#: The outcomes that are verdicts about the bytes (a refusal is one unless it is the host's fault).
VERDICT_STATUSES = (V.TIMING_MEASURED, V.TIMING_MEASURED_INVALID)


def _is_infra(document: Mapping[str, Any]) -> bool:
    from .gates import is_infra_refusal

    annotations = document.get("annotations") or {}
    return bool(
        document.get("infra_worker_lost")
        or document.get("infra_marked")
        or annotations.get("infra_marked")
        or document.get("superseded")
        or is_infra_refusal(document.get("refusal"))
    )


def is_verdict(document: Mapping[str, Any] | None) -> bool:
    """Whether ``document`` says something about the bytes it was measured on (see the module doc)."""
    if not isinstance(document, Mapping):
        return False
    status = document.get("timing_status")
    if status in VERDICT_STATUSES:
        return True
    return status == V.TIMING_REFUSED and not _is_infra(document)


def _set(document: dict[str, Any], dotted: str, value: Any) -> None:
    node: Any = document
    parts = [p for p in str(dotted).split(".") if p]
    for part in parts[:-1]:
        child = node.get(part) if isinstance(node, dict) else None
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    if parts:
        node[parts[-1]] = value


def annotated(document: Mapping[str, Any] | None, directory: Path) -> dict[str, Any] | None:
    """``document`` with the operator's annotations of it (``annotations.json`` beside it) laid over a
    COPY: the infra mark as ``infra_marked``, each corrected citation at its field, every note kept."""
    if document is None:
        return None
    out = copy.deepcopy(dict(document))
    notes = read_json(Path(directory) / J.ANNOTATIONS_FILE)
    if not notes:
        return out
    out["annotations"] = notes
    if notes.get("infra_marked"):
        out["infra_marked"] = notes["infra_marked"]
    for correction in notes.get("citation_corrections") or ():
        _set(out, str(correction.get("field")), correction.get("now"))
    if notes.get("citation_corrections"):
        out["citation_corrections"] = [
            *(document.get("citation_corrections") or ()),
            *notes["citation_corrections"],
        ]
    return out


def history(job_dir: Path) -> list[dict[str, Any]]:
    """Every attempt of the job that left a result, oldest first: ``{attempt, kind, retracted, path,
    result}``.  The current attempt (the job directory itself) is last, ``attempt: "current"``."""
    job_dir = Path(job_dir)
    rows: list[dict[str, Any]] = []
    legacy = []
    for path in sorted(job_dir.iterdir()) if job_dir.is_dir() else ():
        if path.is_dir() and path.name.startswith(J.ARCHIVED_ATTEMPT_PREFIXES):
            legacy.append(path)
    for path in legacy:
        document = read_json(path / J.RESULT_FILE)
        if document is not None:
            rows.append(
                {
                    "attempt": path.name,
                    "kind": path.name.rstrip("0123456789").rstrip("_"),
                    "retracted": False,
                    "path": str(path / J.RESULT_FILE),
                    "result": annotated(document, path),
                    "order": (0, epoch_of(document.get("finished_at")) or 0.0, path.name),
                }
            )
    root = job_dir / J.ATTEMPTS_DIR
    numbered = (
        sorted((p for p in root.iterdir() if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name))
        if root.is_dir()
        else []
    )
    for path in numbered:
        record = read_json(path / J.ATTEMPT_RECORD) or {}
        document = read_json(path / J.RESULT_FILE)
        if document is None:
            continue
        rows.append(
            {
                "attempt": int(path.name),
                "kind": record.get("kind"),
                "retracted": bool(record.get("retracted")),
                "why": record.get("why"),
                "path": str(path / J.RESULT_FILE),
                "result": annotated(document, path),
                "order": (1, int(path.name), ""),
            }
        )
    rows.sort(key=lambda row: row["order"])
    current = read_json(job_dir / J.RESULT_FILE)
    if current is not None:
        rows.append(
            {
                "attempt": "current",
                "kind": "current",
                "retracted": False,
                "path": str(job_dir / J.RESULT_FILE),
                "result": annotated(current, job_dir),
                "order": (2, 0, ""),
            }
        )
    for row in rows:
        row.pop("order")
    return rows


def effective_result(job_dir: Path) -> dict[str, Any] | None:
    """The result that stands for the job (see the module doc), or None when no attempt left one.

    The current attempt's result when it is a verdict; else the newest earlier verdict not retracted,
    marked ``from_attempt`` with the current outcome (if any) under ``current_attempt``; else the current
    result as it is (an infra outcome with nothing earlier to stand)."""
    rows = history(job_dir)
    if not rows:
        return None
    current = rows[-1] if rows[-1]["attempt"] == "current" else None
    if current is not None and is_verdict(current["result"]):
        return current["result"]
    earlier = [
        row for row in rows if row["attempt"] != "current" and not row["retracted"] and is_verdict(row["result"])
    ]
    if earlier:
        chosen = earlier[-1]
        document = dict(chosen["result"])
        document["from_attempt"] = {"attempt": chosen["attempt"], "kind": chosen["kind"], "path": chosen["path"]}
        if current is not None:
            outcome = current["result"]
            document["current_attempt"] = {
                "timing_status": outcome.get("timing_status"),
                "refusal": str(outcome.get("refusal") or "")[:300] or None,
                "finished_at": outcome.get("finished_at"),
                "why_not_standing": "the current attempt's outcome is not a verdict about these bytes",
            }
        return document
    return current["result"] if current is not None else None


def attempt_count(job_dir: Path) -> int:
    """How many attempts the job has had that left a result (the current one included)."""
    return len(history(job_dir))


__all__ = ["VERDICT_STATUSES", "annotated", "attempt_count", "effective_result", "history", "is_verdict"]
