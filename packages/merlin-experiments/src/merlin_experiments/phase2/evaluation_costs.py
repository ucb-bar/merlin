"""Parent-observed queue costs; no numerical, cycle or execution authority.

Only the original job objects can reopen these observations. Cost collection
cannot change a completed grade or reconstruct ownership from a saved receipt.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass

from .contracts import StageGateError, canonical_json, document_sha256, sha256_file, write_json

SCHEMA = "merlin.evaluation_cost.v1"
BOUNDARY = "assigned CPUs before verification/setup through cleanup, durable release and artifact publication"


@dataclass
class _Entry:
    job: object
    requested: float
    enqueued: float
    started: float | None = None
    context: dict | None = None
    cleaned: bool = False
    released: bool = False
    ended: bool = False
    document: bytes | None = None
    pins: tuple = ()
    unavailable: str | None = None


def _context(pool, job):
    corpus, rtl = pool.inputs.corpus, pool.inputs.rtl
    member = None
    if pool.inputs.ordinary and len(job.selected_members) == 1:
        identity = job.selected_members[0]
        selected = next(row for row in corpus.capsules if (row.family, row.capsule) == identity)
        member = {
            "family": selected.family,
            "capsule": selected.capsule,
            "source_sha256": selected.source_sha256,
            "performance_sha256": document_sha256(selected.descriptor["performance"]),
        }
    resources = {
        "cpu_ids": list(job.cpus),
        "grant": asdict(pool.grant),
        "assigned": asdict(pool.resources),
        "enforcement": pool.enforcement,
        "external_boundary": None if pool._external_boundary is None else pool._external_boundary.verify(),
    }
    # Take a small immutable data copy, not compiler, tensor or process copies.
    return json.loads(
        canonical_json(
            {
                "candidate_sha256": job.revision.source_sha256,
                "baseline_sha256": pool.baseline_revision.source_sha256,
                "corpus_sha256": corpus.capsules_sha256,
                "manifest_sha256": corpus.manifest_sha256,
                "certificate_sha256": None if rtl is None else rtl.certificate.sha256,
                "closure_sha256": job.closure,
                "environment_sha256": job.environment,
                "resource_context_sha256": document_sha256(resources),
                "resources": resources,
                "kind": job.kind,
                "selected_members": [list(key) for key in job.selected_members],
                "selection_scope": "selected public members" if job.selected_members else "complete original roster",
                "member": member,
            }
        )
    )


class EvaluationCosts:
    """A bounded observer belonging to one parent pool and its original jobs."""

    def __init__(self, pool):
        self._pool, self._pid, self._entries = pool, os.getpid(), {}

    def _entry(self, job):
        entry = self._entries.get(job.job_id)
        if (
            os.getpid() != self._pid
            or self._pool._jobs.get(job.job_id) is not job
            or entry is None
            or entry.job is not job
        ):
            raise StageGateError("evaluation cost requires its original parent-issued job")
        return entry

    def queued(self, job, *, requested):
        if os.getpid() != self._pid or job.job_id in self._entries or self._pool._jobs.get(job.job_id) is not job:
            raise StageGateError("evaluation cost cannot reconstruct a queued job")
        self._entries[job.job_id] = _Entry(job, requested, time.monotonic())

    def start(self, job):
        now = time.monotonic()
        entry = self._entry(job)
        if entry.started is not None or entry.ended:
            raise StageGateError("evaluation cost cannot restart an original job")
        entry.started = now
        try:
            entry.context = _context(self._pool, job)
        except Exception as error:  # noqa: BLE001 - telemetry must not replace evaluator admission
            entry.unavailable = f"cost context unavailable ({type(error).__name__})"

    def cleaned(self, job):
        self._entry(job).cleaned = True

    def released(self, job):
        self._entry(job).released = True

    def finish(self, job, *, publication=None):
        """Call after original terminal publication; never retry evaluator cleanup."""
        now = time.monotonic()
        try:
            entry = self._entry(job)
            if entry.ended:
                return
            entry.ended = True
            if entry.started is None or entry.context is None or not entry.released:
                return
            if job.state not in ("completed", "refused", "cancelled"):
                return
            if job.state == "completed" and not entry.cleaned:
                entry.unavailable = "complete cost lacks original descendant cleanup"
                return
            clocks = (entry.requested, entry.enqueued, entry.started, now)
            if any(not math.isfinite(value) for value in clocks) or tuple(sorted(clocks)) != clocks:
                entry.unavailable = "parent cost clock is invalid"
                return
            products = [job.root / "cleanup.json"]
            if job.state == "completed":
                products.append(job.root / "feedback.json")
            pins = tuple((path, sha256_file(path)) for path in products if path.is_file())
            if job.state == "completed" and len(pins) != 2:
                entry.unavailable = "complete cost lacks original durable result and cleanup"
                return
            document = {
                "schema": SCHEMA,
                "status": "observed",
                "job_id": job.job_id,
                "outcome": job.state,
                "context": entry.context,
                "wall_seconds": now - entry.started,
                "admission_seconds": entry.enqueued - entry.requested,
                "queue_wait_seconds": entry.started - entry.enqueued,
                "request_wall_seconds": now - entry.requested,
                "boundary": BOUNDARY,
                "excluded": "cost/status serialization and export after this observation; client transport",
                "cpu_seconds": None,
                "per_arm_seconds": None,
                "cache_state": "UNKNOWN",
                "artifact_publication": publication,
                "purpose": "observed evaluation cost only; no prediction, timing qualification or promotion",
            }
            path = job.root / "evaluation-cost.json"
            if path.exists() or path.is_symlink() or job.root.resolve() != job.root:
                raise StageGateError("evaluation cost requires a fresh canonical parent-owned product")
            write_json(path, document)
            path.chmod(0o400)
            entry.document = canonical_json(document)
            entry.pins = (*pins, (path, sha256_file(path)))
        except Exception as error:  # noqa: BLE001 - cost failure cannot repeat release or alter the result
            entry = self._entries.get(job.job_id)
            if entry is not None and entry.job is job:
                entry.document = None
                entry.unavailable = f"cost publication unavailable ({type(error).__name__})"

    def view(self, job):
        """Reopen the live original observation; saved data cannot mint one."""
        try:
            entry = self._entry(job)
            if entry.document is not None:
                document = json.loads(entry.document)
                if (
                    job.state != document["outcome"]
                    or _context(self._pool, job) != document["context"]
                    or any(path.is_symlink() or sha256_file(path) != digest for path, digest in entry.pins)
                ):
                    raise StageGateError("original evaluation cost dependencies changed")
                return document
            reason = entry.unavailable or (
                "original job has not started" if entry.started is None else "original terminal cleanup is incomplete"
            )
        except Exception as error:  # noqa: BLE001 - missing cost never replaces original evaluator checks
            reason = f"original cost unavailable ({type(error).__name__})"
        return {"schema": SCHEMA, "status": "UNKNOWN", "wall_seconds": None, "reason": reason}


def view(pool, job):
    observer = getattr(pool, "_evaluation_costs", None)
    if type(observer) is not EvaluationCosts:
        return {"schema": SCHEMA, "status": "UNKNOWN", "wall_seconds": None, "reason": "no original cost observer"}
    return observer.view(job)


def note(pool, job, event, **kwargs):
    """Fixed parent hooks are optional telemetry, never a second cleanup owner."""
    observer = getattr(pool, "_evaluation_costs", None)
    if type(observer) is not EvaluationCosts or event not in ("queued", "start", "cleaned", "released", "finish"):
        return
    try:
        getattr(observer, event)(job, **kwargs)
    except Exception as error:  # noqa: BLE001 - even a post-release cost error cannot re-enter finalization
        entry = observer._entries.get(job.job_id)
        if entry is not None and entry.job is job:
            entry.document = None
            entry.unavailable = f"parent cost observation unavailable ({type(error).__name__})"
