"""End-of-invocation coverage; these records grant no proof authority."""

from __future__ import annotations

import hashlib
import json
import logging
from contextvars import ContextVar
from pathlib import Path

_ACTIVE = ContextVar("merlin_compiler_coverage", default=None)
_LOG = logging.getLogger(__name__)


def _digest(raw):
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _job_binding(job, before, after, gate):
    """Bind a complete boundary to the actual queued request, without proof credit."""
    from .configuration_elision_bridge import file_ref
    from .speculative_gate import Job

    if type(job) is not Job or after is None:
        raise ValueError("an actual complete-boundary proof job is required")
    if job.directory.resolve() != (Path(gate.directory) / job.id).resolve():
        raise ValueError("coverage proof job belongs to another compiler gate")
    paths = tuple(job.directory / name for name in ("request.json", "before.mlir", "after.mlir"))
    refs = tuple(file_ref(path) for path in paths)
    raw = tuple(path.read_bytes() for path in paths)
    if any(
        hashlib.sha256(data).hexdigest() != ref["sha256"] or len(data) != ref["bytes"]
        for data, ref in zip(raw, refs, strict=True)
    ):
        raise ValueError("proof job bytes changed during coverage binding")
    request = json.loads(raw[0])
    expected = {
        "check": job.check,
        "label": job.label,
        "directory": str(job.directory),
        "params": job.params,
        "before": str(paths[1]),
        "after": str(paths[2]),
    }
    if request != expected or raw[1:] != (before.encode("utf-8"), after.encode("utf-8")):
        raise ValueError("coverage boundary differs from its actual queued proof inputs")
    return refs


class CoverageJournal:
    def __init__(self, validation):
        self.validation = validation
        self.rows = []
        self._jobs = {}

    def boundary(self, pass_name, before, after, *, profile_id, exclusion=None, job=None):
        if exclusion is not None and job is not None:
            raise ValueError("a boundary cannot be both source-excluded and submitted for proof")
        row = {
            "pass": pass_name,
            "profile": profile_id,
            "source_sha256": _digest(before),
            "final_sha256": None if after is None else _digest(after),
            "status": "pending" if job is not None else "unchanged",
            "proof_credit": False,
        }
        if exclusion is not None:
            row.update(status="unverified", code=exclusion.code, reason=exclusion.reason)
        elif job is not None:
            from .speculative_gate import GateFailure

            try:
                refs = _job_binding(job, before, after, self.validation.gate)
            except (OSError, ValueError, TypeError, KeyError) as error:
                raise GateFailure(
                    {
                        "check": "compiler_coverage",
                        "label": pass_name,
                        "verdict": "unavailable",
                        "reason": str(error),
                        "directory": str(self.validation.gate.directory),
                    }
                ) from error
            row.update(job=job.id, check=job.check, label=job.label, input_refs=refs)
            self._jobs[len(self.rows)] = (job, before, after, refs)
        elif before != after:
            raise ValueError("a changed supported boundary needs its actual proof job")
        self.rows.append(row)

    def finish(self, *, compiler_completed):
        """Call only after the outer gate guard has joined or failed."""
        from .speculative_gate import GateFailure

        by_job = {}
        for verdict in self.validation.gate.verdicts:
            by_job.setdefault(verdict["job"], []).append(verdict)
        rows = []
        missing = False
        for number, original in enumerate(self.rows):
            row = dict(original)
            if row["status"] == "unverified" and compiler_completed and row["final_sha256"] is None:
                row.update(status="unavailable", reason="the ordinary compiler produced no captured output")
                missing = True
            if row["status"] == "pending":
                matches = by_job.get(row["job"], [])
                job, before, after, refs = self._jobs[number]
                try:
                    bound = _job_binding(job, before, after, self.validation.gate) == refs
                except (OSError, ValueError, TypeError, KeyError):
                    bound = False
                if (
                    not bound
                    or len(matches) != 1
                    or matches[0].get("check") != row["check"]
                    or matches[0].get("label") != row["label"]
                ):
                    row.update(
                        status="unavailable", reason="no unique completed verdict bound to the actual proof inputs"
                    )
                    missing = True
                else:
                    verdict = matches[0]
                    status = verdict.get("verdict")
                    row.update(
                        status="verified" if status == "accepted" else status,
                        proof_credit=status == "accepted",
                    )
                    if compiler_completed and status != "accepted":
                        missing = True
            rows.append(row)
        document = {
            "schema": "merlin.compiler-coverage.v1",
            "finalized": True,
            "compiler_completed": compiler_completed and not missing,
            "claim_scope": "individual bounded pass boundaries; no whole-compiler or native-fidelity claim",
            "requested_passes": sorted(self.validation.passes),
            "verified_boundaries": sum(row["status"] == "verified" for row in rows),
            "unverified_boundaries": sum(row["status"] == "unverified" for row in rows),
            "unchanged_boundaries": sum(row["status"] == "unchanged" for row in rows),
            "boundaries": rows,
        }
        directory = Path(self.validation.gate.directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "coverage.json"
        temporary = directory / "coverage.json.partial"
        temporary.write_text(json.dumps(document, sort_keys=True, indent=2) + "\n")
        temporary.replace(path)
        unverified = [row for row in rows if row["status"] == "unverified"]
        if unverified:
            reasons = "; ".join(f"{row['pass']} [{row['profile']}]: {row['reason']}" for row in unverified)
            _LOG.warning("Unverified compiler boundaries: %s. Coverage: %s", reasons, path)
        if compiler_completed and missing:
            raise GateFailure(
                {
                    "check": "compiler_coverage",
                    "label": "completed proof jobs",
                    "verdict": "unavailable",
                    "reason": "an admitted boundary has no unique completed proof verdict",
                    "directory": str(directory),
                }
            )
        return document


def begin(validation):
    journal = CoverageJournal(validation)
    return journal, _ACTIVE.set(journal)


def reset(token):
    _ACTIVE.reset(token)


def boundary(pass_name, before, after, *, profile_id, exclusion=None, job=None):
    journal = _ACTIVE.get()
    if journal is not None:
        journal.boundary(pass_name, before, after, profile_id=profile_id, exclusion=exclusion, job=job)
