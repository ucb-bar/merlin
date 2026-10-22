"""Redacted first-grade observations, never a replacement for a complete verdict."""

from __future__ import annotations

import datetime as dt
import json

from . import qa
from .qualification_summary import summarize


class SnapshotProgress:
    def __init__(self, workspace, runs_root, source_sha256, guard):
        self.path = workspace / "qa/progress.json"
        self.runs_root, self.source, self.guard = runs_root, source_sha256, guard
        self.rows = {}
        self.n_selected = None

    def selected(self, count):
        if type(count) is not int or count < 0:
            raise ValueError("selected cohort count must be a nonnegative integer")
        self.n_selected = count
        self._write("running")

    def completed(self, result):
        name = result.get("capsule")
        # Read only the existing durable redaction; raw callback rows may contain
        # private inputs, oracle output or an unredacted error message.
        row = qa._per_capsule_from_results(self.runs_root, only_capsule=name).get(name)
        if row is not None:
            self.rows[name] = {"capsule": name, **row}
            self._write("running")

    def finish(self, verdict):
        superseded = self.guard.reason() is not None
        state = ("superseded_snapshot" if getattr(self.guard, "finish_bounded_snapshot", False)
                 else "deferred") if superseded else "complete"
        self._write(state, verdict=verdict)

    def _write(self, state, *, verdict=None):
        try:
            # A draining obsolete grade must not replace a newer grade's stream.
            if not self.guard.current() and self.path.is_file():
                if json.loads(self.path.read_text()).get("graded_source_sha256") != self.source:
                    return
            rows = list(self.rows.values())
            qualification = summarize(rows)
            current = self.guard.current()
            record = {
                "schema": "merlin.phase1.grade_progress.v1",
                "state": state,
                "all_pass": None,
                "certified_complete": False,
                "original_cohort_complete": False,
                "graded_source_sha256": self.source,
                "snapshot_current": current,
                "count_scope": "immutable_snapshot",
                "current_source_n_passed": sum(row.get("status") == "pass" for row in rows) if current else None,
                "n_selected": self.n_selected,
                "n_completed": len(rows),
                "n_passed": sum(row.get("status") == "pass" for row in rows),
                "n_execution_passed": qualification["execution_passed"],
                "qualification_summary": qualification,
                "per_capsule": rows,
                "updated_at": dt.datetime.now(dt.UTC).isoformat(),
                "scope": "completed redacted rows only; selected full roster and final gates remain mandatory",
            }
            if verdict is not None:
                record["verdict_snapshot_current"] = verdict.get("snapshot_current")
            self.path.parent.mkdir(exist_ok=True)
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps(record, indent=2) + "\n")
            temporary.replace(self.path)
        except (OSError, ValueError, TypeError) as error:
            # Optional observations cannot cancel valid grading or promote it.
            print(f"[first-grade] progress unavailable: {type(error).__name__}", flush=True)
