"""Redacted first-grade observations, never a replacement for a complete verdict."""

from __future__ import annotations

import datetime as dt
import json

from . import qa
from .qualification_summary import summarize


def observed_failures(rows, source):
    """Group only observed, redacted failures; never infer failure of unrun cases."""
    from merlin.targetgen.bundle_harness import unresolved_symbols

    grouped = {}
    total = 0
    for row in rows:
        detail = row.get("failure_detail")
        if not isinstance(detail, str) or not detail:
            continue
        total += 1
        symbols = unresolved_symbols(detail) if "link failed:" in detail else ()
        key = ("link", symbols) if symbols else (row.get("failure_plane"), row.get("failure_category"), detail)
        if key not in grouped and len(grouped) >= 4:
            continue
        group = grouped.setdefault(key, {
            "stage": "link" if symbols else row.get("failure_plane"),
            "missing_symbols": list(symbols[:16]),
            "detail": ("Observed linker cannot resolve: " + ", ".join(symbols[:16])) if symbols else detail[:600],
            "observed_capsules": [], "n_observed": 0,
        })
        group["n_observed"] += 1
        if len(group["observed_capsules"]) < 32:
            group["observed_capsules"].append(row.get("capsule"))
    return {"graded_source_sha256": source, "n_observed_failures": total,
            "groups": list(grouped.values()), "scope": "observed artifacts only; unrun capsules remain unmeasured",
            "numerical_failure_inferred": False, "certification_authority": False}


def _previous_failure_summary(path):
    try:
        if path.is_symlink() or path.stat().st_size > 2 << 20:
            return None
        prior = json.loads(path.read_text())
        summary = observed_failures(prior.get("per_capsule", []), prior.get("graded_source_sha256"))
        if not summary["n_observed_failures"]:
            summary = prior.get("previous_snapshot_failures")
        if not isinstance(summary, dict):
            return None
        source = summary.get("graded_source_sha256")
        if (not isinstance(source, str) or len(source) != 64
                or any(c not in "0123456789abcdef" for c in source)):
            return None
        # Retain only the previously redacted diagnostic fields, not old counts
        # as a result for the new batch or any proof of current-source validity.
        return {key: summary.get(key) for key in (
            "graded_source_sha256", "n_observed_failures", "groups", "scope",
            "numerical_failure_inferred", "certification_authority")}
    except (OSError, ValueError, TypeError, KeyError):
        return None


class SnapshotProgress:
    def __init__(self, workspace, runs_root, source_sha256, guard):
        self.path = workspace / "qa/progress.json"
        self.runs_root, self.source, self.guard = runs_root, source_sha256, guard
        self.rows = {}
        self.n_selected = None
        self.previous_failures = _previous_failure_summary(self.path)
        if self.previous_failures is None:
            # A newer empty live stream must not erase the last completed source-bound failure.
            history = runs_root.parent.parent / "qa_history"
            candidates = sorted(history.glob("verdict_fast_*.json"), key=lambda p: p.stat().st_mtime_ns,
                                reverse=True)[:8]
            for archived in candidates:
                self.previous_failures = _previous_failure_summary(archived)
                if self.previous_failures is not None:
                    break

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
            current = self.guard.current()
            # A draining obsolete grade must not replace a newer grade's stream. The newest owned snapshot
            # may: an author who keeps editing leaves no snapshot current, and requiring currency left the
            # live stream on an old source for 20+ minutes while newer grades ran.
            from .snapshot_publication import owns

            live_matches = (current or not self.path.is_file()
                or json.loads(self.path.read_text()).get("graded_source_sha256") == self.source
                or owns(self.path.parent.parent))
            rows = list(self.rows.values())
            qualification = summarize(rows)
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
                "observed_failure_summary": observed_failures(rows, self.source),
                "previous_snapshot_failures": self.previous_failures,
                "previous_failures_are_current_verdict": False,
                "updated_at": dt.datetime.now(dt.UTC).isoformat(),
                "scope": "completed redacted rows only; selected full roster and final gates remain mandatory",
            }
            if verdict is not None:
                record["verdict_snapshot_current"] = verdict.get("snapshot_current")
            self.runs_root.mkdir(parents=True, exist_ok=True)
            archive = self.runs_root / "snapshot_progress.json"
            archive.write_text(json.dumps(record, indent=2) + "\n")
            if live_matches:
                from .snapshot_publication import publish_live

                def write():
                    self.path.parent.mkdir(exist_ok=True)
                    temporary = self.path.with_suffix(".tmp")
                    temporary.write_text(json.dumps(record, indent=2) + "\n")
                    temporary.replace(self.path)

                publish_live(self.path.parent.parent, write)
        except (OSError, ValueError, TypeError) as error:
            # Optional observations cannot cancel valid grading or promote it.
            print(f"[first-grade] progress unavailable: {type(error).__name__}", flush=True)
