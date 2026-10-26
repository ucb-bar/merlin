"""Publish a finished grade's results to the author even when its snapshot is no longer current.

``qa/verdict.json`` is live feedback and belongs to the newest owned, current snapshot
(:mod:`.snapshot_publication`): a grade whose source the author has since edited is archived, never
written there. That rule is right for the live verdict, but an author who keeps editing leaves no
snapshot current for long, while a grade takes minutes. Measured on a live run: the agent's verdict
held only the model gate for 20+ minutes while archived grades had 252 rows and 120 L2 passes, so the
per-test-case results and the ranked performance brief almost never reached it.

``qa/latest_completed_grade.json`` holds, for every test case, the newest completed result: its
top-level fields are the newest grade's (by snapshot token), and its rows merge across grades. A grade
admits only the cases it can start before its admission window closes, so a newer grade often runs a
few cases and gates the rest; taking it whole would hide the older grade's results. Measured on a live
run: a newer grade with 1 pass and 174 admission-deferred rows displaced an older grade with 52 passes.
So a row with a result is never replaced by one without (gated, deferred, not run), a newer result
replaces an older one, and each row's ``graded`` names the grade and source digest it describes.
``snapshot_current`` is claimed only when every row describes the newest grade's bytes. The live
verdict is never touched for a non-current snapshot; the author's brief names this file, and a current
snapshot's verdict stays authoritative.
"""

from __future__ import annotations

import fcntl
import json
from pathlib import Path

FILENAME = "latest_completed_grade.json"
#: The one-line-per-test-case view written beside it (:mod:`.grade_summary`).
SUMMARY_FILENAME = "latest_grade_summary.txt"
POINTER = "latest_completed_grade"
#: Each row's grade of origin, so merged rows stay attributed to the bytes they graded.
ORIGIN = "graded"
SUMMARY = "completed_rows"
SUMMARY_SCHEMA = "merlin.phase1.completed_grade_rows.v1"
#: Grades recorded into the file, newest last; readers (the console) use it to tell what was delivered.
KEEP_RECORDED = 64
NOTE = (
    "each row is the newest completed result for its test case; a case a newer grade did not run "
    "(gated, deferred) keeps its last result. `graded` on a row names the grade and source digest it "
    "describes, which may differ from your current source. Top-level counts (n_passed, all_pass, "
    "tiers) belong to the newest grade only; `completed_rows` counts the merged rows."
)
#: Row statuses that state no result: the case never ran (admission closed, gate, deferral, or carried
#: by regression sampling, whose earlier result is the one kept).
NO_RESULT = frozenset({None, "", "gated", "declined", "deferred", "incomplete", "carried"})


def _token(value) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def publish(workspace, verdict: dict, token, write_verdict) -> dict:
    """Write the live verdict when current; record the newest completed grade either way."""
    from . import package_rejection

    qa = Path(workspace) / "qa"
    if verdict.get("snapshot_current"):
        verdict = write_verdict(qa / "verdict.json", verdict)
    else:
        package_rejection.publish_notice(workspace, verdict)
    try:
        record(workspace, verdict, token)
    except (OSError, ValueError, TypeError) as error:  # the archive keeps the grade; this is a view
        print(f"[first-grade] latest completed grade unavailable: {type(error).__name__}", flush=True)
    return verdict


def executed(row) -> bool:
    """Whether a per-test-case row carries a result rather than only a reason it did not run."""
    return isinstance(row, dict) and (bool(row.get("tiers")) or row.get("status") not in NO_RESULT)


def _origin(document: dict, token) -> dict:
    return {
        "token": str(token),
        "graded_source_sha256": document.get("graded_source_sha256"),
        "graded_at": document.get("graded_at"),
    }


def _rows(document: dict, origin: dict, *, keep_labels: bool) -> dict:
    """Test case -> row labelled with its grade; a held file's rows keep the label they were merged with."""
    out = {}
    for row in document.get("per_capsule") or ():
        if isinstance(row, dict) and isinstance(row.get("capsule"), str):
            labelled = keep_labels and isinstance(row.get(ORIGIN), dict)
            out[row["capsule"]] = row if labelled else {**row, ORIGIN: origin}
    return out


def _wins(row: dict, held: dict) -> bool:
    """Whether ``row`` replaces ``held``: a result beats no result, then the newer grade's row wins."""
    if executed(row) != executed(held):
        return executed(row)
    return (_token(row[ORIGIN].get("token")) or 0) > (_token(held[ORIGIN].get("token")) or 0)


def merge(held: dict, incoming: dict) -> tuple[dict, bool]:
    """Per-test-case merge of two labelled row maps; returns the rows and whether any changed."""
    merged, changed = dict(held), False
    for name, row in incoming.items():
        if name not in merged or _wins(row, merged[name]):
            merged[name], changed = row, True
    return merged, changed


def _summary(rows: list, newest: dict, current_source, recorded: list) -> dict:
    grades: dict = {}
    for row in rows:
        origin = row[ORIGIN]
        entry = grades.setdefault(
            origin.get("token"),
            {
                "token": origin.get("token"),
                "graded_source_sha256": origin.get("graded_source_sha256"),
                "graded_at": origin.get("graded_at"),
                "rows": 0,
                "executed": 0,
                "passed": 0,
            },
        )
        entry["rows"] += 1
        entry["executed"] += executed(row)
        entry["passed"] += row.get("status") == "pass"
    return {
        "schema": SUMMARY_SCHEMA,
        "rows": len(rows),
        "executed": sum(executed(row) for row in rows),
        "passed": sum(row.get("status") == "pass" for row in rows),
        "from_newest_grade": (grades.get(newest["token"]) or {}).get("rows", 0),
        "current_source_sha256": current_source,
        "from_current_source": sum(
            1 for row in rows if current_source and row[ORIGIN].get("graded_source_sha256") == current_source
        ),
        "by_grade": sorted(grades.values(), key=lambda entry: _token(entry["token"]) or 0, reverse=True),
        "recorded_grades": recorded[-KEEP_RECORDED:],
        "note": NOTE,
    }


def record(workspace, verdict: dict, token) -> bool:
    """Merge a completed grade into ``qa/latest_completed_grade.json``; returns whether the file changed.

    A grade whose rows are all gated (e.g. its snapshot stopped admission before any case ran)
    stays in ``qa_history`` and changes nothing. A newer grade supplies the top-level fields; an older
    grade finishing late only fills in test cases the held rows have no result for.
    """
    order = _token(token)
    if order is None or not any(executed(row) for row in verdict.get("per_capsule") or ()):
        return False  # an empty, aborted or all-gated grade never replaces completed evidence
    qa = Path(workspace) / "qa"
    qa.mkdir(parents=True, exist_ok=True)
    path = qa / FILENAME
    with (qa / ".latest-completed.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            held = json.loads(path.read_text()) if path.is_file() else None
            previous = _token(held.get(POINTER, {}).get("token")) if isinstance(held, dict) else None
        except (OSError, ValueError, AttributeError):
            held, previous = None, None
        held = held if previous is not None else None
        recorded = [str(t) for t in ((held or {}).get(SUMMARY) or {}).get("recorded_grades") or () if _token(t)]
        if previous == order or str(order) in recorded:
            return False
        incoming = _rows(verdict, _origin(verdict, order), keep_labels=False)
        held_rows = _rows(held, _origin(held, previous), keep_labels=True) if held else {}
        merged, changed = merge(held_rows, incoming)
        newer = held is None or order > previous
        if not (newer or changed):
            return False
        top = dict(verdict) if newer else {key: value for key, value in held.items() if key not in (POINTER, SUMMARY)}
        first = [
            row["capsule"]
            for row in top.get("per_capsule") or ()
            if isinstance(row, dict) and row.get("capsule") in merged
        ]
        rows = [merged[name] for name in dict.fromkeys([*first, *sorted(merged)])]
        top["per_capsule"] = rows
        from .failure_clusters import attach as attach_clusters
        from .joint_physical_correctness import attach
        from .package_rejection import enrich
        from .perf_history import Recorder
        from .performance_headroom import attach as attach_headroom

        ledger = Recorder()
        # Clusters span the merged rows, so a cluster can name cases graded on different sources.
        document = attach_clusters(
            workspace, enrich(attach_headroom(Path(workspace), attach(path, top), observe=ledger.observe))
        )
        newest = _origin(top, order if newer else previous)
        same_bytes = all(row[ORIGIN].get("graded_source_sha256") == newest["graded_source_sha256"] for row in rows)
        document["snapshot_current"] = bool(top.get("snapshot_current")) and same_bytes
        document[POINTER] = {
            **newest,
            "path": f"qa/{FILENAME}",
            "summary": f"qa/{SUMMARY_FILENAME}",
            "snapshot_current": document["snapshot_current"],
            "note": (
                "newest completed grade; its results describe graded_source_sha256, which may differ "
                "from your current source if you edited since; rows from other grades say so in `graded`"
            ),
        }
        document[SUMMARY] = _summary(rows, newest, top.get("current_source_sha256"), [*recorded, str(order)])
        from .performance_headline import KEY as HEADLINE

        if isinstance(document.get(HEADLINE), dict):  # name the grade the merged rows' headline belongs to
            document[HEADLINE]["grade"] = {"token": newest["token"], **(document[HEADLINE].get("grade") or {})}
        temporary = path.with_suffix(".next.json")
        temporary.write_text(json.dumps(document, indent=2) + "\n")
        temporary.replace(path)
        from .grade_summary import write as write_summary

        write_summary(workspace, document)  # the compact view, under the same lock as the file it mirrors
        ledger.append(workspace, newest)  # the per-case ledger of what changed (:mod:`.perf_history`)
        from .capability_brief import refresh_completed

        try:
            refresh_completed(workspace, document)
        except Exception as error:  # noqa: BLE001 -- advisory memory cannot discard a durable grade
            print(f"[first-grade] joint brief unavailable: {type(error).__name__}", flush=True)
    return True
