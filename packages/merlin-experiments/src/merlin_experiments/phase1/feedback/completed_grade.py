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

Every publication reconciles each row with the current source (:mod:`.reconcile`): its ``graded`` gains
``currency`` -- ``current`` when the bytes the case reaches are unchanged since its snapshot, else
``older_version`` (a result for the older source it names, still shown as evidence). ``completed_rows``
counts current rows and passes separately; only those are passes of the current compiler.
"""

from __future__ import annotations

import fcntl
import json
from contextlib import contextmanager
from pathlib import Path

from . import reconcile, result_details
from .stage_timing import measure

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
    "describes, which may differ from your current source; its `currency` says whether the bytes the case "
    "reaches are unchanged since then (`current`) or the row is a result for an older version "
    "(`older_version`). Top-level counts (n_passed, all_pass, tiers) belong to the newest grade only; "
    "`completed_rows` counts the merged rows, and `current_passed` only passes of your current source."
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
        **({"reach_sha256": document["graded_reach_sha256"]} if document.get("graded_reach_sha256") else {}),
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
    """Whether ``row`` replaces ``held``: a result beats no result, then the newer grade's row wins.

    A grade's own completed document replaces the rows it already published case by case (:func:`record_rows`).
    """
    if executed(row) != executed(held):
        return executed(row)
    return (_token(row[ORIGIN].get("token")) or 0) >= (_token(held[ORIGIN].get("token")) or 0)


def merge(held: dict, incoming: dict) -> tuple[dict, bool]:
    """Per-test-case merge of two labelled row maps; returns the rows and whether any changed."""
    merged, changed = dict(held), False
    for name, row in incoming.items():
        same = name in merged and reconcile.comparable(row) == reconcile.comparable(merged[name])
        if name not in merged or (not same and _wins(row, merged[name])):
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
        "current_rows": sum(not reconcile.is_older(row) for row in rows),
        "current_passed": sum(row.get("status") == "pass" and not reconcile.is_older(row) for row in rows),
        "older_version_rows": sum(reconcile.is_older(row) for row in rows),
        "older_version_passed": sum(row.get("status") == "pass" and reconcile.is_older(row) for row in rows),
        "current_reach_sha256": (rows[0][ORIGIN].get("reconciled_reach_sha256") if rows else None),
        "by_grade": sorted(grades.values(), key=lambda entry: _token(entry["token"]) or 0, reverse=True),
        "recorded_grades": recorded[-KEEP_RECORDED:],
        "note": NOTE,
    }


@contextmanager
def _held(workspace):
    """``(path, held document or None, its grade token, recorded grades)`` under the completed-grade lock."""
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
        yield path, result_details.hydrate(held, workspace=workspace, read_only=True), previous, recorded


def record(workspace, verdict: dict, token) -> bool:
    """Merge a completed grade into ``qa/latest_completed_grade.json``; returns whether the file changed.

    A grade whose rows are all gated (e.g. its snapshot stopped admission before any case ran)
    stays in ``qa_history`` and changes nothing. A newer grade supplies the top-level fields; an older
    grade finishing late only fills in test cases the held rows have no result for.
    """
    order = _token(token)
    if order is None or not any(executed(row) for row in verdict.get("per_capsule") or ()):
        return False  # an empty, aborted or all-gated grade never replaces completed evidence
    with _held(workspace) as (path, held, previous, recorded):
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
        _store(workspace, path, top, order if newer else previous, recorded=[*recorded, str(order)])
    return True



def record_rows(workspace, rows, token, *, graded_source_sha256, graded_at=None, reach_sha256=None) -> bool:
    """Merge one running grade's completed test cases before the grade itself completes.

    A grade holds its owner until its slowest admitted case finishes; measured on a live run, small
    cases finished within five minutes of the start while whole-model cases held the grade for 25-55
    minutes, so their results reached this file an hour late. Each row is labelled with its own grade
    and source digest and merged by the same rule as :func:`record` (a result beats no result, then the
    newer grade); top-level fields stay the held grade's until this grade completes. Without a held
    document there are no top-level fields to keep, so nothing is written (``qa/progress.json`` streams
    the first grade). Returns whether the file changed.
    """
    order = _token(token)
    if order is None:
        return False
    origin = {"token": str(order), "graded_source_sha256": graded_source_sha256, "graded_at": graded_at,
              **({"reach_sha256": reach_sha256} if reach_sha256 else {})}
    incoming = {row["capsule"]: {**row, ORIGIN: origin} for row in rows
                if isinstance(row, dict) and isinstance(row.get("capsule"), str) and executed(row)}
    if not incoming:
        return False
    with _held(workspace) as (path, held, previous, recorded):
        if held is None or str(order) in recorded:
            return False  # a completed grade already published its own rows
        merged, changed = merge(_rows(held, _origin(held, previous), keep_labels=True), incoming)
        if not changed:
            return False
        top = {key: value for key, value in held.items() if key not in (POINTER, SUMMARY)}
        first = [row["capsule"] for row in top.get("per_capsule") or ()
                 if isinstance(row, dict) and row.get("capsule") in merged]
        top["per_capsule"] = [merged[name] for name in dict.fromkeys([*first, *sorted(merged)])]
        _store(workspace, path, top, previous, recorded=recorded)
    return True


def project_rows(workspace, token, projections: dict) -> bool:
    """Add a grade's late advisory fields (``{case: {field: value}}``) to the rows that grade owns.

    A row another grade has since replaced is left alone: the projection describes only its own grade's
    executable. Counts, statuses and tiers are never among the fields. Returns whether the file changed.
    """
    order = _token(token)
    if order is None or not projections:
        return False
    with _held(workspace) as (path, held, previous, recorded):
        if held is None:
            return False
        rows = [row for row in held.get("per_capsule") or () if isinstance(row, dict)]
        changed = False
        for row in rows:
            fields = projections.get(row.get("capsule"))
            if fields and _token((row.get(ORIGIN) or {}).get("token")) == order:
                row.update(fields)
                changed = True
        if not changed:
            return False
        top = {key: value for key, value in held.items() if key not in (POINTER, SUMMARY)}
        top["per_capsule"] = rows
        _store(workspace, path, top, previous, recorded=recorded)
    return True


def follow_up(workspace, archive, verdict: dict, token, write_verdict, projections: dict) -> None:
    """Publish a grade's late advisory fields (``{case: {field: value}}``) wherever that grade is shown.

    The archived verdict gains them in place (its modification time kept: readers order the archive
    by it), the live verdict only while it still shows this very grade (same source digest and
    ``graded_at``) and this grade still owns live publication, and ``qa/latest_completed_grade.json``
    only on rows this grade still owns (:func:`project_rows`).
    """
    import os

    archive = Path(archive)
    stat = archive.stat()
    document = json.loads(archive.read_text())
    if document.get("graded_at") != verdict.get("graded_at"):
        raise ValueError("archived verdict is not the grade being followed up")
    for row in document.get("per_capsule") or ():
        if isinstance(row, dict) and row.get("capsule") in projections:
            row.update(projections[row["capsule"]])
    temporary = archive.with_name(f".{archive.name}.next")
    temporary.write_text(json.dumps(document, indent=2))
    temporary.replace(archive)
    os.utime(archive, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    if verdict.get("snapshot_current"):

        def same_grade(previous):
            return all(previous.get(key) == verdict.get(key) for key in ("graded_at", "graded_source_sha256"))

        write_verdict(Path(workspace) / "qa" / "verdict.json", document, expect=same_grade)
    project_rows(workspace, token, projections)

def _store(workspace, path: Path, top: dict, token, *, recorded: list) -> None:
    """Enrich ``top`` (its merged rows already in place) and write it with every view derived from it.

    Callers hold the completed-grade lock; ``token`` is the grade the top-level fields belong to.
    """
    from .failure_clusters import attach as attach_clusters
    from .joint_physical_correctness import attach
    from .package_rejection import enrich
    from .perf_history import Recorder
    from .performance_headroom import attach as attach_headroom

    rows = top["per_capsule"]
    _reconcile(workspace, rows, top)
    ledger = Recorder()
    # Clusters span the merged rows, so a cluster can name cases graded on different sources.
    document = top
    with measure("attach_physical_correctness", document):
        document = attach(path, document)
    with measure("attach_performance_headroom", document):
        document = attach_headroom(Path(workspace), document, observe=ledger.observe)
    with measure("attach_package_rejection", document):
        document = enrich(document)
    with measure("attach_failure_clusters", document):
        document = attach_clusters(workspace, document)
    newest = _origin(top, token)
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
    document[SUMMARY] = _summary(rows, newest, top.get("current_source_sha256"), recorded)
    from .performance_headline import KEY as HEADLINE

    if isinstance(document.get(HEADLINE), dict):  # name the grade the merged rows' headline belongs to
        document[HEADLINE]["grade"] = {"token": newest["token"], **(document[HEADLINE].get("grade") or {})}
    events = path.parent / "publication_timings.jsonl"
    with measure("result_details.compact", document, events=events):
        compact = result_details.compact_document(document, workspace=workspace)
    try:
        previous = json.loads(path.read_text()) if path.is_file() else None
    except (OSError, ValueError):
        previous = None
    with measure("completed_grade.index_publication", document, events=events):
        result_details.append_rows(workspace, previous, compact)
        temporary = path.with_suffix(".next.json")
        temporary.write_text(json.dumps(compact, separators=(",", ":")) + "\n")
        temporary.replace(path)
    from .grade_summary import write as write_summary

    write_summary(workspace, document)  # the compact view, under the same lock as the file it mirrors
    ledger.append(workspace, newest)  # the per-case ledger of what changed (:mod:`.perf_history`)
    from .capability_brief import refresh_completed

    try:
        refresh_completed(workspace, document)
    except Exception as error:  # noqa: BLE001 -- advisory memory cannot discard a durable grade
        print(f"[first-grade] joint brief unavailable: {type(error).__name__}", flush=True)


def _reconcile(workspace, rows: list, top: dict) -> None:
    """Label every merged row ``current`` or ``older_version`` against the source on disk now."""
    reach = reconcile.current_reach(workspace)
    source = None
    if reach is None:  # no decomposition: only a byte-identical source is current
        from merlin.common.tree_hash import hash_tree

        try:
            source = hash_tree(Path(workspace) / "submission")["sha256"]
        except OSError:
            source = top.get("current_source_sha256")
    reconcile.label(rows, reach=reach, source=source, origin_key=ORIGIN)
