"""Tell a continuation's new session where the previous session left off.

A relaunch seeds the new workspace with the previous compiler, but the new author session starts with no
memory of the last one: its open threads, the hypothesis it was testing and the failures it was looking
at are gone, and the first part of the new session is spent rediscovering them. ``PREVIOUS_SESSION.md``
is a compact, read-only summary built only from what the previous session itself wrote or was shown:

* its last turn report (``rounds/*.final.txt``);
* the tail of its own ``docs/iteration_notes.md`` from the seeded submission;
* the open failures and largest performance headroom of the newest grade it could see
  (``qa/latest_completed_grade.json``, else ``qa/verdict.json``).

Host paths are scrubbed and every section is bounded. Missing inputs say "not recorded"; nothing here can
fail a launch.
"""
from __future__ import annotations

import json
from pathlib import Path

FILENAME = "PREVIOUS_SESSION.md"
SECTION_CHARS = 4000
MAX_FAILURES = 30
MAX_HEADROOM = 5


def _tail(text: str, limit: int = SECTION_CHARS) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[-limit:]
    newline = cut.find("\n")
    return "…\n" + (cut[newline + 1:] if 0 <= newline < len(cut) - 1 else cut)


def _read(path: Path) -> str | None:
    try:
        if path.is_symlink() or not path.is_file():
            return None
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _last_turn_report(run_dir: Path) -> str | None:
    rounds = run_dir / "rounds"
    try:
        reports = sorted(p for p in rounds.iterdir() if p.name.endswith(".final.txt") and p.is_file())
    except OSError:
        return None
    return _read(reports[-1]) if reports else None


def _latest_grade(workspace: Path) -> dict | None:
    for name in ("latest_completed_grade.json", "verdict.json"):
        text = _read(workspace / "qa" / name)
        if not text:
            continue
        try:
            document = json.loads(text)
        except ValueError:
            continue
        if isinstance(document, dict) and document.get("per_capsule"):
            return document
    return None


def _failures(grade: dict) -> list[str]:
    rows = []
    for row in grade.get("per_capsule") or ():
        if not isinstance(row, dict):
            continue
        tiers = row.get("tiers") if isinstance(row.get("tiers"), dict) else {}
        failed = row.get("status") == "fail" or any(value == "fail" for value in tiers.values())
        if failed:
            reason = row.get("failure_category") or row.get("failure_plane") or "fail"
            rows.append(f"- `{row.get('capsule')}`: {reason}")
    return rows


def _headroom(grade: dict) -> list[str]:
    priority = ((grade.get("feedback_brief") or {}).get("performance_priority") or {})
    block = priority.get("development_headroom") if isinstance(priority, dict) else None
    entries = (block or {}).get("largest_headroom") or [] if isinstance(block, dict) else []
    lines = []
    for entry in entries[:MAX_HEADROOM]:
        if isinstance(entry, dict):
            lines.append(f"- `{entry.get('test_case')}`: {entry.get('cycles')} cycles, "
                         f"{entry.get('gap_ratio')}x its roofline bound")
    return lines


def render(previous_run_dir: Path, previous_workspace: Path, seed_submission: Path) -> str:
    """The note's text; every section says ``not recorded`` when its input is absent."""
    from merlin.common.path_scrub import scrub_host_paths

    report = _last_turn_report(previous_run_dir)
    notes = _read(seed_submission / "docs" / "iteration_notes.md")
    grade = _latest_grade(previous_workspace)
    parts = [
        "# Where the previous session left off",
        "",
        "Operator-generated from the previous session's own outputs. It grants no credit: every result "
        "below is about the previous compiler bytes and is re-checked in this run.",
        "",
        "## Its last turn report",
        "",
        _tail(report) if report else "not recorded",
        "",
        "## End of its iteration notes",
        "",
        _tail(notes) if notes else "not recorded",
        "",
        "## Open failures in the newest grade it could see",
        "",
    ]
    if grade is None:
        parts.append("not recorded")
    else:
        source = str(grade.get("graded_source_sha256") or "")[:12] or "unknown"
        failures = _failures(grade)
        parts.append(f"Graded source `{source}`; {len(failures)} failing test case(s).")
        parts.extend(failures[:MAX_FAILURES])
        if len(failures) > MAX_FAILURES:
            parts.append(f"- … {len(failures) - MAX_FAILURES} more")
        headroom = _headroom(grade)
        parts += ["", "## Largest performance headroom it was shown", ""]
        parts.extend(headroom or ["not recorded"])
    return scrub_host_paths("\n".join(parts) + "\n")


def write(workspace, previous: dict, seed_submission) -> dict:
    """Stage ``PREVIOUS_SESSION.md`` read-only in the new workspace; returns what was written."""
    run_dir, old_workspace = Path(previous["run_dir"]), Path(previous["workspace"])
    text = render(run_dir, old_workspace, Path(seed_submission))
    target = Path(workspace) / FILENAME
    target.unlink(missing_ok=True)
    target.write_text(text, encoding="utf-8")
    target.chmod(0o444)
    return {"file": FILENAME, "bytes": len(text.encode("utf-8")),
            "inputs_found": {"turn_report": _last_turn_report(run_dir) is not None,
                             "iteration_notes": (Path(seed_submission) / "docs" / "iteration_notes.md").is_file(),
                             "grade": _latest_grade(old_workspace) is not None}}
