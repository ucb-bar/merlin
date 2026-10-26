"""Keep ``qa/round_brief.md`` current while one continuous session runs.

The brief was written at round boundaries only (:func:`.brief.write` after a round's grade, and
:func:`.brief.refresh_before_launch` before an agent turn). Under the default continuous schedule a run
is one round, so the brief the author is told to read first was written once at session start and then
never again. Measured on a live run: the brief kept its 17:40 contents through an 8 h session while
background grades kept publishing ``qa/verdict.json`` and ``qa/latest_completed_grade.json``.

:func:`after_grade` rebuilds it from those published files right after a background grade is
published; :func:`refreshing` attaches it to the background owner's grade callbacks
(:func:`.lifecycle.start_background`). It shares the rebuild with :func:`.brief.refresh_before_launch` (a recognized leading
RESUME banner survives, the notes-staleness stamp is not advanced), writes under the brief's own
``qa/`` lock through a temporary file and an atomic rename, and never raises: a brief that cannot be
rebuilt leaves the previous one in place and the grade untouched.
"""

from __future__ import annotations

from pathlib import Path


def refreshing(grade, workspace, run_dir):
    """``grade`` (a background grade callback) followed by :func:`after_grade`, whatever it returned.

    The background owner wraps its first-grade and interval-grade callbacks with this, so every grade it
    publishes, on any schedule, is followed by a brief rebuilt from what it published. A grade that
    raised published nothing new and is re-raised unchanged."""

    def run(*args, **kwargs):
        result = grade(*args, **kwargs)
        after_grade(run_dir, workspace)
        return result

    return run


def after_grade(run_dir, workspace) -> Path | None:
    """Rebuild the round brief after a published grade; returns its path, or ``None`` on any failure."""
    try:
        from . import brief

        path = brief.refresh_before_launch(Path(run_dir), Path(workspace), 0)
    except Exception as error:  # noqa: BLE001 -- a stale brief is better than a failed grade
        print(f"[grade] round brief refresh skipped: {type(error).__name__}: {error}", flush=True)
        return None
    return path
