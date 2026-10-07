"""Experiment tracking views: a static HTML dashboard and a live terminal view, from records only.

``merlin-experiment dashboard <run_dir> | --target T`` writes one self-contained HTML file;
``merlin-experiment watch <run_dir>`` prints the same summary in the terminal until interrupted.
Both read the records the phase owners already wrote (:mod:`.records`) and never measure, grade or
build anything.  The dashboard's home is the storage contract's ``experiment-dashboards`` product
root: ``out/artifacts/experiments/<target>/dashboard/``.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TextIO

from . import html, records, text

#: The storage contract's ``product_roots`` entry for these views.
DASHBOARD_HOME = "experiment-dashboards"
DASHBOARD_DIR = "dashboard"
TARGET_PAGE = "target.html"


def dashboard_dir(target: str, *, artifacts_root: str | Path | None = None) -> Path:
    """``out/artifacts/experiments/<target>/dashboard/`` (the contract's declared home)."""
    from merlin.common.artifacts import declared_home
    from merlin.targetgen import package_records

    return (
        declared_home(DASHBOARD_HOME, artifacts_root=artifacts_root) / package_records.component(target) / DASHBOARD_DIR
    )


def write_dashboard(
    *,
    run_dir: Path | None = None,
    target: str | None = None,
    out: Path | None = None,
    store: Path | None = None,
    stall_hours: float = records.DEFAULT_STALL_HOURS,
    now: float | None = None,
) -> dict[str, Any]:
    """Summarize a run (or every run of a target) and write its page; return where and what it says."""
    from ..spec import SpecError

    if (run_dir is None) == (target is None):
        raise SpecError("dashboard needs exactly one of a run directory or --target")
    if target is not None:
        summary = records.target_summary(target, now=now, stall_hours=stall_hours)
        destination = out or dashboard_dir(target) / TARGET_PAGE
    else:
        summary = records.run_summary(Path(run_dir), now=now, stall_hours=stall_hours, store=store)
        if out is None:
            if not summary.get("target"):
                raise SpecError(f"{run_dir} records no target, so it has no default dashboard home; pass --out")
            out = dashboard_dir(str(summary["target"])) / f"{summary['run_id']}.html"
        destination = out
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.tmp")
    staging.write_text(html.render(summary), encoding="utf-8")
    staging.replace(destination)
    return {
        "dashboard": str(destination),
        "kind": summary["kind"],
        "target": summary.get("target"),
        "state": (summary.get("liveness") or {}).get("state"),
        "detail": (summary.get("liveness") or {}).get("detail"),
    }


def watch(
    run_dir: Path,
    *,
    interval: float = 30.0,
    once: bool = False,
    store: Path | None = None,
    stall_hours: float = records.DEFAULT_STALL_HOURS,
    colour: bool | None = None,
    stream: TextIO | None = None,
    sleep: Callable[[float], None] = time.sleep,
    now: float | None = None,
) -> int:
    """Print the run's summary; unless ``once``, clear and reprint every ``interval`` s until Ctrl-C."""
    stream = stream or sys.stdout
    if colour is None:
        colour = bool(getattr(stream, "isatty", lambda: False)())
    try:
        while True:
            summary = records.run_summary(Path(run_dir), now=now, stall_hours=stall_hours, store=store)
            page = text.render(summary, colour=colour)
            stream.write(page if once else text.CLEAR + page)
            stream.flush()
            if once:
                return 0
            sleep(max(1.0, float(interval)))
    except KeyboardInterrupt:
        stream.write("\n")
        return 0


__all__ = ["DASHBOARD_HOME", "dashboard_dir", "watch", "write_dashboard"]
