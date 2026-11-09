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

from . import live, records, text, views

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


def _options(**kwargs: Any) -> views.Options:
    return views.Options(**{k: v for k, v in kwargs.items() if v is not None})


def default_destination(opts: views.Options, ctx: views.Context, now: float) -> Path:
    """Where a page goes when ``--out`` is not given: the target's declared dashboard home."""
    from ..spec import SpecError

    if opts.target is not None:
        return dashboard_dir(opts.target) / ("explorer.html" if opts.explorer or opts.compare else TARGET_PAGE)
    if opts.phase0 is not None:
        from . import phase0

        target = next(
            (d["requirements"]["target"] for d in phase0.summary(opts.phase0)["derivations"] if d.get("requirements")),
            None,
        )
        if not target:
            raise SpecError(f"{opts.phase0} records no target, so it has no default dashboard home; pass --out")
        return dashboard_dir(str(target)) / f"phase0-{Path(opts.phase0).resolve().name}.html"
    summary = records.run_summary(Path(opts.run_dir), now=now, stall_hours=opts.stall_hours, store=opts.store)
    if not summary.get("target"):
        raise SpecError(f"{opts.run_dir} records no target, so it has no default dashboard home; pass --out")
    return dashboard_dir(str(summary["target"])) / f"{summary['run_id']}.html"


def destination_for(**options: Any) -> Path:
    """The default page path for these options (the target's dashboard home)."""
    import time as _time

    opts = _options(**options)
    _check_subject(opts)
    return default_destination(opts, views.Context(), _time.time())


def _check_subject(opts: views.Options) -> None:
    from ..spec import SpecError

    subjects = [opts.run_dir is not None, opts.target is not None, opts.phase0 is not None]
    if opts.compare is not None:
        return
    if sum(subjects) != 1:
        raise SpecError("dashboard needs exactly one of a run directory, --target or --phase0")
    if opts.explorer and opts.target is None:
        raise SpecError("--explorer needs --target")


def write_dashboard(
    *,
    run_dir: Path | None = None,
    target: str | None = None,
    out: Path | None = None,
    store: Path | None = None,
    stall_hours: float = records.DEFAULT_STALL_HOURS,
    now: float | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Summarize a run, a target, a Phase 0 directory or a comparison and write its page.

    ``extra`` carries the richer views' options (:class:`.views.Options`: ``phase0``, ``monitor``,
    ``load``, ``operator_private``, ``corpus``, ``measurement_root``, ``stage_root``, ``explorer``,
    ``compare``)."""
    import time as _time

    opts = _options(run_dir=run_dir, target=target, store=store, stall_hours=stall_hours, **extra)
    _check_subject(opts)
    now = _time.time() if now is None else now
    ctx = views.Context()
    destination = Path(out) if out is not None else default_destination(opts, ctx, now)
    live.check_output(destination, opts.inputs())
    for name, page in views.render_pages(opts, ctx, now).items():
        live.write_page(destination.parent / name if name else destination, page)
    result: dict[str, Any] = {"dashboard": str(destination)}
    if opts.run_dir is not None and opts.compare is None:
        summary = ctx.tails.state.get("base_summary")
        if summary is None:
            summary = (
                records.run_summary(Path(run_dir), now=now, stall_hours=stall_hours, store=store)
                if not (Path(run_dir) / "state").is_dir()
                else {"kind": "paired"}
            )
        result.update(
            kind=summary.get("kind"),
            target=summary.get("target"),
            state=(summary.get("liveness") or {}).get("state"),
            detail=(summary.get("liveness") or {}).get("detail"),
        )
    elif opts.target is not None:
        result.update(kind="explorer" if opts.explorer else "target", target=opts.target)
    elif opts.phase0 is not None:
        result.update(kind="phase0", mode="operator-private" if opts.operator_private else "public")
    else:
        result.update(kind="compare")
    return result


def isolate(cpus: str | None = None) -> dict[str, Any]:
    """Make this process cheap for whatever else runs here: lowest CPU priority, idle I/O class, and an
    optional CPU set (``"24-31"`` or ``"24,26,28-31"``).  Returns what was applied."""
    import os
    import shutil
    import subprocess

    applied: dict[str, Any] = {}
    try:
        applied["nice"] = os.nice(19 - os.nice(0))
    except OSError as exc:
        applied["nice"] = f"unchanged ({exc})"
    ionice = shutil.which("ionice")
    if ionice:
        done = subprocess.run([ionice, "-c", "3", "-p", str(os.getpid())], capture_output=True, check=False)
        applied["ionice"] = "idle" if done.returncode == 0 else f"unchanged ({done.stderr.decode().strip()})"
    else:
        applied["ionice"] = "unavailable (no ionice binary)"
    if cpus:
        chosen: set[int] = set()
        for part in cpus.split(","):
            lo, _, hi = part.strip().partition("-")
            if not lo.isdigit() or (hi and not hi.isdigit()):
                from ..spec import SpecError

                raise SpecError(f"--cpus takes a list such as 24-31 or 24,26: {cpus!r}")
            chosen.update(range(int(lo), int(hi or lo) + 1))
        os.sched_setaffinity(0, chosen)
        applied["cpus"] = sorted(os.sched_getaffinity(0))
    return applied


def serve_dashboard(
    *,
    out: Path,
    interval: float = live.DEFAULT_INTERVAL,
    port: int = live.DEFAULT_PORT,
    cpus: str | None = None,
    iterations: int | None = None,
    sleep: Callable[[float], None] = time.sleep,
    announce: Callable[[str], None] | None = None,
    ready: Callable[[Any], None] | None = None,
    isolation: bool = True,
    **options: Any,
) -> int:
    """``dashboard --live``: rewrite ``out`` every ``interval`` s from the records and serve it locally.

    Reads only; writes only ``out`` (refused inside any directory it reads).  Runs at the lowest CPU and
    I/O priority, optionally pinned to ``cpus``."""
    announce = announce or (lambda message: print(message, flush=True))
    opts = _options(**options)
    opts.refresh = interval
    _check_subject(opts)
    live.check_output(Path(out), opts.inputs())
    if isolation:
        announce(f"isolation: {isolate(cpus)}")
    ctx = views.Context()
    return live.serve(
        lambda: views.render_pages(opts, ctx),
        Path(out),
        interval=interval,
        port=port,
        iterations=iterations,
        sleep=sleep,
        announce=announce,
        ready=ready,
    )


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
    from . import activity
    from .tail import Tails

    tails = Tails()  # the agent stream is tailed from where the previous refresh stopped
    try:
        while True:
            summary = records.run_summary(Path(run_dir), now=now, stall_hours=stall_hours, store=store)
            page = text.render(summary, colour=colour)
            agent = text.agent(activity.read(Path(run_dir), records.Inventory(), tails), summary["generated"])
            if agent:
                page += "\n".join(agent) + "\n"
            stream.write(page if once else text.CLEAR + page)
            stream.flush()
            if once:
                return 0
            sleep(max(1.0, float(interval)))
    except KeyboardInterrupt:
        stream.write("\n")
        return 0


__all__ = [
    "DASHBOARD_HOME",
    "dashboard_dir",
    "destination_for",
    "isolate",
    "serve_dashboard",
    "watch",
    "write_dashboard",
]
