"""Animated read-only experiment console: cores, jobs, agent tree and performance.

Run ``python -m merlin.experiments.tracking.terminal.watch --task TASK_DIR``.
Sampling is independent of animation; no agents, graders or hardware jobs are
launched, and no campaign files are modified. No new terminal dependency is needed.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import os
import select
import shutil
import sys
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from merlin.experiments.tracking.readers import records as R
from merlin.experiments.tracking.terminal import terminal_views as V

# ---- terminal ----------------------------------------------------------------------------------


class Feed:
    """Slow record reads never block keyboard input or animation."""

    def __init__(self, sampler, metadata, interval, health=None):
        self.sampler, self.metadata, self.interval = sampler, metadata, interval
        self.health = health
        self.lock, self.stop = threading.Lock(), threading.Event()
        self.sample, self.details, self.error = {}, {}, None
        self.generation = 0
        self.thread = threading.Thread(target=self.run, name="terminal-read-only-sampler", daemon=True)

    def run(self):
        next_metadata = 0.0
        run_id = None
        while not self.stop.is_set():
            started = time.monotonic()
            try:
                sample = self.sampler.sample()
                current_run = sample.get("campaign", {}).get("run_id")
                if current_run != run_id:
                    with self.lock:
                        self.details = {}
                    run_id, next_metadata = current_run, 0.0
                with self.lock:
                    self.sample, self.error = sample, None
                    self.generation += 1
                if started >= next_metadata:
                    details = self.metadata.sample(sample)
                    if self.health is not None:
                        details["progress_health"] = self.health.sample(sample)
                    with self.lock:
                        self.details = details
                    next_metadata = started + max(3.0, self.interval)
            except Exception as exc:
                with self.lock:
                    self.error = f"Read unavailable: {type(exc).__name__}: {V.clean(exc, 160)}"
            self.stop.wait(max(0.05, self.interval - (time.monotonic() - started)))

    def latest(self):
        with self.lock:
            return self.sample, self.details, self.error, self.generation


@contextlib.contextmanager
def terminal_mode():
    """Always restore the user's screen, cursor and input mode."""
    import termios
    import tty

    fd = sys.stdin.fileno()
    previous = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        sys.stdout.write("\x1b[?1049h\x1b[?25l\x1b[H")
        sys.stdout.flush()
        yield fd
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, previous)
        sys.stdout.write("\x1b[0m\x1b[?25h\x1b[?1049l")
        sys.stdout.flush()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--task", type=Path, required=True, help="Managed campaign task directory; follow its latest native owner"
    )
    result.add_argument(
        "--selection", type=Path, help="Optional exact selection instead of following the latest launch"
    )
    result.add_argument(
        "--operator-private",
        action="store_true",
        help="Show operator capsule, agent and performance details; never sent to the author loop",
    )
    result.add_argument("--interval", type=float, default=1.0, help="Resource sample interval in seconds (default: 1)")
    result.add_argument("--window", type=int, default=120, help="Seconds displayed in the core timeline (default: 120)")
    result.add_argument(
        "--view",
        choices=["overview", *[v.lower() for v in V.TABS]],
        default="overview",
        help="Unified overview by default; optional focused plain-text reports",
    )
    result.add_argument("--once", action="store_true", help="Print one sampled frame and exit")
    result.add_argument("--no-color", action="store_true")
    result.add_argument("--width", type=int, help="Override detected terminal columns")
    result.add_argument("--height", type=int, help="Override detected terminal rows")
    result.add_argument(
        "--legacy", action="store_true", help="Keep this single-screen view even when the command console is built"
    )
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if not 0.2 <= args.interval <= 60:
        raise SystemExit("--interval must be between 0.2 and 60 seconds")
    if not 10 <= args.window <= 3600:
        raise SystemExit("--window must be between 10 and 3600 seconds")
    if args.width is not None and args.width < 45:
        raise SystemExit("--width must be at least 45 columns")
    if args.height is not None and args.height < 12:
        raise SystemExit("--height must be at least 12 rows")
    interactive = sys.stdin.isatty() and sys.stdout.isatty() and not args.once
    if interactive and not args.legacy and args.view == "overview" and args.selection is None:
        # The command console supersedes this view on a terminal once its frontend is built.
        from merlin.experiments.tracking.console import launch

        binary = launch.binary()
        if binary is not None:
            return launch.exec_frontend(
                launch.frontend_argv(
                    binary, operator_private=args.operator_private, tasks=[args.task], interval=args.interval
                )
            )
    from merlin.experiments.tracking.terminal.metadata import Metadata
    from merlin.experiments.tracking.terminal.progress_alerts import ProgressHealth
    from merlin.experiments.tracking.terminal.resource_probe import Sampler

    sampler = Sampler(args.task, operator_private=args.operator_private, selection=args.selection)
    metadata = Metadata(args.task, operator_private=args.operator_private)
    health = ProgressHealth(args.task, operator_private=args.operator_private)
    history = V.History(args.window)
    color = not args.no_color and "NO_COLOR" not in os.environ and sys.stdout.isatty()
    tab = -1 if args.view == "overview" else V.TABS.index(args.view.upper())
    offset, frame, paused = 0, 0, False
    paused_history = None

    def draw(sample, details, error=None):
        size = shutil.get_terminal_size((120, 40))
        return V.render(
            sample,
            details,
            paused_history if paused else history,
            tab=tab,
            width=args.width or size.columns,
            height=args.height or (size.lines if interactive else max(34, size.lines)),
            color=color,
            frame=frame,
            paused=paused,
            offset=offset,
            error=error,
        )

    if not interactive:
        sampler.sample()  # Establish counters; an unmeasured first sample is never zero utilization.
        time.sleep(args.interval)
        sample = sampler.sample()
        sample["operator_private"] = args.operator_private
        details = metadata.sample(sample)
        details["progress_health"] = health.sample(sample)
        history.add(sample)
        print(draw(sample, details))
        return 0

    feed = Feed(sampler, metadata, args.interval, health)
    feed.thread.start()
    shown, shown_details, last_generation, last_frame = {}, {}, -1, ""
    try:
        with terminal_mode() as fd:
            while True:
                sample, details, error, generation = feed.latest()
                if generation != last_generation:
                    sample = {**sample, "operator_private": args.operator_private}
                    history.add(sample)
                    last_generation = generation
                if not paused:
                    shown, shown_details = {**sample, "operator_private": args.operator_private}, details
                text = draw(shown, shown_details, error)
                if text != last_frame:
                    # Only changed rows travel over SSH; never clear between animation frames.
                    rows, previous_rows = text.splitlines(), last_frame.splitlines()
                    changes = []
                    for index in range(max(len(rows), len(previous_rows))):
                        row = rows[index] if index < len(rows) else ""
                        previous = previous_rows[index] if index < len(previous_rows) else None
                        if row != previous:
                            changes.append(f"\x1b[{index + 1};1H" + row + "\x1b[K")
                    sys.stdout.write("".join(changes))
                    sys.stdout.flush()
                    last_frame = text
                ready, _, _ = select.select([fd], [], [], 0.12)
                if ready:
                    key = os.read(fd, 32).decode(errors="ignore")
                    if "q" in key or "\x03" in key:
                        break
                    if tab != -1 and key in ("1", "2", "3", "4", "5"):
                        tab, offset = int(key) - 1, 0
                    elif tab != -1 and key == "\t":
                        tab, offset = (tab + 1) % len(V.TABS), 0
                    elif key in ("j", "\x1b[B"):
                        offset = min(10000, offset + 1)
                    elif key in ("k", "\x1b[A"):
                        offset = max(0, offset - 1)
                    elif key == " ":
                        paused = not paused
                        paused_history = copy.deepcopy(history) if paused else None
                frame += 1
    except KeyboardInterrupt:
        pass
    finally:
        feed.stop.set()
        feed.thread.join(timeout=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ---- text --------------------------------------------------------------------------------------
# The terminal view of a tracking summary: plain text, ANSI colour optional, no curses.
#
# The same summary the HTML page draws, as a few dense lines a person can leave running in a tmux pane.
# A value the records do not hold is printed as "not recorded".

NR = "not recorded"
_COLOUR = {
    R.UNFINALIZED: "33",
    R.FROZEN: "33",
    R.LIVE: "32",
    R.FINISHED: "32",
    R.STALLED: "1;31",
    R.STOPPED: "33",
    R.ENDED: "33",
    R.RELAUNCHED: "36",
    R.UNKNOWN: "2",
}
CLEAR = "\x1b[H\x1b[2J"


def _paint(text: str, code: str | None, colour: bool) -> str:
    return f"\x1b[{code}m{text}\x1b[0m" if colour and code else text


def _n(value: Any) -> str:
    return f"{value:,}" if isinstance(value, int) and not isinstance(value, bool) else NR


def _h(value: float | None) -> str:
    if value is None:
        return NR
    return f"{value:.1f} h" if value < 48 else f"{value / 24:.1f} d"


def _when(value: float | None) -> str:
    return R.stamp(value) or NR


def _state(liveness: Mapping[str, Any] | None, colour: bool) -> str:
    state = str((liveness or {}).get("state") or R.UNKNOWN)
    return f"{_paint(state, _COLOUR.get(state), colour)}  {(liveness or {}).get('detail') or ''}".rstrip()


def _reading(c: Mapping[str, Any]) -> str:
    if c.get("cycles"):
        return _n(c["cycles"])
    return _n(c["whole_window"]) + " wrong" if c.get("whole_window") else ""


def _phase2(p2: Mapping[str, Any], now: float, colour: bool) -> list[str]:
    counts, in_run = p2.get("counts") or {}, p2.get("counts_in_run")
    best, bar = p2.get("best") or {}, p2.get("bar") or {}
    store, board, run = p2.get("store") or {}, p2.get("board"), p2.get("run") or {}
    ratio = f" ({best['cycles'] / bar['cycles']:.3f}x bar)" if best.get("cycles") and bar.get("cycles") else ""
    roles = ", ".join(run["prohibited_roles"]) if run.get("prohibited_roles") else NR
    absent = "" if store.get("loaded") or not store.get("root") else " (absent)"
    this_run = f" (this run {in_run.get(R.MEASURED, 0)})" if in_run else ""
    last = _when((p2.get("liveness") or {}).get("last_measured"))
    lines = [
        "PHASE 2  " + _state(p2.get("liveness"), colour),
        f"  method {run.get('method') or NR}  prohibited roles {roles}",
        f"  store {store.get('root') or NR}{absent}",
        f"  candidates {len(p2.get('candidates') or [])}  measured {counts.get(R.MEASURED, 0)}{this_run}"
        f"  pending {counts.get(R.OPEN, 0)}",
        f"  best {_n(best.get('cycles'))}{ratio} {str(best.get('key') or '')[:12]}  bar {_n(bar.get('cycles'))}"
        f"  last measured {last}",
        "  failures  " + "  ".join(f"{name} {counts.get(name, 0)}" for name in R.FAILURE_CLASSES),
    ]
    plateau = p2.get("plateau")
    if plateau:
        rule, improved = plateau.get("rule") or {}, plateau.get("last_improvement") or {}
        anchor = R.epoch(improved.get("epoch")) or R.epoch(improved.get("at"))
        lines.append(
            f"  plateau  rule {rule.get('hours', NR)} h / {rule.get('min_sessions', NR)} sessions;"
            f" last improvement {_when(anchor)} ({_n(improved.get('cycles'))});"
            f" now {_h((now - anchor) / 3600 if anchor else None)}"
        )
    else:
        lines.append(f"  plateau  {NR}")
    if board:
        outage = board.get("open_outage")
        state = f"OPEN since {_when(outage['opened'])}" if outage else "none open"
        lines.append(
            f"  board  queue {board.get('queue', 0)} (oldest {_when(board.get('oldest_waiting'))})"
            f"  running {board.get('running', 0)}  outage {state}; {len(board.get('closed_outages') or [])} closed"
            f"  solo streak {_n(board.get('solo_streak'))}"
        )
    else:
        lines.append(f"  board  {NR}")
    recent = [c for c in p2.get("candidates") or () if c.get("finished")][-6:]
    if recent:
        lines.append("  recent:")
        for c in reversed(recent):
            note = (c.get("reason") or c.get("label") or "")[:60]
            lines.append(
                f"    {_when(c['finished'])}  {c['class']:<22} {_reading(c):>16}  {str(c['key'])[:12]}  {note}"
            )
    return lines


def _phase1(p1: Mapping[str, Any], colour: bool) -> list[str]:
    latest, plateau, freeze = p1.get("latest") or {}, p1.get("plateau"), p1.get("freeze")
    tiers = p1.get("tiers") or []
    lines = ["PHASE 1  " + _state(p1.get("liveness"), colour)]
    if latest:
        reached = {t: sum(1 for c in latest.get("capsules") or () if c["tiers"].get(t) == "pass") for t in tiers}
        lines.append(
            f"  latest grade {latest.get('name')} at {_when(latest.get('at'))}: "
            f"{_n(latest.get('n_passed'))}/{_n(latest.get('n_capsules'))} passed;"
            f" highest tier {latest.get('highest_tier') or NR}"
        )
        if reached:
            lines.append("  tier passes  " + "  ".join(f"{t} {n}" for t, n in reached.items()))
        planes = sorted((latest.get("first_failure_planes") or {}).items(), key=lambda kv: -int(kv[1]))
        if planes:
            lines.append("  first failure planes  " + "  ".join(f"{k} {v}" for k, v in planes))
    else:
        lines.append(f"  grades  {NR}")
    lines.append(f"  grades recorded {len(p1.get('grades') or [])}")
    lines.append(f"  plateau  {(plateau or {}).get('sentence') or NR}")
    if freeze:
        lines.append(f"  freeze  frozen {_when(freeze['at'])} commit {str(freeze.get('frozen_commit') or NR)[:12]}")
    else:
        lines.append(f"  freeze  {NR}")
    for c in (p1.get("failing") or [])[:5]:
        detail = (c.get("failure_detail") or "")[:50]
        lines.append(f"    {c['capsule'][:48]:<48} {c['status']!s:<10} {c.get('failure_plane') or '':<18} {detail}")
    return lines


def render(summary: Mapping[str, Any], *, colour: bool = False) -> str:
    now = float(summary.get("generated") or 0.0)
    phases = ", ".join(summary.get("phases") or []) or NR
    lines = [
        f"{summary.get('run_id')}  target {summary.get('target') or NR}  phase {phases}  ({R.stamp(now)})",
        "STATE  " + _state(summary.get("liveness"), colour),
    ]
    orchestration = summary.get("orchestration")
    if orchestration:
        lines.append(
            f"ORCHESTRATION  {orchestration.get('experiment')}  state {orchestration.get('state')}"
            f"  plan {orchestration.get('plan_binding')}"
        )
    if summary.get("phase2"):
        lines += _phase2(summary["phase2"], now, colour)
    if summary.get("phase1"):
        lines += _phase1(summary["phase1"], colour)
    for number, engine in sorted((summary.get("engines") or {}).items()):
        lines.append(f"-- phase {number} engine output {engine.get('run_id')}")
        lines += render(engine, colour=colour).splitlines()[1:]
    if not summary.get("phase1") and not summary.get("phase2") and not summary.get("engines"):
        lines.append(f"no phase records: {NR}")
    absent = sum(1 for row in summary.get("inventory") or () if row["state"] != "read")
    lines.append(f"records: {len(summary.get('inventory') or [])} consulted, {absent} absent/unreadable")
    return "\n".join(lines) + "\n"


def agent(activity: Mapping[str, Any] | None, now: float) -> list[str]:
    """What the authoring agent is doing now, from its event stream (:mod:`.activity`)."""
    if not activity:
        return []
    current = activity["now"]
    last = current.get("last_event")
    age = f"{(now - last) / 60.0:.0f} min ago" if last is not None else NR
    lines = [
        f"AGENT  last event {R.stamp(last) or NR} ({age}); {activity['n_commands']} commands, "
        f"{activity['failed_commands']} non-zero"
    ]
    for call in current.get("running") or ():
        lines.append(f"  running  {call.get('label') or call['type']}: {str(call.get('command') or '')[:110]}")
    message = current.get("last_message")
    if message:
        lines.append(f"  said     {message['text'][:140]}")
    return lines


__all__ = [
    "CLEAR",
    "agent",
    "render",
]
