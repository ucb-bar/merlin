"""``merlin-experiment console``: start the frontend on a terminal, or print text everywhere else."""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from argparse import Namespace

from . import launch, placement, plain
from .discovery import Discovery, default_selection
from .model import REFRESH_BUDGET, Collector

SLOW_INTERVAL_S = 8.0


def _select(items: list[dict], wanted: str | None) -> dict | None:
    if wanted:
        for item in items:
            if wanted in (item["id"], item["run_id"]):
                return item
        raise SystemExit(f"no experiment matches {wanted!r}; try --list")
    return default_selection(items)


def run(args: Namespace) -> int:
    color = not args.no_color and "NO_COLOR" not in os.environ and sys.stdout.isatty()
    width = args.width or shutil.get_terminal_size((160, 40)).columns
    discovery = Discovery(args.tasks_root, tasks=args.task, extra_runs=args.run_dir)
    if args.list:
        items = discovery.scan()
        print(
            json.dumps(items, indent=1, default=str)
            if args.json
            else plain.experiments_text(items, color=color, width=width)
        )
        return 0
    if args.build:
        print(f"built {launch.build()}", file=sys.stderr)
    interactive = sys.stdin.isatty() and sys.stdout.isatty() and not args.once and not args.plain
    binary = launch.binary() if interactive else None
    if binary is not None:
        return launch.exec_frontend(
            launch.frontend_argv(
                binary,
                operator_private=args.operator_private,
                tasks=args.task,
                run_dirs=args.run_dir,
                experiment=args.experiment,
                tasks_root=args.tasks_root,
                interval=args.interval,
            )
        )
    items = discovery.scan()
    item = _select(items, args.experiment)
    if item is None:
        print(plain.experiments_text(items, color=color, width=width))
        return 1
    once = args.once or not interactive
    # A one-shot snapshot reads every record now; the interactive loop keeps its per-refresh budget.
    collector = Collector(item, operator_private=args.operator_private, budget=None if once else REFRESH_BUDGET)
    if once:
        snapshot = collector.refresh("all")
        if args.json:
            print(json.dumps(snapshot, indent=1, default=str))
        else:
            print(plain.render(snapshot, color=color, width=width))
        return 0
    if not args.plain:
        print(
            "merlin-console is not built; showing plain text. Build it once with: merlin-experiment console"
            " --build (needs Go >= 1.26; see docs/guides/experiment_console.md)",
            file=sys.stderr,
        )
        time.sleep(1.5)
    # The text loop follows the same rules as the data process: off the experiment's CPUs, catalogs
    # on the slow cadence.
    placement.place_console()
    next_slow = 0.0
    try:
        while True:
            now = time.monotonic()
            tier = "all" if now >= next_slow else "fast"
            if tier == "all":
                next_slow = now + SLOW_INTERVAL_S
            snapshot = collector.refresh(tier)
            sys.stdout.write("\x1b[H\x1b[2J" + plain.render(snapshot, color=color, width=width) + "\n")
            sys.stdout.flush()
            time.sleep(args.interval or 2.0)
    except KeyboardInterrupt:
        return 0


__all__ = ["run"]
