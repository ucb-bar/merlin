"""The console's data process: JSON lines out, commands in. The Bubble Tea frontend spawns it.

Run ``python -m merlin_experiments.tracking.console.serve [--operator-private] [--task T ...]``.

Messages on stdout, one JSON object per line:

* ``{"type": "hello", "schema", "operator_private", "pid"}``
* ``{"type": "experiments", "schema": "merlin.console.experiments.v1", "items": [...], "selected", "notices"}``
* ``{"type": "snapshot", "full": bool, "console": {...}, ...snapshot}`` -- ``sections`` holds only the
  sections whose content changed since the last message for this experiment (all of them when
  ``full``); ``console`` is the console's own cost and placement (see below)
* ``{"type": "inspect", "request_id", "kind", "key", "title", "sections", "paths"}``
* ``{"type": "error", "message"}``

Commands on stdin, one JSON object per line: ``{"cmd": "select", "id": <experiment id>}``,
``{"cmd": "inspect", "request_id": n, "kind": <source>, "key": <row key>}``, ``{"cmd": "refresh"}``,
``{"cmd": "quit"}``. ``--dump`` prints one full snapshot and exits; ``--list`` prints the experiments.

The process only reads records. It writes nothing but its own stdout.

The console must never compete with the experiment it watches. On start and every
:data:`PLACE_EVERY_S` seconds it moves itself and its frontend to CPUs no live experiment holds
(:func:`.placement.place_console`, ``nice`` 10, ``SCHED_IDLE`` when every CPU is held), and it
measures its own CPU use: above :data:`CPU_BUDGET_PCT` of one core the fast interval doubles (up to
:data:`MAX_INTERVAL_S`) and recovers when the cost falls. ``console`` in every snapshot reports
``{cpu_pct, rss_bytes, cpus, avoided, policy, shared, interval_s, throttled}``. ``--no-placement``
leaves scheduling alone.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any

from . import EXPERIMENTS_SCHEMA, SNAPSHOT_SCHEMA, placement
from .discovery import Discovery, default_selection
from .io import Reader
from .model import Collector

LIVE_STATES = ("LIVE", "STALLED", "ACTIVE")
PLACE_EVERY_S = 30.0
#: The console's own CPU budget, in percent of one core (data process plus frontend).
CPU_BUDGET_PCT = 15.0
MAX_INTERVAL_S = 10.0


def _finite(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_finite(v) for v in value]
    return value


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--tasks-root", type=Path, help="Directory holding the task folders (default: derived)")
    p.add_argument("--task", type=Path, action="append", help="Limit discovery to this task folder (repeatable)")
    p.add_argument("--run-dir", type=Path, action="append", help="Also show this run or record directory (repeatable)")
    p.add_argument("--experiment", help="Open this experiment first (run id or run directory)")
    p.add_argument("--operator-private", action="store_true", help="Show capsule names, paths and golden references")
    p.add_argument("--interval", type=float, default=1.0, help="Fast refresh seconds for a running experiment")
    p.add_argument("--slow-interval", type=float, default=8.0, help="Slow refresh seconds (catalogs, perf, passes)")
    p.add_argument("--discovery-interval", type=float, default=15.0, help="Seconds between experiment scans")
    p.add_argument("--dump", action="store_true", help="Print one full snapshot of the selected experiment and exit")
    p.add_argument("--list", action="store_true", help="Print the discovered experiments and exit")
    p.add_argument("--no-placement", action="store_true", help="Do not move or deprioritize the console processes")
    return p


def _incarnation(item: dict[str, Any]) -> tuple[int, int] | None:
    """The identity of the run directory itself; a relaunch that reuses the path gets a new one."""
    run_dir = item.get("run_dir")
    try:
        info = os.stat(run_dir) if run_dir else None
    except OSError:
        return None
    return None if info is None else (info.st_dev, info.st_ino)


class Server:
    def __init__(self, args: argparse.Namespace, out=sys.stdout) -> None:
        self.args = args
        self.out = out
        self.lock = threading.Lock()
        self.discovery = Discovery(args.tasks_root, tasks=args.task, extra_runs=args.run_dir)
        self.experiments: list[dict[str, Any]] = []
        self.collector: Collector | None = None
        self.incarnation: tuple[int, int] | None = None
        self.sent: dict[str, str] = {}
        self.sent_objects: dict[str, Any] = {}
        self.commands: queue.Queue = queue.Queue()
        self.cost = placement.SelfCost()
        self.placement: dict[str, Any] = {}
        self.next_place = 0.0
        self.scale = 1.0  # Fast-interval multiplier while over the CPU budget.
        self.cost_now: dict[str, Any] = {}

    def place(self, now: float) -> None:
        if self.args.no_placement or now < self.next_place:
            return
        self.next_place = now + PLACE_EVERY_S
        try:
            self.placement = placement.place_console()
        except Exception as exc:  # noqa: BLE001 -- placement is best effort, never fatal
            self.placement = {"error": f"{type(exc).__name__}: {exc}"}

    def measure(self, interval: float) -> dict[str, Any]:
        pids = self.placement.get("pids") or [os.getpid()]
        cost = self.cost.sample(pids)
        pct = cost.get("cpu_pct")
        if isinstance(pct, float):
            if pct > CPU_BUDGET_PCT:
                self.scale = min(self.scale * 2, MAX_INTERVAL_S / max(interval, 0.2))
            elif pct < CPU_BUDGET_PCT / 3 and self.scale > 1:
                self.scale = max(1.0, self.scale / 2)
        keep = ("cpus", "avoided", "policy", "shared", "error")
        self.cost_now = {
            **cost,
            **{k: self.placement.get(k) for k in keep if k in self.placement},
            "interval_s": round(interval * self.scale, 2),
            "throttled": self.scale > 1,
            "budget_pct": CPU_BUDGET_PCT,
        }
        return self.cost_now

    def emit(self, message: dict[str, Any]) -> None:
        try:
            text = json.dumps(message, separators=(",", ":"), default=str, allow_nan=False)
        except ValueError:
            # A non-finite number from some record is "not recorded", never NaN on the wire.
            text = json.dumps(_finite(message), separators=(",", ":"), default=str, allow_nan=False)
        with self.lock:
            self.out.write(text + "\n")
            self.out.flush()

    def scan(self) -> None:
        self.experiments = self.discovery.scan()
        if self.collector is not None:
            # A relaunch under the same run id replaces the run directory at the same path; follow the
            # new run instead of the caches, tails and grade history of the one that was moved away.
            current = next((item for item in self.experiments if item["id"] == self.collector.experiment["id"]), None)
            if current is not None and _incarnation(current) != self.incarnation:
                self.select(current["id"])
        if self.collector is not None:
            for item in self.experiments:
                self.collector.update_experiment(item)
        self.emit(
            {
                "type": "experiments",
                "schema": EXPERIMENTS_SCHEMA,
                "items": self.experiments,
                "selected": self.collector.experiment["id"] if self.collector else None,
                "notices": self.discovery.notices,
                "scanned_at": time.time(),
            }
        )

    def find(self, wanted: str | None) -> dict[str, Any] | None:
        if wanted:
            for item in self.experiments:
                if wanted in (item["id"], item["run_id"]):
                    return item
        return default_selection(self.experiments)

    def select(self, wanted: str | None) -> None:
        item = self.find(wanted)
        if item is None:
            self.collector = None
            return
        incarnation = _incarnation(item)
        if (
            self.collector is None
            or self.collector.experiment["id"] != item["id"]
            or self.incarnation != incarnation
        ):
            self.collector = Collector(item, operator_private=self.args.operator_private)
            self.sent, self.sent_objects = {}, {}
            self.incarnation = incarnation

    def snapshot(self, tier: str) -> None:
        if self.collector is None:
            return
        snap = self.collector.refresh(tier)
        full = not self.sent
        sections = {}
        for name, section in snap["sections"].items():
            if not full and self.sent_objects.get(name) is section:
                continue  # A source that returned its memoized section has nothing new.
            self.sent_objects[name] = section
            text = json.dumps(section, sort_keys=True, default=str)
            if full or self.sent.get(name) != text:
                sections[name] = section
                self.sent[name] = text
        self.emit({**snap, "type": "snapshot", "full": full, "sections": sections, "console": self.cost_now})
        if full:
            # The first full read is a one-off burst; the reported cost starts after it.
            self.cost.sample(self.placement.get("pids") or [os.getpid()])
            self.cost_now = {**self.cost_now, "cpu_pct": None}

    def inspect(self, command: dict[str, Any]) -> None:
        if self.collector is None:
            return
        kind, key = str(command.get("kind") or ""), str(command.get("key") or "")
        result = self.collector.inspect(kind, key)
        self.emit({"type": "inspect", "request_id": command.get("request_id"), "kind": kind, "key": key, **result})

    def read_commands(self, stream) -> None:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                command = json.loads(line)
            except ValueError:
                self.emit({"type": "error", "message": "unreadable command line"})
                continue
            if isinstance(command, dict):
                self.commands.put(command)
        self.commands.put({"cmd": "quit"})

    def run(self) -> int:
        self.emit(
            {
                "type": "hello",
                "schema": SNAPSHOT_SCHEMA,
                "operator_private": self.args.operator_private,
                "pid": os.getpid(),
            }
        )
        self.place(time.monotonic())
        self.experiments = self.discovery.scan()
        self.select(self.args.experiment)
        self.scan()
        threading.Thread(target=self.read_commands, args=(sys.stdin,), name="console-commands", daemon=True).start()
        next_fast = next_slow = 0.0
        next_scan = time.monotonic() + self.args.discovery_interval
        while True:
            now = time.monotonic()
            live = self.collector is not None and self.collector.experiment.get("state") in LIVE_STATES
            base = self.args.interval if live else max(self.args.interval, 5.0)
            self.place(now)
            fast_interval = base * self.scale
            if now >= min(next_fast, next_slow):
                self.measure(base)
            if self.collector is not None and now >= next_slow:
                self.snapshot("all")
                next_slow = now + self.args.slow_interval * self.scale * (1 if live else 4)
                next_fast = now + fast_interval
            elif self.collector is not None and now >= next_fast:
                self.snapshot("fast")
                next_fast = now + fast_interval
            if now >= next_scan:
                self.scan()
                next_scan = now + self.args.discovery_interval
            wait = max(0.05, min(next_fast, next_slow, next_scan) - time.monotonic())
            try:
                command = self.commands.get(timeout=wait)
            except queue.Empty:
                continue
            name = command.get("cmd")
            if name == "quit":
                return 0
            if name == "select":
                self.select(str(command.get("id") or ""))
                self.scan()
                next_slow = 0.0
            elif name == "inspect":
                self.inspect(command)
            elif name == "refresh":
                next_slow = 0.0
                next_scan = 0.0


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not 0.2 <= args.interval <= 60 or not 1 <= args.slow_interval <= 600:
        raise SystemExit("--interval must be 0.2-60 s and --slow-interval 1-600 s")
    server = Server(args)
    if args.list:
        print(json.dumps(server.discovery.scan(), indent=1, default=str))
        return 0
    if args.dump:
        # One refresh, once: read every record rather than leave panels waiting for a next refresh.
        server.discovery.reader = Reader(budget=None)
        server.experiments = server.discovery.scan()
        server.select(args.experiment)
        if server.collector is None:
            raise SystemExit("no experiment found")
        server.collector = Collector(server.collector.experiment, operator_private=args.operator_private, budget=None)
        print(json.dumps(server.collector.refresh("all"), indent=1, default=str))
        return 0
    try:
        return server.run()
    except (BrokenPipeError, KeyboardInterrupt):
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
