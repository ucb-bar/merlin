#!/usr/bin/env python3
"""Sample one process tree's CPU, RSS, virtual memory, threads, and procfs I/O as JSONL."""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path


def _proc_table() -> dict[int, tuple[int, list[str]]]:
    table = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().split()
            table[int(entry.name)] = (int(fields[3]), fields)
        except (OSError, ValueError, IndexError):
            continue
    return table


def _tree(root: int, table: dict[int, tuple[int, list[str]]]) -> set[int]:
    selected, frontier = {root}, [root]
    while frontier:
        parent = frontier.pop()
        children = [pid for pid, (ppid, _fields) in table.items()
                    if ppid == parent and pid not in selected]
        selected.update(children)
        frontier.extend(children)
    return selected


def sample(root: int) -> dict | None:
    table = _proc_table()
    if root not in table:
        return None
    pids = _tree(root, table)
    ticks = os.sysconf("SC_CLK_TCK")
    page = os.sysconf("SC_PAGE_SIZE")
    totals = {"user_cpu_s": 0.0, "system_cpu_s": 0.0, "rss_bytes": 0,
              "virtual_bytes": 0, "threads": 0, "read_bytes": 0, "write_bytes": 0}
    for pid in pids:
        try:
            fields = table[pid][1]
            totals["user_cpu_s"] += int(fields[13]) / ticks
            totals["system_cpu_s"] += int(fields[14]) / ticks
            totals["virtual_bytes"] += int(fields[22])
            totals["rss_bytes"] += int(fields[23]) * page
            totals["threads"] += int(fields[19])
            io = {}
            for line in (Path("/proc") / str(pid) / "io").read_text().splitlines():
                key, _, value = line.partition(":")
                io[key] = int(value.strip())
            totals["read_bytes"] += io.get("read_bytes", 0)
            totals["write_bytes"] += io.get("write_bytes", 0)
        except (OSError, ValueError, IndexError):
            continue
    load = os.getloadavg()
    return {"sampled_at": datetime.now(timezone.utc).isoformat(), "root_pid": root,
            "processes": len(pids), **totals,
            "host_load_1m": load[0], "host_load_5m": load[1], "host_load_15m": load[2]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=5.0)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8", buffering=1) as out:
        while True:
            record = sample(args.pid)
            if record is None:
                break
            out.write(json.dumps(record, sort_keys=True) + "\n")
            time.sleep(max(args.interval, 0.25))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
