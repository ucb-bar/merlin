#!/usr/bin/env python3
"""Show the firesim-queue with MODEL NAMES resolved.

The native `firesim_queue.py status` labels every merlin job by its staged bootbinary
(`zephyr0-zephyr.elf`), so our model×dtype runs are indistinguishable. This reads the
queue DB and prints each job with the model bundle recovered from its `stage_from` path
(e.g. .../fs/groot_n1d7_fp32_consistent/build/zephyr/zephyr.elf -> groot_n1d7_fp32).

Usage:  .venv/bin/python build_tools/scripts/fsq.py [--all] [--mine]
"""
import argparse, json, re, sqlite3, time

DB = "/scratch2/agustin/firesim_queue/queue.db"
ACTIVE = ("RUNNING", "QUEUED")


def model_of(kind_args: str) -> str:
    m = re.search(r'"stage_from":\s*"([^"]+)"', kind_args or "")
    if not m:
        return "-"
    p = m.group(1)
    mm = re.search(r"/fs[x]?/([^/]+)/", p)               # our sweep workroots
    if mm:
        return mm.group(1).replace("_consistent", "")
    return p.split("/")[-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="show recent terminal jobs too")
    ap.add_argument("--mine", action="store_true", help="only jobs staged from our sweeps")
    args = ap.parse_args()
    c = sqlite3.connect(DB)
    hb = c.execute("SELECT value FROM kv WHERE key='daemon_heartbeat'").fetchone()
    age = (time.time() - float(hb[0])) if hb else None
    print(f"daemon: {'ALIVE' if age is not None and age < 120 else 'STALE/DOWN'}"
          + (f" (heartbeat {age:.0f}s ago)" if age is not None else ""))
    q = ("SELECT id,user,state,phase,started_at,ended_at,exit_code,kind_args FROM jobs "
         + ("WHERE state IN ('RUNNING','QUEUED') " if not args.all else "")
         + "ORDER BY id DESC " + ("LIMIT 20" if args.all else ""))
    rows = list(c.execute(q))
    if not args.all:
        rows = rows[::-1]                                # chronological for the live view
    print(f"{'id':>4} {'state':<9} {'phase':<9} {'wall':>7}  model")
    for r in rows:
        model = model_of(r[7])
        if args.mine and model == "-":
            continue
        end = r[5] or time.time()
        wall = f"{end - r[4]:.0f}s" if r[4] else "-"
        print(f"{r[0]:>4} {r[2]:<9} {r[3] or '':<9} {wall:>7}  {model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
