#!/usr/bin/env python3
"""Remote pilot: run every baseline valid.elf once across all slots to size the candidate watchdog settings.

Outside the tuning ledger (reported separately as pilot occupancy). Writes f2/pilot/summary.json.
"""
import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

EXP = Path("/work/exp/tvmfs")
sys.path.insert(0, os.environ["COMPARISON"])
from firesim_job_pool import FireSimPool  # noqa: E402

pool = FireSimPool(EXP / "f2/pool.json", deadline_epoch=time.time() + 4 * 3600)
out = EXP / "f2/pilot"
out.mkdir(exist_ok=True)
elfs = sorted((EXP / "work/baseline").glob("*/guest/valid.elf"))
big_first = sorted(elfs, key=lambda p: -p.stat().st_size)


def one(elf):
    name = elf.parent.parent.name
    t0 = time.monotonic()
    try:
        warm = pool.run(elf, out / name, wall_timeout=900)
        err = None
    except Exception as e:  # noqa: BLE001
        warm, err = None, repr(e)
    return dict(name=name, warm_cycles=warm, wall_seconds=time.monotonic() - t0, error=err,
                elf_bytes=elf.stat().st_size)


with ThreadPoolExecutor(len(pool.slots)) as ex:
    rows = list(ex.map(one, big_first))
occ = {}
for r in rows:
    f = out / r["name"] / "occupancy.json"
    if f.exists():
        r["occupancy"] = json.loads(f.read_text())
json.dump(rows, open(out / "summary.json", "w"), indent=1)
for r in sorted(rows, key=lambda r: r["name"]):
    print(r["name"], r["warm_cycles"], round(r["wall_seconds"], 1), r["error"])
