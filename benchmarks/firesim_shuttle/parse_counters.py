#!/usr/bin/env python3
"""Parse Gemmini per-dispatch counter lines from a FireSim uartlog.

The merlin_model_runner Zephyr worker emits one COUNTER line per dispatch
when built with MERLIN_PROFILE_COUNTERS=1. Format:

  COUNTER, begin
  COUNTER, <ordinal>, <symbol>, <slot0>, <slot1>, ..., <slot7>
  ...
  COUNTER, end

The 8 slots correspond to the panel defined in
  runtime/src/iree/hal/local/loaders/merlin_gemmini_counter.h
(MERLIN_GEMMINI_COUNTER_PANEL macro).

Output: CSV with columns (ordinal, symbol, slot_name, value).

Usage:
  parse_counters.py <uartlog> [--csv-out path] [--print-top N]
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

# Must match the panel in merlin_gemmini_counter.h.
SLOT_NAMES = [
    "MAIN_LD_CYCLES",  # 0
    "MAIN_EX_CYCLES",  # 1
    "MAIN_ST_CYCLES",  # 2
    "MAIN_LD_ST_EX_CYCLES",  # 3 — three-way overlap
    "LOAD_DMA_WAIT_CYCLE",  # 4
    "EXE_PRELOAD_HAZ_CYCLE",  # 5
    "RESERVATION_STATION_FULL_CYCLES",  # 6
    "WDMA_TL_WAIT_CYCLES",  # 7
]

_COUNTER_RE = re.compile(r"^COUNTER,\s*(\d+),\s*([^,]+)" + ",\\s*(\\d+)" * 8 + r"\s*$")


def parse(uartlog: Path):
    """Yield (ordinal:int, symbol:str, slot_name:str, value:int) tuples."""
    in_block = False
    for line in uartlog.read_text(errors="replace").splitlines():
        line = line.strip()
        if line == "COUNTER, begin":
            in_block = True
            continue
        if line == "COUNTER, end":
            in_block = False
            continue
        if not in_block:
            continue
        m = _COUNTER_RE.match(line)
        if not m:
            continue
        ordinal = int(m.group(1))
        sym = m.group(2).strip()
        slots = [int(m.group(3 + i)) for i in range(8)]
        # Skip all-zero rows — emitted when MERLIN_PROFILE_COUNTERS is unset
        # (the dump still runs but slots stay at their zero init values).
        if all(v == 0 for v in slots):
            continue
        for slot_idx, value in enumerate(slots):
            yield ordinal, sym, SLOT_NAMES[slot_idx], value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("uartlog", type=Path)
    p.add_argument("--csv-out", type=Path, help="Write CSV to this path (otherwise stdout).")
    p.add_argument(
        "--print-top",
        type=int,
        default=0,
        help="Print top N dispatches by MAIN_LD_CYCLES + MAIN_EX_CYCLES + MAIN_ST_CYCLES.",
    )
    args = p.parse_args()

    rows = list(parse(args.uartlog))
    if not rows:
        print(
            f"[parse_counters] no COUNTER lines found in {args.uartlog} "
            "(MERLIN_PROFILE_COUNTERS not enabled? Or no dispatches ran?)",
            file=sys.stderr,
        )
        sys.exit(1)

    # CSV out
    sink = args.csv_out.open("w") if args.csv_out else sys.stdout
    try:
        w = csv.writer(sink)
        w.writerow(["ordinal", "symbol", "slot_name", "value"])
        for r in rows:
            w.writerow(r)
    finally:
        if args.csv_out:
            sink.close()

    if args.print_top > 0:
        # Aggregate by (ordinal, symbol) for the top-N print.
        per_dispatch = {}
        for ordinal, sym, slot_name, value in rows:
            key = (ordinal, sym)
            d = per_dispatch.setdefault(key, {})
            d[slot_name] = value
        scored = []
        for (ordinal, sym), slots in per_dispatch.items():
            total = slots.get("MAIN_LD_CYCLES", 0) + slots.get("MAIN_EX_CYCLES", 0) + slots.get("MAIN_ST_CYCLES", 0)
            scored.append((total, ordinal, sym, slots))
        scored.sort(reverse=True)
        print(f"\nTop {args.print_top} dispatches by total LD+EX+ST cycles:", file=sys.stderr)
        print(
            f"{'ord':>4} {'symbol':<70} {'LD':>10} {'EX':>10} {'ST':>10} "
            f"{'overlap':>10} {'DMA_wait':>10} {'PRELOAD_haz':>12} "
            f"{'RS_full':>10} {'WDMA_wait':>10}",
            file=sys.stderr,
        )
        for total, ordinal, sym, slots in scored[: args.print_top]:
            print(
                f"{ordinal:>4} {sym[:70]:<70} "
                f"{slots.get('MAIN_LD_CYCLES', 0):>10} "
                f"{slots.get('MAIN_EX_CYCLES', 0):>10} "
                f"{slots.get('MAIN_ST_CYCLES', 0):>10} "
                f"{slots.get('MAIN_LD_ST_EX_CYCLES', 0):>10} "
                f"{slots.get('LOAD_DMA_WAIT_CYCLE', 0):>10} "
                f"{slots.get('EXE_PRELOAD_HAZ_CYCLE', 0):>12} "
                f"{slots.get('RESERVATION_STATION_FULL_CYCLES', 0):>10} "
                f"{slots.get('WDMA_TL_WAIT_CYCLES', 0):>10}",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
